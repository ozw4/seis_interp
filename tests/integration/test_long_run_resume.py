"""Exact recovery through the method adapters shared by both production runners."""

import random
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch

from seis_interp.data.forge_mvp_artifacts import ForgeModelInputs
from seis_interp.training import ccnet5d_observed_training, forge_mvp_methods
from seis_interp.training import relational_trace_graph_poc_trainer as graph_trainer
from seis_interp.training.c3_region_methods import run_c3_region_method
from seis_interp.training.exact_resume import ExactResume
from seis_interp.training.forge_c3_methods import run_forge_c3_method
from tests.fixtures.c3_fair_grid import tiny_c3_configs


class Interrupted(Exception):
    pass


class StopAt100(ExactResume):
    def save(self, step, *args, **kwargs):
        super().save(step, *args, **kwargs)
        if step == 100:
            raise Interrupted


@pytest.fixture
def tiny_inputs():
    previous = torch.get_num_threads()
    previous_mkldnn = torch.backends.mkldnn.enabled
    torch.backends.mkldnn.enabled = False
    torch.set_num_threads(1)
    shape = (2, 3, 2, 3)
    indices = np.array(list(np.ndindex(shape)))
    geometry = pd.DataFrame({"cell_id": np.arange(len(indices))})
    for axis, (role, name) in enumerate(
        [(role, name) for role in ("source", "receiver") for name in ("line", "point")]
    ):
        geometry[f"{role}_grid_{name}"] = indices[:, axis]
        geometry[f"{role}_{name}_continuous"] = indices[:, axis] + 0.03
    for role in ("source", "receiver"):
        geometry[f"{role}_x_m"] = geometry[f"{role}_grid_point"] * 50.0
        geometry[f"{role}_y_m"] = geometry[f"{role}_grid_line"] * 200.0
    observed = np.arange(25)
    inputs = ForgeModelInputs(
        geometry=geometry,
        observed_cells=observed,
        test_cells=np.arange(25, 36),
        excluded_cells=np.array([], dtype=np.int64),
        observed_values=np.random.default_rng(9).normal(size=(25, 9)).astype(np.float32),
        spatial_shape=shape,
        time_s=np.arange(9) * 0.008,
        global_rms=1.0,
        binding={"fixture": "exact_resume"},
    )
    yield inputs
    torch.set_num_threads(previous)
    torch.backends.mkldnn.enabled = previous_mkldnn


def configuration(method, precision, device):
    config = tiny_c3_configs()[method]
    config["device"] = device
    config["training"].update(
        max_steps=200,
        report_interval=37,
        optimizer="adamw",
        mixed_precision=precision,
        ema_decay=0.999,
        ema_eval_decays=[0.999, 0.9995, 0.9998],
        ema_snapshot_updates=[73, 150],
    )
    if method == "ccnet5d_grid":
        config["patches"]["shape"] = [9, 2, 2, 2, 2]
        config["prediction"]["core_shape"] = [9, 2, 2, 2, 2]
    if method == "regsi_real":
        config["training"]["execution"] = {
            "neighbor_table": True,
            "prefetch": True,
            "deferred_checks": True,
            "foreach_ema": True,
            "shared_prediction": True,
        }
    if method == "nersi_profile":
        config["training"]["coordinate_jitter_cells"] = 0.25
    return config


def assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left.cpu(), right.cpu())
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, np.random.Generator):
        assert_equal(left.bit_generator.state, right.bit_generator.state)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            if not str(key).endswith("seconds"):
                assert_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_equal(a, b)
    elif hasattr(left, "__dict__"):
        assert_equal(vars(left), vars(right))
    else:
        assert left == right


@pytest.mark.parametrize("dataset", ["c3", "forge"])
@pytest.mark.parametrize("method", ["regsi_real", "ccnet5d_grid", "nersi_profile"])
@pytest.mark.parametrize("precision", ["off", "bf16"])
def test_200_equals_100_resume_100(tmp_path, monkeypatch, tiny_inputs, dataset, method, precision):
    # Exercise CPU bf16 arithmetic while retaining the public CUDA-only AMP contract.
    for module in (ccnet5d_observed_training, graph_trainer, forge_mvp_methods):
        monkeypatch.setattr(
            module,
            "mixed_precision_dtype",
            lambda mode, _: torch.bfloat16 if mode == "bf16" else None,
        )
    run = run_c3_region_method if dataset == "c3" else run_forge_c3_method
    config = configuration(method, precision, "cpu")
    paths = [tmp_path / "straight.pt", tmp_path / "split.pt"]
    random.seed(51)
    np.random.seed(52)
    straight = run(
        tiny_inputs, deepcopy(config), lambda _: None, recovery=ExactResume(paths[0], updates=100)
    )
    random.seed(51)
    np.random.seed(52)
    with pytest.raises(Interrupted):
        run(
            tiny_inputs, deepcopy(config), lambda _: None, recovery=StopAt100(paths[1], updates=100)
        )
    split = run(
        tiny_inputs, deepcopy(config), lambda _: None, recovery=ExactResume(paths[1], updates=100)
    )
    assert_equal(straight.prediction, split.prediction)
    assert_equal(straight.checkpoint, split.checkpoint)
    assert_equal(straight.shadow_predictions, split.shadow_predictions)
    assert_equal(straight.ema_states, split.ema_states)
    assert_equal(straight.history, split.history)
    a, b = [torch.load(path, weights_only=False) for path in paths]
    assert all(item["step"].is_cpu for item in b["optimizer"]["state"].values())
    assert a["phase"] == b["phase"] == "before_prediction"
    assert_equal(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("precision", ["off", "bf16"])
@pytest.mark.parametrize("method", ["regsi_real", "ccnet5d_grid", "nersi_profile"])
def test_cuda_exact_resume(tmp_path, tiny_inputs, method, precision):
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA bf16 unavailable")

    class CheckCPUStateResume(ExactResume):
        def load(self, model, optimizer):
            state = super().load(model, optimizer)
            assert all(item["step"].is_cpu for item in optimizer.state.values())
            return state

    config = configuration(method, precision, "cuda:0")
    paths = [tmp_path / "straight.pt", tmp_path / "split.pt"]
    run = run_c3_region_method
    straight = run(
        tiny_inputs, deepcopy(config), lambda _: None, recovery=ExactResume(paths[0], updates=100)
    )
    with pytest.raises(Interrupted):
        run(
            tiny_inputs, deepcopy(config), lambda _: None, recovery=StopAt100(paths[1], updates=100)
        )
    split = run(
        tiny_inputs,
        deepcopy(config),
        lambda _: None,
        recovery=CheckCPUStateResume(paths[1], updates=100),
    )
    assert_equal(straight.prediction, split.prediction)
    assert_equal(straight.checkpoint, split.checkpoint)


def test_cpu_patch_prefetch_keeps_weights_rng_and_draw_order_bitwise(tmp_path, tiny_inputs):
    config = configuration("ccnet5d_grid", "off", "cpu")
    config["training"].update(max_steps=17)
    config["training"].pop("ema_snapshot_updates")
    results, payloads = [], []
    for prefetch in (False, True):
        config["execution"] = {"patch_prefetch": prefetch}
        path = tmp_path / f"prefetch_{prefetch}.pt"
        results.append(
            run_c3_region_method(
                tiny_inputs, deepcopy(config), lambda _: None, recovery=ExactResume(path)
            )
        )
        payloads.append(torch.load(path, weights_only=False))
    assert_equal(results[0].prediction, results[1].prediction)
    assert_equal(results[0].history, results[1].history)
    assert_equal(results[0].checkpoint, results[1].checkpoint)
    assert_equal(payloads[0]["model"], payloads[1]["model"])
    assert_equal(payloads[0]["optimizer"], payloads[1]["optimizer"])
    assert_equal(payloads[0]["training"]["source"], payloads[1]["training"]["source"])
    assert_equal(
        payloads[0]["training"]["patches_per_update"], payloads[1]["training"]["patches_per_update"]
    )
