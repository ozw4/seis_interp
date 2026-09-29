import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from seis_interp.data.forge_aligned_inputs import geometry_delays
from seis_interp.data.forge_mvp_artifacts import load_model_inputs, verify_hashes
from seis_interp.forge_mvp_config import validate_classical_execution
from seis_interp.pipelines.prepare_forge_mvp import prepare_forge_mvp
from seis_interp.pipelines.tune_forge_matched import (
    _classical_batched,
    matched_candidate_config,
    tune_forge_matched,
)
from seis_interp.processing.forge_mvp_contract import METHODS
from seis_interp.training.forge_mvp_methods import run_method
from seis_interp.training.forge_mvp_preflight import check_method_shapes
from tests.fixtures.forge_mvp import tiny_forge_study

VELOCITY = 40000.0
BUDGET = {
    "max_steps": 1,
    "learning_rate": 0.001,
    "loss_traces_per_update": 2,
    "parameter_count_target": 1300,
    "parameter_count_tolerance": 0.95,
}


@pytest.fixture
def aligned(tmp_path, monkeypatch):
    study, contract, _, configs, *_ = tiny_forge_study(tmp_path)
    contract["preprocessing"].update(
        amplitude_scaling="trace_oracle", trace_rms=True, alignment={"velocity_m_per_s": VELOCITY}
    )
    contract["matched_training"] = BUDGET
    (study / "config.yaml").write_text(yaml.safe_dump(contract))
    # Two loss traces per update for every neural method.
    configs["nersi_real"]["training"]["batch_size"] = 2
    configs["ccnet5d_grid"]["training"]["supervised_traces_per_update"] = 2
    for name in ("regsi_real", "regsi_grid"):
        # Seven observed traces: an episode hides round(7 * 0.5) = 4 >= two queries.
        configs[name]["training"].update(inner_mask_fraction=0.5, query_batch_size=2)
    for name in ("nersi_real", "ccnet5d_grid", "regsi_real", "regsi_grid"):
        (study / f"{name}.yaml").write_text(yaml.safe_dump(configs[name]))
    monkeypatch.setattr(
        "seis_interp.pipelines.prepare_forge_mvp.subprocess.check_output",
        lambda args, **kwargs: "testcommit\n" if args[-1] == "HEAD" else "",
    )
    monkeypatch.setattr(
        "seis_interp.pipelines.tune_forge_matched.execution_identity",
        lambda _: {"git_commit": "testcommit", "dirty_worktree": True},
    )
    script = Path(__file__).resolve().parents[2] / "scripts/forge_matched_tuning.py"
    (tmp_path / "scripts" / script.name).write_bytes(script.read_bytes())
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield tmp_path, prepare_forge_mvp(tmp_path, study), contract
    torch.set_num_threads(previous)


def test_preparation_shares_one_geometry_only_canvas_and_matched_counts(aligned):
    _, preparation, contract = aligned
    seal = json.loads((preparation / "preparation_manifest.json").read_text())
    inputs = {m: load_model_inputs(preparation, m) for m in ("regsi_real", "regsi_grid")}
    maximum = max(
        int(geometry_delays(i.geometry, VELOCITY, 0.001).max()) for i, _ in inputs.values()
    )
    expected = {"velocity_m_per_s": VELOCITY, "max_delay_samples": maximum}
    assert maximum > 0
    assert seal["alignment"] == expected
    assert seal["canvas_sample_count"] == contract["time_sample_count"] + maximum
    assert set(seal["parameter_counts"]) == {
        "ccnet5d_grid",
        "nersi_real",
        "regsi_real",
        "regsi_grid",
    }
    for method in METHODS:
        _, config = load_model_inputs(preparation, method)
        assert config["alignment"] == expected
    gates = json.loads((preparation / "preparation_gates.json").read_text())
    assert gates["shared_alignment"] == expected
    assert gates["matched_parameter_counts"] == seal["parameter_counts"]


def test_every_method_predicts_original_samples_from_the_aligned_canvas(aligned):
    _, preparation, _ = aligned
    for method in METHODS:
        inputs, config = load_model_inputs(preparation, method)
        result = run_method(inputs, config, reporter=lambda message: None)
        assert result.prediction.shape == (len(inputs.test_cells), len(inputs.time_s))
        assert result.prediction.dtype == np.float32
        assert np.isfinite(result.prediction).all()
        if result.checkpoint is not None:
            assert result.checkpoint["alignment"] == config["alignment"]
            assert result.diagnostics["alignment_canvas_sample_count"] == (
                len(inputs.time_s) + config["alignment"]["max_delay_samples"]
            )
        report = check_method_shapes(inputs, config)
        assert report["scope"]


def test_alignment_changes_the_classical_prediction_but_not_its_samples(aligned):
    _, preparation, _ = aligned
    inputs, config = load_model_inputs(preparation, "pocs_grid")
    shifted = run_method(inputs, config).prediction
    flat = run_method(inputs, {**config, "alignment": None}).prediction
    assert shifted.shape == flat.shape
    assert not np.array_equal(shifted, flat)
    batched = _classical_batched(inputs, config, {"device": "cpu"})
    np.testing.assert_allclose(batched.prediction, shifted, rtol=1e-5, atol=1e-6)


def test_matched_search_records_candidates_that_reproduce_run_method(aligned):
    root, preparation, _ = aligned
    candidate = {"name": "narrow", "model": {"encoder_width": 3}, "training": {"ema_decay": 0.9}}
    search = root / "search.yaml"
    search.write_text(
        yaml.safe_dump(
            {
                "method_id": "nersi_real",
                "target_relative_mse": 100.0,
                "stop_when_target_met": True,
                "device": "cpu",
                "candidates": [candidate, {"name": "must_not_run"}],
            }
        )
    )
    output = tune_forge_matched(root, preparation, search)
    record = json.loads((output / "manifest.json").read_text())
    verify_hashes(output, record["artifacts"])
    assert record["status"] == "completed"
    assert record["target_met"] is True
    assert record["independent_test_estimate"] is False
    assert record["alignment"]["velocity_m_per_s"] == VELOCITY
    assert not (output / "must_not_run").exists()
    inputs, config = load_model_inputs(preparation, "nersi_real")
    config["model"]["encoder_width"] = 3
    config["training"]["ema_decay"] = 0.9
    expected = run_method(inputs, config, reporter=lambda message: None)
    np.testing.assert_array_equal(np.load(output / "narrow/prediction.npy"), expected.prediction)
    summary = json.loads((output / "narrow/run_manifest.json").read_text())
    assert summary["parameter_count"] > 0


@pytest.mark.parametrize(
    "method,candidate",
    [
        ("nersi_real", {"training": {"learning_rate": 0.01}}),
        ("nersi_real", {"training": {"batch_size": 8}}),
        ("nersi_real", {"training": {"max_steps": 2}}),
        ("nersi_real", {"model": {"moveout_samples_per_radius": 10.0}}),
        ("nersi_real", {"model": {"encoder_width": 400}}),
        ("nersi_real", {"alignment": {"velocity_m_per_s": 100.0}}),
        ("regsi_real", {"training": {"query_batch_size": 4}}),
        ("regsi_real", {"training": {"optimizer": "adam"}}),
        ("regsi_real", {"training": {"inner_mask_fraction": 0.1}}),
        ("ccnet5d_grid", {"training": {"supervised_traces_per_update": 4}}),
        ("ccnet5d_grid", {"patches": {"shape": [4, 2, 3, 2, 3]}}),
        ("drr_grid", {"algorithm": {"frequency_min_hz": 5.0}}),
        ("pocs_grid", {"model": {}}),
    ],
)
def test_matched_search_rejects_changes_fixed_by_the_comparison(aligned, method, candidate):
    _, preparation, _ = aligned
    _, base = load_model_inputs(preparation, method)
    original = deepcopy(base)
    seal = json.loads((preparation / "preparation_manifest.json").read_text())
    with pytest.raises(ValueError):
        matched_candidate_config(
            base,
            {"name": "bad", **candidate},
            "cpu",
            BUDGET,
            seal["canvas_sample_count"],
            seal["observed_trace_count"],
        )
    assert base == original


@pytest.mark.parametrize(
    "method,execution",
    [
        ("pocs_grid", {"backend": "batched"}),
        ("drr_grid", {"backend": "batched", "workers": 2}),
    ],
)
def test_batched_cpu_execution_reproduces_the_sequential_classical_prediction(
    aligned, method, execution
):
    _, preparation, _ = aligned
    inputs, config = load_model_inputs(preparation, method)
    sequential = run_method(inputs, config).prediction
    batched = run_method(inputs, {**config, "execution": execution}).prediction
    np.testing.assert_allclose(batched, sequential, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize(
    "execution",
    [
        {"backend": "gpu"},
        {"backend": "sequential", "workers": 2},
        {"backend": "batched", "workers": 0},
        {"backend": "batched", "threads": 2},
    ],
)
def test_invalid_classical_execution_is_rejected(execution):
    config = {"method_id": "drr_grid", "device": "cpu", "execution": execution}
    with pytest.raises(ValueError):
        validate_classical_execution(config)
