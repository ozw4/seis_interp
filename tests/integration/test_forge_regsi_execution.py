"""FORGE ReGSI runs with the exact execution switches reproduce the default run."""

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from seis_interp.data.forge_mvp_artifacts import load_model_inputs
from seis_interp.pipelines.prepare_forge_mvp import prepare_forge_mvp
from seis_interp.training.forge_mvp_methods import run_method
from tests.fixtures.forge_mvp import tiny_forge_study

_EXACT = {
    "neighbor_table": True,
    "prefetch": True,
    "deferred_checks": True,
    "foreach_ema": True,
    "shared_prediction": True,
}


@pytest.fixture
def preparation(tmp_path, monkeypatch):
    study, _, _, configs, *_ = tiny_forge_study(tmp_path)
    for name in ("regsi_real", "regsi_grid"):
        configs[name]["training"].update(max_steps=5, report_interval=2)
        (study / f"{name}.yaml").write_text(yaml.safe_dump(configs[name]))
    monkeypatch.setattr(
        "seis_interp.pipelines.prepare_forge_mvp.subprocess.check_output",
        lambda args, **kwargs: "testcommit\n" if args[-1] == "HEAD" else "",
    )
    script = Path(__file__).resolve().parents[2] / "scripts/forge_matched_tuning.py"
    (tmp_path / "scripts" / script.name).write_bytes(script.read_bytes())
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield prepare_forge_mvp(tmp_path, study)
    torch.set_num_threads(previous)


def _run(preparation, method, execution):
    inputs, config = load_model_inputs(preparation, method)
    config["training"].update(
        ema_decay=0.8, ema_eval_decays=[0.5, 0.8, 0.95], ema_snapshot_updates=[3]
    )
    if execution is not None:
        config["training"]["execution"] = execution
    return run_method(inputs, config, reporter=lambda message: None)


def _without_timings(history):
    return [{k: v for k, v in row.items() if not k.endswith("seconds")} for row in history]


@pytest.mark.parametrize("method", ["regsi_real", "regsi_grid"])
def test_exact_execution_switches_reproduce_predictions_shadows_and_states(preparation, method):
    baseline = _run(preparation, method, None)
    result = _run(preparation, method, _EXACT)
    np.testing.assert_array_equal(result.prediction, baseline.prediction)
    assert list(result.shadow_predictions) == list(baseline.shadow_predictions)
    for key, value in baseline.shadow_predictions.items():
        np.testing.assert_array_equal(result.shadow_predictions[key], value)
    assert _without_timings(result.history) == _without_timings(baseline.history)
    for name, value in baseline.checkpoint["model_state_dict"].items():
        assert torch.equal(result.checkpoint["model_state_dict"][name], value)
    assert result.ema_states.keys() == baseline.ema_states.keys()
    for key, by_update in baseline.ema_states.items():
        assert by_update.keys() == result.ema_states[key].keys()
        for update, state in by_update.items():
            assert all(torch.equal(result.ema_states[key][update][k], v) for k, v in state.items())
    for name in ("final_training_loss", "parameter_count", "initialization_hash"):
        assert result.diagnostics[name] == baseline.diagnostics[name]
    assert result.diagnostics["runtime_prediction_seconds"] > 0
    assert result.diagnostics["runtime_shadow_prediction_seconds"] > 0


def test_invalid_execution_is_rejected_before_training(preparation):
    with pytest.raises(ValueError, match="training.execution"):
        _run(preparation, "regsi_real", {"prefetch": "yes"})


def test_neighbor_table_with_brute_force_search_is_rejected_before_reading_data(
    preparation, monkeypatch
):
    inputs, config = load_model_inputs(preparation, "regsi_real")
    config["graph"]["neighbor_search"] = "brute_force"
    config["training"]["execution"] = {"neighbor_table": True}

    def unexpected_read(*args, **kwargs):
        raise AssertionError("graph data must not be read for an invalid execution")

    monkeypatch.setattr("seis_interp.training.forge_mvp_methods.graph_data", unexpected_read)
    with pytest.raises(ValueError, match="requires graph.neighbor_search=exact_index"):
        run_method(inputs, config, reporter=lambda message: None)


def test_cpu_threads_on_a_cpu_device_are_rejected_before_reading_data(preparation, monkeypatch):
    inputs, config = load_model_inputs(preparation, "regsi_real")
    assert config["device"] == "cpu"
    config["training"]["execution"] = {**_EXACT, "cpu_threads": 1}

    def unexpected_read(*args, **kwargs):
        raise AssertionError("graph data must not be read for an invalid execution")

    monkeypatch.setattr("seis_interp.training.forge_mvp_methods.graph_data", unexpected_read)
    with pytest.raises(ValueError, match="cpu_threads is not supported on a CPU device"):
        run_method(inputs, config, reporter=lambda message: None)
