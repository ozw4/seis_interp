"""End-to-end observed-only GNN training and common dense-volume evaluation."""

import json
from dataclasses import replace

import numpy as np
import pytest
import torch
import yaml

from seis_interp.pipelines import interpolate_relational_trace_graph as pipeline
from seis_interp.training.amplitude_scaling import compute_observed_global_rms
from tests.fixtures.c3_poc_trace_graph import prepare_poc_trace_graph_inputs


def _run(tmp_path, paths, config, name="run"):
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config))
    output = tmp_path / name
    metrics = pipeline.interpolate_relational_trace_graph_run(
        config_path=path, output_dir=output, **paths
    )
    return metrics, output


@pytest.mark.parametrize("search", ["exact_index", "brute_force"])
def test_poc_graph_final_state_full_coverage_and_common_metrics(tmp_path, monkeypatch, search):
    inputs, config, paths = prepare_poc_trace_graph_inputs(tmp_path / "data")
    config["graph"]["neighbor_search"] = search
    original_predict = pipeline.predict_relational_trace_graph
    captured = []

    def predict(model, domain, preprocessing, **kwargs):
        assert domain.amplitudes_path is None
        assert domain.array_rows is None
        np.testing.assert_array_equal(
            domain.trace_ids[domain.query_mask],
            inputs.observed_volume.array_rows[inputs.observed_volume.evaluation_target_trace_mask],
        )
        checkpoint = torch.load(tmp_path / "run/final.pt", weights_only=True)
        for key, value in model.state_dict().items():
            assert torch.equal(value.cpu(), checkpoint["model_state_dict"][key])
        result = original_predict(model, domain, preprocessing, **kwargs)
        captured.append(result)
        return result

    def forbidden(*args, **kwargs):
        pytest.fail("PoC must not call a validation or graph-specific evaluator")

    monkeypatch.setattr(pipeline, "predict_relational_trace_graph", predict)
    monkeypatch.setattr(
        "seis_interp.training.relational_trace_graph_trainer._evaluate_validation", forbidden
    )
    monkeypatch.setattr(
        "seis_interp.evaluation.trace_graph_metrics.evaluate_trace_graph_prediction", forbidden
    )
    metrics, output = _run(tmp_path, paths, config)
    checkpoint = torch.load(output / "final.pt", weights_only=True)
    assert not (output / "artifacts/best.pt").exists()
    assert checkpoint["steps_completed"] == metrics["optimizer_updates"] == 3
    assert checkpoint["checkpoint_role"] == "final"
    assert checkpoint["loss"] == "masked_trace_relative_mse"
    assert checkpoint["inner_mask_fraction"] == 0.5
    assert not {"best_step", "validation_history"} & checkpoint.keys()
    volume = inputs.observed_volume
    scale = compute_observed_global_rms(volume.values, volume.observed_trace_mask)
    assert (
        checkpoint["normalization"]["scale"]
        == checkpoint["preprocessing"]["amplitude_scale"]
        == scale
    )
    assert checkpoint["inputs_lock"] == inputs.inputs_lock
    prediction = np.load(output / "prediction.npy")
    coverage = np.load(output / "artifacts/target_coverage.npy")
    ids = np.load(output / "artifacts/query_trace_ids.npy")
    np.testing.assert_array_equal(ids, volume.array_rows[volume.evaluation_target_trace_mask])
    np.testing.assert_array_equal(coverage, volume.evaluation_target_trace_mask)
    assert prediction.shape == volume.values.shape
    assert np.isfinite(prediction).all()
    np.testing.assert_array_equal(
        prediction[:, volume.observed_trace_mask], volume.values[:, volume.observed_trace_mask]
    )
    np.testing.assert_array_equal(
        prediction[:, volume.evaluation_target_trace_mask].T, captured[0].prediction
    )
    for key in ("snr_db", "rmse", "relative_l2", "mean_trace_relative_mse"):
        assert key in metrics["evaluation_target"]
    assert metrics["evaluation_target"]["trace_count"] == len(ids)
    run = json.loads((output / "metadata.json").read_text())
    assert run["status"] == "success"
    assert run["coverage"]["covered_target_trace_count"] == len(ids)
    assert run["method_details"]["parameter_count"] > 0
    assert run["training_or_reconstruction"]["steps_completed"] == 3
    assert run["timing"]["training_seconds"] > 0
    assert (
        json.loads((output / "metrics.json").read_text())["evaluation_target"]
        == metrics["evaluation_target"]
    )
    with pytest.raises(FileExistsError, match="already exists"):
        _run(tmp_path, paths, config)


def test_target_truth_changes_only_metrics_not_training_or_pre_evaluation_prediction(
    tmp_path, monkeypatch
):
    original_evaluate = pipeline.evaluate_c3_volume_prediction
    captures = []

    def evaluate(values, *args, **kwargs):
        captures.append(values.copy())
        return original_evaluate(values, *args, **kwargs)

    monkeypatch.setattr(pipeline, "evaluate_c3_volume_prediction", evaluate)
    states, scores, histories = [], [], []
    for name, offset in (("base", 0.0), ("changed", 1000000.0)):
        _, config, paths = prepare_poc_trace_graph_inputs(tmp_path / name, target_offset=offset)
        metrics, output = _run(tmp_path, paths, config, name=f"{name}_run")
        states.append(torch.load(output / "final.pt", weights_only=True)["model_state_dict"])
        scores.append(metrics["evaluation_target"])
        histories.append(
            [
                {k: v for k, v in row.items() if k != "seconds"}
                for row in metrics["training"]["history"]
            ]
        )
    assert states[0].keys() == states[1].keys()
    assert all(torch.equal(states[0][key], states[1][key]) for key in states[0])
    assert histories[0] == histories[1]
    np.testing.assert_array_equal(captures[0], captures[1])
    assert scores[0] != scores[1]


def test_no_context_queries_are_trained_predicted_and_evaluated(tmp_path):
    inputs, config, paths = prepare_poc_trace_graph_inputs(tmp_path / "data")
    config["graph"]["max_normalized_distance"] = 1e-12
    metrics, output = _run(tmp_path, paths, config)
    training = metrics["training"]
    assert training["no_context_query_count"] == training["query_count"] > 0
    assert metrics["no_context_query_count"] == metrics["evaluation_target"]["trace_count"]
    prediction = np.load(output / "prediction.npy")
    assert np.isfinite(prediction).all()
    assert np.count_nonzero(prediction[:, inputs.observed_volume.evaluation_target_trace_mask]) == 0


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "foreign", "nonfinite"])
def test_invalid_prediction_fails_whole_run_before_evaluation(tmp_path, monkeypatch, corruption):
    _, config, paths = prepare_poc_trace_graph_inputs(tmp_path / "data")
    original = pipeline.predict_relational_trace_graph

    def broken(*args, **kwargs):
        result = original(*args, **kwargs)
        if corruption == "missing":
            return replace(
                result,
                query_trace_ids=result.query_trace_ids[:-1],
                prediction=result.prediction[:-1],
            )
        if corruption in ("duplicate", "foreign"):
            ids = result.query_trace_ids.copy()
            ids[0] = ids[1] if corruption == "duplicate" else -99999
            return replace(result, query_trace_ids=ids)
        result.prediction[0] = np.nan
        return result

    monkeypatch.setattr(pipeline, "predict_relational_trace_graph", broken)
    monkeypatch.setattr(
        pipeline,
        "evaluate_c3_volume_prediction",
        lambda *a, **k: pytest.fail("invalid coverage must fail before evaluation"),
    )
    with pytest.raises(ValueError):
        _run(tmp_path, paths, config)
    output = tmp_path / "run"
    assert json.loads((output / "metadata.json").read_text())["status"] == "failed"
    assert (output / "final.pt").exists()
    assert not (output / "prediction.npy").exists()


@pytest.mark.parametrize(
    "key,value",
    [
        ("validation_interval", 1),
        ("loss", "masked_mse"),
        ("inner_mask_fraction", 1.0),
        ("amplitude_scaling", "per_trace_rms"),
    ],
)
def test_invalid_training_contract_fails_before_input_read(tmp_path, monkeypatch, key, value):
    _, config, paths = prepare_poc_trace_graph_inputs(tmp_path / "data")
    config["training"][key] = value
    monkeypatch.setattr(
        pipeline,
        "load_c3_random80_poc_inputs",
        lambda **k: pytest.fail("must validate config first"),
    )
    with pytest.raises(ValueError):
        _run(tmp_path, paths, config)
    assert not (tmp_path / "run").exists()
