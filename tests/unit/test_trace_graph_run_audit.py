"""Completed-run audit boundaries on tiny irregular IDs, rows and dense volumes."""

from __future__ import annotations

import json
import runpy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from seis_interp.data.c3_volume_adapter import ObservedC3Volume
from seis_interp.data.file_checksums import file_sha256
from seis_interp.data.trace_graph_prediction_store import trace_graph_query_index
from seis_interp.evaluation import trace_graph_run_audit as audit
from seis_interp.evaluation.trace_graph_metrics import evaluate_trace_graph_prediction
from seis_interp.models.relational_trace_graph import RelationalTraceGraphInterpolator
from seis_interp.processing.trace_graph_preprocessing import fit_trace_graph_preprocessing
from seis_interp.relational_trace_graph_config import (
    validate_relational_trace_graph_training_config,
)
from seis_interp.training.relational_trace_graph_checkpoints import (
    save_relational_trace_graph_checkpoint,
)
from seis_interp.training.relational_trace_graph_prediction import predict_relational_trace_graph
from tests.fixtures.relational_trace_graph import make_relational_trace_domains
from tests.fixtures.relational_trace_graph_runs import trace_graph_training_config


def _write(path, data):
    path.write_text(json.dumps(data, allow_nan=False), encoding="utf-8")


def _fixture(tmp_path, monkeypatch, *, final_metric_multiplier=1.0):
    suite, training, prediction, interim = (
        tmp_path / name for name in ("suite", "train", "predict", "interim")
    )
    for path in (suite, interim, training / "artifacts", prediction / "artifacts"):
        path.mkdir(parents=True)
    domain, fit_domain, amplitudes = make_relational_trace_domains()
    np.save(interim / "amplitudes.npy", amplitudes)
    files = {
        "interim": {"amplitudes.npy": {"sha256": file_sha256(interim / "amplitudes.npy")}},
        "processed": {"trace_split.parquet": {"sha256": "c" * 64}},
    }
    domain = replace(
        domain,
        amplitudes_path=interim / "amplitudes.npy",
        inputs_lock={"partition": "validation", "benchmark_case": {"input_files": files}},
    )
    provenance = {"source_inputs_lock": {"partition": "train", "input_files": files}}
    config = trace_graph_training_config()
    config["training_data"]["time_samples"] = [1, 6]
    config["model"]["method_variant"] = "relational"
    config["model"]["explicit_azimuth_features"] = True
    model_config, settings, options = validate_relational_trace_graph_training_config(config)
    preprocessing = fit_trace_graph_preprocessing(
        fit_domain, amplitudes, **config["geometry_features"]
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        model = RelationalTraceGraphInterpolator(**model_config)
    ids = np.array([30, 20])  # Deliberately different from domain and dense order.
    array_rows = np.array([6, 2, 4, 0, 1, 5, 3])
    spatial_shape = (1, 1, 1, 7)
    observed_mask = np.isin(array_rows, domain.array_rows[domain.observed_mask]).reshape(
        spatial_shape
    )
    values = amplitudes[array_rows, 1:6].T.copy().reshape((5, *spatial_shape))
    values.reshape(5, -1)[:, ~observed_mask.ravel()] = 0
    observed = ObservedC3Volume(
        values, domain.time_s, array_rows.reshape(spatial_shape), observed_mask, ~observed_mask
    )
    volume_metadata = {"shape": list(values.shape), "selection": {"time": [1, 6]}}
    inputs = SimpleNamespace(observed_volume=observed, volume_metadata=volume_metadata)
    scores = {}
    for name, directory, bias in (("best", training, 0.1), ("final", prediction, 0.2)):
        with torch.no_grad():
            model.decoder.head[-1].bias.fill_(bias)
        result = predict_relational_trace_graph(
            model,
            domain,
            preprocessing,
            graph_settings=settings,
            query_trace_ids=ids,
            query_batch_size=1,
            amplitudes=amplitudes,
            device="cpu",
            measure_resources=False,
        )
        table = trace_graph_query_index(
            domain, query_trace_ids=ids, has_observed_context=result.has_observed_context
        )
        table.to_parquet(directory / "artifacts/query_index.parquet", index=False)
        dense = values.copy()
        for row, wave in zip(table["array_row"], result.prediction, strict=True):
            dense.reshape(5, -1)[:, np.flatnonzero(array_rows == row)[0]] = wave
        np.save(directory / "artifacts/prediction.npy", dense)
        metrics = evaluate_trace_graph_prediction(
            result.prediction,
            domain,
            query_trace_ids=ids,
            has_observed_context=result.has_observed_context,
        )
        scores[name] = metrics
        selection = json.loads(json.dumps(metrics))
        if name == "final":
            selection["evaluation_target"]["error_energy"] *= final_metric_multiplier
        save_relational_trace_graph_checkpoint(
            training / f"artifacts/{name}.pt",
            model_config=model.constructor_config(),
            state_dict=model.state_dict(),
            preprocessing=preprocessing,
            graph_settings=settings,
            training_mask=config["training_mask"],
            training_provenance=provenance,
            training_random_seed=options["random_seed"],
            checkpoint_role="final" if name == "final" else "best_validation",
            global_step=2 if name == "final" else 1,
            selection_metrics=selection,
        )
        scores[f"{name}_selection"] = selection
    checkpoint = {
        "path": str(training / "artifacts/final.pt"),
        "sha256": file_sha256(training / "artifacts/final.pt"),
        "role": "final",
        "step": 2,
        "training_provenance": provenance,
    }
    amplitude_record = {"amplitude_scale": preprocessing.amplitude_scale}
    _write(
        training / "run.json",
        {
            "status": "success",
            "prediction": {"layout": "dense_volume", "checkpoint_role": "best_validation"},
            "amplitude": amplitude_record,
        },
    )
    _write(
        prediction / "run.json",
        {
            "status": "success",
            "prediction": {"layout": "dense_volume", "query_batch_size": 1},
            "checkpoint": checkpoint,
            "input": {"case_id": "validation"},
            "amplitude": amplitude_record,
        },
    )
    _write(
        training / "metrics.json",
        {
            "steps_completed": 2,
            "best_step": 1,
            "best_validation_metrics": scores["best_selection"],
            "final_validation_metrics": scores["final_selection"],
        },
    )
    _write(prediction / "metrics.json", scores["final"])
    _write(training / "inputs.lock.json", provenance)
    _write(prediction / "inputs.lock.json", {**domain.inputs_lock, "checkpoint": checkpoint})
    for directory in (training, prediction):
        (directory / "config.resolved.yaml").write_text(yaml.safe_dump(config))
    manifest = {
        "cases": [{"case_id": "validation", "partition": "validation", "volume_dir": "volume"}],
        "interim": "../interim",
        "files": [],
    }
    _write(suite / "benchmark_suite.json", manifest)
    monkeypatch.setattr(audit, "load_c3_benchmark_input_manifest", lambda *args, **kwargs: manifest)
    monkeypatch.setattr(audit, "load_c3_benchmark_graph_domain", lambda *args, **kwargs: domain)
    monkeypatch.setattr(audit, "load_c3_benchmark_volume_inputs", lambda *args, **kwargs: inputs)
    arguments = {
        "suite": suite,
        "case": "validation",
        "training_run": training,
        "prediction_run": prediction,
        "output": tmp_path / "audit",
        "expected_suite_sha256": file_sha256(suite / "benchmark_suite.json"),
    }
    return arguments, domain, inputs, scores


def test_audit_rescores_complete_dense_targets_and_restores_fixed_batches(tmp_path, monkeypatch):
    arguments, domain, inputs, scores = _fixture(tmp_path, monkeypatch)
    original = predict_relational_trace_graph
    calls = []

    def guarded(model, selected_domain, preprocessing, **kwargs):
        request = json.loads((arguments["output"] / "request.json").read_text())
        assert request["cpu_restoration_thresholds"] == audit.CPU_THRESHOLDS
        assert request["cross_run_energy_tolerance"] == {"rtol": 1e-6, "atol": 1e-12}
        package = Path(audit.__file__).resolve().parents[1]
        for relative in (
            "models/relational_trace_graph.py",
            "processing/trace_graph_subgraphs.py",
            "training/relational_trace_graph_prediction.py",
            "training/relational_trace_graph_checkpoints.py",
            "data/c3_benchmark_inputs.py",
            "evaluation/c3_volume_metrics.py",
        ):
            source = package / relative
            assert request["source_sha256"][str(source)] == file_sha256(source)
        assert kwargs["device"] == "cpu"
        assert "amplitudes" not in kwargs
        rows = domain.array_rows[domain.observed_mask]
        amplitudes = np.load(domain.amplitudes_path)
        np.testing.assert_array_equal(kwargs["observed_waveforms"], amplitudes[rows, 1:6])
        calls.extend(kwargs["query_trace_ids"].tolist())
        return original(model, selected_domain, preprocessing, **kwargs)

    monkeypatch.setattr(audit, "predict_relational_trace_graph", guarded)
    rng = torch.get_rng_state().clone()
    result = audit.audit_c3_proposed_gnn(**arguments)
    assert result["status"] == "success", result
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert calls == [30, 20]
    assert result["saved_output_scores"]["prediction"]["native_order_metrics"] == scores["final"]
    assert result["saved_output_scores"]["prediction"]["observed_reinsertion_max_abs_error"] == 0
    assert (
        result["saved_output_scores"]["prediction"]["dense_order_metrics"]["evaluation_target"][
            "sample_count"
        ]
        == 10
    )
    assert result["checkpoint_integrity"]["best_and_final_weights_equal"] is False
    assert result["cpu_checkpoint_restoration"]["query_count"] == 2
    assert result["cpu_checkpoint_restoration"]["metrics"]["physical_relative_l2"] == 0
    assert result["source_hashes_unchanged"] and result["request_unchanged"]
    assert json.loads((arguments["output"] / "result.json").read_text()) == result


def test_failed_cpu_tolerance_keeps_successful_saved_rescore_separate(tmp_path, monkeypatch):
    arguments, _, _, _ = _fixture(tmp_path, monkeypatch)

    def perturbed(model, domain, preprocessing, **kwargs):
        result = predict_relational_trace_graph(model, domain, preprocessing, **kwargs)
        return replace(
            result, prediction=result.prediction + np.float32(preprocessing.amplitude_scale * 0.02)
        )

    monkeypatch.setattr(audit, "predict_relational_trace_graph", perturbed)
    result = audit.audit_c3_proposed_gnn(**arguments)
    assert result["status"] == "failed_checks", result
    assert all(check["status"] == "passed" for check in result["metric_comparisons"].values())
    assert result["cpu_checkpoint_restoration"]["status"] == "failed_tolerance"
    assert result["cpu_checkpoint_restoration"]["thresholds"] == audit.CPU_THRESHOLDS


def test_original_energy_tolerance_is_not_relaxed_by_exact_cpu_restoration(tmp_path, monkeypatch):
    arguments, _, _, _ = _fixture(tmp_path, monkeypatch, final_metric_multiplier=1.0002)
    result = audit.audit_c3_proposed_gnn(**arguments)
    assert result["status"] == "failed_checks", result
    assert result["metric_comparisons"]["frozen_vs_training_final"]["status"] == "failed_tolerance"
    assert result["metric_comparisons"]["auxiliary_best_vs_training_best"]["status"] == "passed"
    assert result["cpu_checkpoint_restoration"]["status"] == "passed"


@pytest.mark.parametrize(
    "fault", ["hash", "unfinished", "test_partition", "duplicate_ids", "existing_output"]
)
def test_invalid_inputs_do_not_create_or_replace_an_audit(tmp_path, monkeypatch, fault):
    arguments, _, _, _ = _fixture(tmp_path, monkeypatch)
    if fault == "hash":
        arguments["expected_suite_sha256"] = "0" * 64
    elif fault == "unfinished":
        path = arguments["prediction_run"] / "run.json"
        record = json.loads(path.read_text())
        record["status"] = "running"
        _write(path, record)
    elif fault == "test_partition":
        path = arguments["suite"] / "benchmark_suite.json"
        record = json.loads(path.read_text())
        record["cases"][0]["partition"] = "test"
        _write(path, record)
        arguments["expected_suite_sha256"] = file_sha256(path)
    elif fault == "duplicate_ids":
        path = arguments["prediction_run"] / "artifacts/query_index.parquet"
        table = pd.read_parquet(path)
        table.loc[1, "trace_id"] = table.loc[0, "trace_id"]
        table.to_parquet(path, index=False)
    else:
        arguments["output"].mkdir()
        (arguments["output"] / "sentinel").write_text("immutable")
    with pytest.raises((ValueError, FileExistsError)):
        audit.audit_c3_proposed_gnn(**arguments)
    if fault == "existing_output":
        assert [p.name for p in arguments["output"].iterdir()] == ["sentinel"]
    else:
        assert not arguments["output"].exists()


@pytest.mark.parametrize(
    "fault",
    ["missing_query", "wrong_array_row", "observed_changed", "nonfinite", "checkpoint_hash"],
)
def test_completed_but_inconsistent_outputs_fail_before_cpu_forward(tmp_path, monkeypatch, fault):
    arguments, _, _, _ = _fixture(tmp_path, monkeypatch)
    directory = arguments["prediction_run"]
    if fault in ("missing_query", "wrong_array_row"):
        path = directory / "artifacts/query_index.parquet"
        table = pd.read_parquet(path)
        if fault == "missing_query":
            table = table.iloc[:1]
        else:
            table.loc[0, "array_row"] = 6
        table.to_parquet(path, index=False)
    elif fault in ("observed_changed", "nonfinite"):
        path = directory / "artifacts/prediction.npy"
        values = np.load(path)
        values.reshape(5, -1)[0, 0] = np.nan if fault == "nonfinite" else 100.0
        np.save(path, values)
    else:
        path = directory / "run.json"
        record = json.loads(path.read_text())
        record["checkpoint"]["sha256"] = "0" * 64
        _write(path, record)
    monkeypatch.setattr(
        audit,
        "predict_relational_trace_graph",
        lambda *args, **kwargs: pytest.fail("inconsistent saved output must not reach forward"),
    )
    result = audit.audit_c3_proposed_gnn(**arguments)
    assert result["status"] == "failed", result
    expected_error = {
        "missing_query": "every query",
        "wrong_array_row": "array_row",
        "observed_changed": "reinsertion",
        "nonfinite": "finite",
        "checkpoint_hash": "path and hash",
    }[fault]
    assert expected_error in result["error"]["message"]
    assert "cpu_checkpoint_restoration" not in result


def test_batches_cover_native_first_central_last_including_partial_last():
    table = pd.DataFrame({"trace_id": np.arange(10, 19)})
    batches = audit._native_batches(table, 2)
    assert [batch["batch_index"] for batch in batches] == [0, 2, 4]
    assert [batch["query_trace_ids"] for batch in batches] == [[10, 11], [14, 15], [18]]
    assert len(audit._native_batches(table.iloc[:1], 2)) == 1
    with pytest.raises(ValueError, match="positive integer"):
        audit._native_batches(table, True)


def test_cli_requires_all_explicit_paths_and_returns_failed_status(tmp_path, monkeypatch, capsys):
    module = runpy.run_path(str(Path(__file__).parents[2] / "scripts/audit_c3_proposed_gnn.py"))
    main = module["main"]
    with pytest.raises(SystemExit) as error:
        main(["--suite", str(tmp_path)])
    assert error.value.code == 2
    calls = []

    def failed(**kwargs):
        calls.append(kwargs)
        return {"status": "failed_checks"}

    monkeypatch.setitem(main.__globals__, "audit_c3_proposed_gnn", failed)
    result = main(
        [
            "--suite",
            str(tmp_path),
            "--case",
            "validation",
            "--training-run",
            "train",
            "--prediction-run",
            "predict",
            "--output",
            "audit",
            "--expected-suite-sha256",
            "a" * 64,
        ]
    )
    assert result == 1
    assert calls[0]["training_run"] == Path("train")
    assert json.loads(capsys.readouterr().out)["status"] == "failed_checks"


@pytest.mark.parametrize("fault", ["source", "request"])
def test_changed_source_or_request_cannot_produce_success(tmp_path, monkeypatch, fault):
    arguments, _, _, _ = _fixture(tmp_path, monkeypatch)

    def changing(model, domain, preprocessing, **kwargs):
        result = predict_relational_trace_graph(model, domain, preprocessing, **kwargs)
        path = (
            arguments["prediction_run"] / "run.json"
            if fault == "source"
            else arguments["output"] / "request.json"
        )
        with path.open("a") as stream:
            stream.write(" ")
        return result

    monkeypatch.setattr(audit, "predict_relational_trace_graph", changing)
    result = audit.audit_c3_proposed_gnn(**arguments)
    assert result["status"] == "failed_source_changed", result
    assert (
        not result["source_hashes_unchanged"]
        if fault == "source"
        else not result["request_unchanged"]
    )
