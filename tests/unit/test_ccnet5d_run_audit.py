"""Audit complete synthetic CCNet outputs and reject corrupt or incomplete runs."""

from __future__ import annotations

import json

import numpy as np
import pytest

from seis_interp.data.c3_volume_run_inputs import load_c3_volume_run_inputs
from seis_interp.data.file_checksums import file_sha256
from seis_interp.evaluation import ccnet5d_run_audit as audit
from seis_interp.pipelines.interpolate_ccnet5d import interpolate_ccnet5d_run
from seis_interp.pipelines.train_ccnet5d import train_ccnet5d_run
from tests.fixtures.ccnet5d_artifacts import prepare_ccnet5d_artifacts
from tests.fixtures.ccnet5d_runs import (
    ccnet5d_inference_config,
    ccnet5d_training_config,
    prepare_ccnet5d_benchmark,
    write_ccnet5d_config,
)


@pytest.fixture
def completed_runs(tmp_path, monkeypatch):
    source = prepare_ccnet5d_artifacts(tmp_path / "data")
    artifacts = prepare_ccnet5d_benchmark(source)
    training, prediction = tmp_path / "train", tmp_path / "prediction"
    train_config = ccnet5d_training_config(source)
    train_ccnet5d_run(
        config_path=write_ccnet5d_config(tmp_path / "train.yaml", train_config),
        interim_dir=source.interim,
        processed_dir=source.processed,
        output_dir=training,
    )
    config = ccnet5d_inference_config(artifacts)
    paths = {
        "interim_dir": artifacts.interim,
        "processed_dir": artifacts.processed,
        "mask_dir": artifacts.mask,
        "case_dir": artifacts.case,
        "volume_dir": artifacts.volume,
    }
    interpolate_ccnet5d_run(
        config_path=write_ccnet5d_config(tmp_path / "predict.yaml", config),
        checkpoint_path=training / "artifacts/final.pt",
        output_dir=prediction,
        **paths,
    )
    inputs = load_c3_volume_run_inputs(config=config, **paths)
    suite = tmp_path / "suite"
    suite.mkdir()
    manifest = {
        "interim": "../data/interim",
        "cases": [{"case_id": inputs.case["case_id"], "partition": "validation"}],
        "files": [
            {
                "path": "../data/interim/amplitudes.npy",
                "sha256": file_sha256(source.interim / "amplitudes.npy"),
            }
        ],
    }
    manifest_path = suite / "benchmark_suite.json"
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(audit, "load_c3_benchmark_input_manifest", lambda *a, **kw: manifest)
    monkeypatch.setattr(audit, "load_c3_benchmark_volume_inputs", lambda *a, **kw: inputs)
    arguments = {
        "suite": suite,
        "case": inputs.case["case_id"],
        "training_run": training,
        "prediction_run": prediction,
        "output": tmp_path / "audit",
        "expected_suite_sha256": file_sha256(manifest_path),
    }
    return arguments, inputs


def test_completed_audit_restores_fixed_native_tiles_and_all_targets(completed_runs, monkeypatch):
    arguments, inputs = completed_runs
    original = audit.restore_ccnet5d_cpu_tiles

    def restore(loaded, observed, prediction, tiles):
        plan = json.loads((arguments["output"] / "plan.json").read_text())
        assert [row["tile_index"] for row in plan["tiles"]] == [0, 8, 15]
        # Poison hidden placeholders after native prediction; restoration must mask them.
        observed.values[:, observed.evaluation_target_trace_mask] = np.nan
        return original(loaded, observed, prediction, tiles)

    monkeypatch.setattr(audit, "restore_ccnet5d_cpu_tiles", restore)
    report = audit.audit_c3_ccnet5d(**arguments)
    assert report["status"] == "success"
    assert all(report["checks"].values())
    assert report["CPU_restoration"]["measured"]["normalized_difference_max_abs"] == 0
    assert report["CPU_restoration"]["target_truth_read"] is False
    target = report["independent_saved_output_metrics"]["evaluation_target"]
    assert target["sample_count"] == inputs.observed_volume.evaluation_target_trace_mask.sum() * 4
    with pytest.raises(ValueError, match="must not exist"):
        audit.audit_c3_ccnet5d(**arguments)


@pytest.mark.parametrize("corruption", ["observed", "target", "metrics", "reported_snr"])
def test_audit_rejects_changed_saved_outputs_or_metrics(completed_runs, corruption):
    arguments, inputs = completed_runs
    prediction = arguments["prediction_run"]
    if corruption in ("metrics", "reported_snr"):
        path = prediction / "metrics.json"
        metrics = json.loads(path.read_text())
        if corruption == "metrics":
            metrics["evaluation_target"]["error_energy"] *= 2
        else:
            metrics["evaluation_target"]["snr_db"] = 999.0
        path.write_text(json.dumps(metrics))
    else:
        path = prediction / "artifacts/prediction.npy"
        values = np.load(path)
        mask = (
            inputs.observed_volume.observed_trace_mask
            if corruption == "observed"
            else inputs.observed_volume.evaluation_target_trace_mask
        )
        values[:, mask] += 10
        np.save(path, values)
    report = audit.audit_c3_ccnet5d(**arguments)
    assert report["status"] == "failed_verification"
    key = {
        "observed": "observed_reinsertion_exact",
        "target": "CPU_restoration",
        "metrics": "error_energy",
        "reported_snr": "native_target_metrics_exact",
    }[corruption]
    assert report["checks"][key] is False


@pytest.mark.parametrize(
    "corruption", ["budget", "role", "suite_hash", "nonfinite", "halo", "patch_plan"]
)
def test_audit_rejects_invalid_contract_before_restoration(completed_runs, corruption):
    arguments, _ = completed_runs
    if corruption == "suite_hash":
        arguments["expected_suite_sha256"] = "0" * 64
    elif corruption == "nonfinite":
        path = arguments["prediction_run"] / "artifacts/prediction.npy"
        values = np.load(path)
        values.flat[0] = np.nan
        np.save(path, values)
    elif corruption == "patch_plan":
        (arguments["training_run"] / "artifacts/patch_plan.json").write_text("{}")
    else:
        directory = arguments["training_run" if corruption == "budget" else "prediction_run"]
        path = directory / ("metrics.json" if corruption == "budget" else "run.json")
        record = json.loads(path.read_text())
        if corruption == "budget":
            record["steps_completed"] -= 1
        elif corruption == "halo":
            record["prediction"]["halo_radius"] += 1
        else:
            record["checkpoint"]["role"] = "best_selection"
        path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        audit.audit_c3_ccnet5d(**arguments)
    assert not arguments["output"].exists()


def test_test_partition_rejected_before_volume_reader(completed_runs, monkeypatch):
    arguments, _ = completed_runs
    manifest = {"cases": [{"case_id": arguments["case"], "partition": "test"}]}
    monkeypatch.setattr(audit, "load_c3_benchmark_input_manifest", lambda *a, **kw: manifest)
    monkeypatch.setattr(
        audit,
        "load_c3_benchmark_volume_inputs",
        lambda *a, **kw: pytest.fail("test amplitudes must not be loaded"),
    )
    with pytest.raises(ValueError, match="restricted to validation"):
        audit.audit_c3_ccnet5d(**arguments)


def test_suite_input_mutation_during_audit_fails(completed_runs, monkeypatch):
    arguments, _ = completed_runs
    original = audit.restore_ccnet5d_cpu_tiles

    def restore(*args):
        result = original(*args)
        path = arguments["suite"].parent / "data/interim/amplitudes.npy"
        # Alter an otherwise valid container after initial input verification.
        with path.open("ab") as stream:
            stream.write(b"changed")
        return result

    monkeypatch.setattr(audit, "restore_ccnet5d_cpu_tiles", restore)
    report = audit.audit_c3_ccnet5d(**arguments)
    assert report["status"] == "failed_verification"
    assert report["checks"]["inputs_unchanged"] is False
