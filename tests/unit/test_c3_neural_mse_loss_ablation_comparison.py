import importlib.util
import json

import numpy as np
import pytest
import torch
import yaml

from seis_interp.configuration import REPOSITORY_ROOT
from seis_interp.data.file_checksums import file_sha256
from tests.fixtures.c3_poc_runs import poc_input_lock, write_poc_run_artifacts


@pytest.fixture
def comparison(tmp_path, monkeypatch):
    path = REPOSITORY_ROOT / "studies/study_037_c3_neural_mse_loss_ablation/compare.py"
    spec = importlib.util.spec_from_file_location("mse_comparison", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    baseline = tmp_path / "baseline"
    root = tmp_path / "mse"
    for method in module.METHODS:
        for output, loss in (
            (baseline / "run" / method, "masked_trace_relative_mse"),
            (root / method, "masked_trace_mse"),
        ):
            write_poc_run_artifacts(output, method)
            config = {"training": {"loss": loss, "max_steps": 5000, "learning_rate": 0.001}}
            (output / "config.resolved.yaml").write_text(yaml.safe_dump(config))
            metadata = json.loads((output / "metadata.json").read_text())
            metadata.update(
                loss_or_native_objective=loss,
                method_details={},
                training_or_reconstruction={"loss": loss},
            )
            metadata["compute"]["optimizer_updates"] = 5000
            (output / "metadata.json").write_text(json.dumps(metadata))
    frozen = {
        "run_directory": "run",
        "inputs_lock": poc_input_lock(),
        "run_file_sha256": {
            p.relative_to(baseline / "run").as_posix(): file_sha256(p)
            for p in (baseline / "run").rglob("*")
            if p.is_file()
        },
    }
    (baseline / "stage_1_baseline.lock.json").write_text(json.dumps(frozen))
    monkeypatch.setattr(module, "BASELINE", baseline)
    return module, root


def test_comparison_is_artifact_only_and_reports_metrics_and_cost(comparison, monkeypatch):
    module, root = comparison
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("no amplitude loading"))
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("no checkpoint inference"))
    rows = module.comparison_rows(root)
    assert [row["method"] for row in rows] == list(module.METHODS)
    for row in rows:
        assert row["baseline_snr_db"] == row["mse_snr_db"] == 10.0
        assert row["delta_snr_db"] == row["delta_rmse"] == 0.0
        assert row["baseline_mean_trace_relative_mse"] == row["mse_mean_trace_relative_mse"] == 0.1
        assert row["training_or_reconstruction_seconds"] == 2.0
        assert row["prediction_seconds"] == 1.0
        assert row["cuda_max_memory_allocated_bytes"] is None


@pytest.mark.parametrize(
    "filename,keys,value,error",
    [
        ("metadata.json", ["status"], "failed", "status"),
        ("metadata.json", ["coverage", "complete"], False, "complete"),
        ("metadata.json", ["loss_or_native_objective"], "masked_trace_relative_mse", "loss"),
        ("metadata.json", ["method_details", "loss"], "unknown", "loss"),
        ("metadata.json", ["compute", "optimizer_updates"], 2, "budget"),
        ("metadata.json", ["artifacts", "prediction", "sha256"], "wrong", "SHA-256"),
        ("metadata.json", ["artifacts", "checkpoint", "sha256"], "wrong", "SHA-256"),
        (
            "inputs.lock.json",
            ["benchmark_volume", "files", "volume.json", "sha256"],
            "wrong",
            "input lock",
        ),
        ("config.resolved.yaml", ["training", "learning_rate"], 0.0001, "config"),
        ("config.resolved.yaml", ["training", "loss"], "masked_trace_relative_mse", "config"),
    ],
)
def test_comparison_rejects_incomparable_runs(comparison, filename, keys, value, error):
    module, root = comparison
    path = root / "ccnet5d" / filename
    data = yaml.safe_load(path.read_text())
    section = data
    for key in keys[:-1]:
        section = section[key]
    section[keys[-1]] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=error):
        module.comparison_rows(root)


def test_comparison_rejects_altered_baseline_evidence(comparison):
    module, root = comparison
    (module.BASELINE / "run/nersi/metrics.json").write_text("{}")
    with pytest.raises(ValueError, match="Frozen baseline artifact"):
        module.comparison_rows(root)
