import json
import runpy
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from seis_interp import configuration
from seis_interp.data import file_checksums
from seis_interp.evaluation import c3_poc_comparison as comparison


@pytest.mark.parametrize("snr,reached", [(15.0, False), (15.01, True)])
def test_tuning_summary_uses_strict_threshold_and_preserves_previous_runs(
    tmp_path, monkeypatch, snr, reached
):
    _run_summary(tmp_path, monkeypatch, snr=snr)
    summary = json.loads((tmp_path / "output/summary.json").read_text())
    assert summary["target_reached"] is reached
    assert len(summary["runs"]) == 2
    assert summary["best_candidate"] == "new"
    assert summary["total_optimizer_updates"] == 20
    assert summary["independent_evaluation"] is False
    assert (tmp_path / "output/summary.csv").exists()


@pytest.mark.parametrize(
    "mismatch",
    ["lock", "alignment", "augmentation", "config", "optimization", "encoding", "bandlimit"],
)
def test_tuning_summary_rejects_inconsistent_run_contract(tmp_path, monkeypatch, mismatch):
    with pytest.raises(ValueError):
        _run_summary(tmp_path, monkeypatch, snr=16.0, mismatch=mismatch)
    assert not (tmp_path / "output").exists()


def _run_summary(tmp_path, monkeypatch, *, snr, mismatch=None):
    lock = {"nested": {"volume_sha256": "a" * 64}}
    config = {"training": {"loss": "masked_trace_mse", "max_steps": 10}}
    previous = {
        "baseline": {"output_directory": str(tmp_path / "baseline")},
        "runs": [{"output_directory": str(tmp_path / "old")}],
        "inputs_lock": lock,
        "source_archives": {},
        "independent_evaluation": False,
    }

    def validate(method, path):
        assert method == "nersi"
        row = comparison.empty_poc_summary_row(method, path)
        row.update(
            status="success",
            snr_db=snr if path.name == "new" else 10.0,
            optimizer_updates=10,
            training_or_reconstruction_seconds=1.0,
        )
        current = deepcopy(lock)
        if path.name == "new" and mismatch == "lock":
            current["nested"]["volume_sha256"] = "b" * 64
        return row, current

    def read(path):
        if path.name == "previous.json":
            return previous
        if path.name == "stage_1b_results.lock.json":
            return {"inputs_lock": lock}
        if path.name == "metrics.json":
            return {"observed_max_abs_error": 0}
        details = {"time_alignment": {"boundary": "circular"}} if mismatch == "alignment" else {}
        if mismatch == "optimization":
            details["optimization"] = {"ema_decay": 0.999}
        if mismatch == "encoding":
            details["model"] = {"coordinate_mapping": {}}
        if mismatch == "bandlimit":
            details["model"] = {"axis_frequency_limits": [1, 2, 3]}
        training = {"loss": "masked_trace_mse"}
        if mismatch == "augmentation":
            training["augmentation"] = {"profile_mixup_max_fraction": 0.2, "random_seed": 501}
        return {
            "loss_or_native_objective": "masked_trace_mse",
            "training_or_reconstruction": training,
            "method_details": details,
        }

    def load(path):
        result = deepcopy(config)
        if mismatch == "config" and path.name == "new.yaml":
            result["training"]["max_steps"] = 11
        return result

    monkeypatch.setattr(comparison, "validate_poc_run_artifacts", validate)
    monkeypatch.setattr(comparison, "read_poc_json", read)
    monkeypatch.setattr(configuration, "load_resolved_config", load)
    monkeypatch.setattr(file_checksums, "file_sha256", lambda path: "a" * 64)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_candidates",
            "--previous-summary",
            str(tmp_path / "previous.json"),
            "--run",
            str(tmp_path / "new"),
            "--output",
            str(tmp_path / "output"),
        ],
    )
    script = (
        configuration.REPOSITORY_ROOT
        / "studies/study_040_c3_nersi_target_tuning/summarize_candidates.py"
    )
    runpy.run_path(str(Path(script)), run_name="__main__")
