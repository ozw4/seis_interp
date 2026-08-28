from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from seis_interp.pipelines import batching_ablation as pipeline
from seis_interp.pipelines.prepare_baseline import prepare_baseline_dataset
from tests.integration.test_domain_scaling_pipeline import build_experiment_fixture
from tests.integration.test_official_siren_baseline_pipeline import (
    _assert_finite_json,
    _git_head,
)

_LABEL = "full_trace_batch_per_trace_rms"


def _build_budget_extension_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    config, interim, _ = build_experiment_fixture(tmp_path)
    processed = tmp_path / "processed_seed42"
    prepare_baseline_dataset(
        interim,
        processed,
        holdout_fraction=0.5,
        validation_fraction_of_holdout=0.5,
        random_seed=42,
        config_source="studies/synthetic/config.yaml",
    )
    config_data = yaml.safe_load(config.read_text(encoding="utf-8"))
    config_data["project"]["random_seed"] = 42
    config_data["model"].update(
        {
            "hidden_layers": 2,
            "omega_0": 30.0,
            "hidden_omega": 30.0,
        }
    )
    # Two training traces with five samples each: the complete-trace batch is ten points.
    config_data["training"].update(
        {
            "batch_size": 10,
            "total_updates": 2,
            "report_interval": 1,
            "prediction_batch_size": 3,
        }
    )
    config_data["experiment"] = {
        "trace_count": 2,
        "baseline_window_updates": 1,
        "baseline_best_median_trace_snr_db": 1.0,
        "baseline_tolerance_db": 1000.0,
        "conditions": [
            {
                "label": label,
                "batch_mode": batch_mode,
                "correlation": correlation,
                "amplitude_scaling": amplitude_scaling,
            }
            for label, batch_mode, correlation, amplitude_scaling in (
                pipeline._STRONG_FIT_BUDGET_EXTENSION_CONDITIONS
            )
        ],
    }
    config.write_text(yaml.safe_dump(config_data, sort_keys=False), encoding="utf-8")
    return config, interim, processed


def test_budget_extension_writes_one_isolated_immutable_cpu_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, interim, processed = _build_budget_extension_fixture(tmp_path)
    output_root = tmp_path / "runs" / pipeline.STRONG_FIT_BUDGET_EXTENSION_STUDY_ID
    monkeypatch.setattr(pipeline, "_run_id_timestamp", lambda: "20260827T060708Z")

    summary = pipeline.run_strong_fit_budget_extension(
        config_path=config,
        interim_dir=interim,
        processed_dir=processed,
        output_root=output_root,
        device_override="cpu",
    )

    git_commit = _git_head()
    prefix = f"20260827T060708Z_{git_commit[:7]}"
    run_directory = output_root / f"{prefix}_{_LABEL}"
    summary_path = output_root / f"{prefix}_summary.json"
    assert [path.name for path in sorted(output_root.iterdir())] == [
        run_directory.name,
        summary_path.name,
    ]
    assert json.loads(summary_path.read_text(encoding="utf-8")) == summary
    assert set(summary) == {
        "study_id",
        "git_commit",
        "generated_at_utc",
        "decision",
        "baseline_reproduced",
        "baseline_window_updates",
        "baseline_expected_best_median_trace_snr_db",
        "baseline_observed_best_median_trace_snr_db",
        "baseline_tolerance_db",
        "first_strong_fit_step",
        "per_trace_scale_stats",
        "conditions",
    }
    assert summary["study_id"] == pipeline.STRONG_FIT_BUDGET_EXTENSION_STUDY_ID
    assert summary["git_commit"] == git_commit
    assert summary["baseline_window_updates"] == 1
    assert summary["baseline_expected_best_median_trace_snr_db"] == 1.0
    assert summary["baseline_tolerance_db"] == 1000.0
    # The huge fixture tolerance forces the gate to pass regardless of the tiny fit.
    assert summary["baseline_reproduced"] is True

    assert sorted(path.name for path in run_directory.iterdir()) == [
        "config.resolved.yaml",
        "inputs.lock.json",
        "metrics.json",
        "run.json",
    ]
    saved_config = yaml.safe_load(
        (run_directory / "config.resolved.yaml").read_text(encoding="utf-8")
    )
    metrics = json.loads((run_directory / "metrics.json").read_text(encoding="utf-8"))
    inputs_lock = json.loads((run_directory / "inputs.lock.json").read_text(encoding="utf-8"))
    run_metadata = json.loads((run_directory / "run.json").read_text(encoding="utf-8"))

    assert saved_config["training"]["device"] == "cpu"
    assert saved_config["experiment"]["active_condition"] == {
        "label": _LABEL,
        "batch_mode": "random_complete_traces",
        "correlation_weight": 0.0,
        "amplitude_scaling": "per_trace_rms",
    }

    assert metrics["condition"] == _LABEL
    assert metrics["batch_mode"] == "random_complete_traces"
    assert metrics["batch_size"] == 10
    assert metrics["trace_count"] == 2
    assert metrics["traces_per_update"] == 2
    assert metrics["replacement"] is False
    assert metrics["updates_completed"] == 2
    assert [row["step"] for row in metrics["history"]] == [1, 2]
    assert metrics["amplitude_scaling"] == "per_trace_rms"
    assert metrics["per_trace_scale_stats"] == summary["per_trace_scale_stats"]
    assert "correlation_weight" not in metrics
    assert "loss_semantics" not in metrics

    window_rows = [row for row in metrics["history"] if row["step"] <= 1]
    assert summary["baseline_observed_best_median_trace_snr_db"] == max(
        row["training_median_trace_snr_db"] for row in window_rows
    )
    expected_first_strong_fit = next(
        (row["step"] for row in metrics["history"] if row["training_median_trace_snr_db"] >= 20.0),
        None,
    )
    assert summary["first_strong_fit_step"] == expected_first_strong_fit
    assert summary["decision"] == pipeline.strong_fit_budget_extension_summary_decision(
        baseline_reproduced=True,
        extension_classification=str(metrics["classification"]),
    )

    training_contract = inputs_lock["training"]
    assert training_contract == {
        "condition": _LABEL,
        "batch_mode": "random_complete_traces",
        "replacement": False,
        "batch_size": 10,
        "total_updates": 2,
        "point_evaluations": 20,
        "amplitude_scaling": "per_trace_rms",
        "traces_per_update": 2,
    }

    assert run_metadata["study_id"] == pipeline.STRONG_FIT_BUDGET_EXTENSION_STUDY_ID
    assert run_metadata["batch_mode"] == "random_complete_traces"
    assert run_metadata["amplitude_scaling"] == "per_trace_rms"
    assert run_metadata["traces_per_update"] == 2
    assert "correlation_weight" not in run_metadata

    (condition_summary,) = summary["conditions"]
    assert condition_summary["label"] == _LABEL
    assert condition_summary["run_directory"] == run_directory.name
    assert condition_summary["batch_size"] == 10
    assert condition_summary["correlation_weight"] == 0.0
    assert condition_summary["classification"] == metrics["classification"]
    best_row = max(metrics["history"], key=lambda row: row["training_median_trace_snr_db"])
    assert condition_summary["best_report"] == best_row
    assert condition_summary["final_report"] == metrics["history"][-1]

    for path in run_directory.iterdir():
        assert str(tmp_path) not in path.read_text(encoding="utf-8")
    _assert_finite_json(summary)
    for path in output_root.rglob("*.json"):
        _assert_finite_json(json.loads(path.read_text(encoding="utf-8")))

    with pytest.raises(FileExistsError, match="already exist"):
        pipeline.run_strong_fit_budget_extension(
            config_path=config,
            interim_dir=interim,
            processed_dir=processed,
            output_root=output_root,
            device_override="cpu",
        )


def test_budget_extension_rejects_a_batch_size_below_the_full_trace_batch(
    tmp_path: Path,
) -> None:
    config, interim, processed = _build_budget_extension_fixture(tmp_path)
    config_data = yaml.safe_load(config.read_text(encoding="utf-8"))
    config_data["training"]["batch_size"] = 5
    config.write_text(yaml.safe_dump(config_data, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="must equal the 10 points"):
        pipeline.run_strong_fit_budget_extension(
            config_path=config,
            interim_dir=interim,
            processed_dir=processed,
            output_root=tmp_path / "runs" / "batch_size_mismatch",
            device_override="cpu",
        )
