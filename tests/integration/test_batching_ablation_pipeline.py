from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from seis_interp.configuration import (
    REPOSITORY_ROOT,
    get_required_config_value,
    load_resolved_config,
)
from seis_interp.pipelines import batching_ablation as pipeline
from seis_interp.processing.trace_splits import TRAIN_SPLIT
from tests.integration.test_domain_scaling_pipeline import build_experiment_fixture


def test_study_006_config_locks_the_fixed_conditions() -> None:
    config = load_resolved_config(
        REPOSITORY_ROOT / "studies" / "study_006_batching_ablation" / "config.yaml",
        repository_root=REPOSITORY_ROOT,
    )
    expected = {
        "study.status": "completed",
        "project.random_seed": 42,
        "sampling.random_trace_holdout_fraction": 0.20,
        "sampling.validation_fraction_of_holdout": 0.25,
        "normalization.coordinates": "train_minmax_linear_plus_azimuth_sin_cos",
        "normalization.amplitude": "train_global_rms",
        "model.name": "siren",
        "model.input_features": 6,
        "model.hidden_width": 256,
        "model.hidden_layers": 4,
        "model.omega_0": 300.0,
        "training.loss": "l2",
        "training.optimizer": "adam",
        "training.learning_rate": 1.0e-3,
        "training.updates": 50000,
        "training.report_interval": 500,
        "training.batch_size": 5000,
        "training.device": "cuda:0",
        "experiment.trace_count": 8,
        "experiment.conditions": [
            {
                "label": "exact_full_batch",
                "batch_mode": "exact_full_batch",
                "full_batch": True,
                "replacement": False,
            },
            {
                "label": "random_replacement_5000",
                "batch_mode": "random_replacement",
                "full_batch": False,
                "replacement": True,
            },
        ],
    }

    assert {path: get_required_config_value(config, path) for path in expected} == expected


def _build_batching_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    config, interim, processed = build_experiment_fixture(tmp_path)
    config_data = yaml.safe_load(config.read_text(encoding="utf-8"))
    config_data["training"].update(
        {
            "updates": 2,
            "report_interval": 1,
            "batch_size": 10,
        }
    )
    config_data["experiment"] = {
        "trace_count": 2,
        "conditions": [
            {
                "label": "exact_full_batch",
                "batch_mode": "exact_full_batch",
                "full_batch": True,
                "replacement": False,
            },
            {
                "label": "random_replacement_5000",
                "batch_mode": "random_replacement",
                "full_batch": False,
                "replacement": True,
            },
        ],
    }
    config.write_text(yaml.safe_dump(config_data, sort_keys=False), encoding="utf-8")
    return config, interim, processed


def test_batching_ablation_writes_paired_immutable_cpu_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, interim, processed = _build_batching_fixture(tmp_path)
    output_root = tmp_path / "runs" / "study_006_batching_ablation"
    monkeypatch.setattr(pipeline, "_run_id_timestamp", lambda: "20260826T010203Z")

    actual_sampler = pipeline.RandomPointSampler
    sampler_rows: list[np.ndarray] = []

    def recording_sampler(*args: Any, **kwargs: Any) -> Any:
        sampler_rows.append(np.asarray(args[3]).copy())
        return actual_sampler(*args, **kwargs)

    monkeypatch.setattr(pipeline, "RandomPointSampler", recording_sampler)

    summary = pipeline.run_batching_ablation(
        config_path=config,
        interim_dir=interim,
        processed_dir=processed,
        output_root=output_root,
        device_override="cpu",
    )

    run_directories = sorted(path for path in output_root.iterdir() if path.is_dir())
    summary_paths = list(output_root.glob("*_summary.json"))
    assert len(run_directories) == 2
    assert len(summary_paths) == 1
    assert json.loads(summary_paths[0].read_text(encoding="utf-8")) == summary

    split_table = pd.read_parquet(processed / "trace_split.parquet")
    training_rows = set(split_table.loc[split_table["split"] == TRAIN_SPLIT, "array_row"].tolist())
    assert len(sampler_rows) == 1
    assert set(sampler_rows[0].tolist()) == training_rows

    required_history_keys = {
        "step",
        "mean_train_loss_since_last_report",
        "training_median_trace_snr_db",
        "training_global_snr_db",
        "training_median_trace_correlation",
        "training_prediction_target_rms_ratio",
    }
    records: dict[str, tuple[dict[str, object], dict[str, object], dict[str, object]]] = {}
    for run_directory in run_directories:
        assert sorted(path.name for path in run_directory.iterdir()) == [
            "config.resolved.yaml",
            "inputs.lock.json",
            "metrics.json",
            "run.json",
        ]
        metrics = json.loads((run_directory / "metrics.json").read_text(encoding="utf-8"))
        inputs_lock = json.loads((run_directory / "inputs.lock.json").read_text(encoding="utf-8"))
        run_metadata = json.loads((run_directory / "run.json").read_text(encoding="utf-8"))
        condition = str(metrics["condition"])
        records[condition] = (metrics, inputs_lock, run_metadata)

        assert metrics["selected_array_rows"] == inputs_lock["selection"]["selected_array_rows"]
        assert set(metrics["selected_array_rows"]) == training_rows
        assert metrics["trace_count"] == 2
        assert metrics["sample_count"] == 5
        assert metrics["point_count"] == 10
        assert metrics["point_evaluations"] == 20
        assert metrics["updates_completed"] == 2
        assert [row["step"] for row in metrics["history"]] == [1, 2]
        assert all(set(row) == required_history_keys for row in metrics["history"])
        assert all(
            math.isfinite(float(value)) for row in metrics["history"] for value in row.values()
        )
        best_row = max(metrics["history"], key=lambda row: row["training_median_trace_snr_db"])
        assert metrics["best_step"] == best_row["step"]
        assert run_metadata["study_id"] == "study_006_batching_ablation"
        assert run_metadata["condition"] == condition
        assert run_metadata["batch_size"] == 10
        assert run_metadata["trace_count"] == 2
        assert run_metadata["sample_count"] == 5
        assert run_metadata["point_count"] == 10
        assert run_metadata["point_evaluations"] == 20
        assert run_metadata["device"] == "cpu"
        assert run_metadata["status"] == "success"
        assert run_metadata["updates_completed"] == 2
        assert inputs_lock["selection"]["source_split"] == TRAIN_SPLIT
        assert inputs_lock["selection"]["point_count"] == 10
        for path in run_directory.iterdir():
            assert str(tmp_path) not in path.read_text(encoding="utf-8")

    assert set(records) == {"exact_full_batch", "random_replacement_5000"}
    exact_metrics, exact_lock, exact_run = records["exact_full_batch"]
    random_metrics, random_lock, random_run = records["random_replacement_5000"]
    assert exact_metrics["selected_array_rows"] == random_metrics["selected_array_rows"]
    assert exact_lock["selection"] == random_lock["selection"]
    assert exact_run["point_evaluations"] == random_run["point_evaluations"]
    assert exact_run["full_batch"] is True
    assert exact_run["replacement"] is False
    assert random_run["full_batch"] is False
    assert random_run["replacement"] is True
    assert summary["point_evaluations_per_condition"] == 20
    assert {run["run_id"] for run in summary["runs"]} == {path.name for path in run_directories}
    assert summary["decision"] in {
        "random_replacement_succeeds",
        "random_replacement_partially_succeeds",
        "exact_coverage_required",
        "control_failed_unexpected",
    }

    original_files = {path: path.read_bytes() for path in output_root.rglob("*") if path.is_file()}
    with pytest.raises(FileExistsError, match="already exist"):
        pipeline.run_batching_ablation(
            config_path=config,
            interim_dir=interim,
            processed_dir=processed,
            output_root=output_root,
            device_override="cpu",
        )
    assert {
        path: path.read_bytes() for path in output_root.rglob("*") if path.is_file()
    } == original_files
