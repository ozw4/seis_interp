from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from seis_interp.configuration import (
    REPOSITORY_ROOT,
    get_required_config_value,
    load_resolved_config,
)
from seis_interp.pipelines.domain_scaling import run_experiment_a
from seis_interp.processing.trace_splits import TRAIN_SPLIT
from tests.fixtures.siren_experiment import build_experiment_fixture


def test_study_004_config_locks_the_fixed_experiment_a_conditions() -> None:
    config = load_resolved_config(
        REPOSITORY_ROOT / "studies" / "study_004_domain_scaling" / "config.yaml",
        repository_root=REPOSITORY_ROOT,
    )
    expected = {
        "study.status": "active",
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
        "training.batch_size": 1024,
        "training.steps_per_epoch": 500,
        "training.max_epochs": 100,
        "training.validation_batch_size": 65536,
        "training.device": "cuda:0",
        "experiment_a.trace_counts": [1, 8, 32, 128, 435],
    }

    assert {path: get_required_config_value(config, path) for path in expected} == expected


def test_experiment_a_writes_nested_immutable_training_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, interim, processed = build_experiment_fixture(tmp_path)
    output_root = tmp_path / "runs" / "study_004_domain_scaling"
    monkeypatch.setattr(
        "seis_interp.pipelines.domain_scaling._run_id_timestamp",
        lambda: "20260826T010203Z",
    )

    summary = run_experiment_a(
        config_path=config,
        interim_dir=interim,
        processed_dir=processed,
        output_root=output_root,
        device_override="cpu",
    )

    run_directories = sorted(path for path in output_root.iterdir() if path.is_dir())
    summary_paths = list(output_root.glob("*_experiment_a_summary.json"))
    assert len(run_directories) == 2
    assert len(summary_paths) == 1
    assert json.loads(summary_paths[0].read_text(encoding="utf-8")) == summary
    assert [path.name.rsplit("_", 1)[-1] for path in run_directories] == [
        "trace001",
        "trace002",
    ]

    split_table = pd.read_parquet(processed / "trace_split.parquet")
    training_rows = set(split_table.loc[split_table["split"] == TRAIN_SPLIT, "array_row"].tolist())
    metrics_by_count: dict[int, dict[str, object]] = {}
    original_files: dict[Path, bytes] = {}
    required_history_keys = {
        "step",
        "mean_train_loss_since_last_report",
        "training_median_trace_snr_db",
        "training_global_snr_db",
        "training_prediction_target_rms_ratio",
    }
    for run_directory in run_directories:
        assert sorted(path.name for path in run_directory.iterdir()) == [
            "config.resolved.yaml",
            "inputs.lock.json",
            "metrics.json",
            "run.json",
        ]
        saved_config = yaml.safe_load(
            (run_directory / "config.resolved.yaml").read_text(encoding="utf-8")
        )
        assert saved_config["training"]["device"] == "cpu"
        metrics = json.loads((run_directory / "metrics.json").read_text(encoding="utf-8"))
        inputs_lock = json.loads((run_directory / "inputs.lock.json").read_text(encoding="utf-8"))
        run_metadata = json.loads((run_directory / "run.json").read_text(encoding="utf-8"))
        trace_count = metrics["trace_count"]
        metrics_by_count[trace_count] = metrics

        assert metrics["updates_completed"] == 4
        assert [row["step"] for row in metrics["history"]] == [2, 4]
        assert all(set(row) == required_history_keys for row in metrics["history"])
        assert {
            "best_step",
            "best_training_median_trace_snr_db",
            "best_training_global_snr_db",
            "best_training_prediction_target_rms_ratio",
            "final_training_median_trace_snr_db",
            "final_training_global_snr_db",
            "final_training_prediction_target_rms_ratio",
            "history",
        } < set(metrics)
        best_row = max(metrics["history"], key=lambda row: row["training_median_trace_snr_db"])
        final_row = metrics["history"][-1]
        assert metrics["best_step"] == best_row["step"]
        assert (
            metrics["best_training_median_trace_snr_db"] == best_row["training_median_trace_snr_db"]
        )
        assert metrics["best_training_global_snr_db"] == best_row["training_global_snr_db"]
        assert (
            metrics["best_training_prediction_target_rms_ratio"]
            == best_row["training_prediction_target_rms_ratio"]
        )
        assert (
            metrics["final_training_median_trace_snr_db"]
            == final_row["training_median_trace_snr_db"]
        )
        assert metrics["final_training_global_snr_db"] == final_row["training_global_snr_db"]
        assert (
            metrics["final_training_prediction_target_rms_ratio"]
            == final_row["training_prediction_target_rms_ratio"]
        )
        assert set(metrics["selected_array_rows"]) <= training_rows
        assert inputs_lock["selection"]["selected_array_rows"] == metrics["selected_array_rows"]
        assert inputs_lock["selection"]["sample_count"] == 5
        assert inputs_lock["selection"]["trace_count"] == trace_count
        assert set(inputs_lock["interim_files"]) == {
            "traces.parquet",
            "amplitudes.npy",
            "time_s.npy",
            "dataset.json",
        }
        assert set(inputs_lock["processed_files"]) == {
            "trace_split.parquet",
            "normalization.json",
            "preparation.json",
        }
        assert run_metadata["study_id"] == "study_004_domain_scaling"
        assert run_metadata["experiment"] == "experiment_a"
        assert run_metadata["trace_count"] == trace_count
        assert run_metadata["device"] == "cpu"
        assert run_metadata["status"] == "success"
        assert run_metadata["updates_completed"] == 4
        for path in run_directory.iterdir():
            original_files[path] = path.read_bytes()
            if path.suffix in {".json", ".yaml"}:
                assert str(tmp_path) not in path.read_text(encoding="utf-8")

    np.testing.assert_array_equal(
        metrics_by_count[1]["selected_array_rows"],
        metrics_by_count[2]["selected_array_rows"][:1],
    )
    assert set(metrics_by_count[2]["selected_array_rows"]) == training_rows
    assert len(summary["runs"]) == 2
    assert {run["run_id"] for run in summary["runs"]} == {path.name for path in run_directories}
    assert str(tmp_path) not in summary_paths[0].read_text(encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exist"):
        run_experiment_a(
            config_path=config,
            interim_dir=interim,
            processed_dir=processed,
            output_root=output_root,
            device_override="cpu",
        )
    assert {path: path.read_bytes() for path in original_files} == original_files


def test_experiment_a_requires_the_final_subset_to_cover_all_training_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, interim, processed = build_experiment_fixture(tmp_path)
    config_data = yaml.safe_load(config.read_text(encoding="utf-8"))
    config_data["experiment_a"]["trace_counts"] = [1]
    config.write_text(yaml.safe_dump(config_data, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(
        "seis_interp.pipelines.domain_scaling._run_id_timestamp",
        lambda: "20260826T010203Z",
    )

    with pytest.raises(ValueError, match="must equal all 2 available training rows"):
        run_experiment_a(
            config_path=config,
            interim_dir=interim,
            processed_dir=processed,
            output_root=tmp_path / "runs",
            device_override="cpu",
        )

    assert not (tmp_path / "runs").exists()
