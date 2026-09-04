from __future__ import annotations

import json
import math
import subprocess
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
from seis_interp.data.file_checksums import file_sha256
from seis_interp.pipelines import batching_ablation as pipeline
from seis_interp.processing.trace_splits import TRAIN_SPLIT
from tests.fixtures.siren_experiment import build_experiment_fixture


def test_study_007_config_locks_the_single_full_ffid_condition() -> None:
    config = load_resolved_config(
        REPOSITORY_ROOT / "studies" / "study_007_full_ffid_large_batch" / "config.yaml",
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
        "training.batch_size": 5000,
        "training.total_updates": 50000,
        "training.report_interval": 500,
        "training.prediction_batch_size": 65536,
        "training.device": "cuda:0",
        "experiment.trace_count": 435,
        "experiment.batch_mode": "random_replacement",
        "experiment.replacement": True,
    }

    assert {path: get_required_config_value(config, path) for path in expected} == expected


def _build_full_ffid_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    config, interim, processed = build_experiment_fixture(tmp_path)
    config_data = yaml.safe_load(config.read_text(encoding="utf-8"))
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
        "batch_mode": "random_replacement",
        "replacement": True,
    }
    config.write_text(yaml.safe_dump(config_data, sort_keys=False), encoding="utf-8")
    return config, interim, processed


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_full_ffid_probe_writes_one_split_isolated_immutable_cpu_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, interim, processed = _build_full_ffid_fixture(tmp_path)
    output_root = tmp_path / "runs" / "study_007_full_ffid_large_batch"
    monkeypatch.setattr(pipeline, "_run_id_timestamp", lambda: "20260826T010203Z")

    actual_sampler = pipeline.RandomPointSampler
    actual_build_trace_points = pipeline.build_trace_points
    actual_predict_points = pipeline.predict_points
    sampler_rows: list[np.ndarray] = []
    evaluation_rows: list[np.ndarray] = []
    prediction_batch_sizes: list[int] = []

    def recording_sampler(*args: Any, **kwargs: Any) -> Any:
        sampler_rows.append(np.asarray(args[3]).copy())
        return actual_sampler(*args, **kwargs)

    def recording_build_trace_points(*args: Any, **kwargs: Any) -> Any:
        evaluation_rows.append(np.asarray(args[3]).copy())
        return actual_build_trace_points(*args, **kwargs)

    def recording_predict_points(*args: Any, **kwargs: Any) -> Any:
        prediction_batch_sizes.append(int(kwargs["batch_size"]))
        return actual_predict_points(*args, **kwargs)

    monkeypatch.setattr(pipeline, "RandomPointSampler", recording_sampler)
    monkeypatch.setattr(pipeline, "build_trace_points", recording_build_trace_points)
    monkeypatch.setattr(pipeline, "predict_points", recording_predict_points)

    summary = pipeline.run_full_ffid_large_batch(
        config_path=config,
        interim_dir=interim,
        processed_dir=processed,
        output_root=output_root,
        device_override="cpu",
    )

    git_commit = _git_head()
    expected_run_id = f"20260826T010203Z_{git_commit[:7]}_random10_trace2"
    run_directories = [path for path in output_root.iterdir() if path.is_dir()]
    summary_paths = list(output_root.glob("*_summary.json"))
    assert [path.name for path in run_directories] == [expected_run_id]
    assert [path.name for path in summary_paths] == [
        f"20260826T010203Z_{git_commit[:7]}_summary.json"
    ]
    assert json.loads(summary_paths[0].read_text(encoding="utf-8")) == summary
    assert len(summary["runs"]) == 1
    assert summary["runs"][0]["run_id"] == expected_run_id

    split_table = pd.read_parquet(processed / "trace_split.parquet")
    training_rows = set(split_table.loc[split_table["split"] == TRAIN_SPLIT, "array_row"])
    held_out_rows = set(split_table.loc[split_table["split"] != TRAIN_SPLIT, "array_row"])
    assert len(sampler_rows) == 1
    assert len(evaluation_rows) == 1
    assert set(sampler_rows[0]) == training_rows
    assert set(evaluation_rows[0]) == training_rows
    assert set(sampler_rows[0]).isdisjoint(held_out_rows)
    assert set(evaluation_rows[0]).isdisjoint(held_out_rows)
    assert prediction_batch_sizes == [3, 3]

    run_directory = run_directories[0]
    assert sorted(path.name for path in run_directory.iterdir()) == [
        "config.resolved.yaml",
        "inputs.lock.json",
        "metrics.json",
        "run.json",
    ]
    assert {path.name for path in output_root.rglob("*") if path.is_file()} == {
        "config.resolved.yaml",
        "inputs.lock.json",
        "metrics.json",
        "run.json",
        summary_paths[0].name,
    }

    saved_config = yaml.safe_load(
        (run_directory / "config.resolved.yaml").read_text(encoding="utf-8")
    )
    assert saved_config["training"]["device"] == "cpu"
    assert saved_config["training"]["batch_size"] == 10
    assert saved_config["training"]["total_updates"] == 2
    assert saved_config["training"]["prediction_batch_size"] == 3

    metrics = json.loads((run_directory / "metrics.json").read_text(encoding="utf-8"))
    inputs_lock = json.loads((run_directory / "inputs.lock.json").read_text(encoding="utf-8"))
    run_metadata = json.loads((run_directory / "run.json").read_text(encoding="utf-8"))
    required_history_keys = {
        "step",
        "mean_train_loss_since_last_report",
        "training_median_trace_snr_db",
        "training_global_snr_db",
        "training_median_trace_correlation",
        "training_prediction_target_rms_ratio",
    }
    assert metrics["trace_count"] == 2
    assert metrics["sample_count"] == 5
    assert metrics["point_count"] == 10
    assert metrics["batch_size"] == 10
    assert metrics["point_evaluations"] == 20
    assert metrics["updates_completed"] == 2
    assert [row["step"] for row in metrics["history"]] == [1, 2]
    assert all(set(row) == required_history_keys for row in metrics["history"])
    assert all(math.isfinite(float(value)) for row in metrics["history"] for value in row.values())
    best_row = max(metrics["history"], key=lambda row: row["training_median_trace_snr_db"])
    final_row = metrics["history"][-1]
    assert metrics["best_step"] == best_row["step"]
    assert metrics["best_training_median_trace_snr_db"] == best_row["training_median_trace_snr_db"]
    assert metrics["best_training_global_snr_db"] == best_row["training_global_snr_db"]
    assert (
        metrics["best_training_median_trace_correlation"]
        == best_row["training_median_trace_correlation"]
    )
    assert (
        metrics["best_training_prediction_target_rms_ratio"]
        == best_row["training_prediction_target_rms_ratio"]
    )
    assert (
        metrics["final_training_median_trace_snr_db"] == final_row["training_median_trace_snr_db"]
    )
    assert metrics["classification"] in {
        "strong_fit",
        "escaped_zero_predictor",
        "near_zero",
    }

    assert inputs_lock["selection"]["source_split"] == TRAIN_SPLIT
    assert inputs_lock["selection"]["selected_array_rows"] == sorted(training_rows)
    assert inputs_lock["selection"]["trace_count"] == 2
    assert inputs_lock["selection"]["sample_count"] == 5
    assert inputs_lock["split"] == {
        "counts": {"train": 2, "validation": 1, "test": 1},
        "training_source": TRAIN_SPLIT,
    }
    assert inputs_lock["preparation"]["normalization"] == {
        "coordinates": "train_minmax_linear_plus_azimuth_sin_cos",
        "amplitude": "train_global_rms",
    }
    assert inputs_lock["training"] == {
        "batch_mode": "random_replacement",
        "replacement": True,
        "batch_size": 10,
        "total_updates": 2,
        "point_evaluations": 20,
    }
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
    for file_name, record in inputs_lock["interim_files"].items():
        assert record["sha256"] == file_sha256(interim / file_name)
    for file_name, record in inputs_lock["processed_files"].items():
        assert record["sha256"] == file_sha256(processed / file_name)

    assert run_metadata["study_id"] == "study_007_full_ffid_large_batch"
    assert run_metadata["git_commit"] == git_commit
    assert run_metadata["status"] == "success"
    assert run_metadata["device"] == "cpu"
    assert run_metadata["random_seed"] == 5
    assert run_metadata["batch_size"] == 10
    assert run_metadata["trace_count"] == 2
    assert run_metadata["sample_count"] == 5
    assert run_metadata["point_evaluations"] == 20
    assert run_metadata["updates_completed"] == 2
    assert run_metadata["python_version"]
    assert run_metadata["torch_version"]
    assert run_metadata["started_at_utc"].endswith("Z")
    assert run_metadata["finished_at_utc"].endswith("Z")
    assert summary["point_evaluations"] == 20
    assert summary["decision"] == pipeline.full_ffid_summary_decision(metrics["classification"])

    for path in output_root.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".yaml"}:
            assert str(tmp_path) not in path.read_text(encoding="utf-8")

    original_files = {path: path.read_bytes() for path in output_root.rglob("*") if path.is_file()}
    with pytest.raises(FileExistsError, match="already exist"):
        pipeline.run_full_ffid_large_batch(
            config_path=config,
            interim_dir=interim,
            processed_dir=processed,
            output_root=output_root,
            device_override="cpu",
        )
    assert {path: path.read_bytes() for path in original_files} == original_files
    assert len(sampler_rows) == 1


def test_full_ffid_probe_requires_every_training_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, interim, processed = _build_full_ffid_fixture(tmp_path)
    config_data = yaml.safe_load(config.read_text(encoding="utf-8"))
    config_data["experiment"]["trace_count"] = 1
    config.write_text(yaml.safe_dump(config_data, sort_keys=False), encoding="utf-8")
    output_root = tmp_path / "runs"
    monkeypatch.setattr(pipeline, "_run_id_timestamp", lambda: "20260826T010203Z")

    with pytest.raises(ValueError, match="must equal all 2 available training rows"):
        pipeline.run_full_ffid_large_batch(
            config_path=config,
            interim_dir=interim,
            processed_dir=processed,
            output_root=output_root,
            device_override="cpu",
        )

    assert not output_root.exists()
