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
from tests.integration.test_domain_scaling_pipeline import build_experiment_fixture


def test_study_008_config_locks_complete_trace_batches_with_pure_l2() -> None:
    config = load_resolved_config(
        REPOSITORY_ROOT / "studies" / "study_008_full_ffid_trace_batches" / "config.yaml",
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
        "experiment.batch_mode": "random_complete_traces",
        "experiment.traces_per_update": 8,
        "experiment.replacement": False,
    }

    assert {path: get_required_config_value(config, path) for path in expected} == expected
    training = config["training"]
    experiment = config["experiment"]
    assert isinstance(training, dict)
    assert isinstance(experiment, dict)
    assert "correlation_weight" not in training
    assert "correlation_weight" not in experiment
    assert "correlation_eps" not in experiment


def _build_trace_batch_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    config, interim, processed = build_experiment_fixture(tmp_path)
    config_data = yaml.safe_load(config.read_text(encoding="utf-8"))
    config_data["training"].update(
        {
            "batch_size": 5,
            "total_updates": 2,
            "report_interval": 1,
            "prediction_batch_size": 3,
        }
    )
    config_data["experiment"] = {
        "trace_count": 2,
        "batch_mode": "random_complete_traces",
        "traces_per_update": 1,
        "replacement": False,
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


def test_trace_batch_probe_writes_one_split_isolated_immutable_cpu_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, interim, processed = _build_trace_batch_fixture(tmp_path)
    output_root = tmp_path / "runs" / "study_008_full_ffid_trace_batches"
    monkeypatch.setattr(pipeline, "_run_id_timestamp", lambda: "20260826T010203Z")

    actual_sampler = pipeline.RandomTraceBatchSampler
    actual_build_trace_points = pipeline.build_trace_points
    actual_predict_points = pipeline.predict_points
    sampler_rows: list[np.ndarray] = []
    sampled_trace_counts: list[int] = []
    sampled_rows: list[list[int]] = []
    evaluation_rows: list[np.ndarray] = []
    prediction_batch_sizes: list[int] = []

    class RecordingSampler:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._time = np.asarray(args[0])
            self._spatial = np.asarray(args[1])
            self._allowed_rows = np.asarray(args[3]).copy()
            sampler_rows.append(self._allowed_rows.copy())
            self._sampler = actual_sampler(*args, **kwargs)

        def sample(self, traces_per_update: int) -> tuple[np.ndarray, np.ndarray]:
            sampled_trace_counts.append(traces_per_update)
            coordinates, targets = self._sampler.sample(traces_per_update)
            trace_coordinates = coordinates.reshape(traces_per_update, len(self._time), 6)
            np.testing.assert_array_equal(
                trace_coordinates[:, :, 0],
                np.tile(self._time, (traces_per_update, 1)),
            )
            batch_rows: list[int] = []
            for trace_index in range(traces_per_update):
                spatial_coordinate = trace_coordinates[trace_index, 0, 1:]
                matches = np.flatnonzero(
                    np.all(self._spatial[self._allowed_rows] == spatial_coordinate, axis=1)
                )
                assert len(matches) == 1
                batch_rows.append(int(self._allowed_rows[int(matches[0])]))
            assert len(set(batch_rows)) == traces_per_update
            sampled_rows.append(batch_rows)
            return coordinates, targets

    def recording_build_trace_points(*args: Any, **kwargs: Any) -> Any:
        evaluation_rows.append(np.asarray(args[3]).copy())
        return actual_build_trace_points(*args, **kwargs)

    def recording_predict_points(*args: Any, **kwargs: Any) -> Any:
        prediction_batch_sizes.append(int(kwargs["batch_size"]))
        return actual_predict_points(*args, **kwargs)

    monkeypatch.setattr(pipeline, "RandomTraceBatchSampler", RecordingSampler)
    monkeypatch.setattr(pipeline, "build_trace_points", recording_build_trace_points)
    monkeypatch.setattr(pipeline, "predict_points", recording_predict_points)

    summary = pipeline.run_full_ffid_trace_batches(
        config_path=config,
        interim_dir=interim,
        processed_dir=processed,
        output_root=output_root,
        device_override="cpu",
    )

    git_commit = _git_head()
    expected_run_id = f"20260826T010203Z_{git_commit[:7]}_tracebatch1_trace2"
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
    assert sampled_trace_counts == [1, 1]
    assert all(set(rows) <= training_rows for rows in sampled_rows)
    assert all(set(rows).isdisjoint(held_out_rows) for rows in sampled_rows)
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
    assert saved_config["training"]["loss"] == "l2"
    assert saved_config["training"]["device"] == "cpu"
    assert saved_config["training"]["batch_size"] == 5
    assert saved_config["training"]["total_updates"] == 2
    assert saved_config["training"]["prediction_batch_size"] == 3
    assert saved_config["experiment"] == {
        "trace_count": 2,
        "batch_mode": "random_complete_traces",
        "traces_per_update": 1,
        "replacement": False,
    }

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
    assert metrics["condition"] == "tracebatch1_trace2"
    assert metrics["batch_mode"] == "random_complete_traces"
    assert metrics["full_batch"] is False
    assert metrics["replacement"] is False
    assert metrics["traces_per_update"] == 1
    assert metrics["trace_count"] == 2
    assert metrics["sample_count"] == 5
    assert metrics["point_count"] == 10
    assert metrics["batch_size"] == 5
    assert metrics["point_evaluations"] == 10
    assert metrics["updates_completed"] == 2
    assert [row["step"] for row in metrics["history"]] == [1, 2]
    assert all(set(row) == required_history_keys for row in metrics["history"])
    assert all(math.isfinite(float(value)) for row in metrics["history"] for value in row.values())
    best_row = max(metrics["history"], key=lambda row: row["training_median_trace_snr_db"])
    assert metrics["best_step"] == best_row["step"]
    assert metrics["classification"] in {
        "strong_fit",
        "escaped_zero_predictor",
        "near_zero",
    }

    assert inputs_lock["selection"]["source_split"] == TRAIN_SPLIT
    assert inputs_lock["selection"]["selected_array_rows"] == sorted(training_rows)
    assert inputs_lock["selection"]["trace_count"] == 2
    assert inputs_lock["selection"]["sample_count"] == 5
    assert inputs_lock["selection"]["point_count"] == 10
    assert inputs_lock["split"] == {
        "counts": {"train": 2, "validation": 1, "test": 1},
        "training_source": TRAIN_SPLIT,
    }
    assert inputs_lock["preparation"]["normalization"] == {
        "coordinates": "train_minmax_linear_plus_azimuth_sin_cos",
        "amplitude": "train_global_rms",
    }
    assert inputs_lock["training"] == {
        "batch_mode": "random_complete_traces",
        "replacement": False,
        "batch_size": 5,
        "total_updates": 2,
        "point_evaluations": 10,
        "traces_per_update": 1,
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

    assert run_metadata["study_id"] == "study_008_full_ffid_trace_batches"
    assert run_metadata["condition"] == "tracebatch1_trace2"
    assert run_metadata["git_commit"] == git_commit
    assert run_metadata["status"] == "success"
    assert run_metadata["device"] == "cpu"
    assert run_metadata["random_seed"] == 5
    assert run_metadata["batch_mode"] == "random_complete_traces"
    assert run_metadata["full_batch"] is False
    assert run_metadata["replacement"] is False
    assert run_metadata["traces_per_update"] == 1
    assert run_metadata["batch_size"] == 5
    assert run_metadata["trace_count"] == 2
    assert run_metadata["sample_count"] == 5
    assert run_metadata["point_count"] == 10
    assert run_metadata["point_evaluations"] == 10
    assert run_metadata["updates_completed"] == 2
    assert run_metadata["python_version"]
    assert run_metadata["torch_version"]
    assert run_metadata["started_at_utc"].endswith("Z")
    assert run_metadata["finished_at_utc"].endswith("Z")
    assert summary["study_id"] == "study_008_full_ffid_trace_batches"
    assert summary["point_evaluations"] == 10
    assert summary["decision"] == pipeline.full_ffid_summary_decision(metrics["classification"])

    for path in output_root.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".yaml"}:
            assert str(tmp_path) not in path.read_text(encoding="utf-8")

    original_files = {path: path.read_bytes() for path in output_root.rglob("*") if path.is_file()}
    with pytest.raises(FileExistsError, match="already exist"):
        pipeline.run_full_ffid_trace_batches(
            config_path=config,
            interim_dir=interim,
            processed_dir=processed,
            output_root=output_root,
            device_override="cpu",
        )
    assert {
        path: path.read_bytes() for path in output_root.rglob("*") if path.is_file()
    } == original_files
    assert len(sampler_rows) == 1


def test_trace_batch_probe_requires_consistent_batch_size_and_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, interim, processed = _build_trace_batch_fixture(tmp_path)
    config_data = yaml.safe_load(config.read_text(encoding="utf-8"))
    config_data["training"]["batch_size"] = 6
    config.write_text(yaml.safe_dump(config_data, sort_keys=False), encoding="utf-8")
    output_root = tmp_path / "runs"
    monkeypatch.setattr(pipeline, "_run_id_timestamp", lambda: "20260826T010203Z")

    with pytest.raises(
        pipeline.ConfigurationError,
        match=r"training.batch_size must equal experiment.traces_per_update \* sample_count \(5\)",
    ):
        pipeline.run_full_ffid_trace_batches(
            config_path=config,
            interim_dir=interim,
            processed_dir=processed,
            output_root=output_root,
            device_override="cpu",
        )

    assert not output_root.exists()
