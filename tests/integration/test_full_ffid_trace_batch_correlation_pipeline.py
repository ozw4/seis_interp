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
from seis_interp.training.correlation_loss import trace_correlation_loss
from tests.fixtures.siren_experiment import build_experiment_fixture


def test_study_009_config_locks_trace_batches_with_correlation_loss() -> None:
    config = load_resolved_config(
        REPOSITORY_ROOT / "studies" / "study_009_full_ffid_trace_batch_correlation" / "config.yaml",
        repository_root=REPOSITORY_ROOT,
    )
    expected = {
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
        "experiment.correlation_weight": 0.1,
        "experiment.correlation_eps": 1.0e-4,
    }

    assert get_required_config_value(config, "study.status") in {"active", "completed"}
    assert {path: get_required_config_value(config, path) for path in expected} == expected


def _build_trace_batch_correlation_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
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
        "batch_mode": "random_complete_traces",
        "traces_per_update": 2,
        "replacement": False,
        "correlation_weight": 0.1,
        "correlation_eps": 1.0e-4,
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


def test_trace_batch_correlation_probe_writes_one_split_isolated_immutable_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, interim, processed = _build_trace_batch_correlation_fixture(tmp_path)
    output_root = tmp_path / "runs" / "study_009_full_ffid_trace_batch_correlation"
    monkeypatch.setattr(pipeline, "_run_id_timestamp", lambda: "20260826T010203Z")

    actual_sampler = pipeline.RandomTraceBatchSampler
    actual_build_trace_points = pipeline.build_trace_points
    sampler_rows: list[np.ndarray] = []
    evaluation_rows: list[np.ndarray] = []
    sampled_trace_counts: list[int] = []
    correlation_shapes: list[tuple[int, ...]] = []
    correlation_epsilons: list[float] = []

    class RecordingSampler:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            sampler_rows.append(np.asarray(args[3]).copy())
            self._sampler = actual_sampler(*args, **kwargs)

        def sample(self, traces_per_update: int) -> tuple[np.ndarray, np.ndarray]:
            sampled_trace_counts.append(traces_per_update)
            return self._sampler.sample(traces_per_update)

    def recording_build_trace_points(*args: Any, **kwargs: Any) -> Any:
        evaluation_rows.append(np.asarray(args[3]).copy())
        return actual_build_trace_points(*args, **kwargs)

    def recording_correlation_loss(
        prediction: Any,
        target: Any,
        *,
        eps: float,
    ) -> Any:
        assert prediction.shape == target.shape
        correlation_shapes.append(tuple(prediction.shape))
        correlation_epsilons.append(eps)
        return trace_correlation_loss(prediction, target, eps=eps)

    monkeypatch.setattr(pipeline, "RandomTraceBatchSampler", RecordingSampler)
    monkeypatch.setattr(pipeline, "build_trace_points", recording_build_trace_points)
    monkeypatch.setattr(pipeline, "trace_correlation_loss", recording_correlation_loss)

    summary = pipeline.run_full_ffid_trace_batch_correlation(
        config_path=config,
        interim_dir=interim,
        processed_dir=processed,
        output_root=output_root,
        device_override="cpu",
    )

    git_commit = _git_head()
    expected_run_id = f"20260826T010203Z_{git_commit[:7]}_tracebatch2_corr0p1_trace2"
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
    assert sampled_trace_counts == [2, 2]
    assert correlation_shapes == [(2, 5), (2, 5)]
    assert correlation_epsilons == [1.0e-4, 1.0e-4]

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
    assert saved_config["experiment"] == {
        "trace_count": 2,
        "batch_mode": "random_complete_traces",
        "traces_per_update": 2,
        "replacement": False,
        "correlation_weight": 0.1,
        "correlation_eps": 1.0e-4,
    }

    metrics = json.loads((run_directory / "metrics.json").read_text(encoding="utf-8"))
    inputs_lock = json.loads((run_directory / "inputs.lock.json").read_text(encoding="utf-8"))
    run_metadata = json.loads((run_directory / "run.json").read_text(encoding="utf-8"))
    required_history_keys = {
        "step",
        "mean_train_loss_since_last_report",
        "mean_train_mse_loss_since_last_report",
        "mean_train_correlation_loss_since_last_report",
        "training_median_trace_snr_db",
        "training_global_snr_db",
        "training_median_trace_correlation",
        "training_prediction_target_rms_ratio",
    }
    assert metrics["condition"] == "tracebatch2_corr0p1_trace2"
    assert metrics["batch_mode"] == "random_complete_traces"
    assert metrics["full_batch"] is False
    assert metrics["replacement"] is False
    assert metrics["traces_per_update"] == 2
    assert metrics["trace_count"] == 2
    assert metrics["sample_count"] == 5
    assert metrics["point_count"] == 10
    assert metrics["batch_size"] == 10
    assert metrics["point_evaluations"] == 20
    assert metrics["updates_completed"] == 2
    assert metrics["correlation_weight"] == 0.1
    assert metrics["correlation_eps"] == 1.0e-4
    assert metrics["loss_semantics"] == "mse_plus_trace_correlation"
    assert [row["step"] for row in metrics["history"]] == [1, 2]
    assert all(set(row) == required_history_keys for row in metrics["history"])
    assert all(math.isfinite(float(value)) for row in metrics["history"] for value in row.values())
    for row in metrics["history"]:
        assert row["mean_train_loss_since_last_report"] == pytest.approx(
            row["mean_train_mse_loss_since_last_report"]
            + 0.1 * row["mean_train_correlation_loss_since_last_report"]
        )

    assert inputs_lock["selection"]["source_split"] == TRAIN_SPLIT
    assert inputs_lock["selection"]["selected_array_rows"] == sorted(training_rows)
    assert inputs_lock["selection"]["trace_count"] == 2
    assert inputs_lock["selection"]["sample_count"] == 5
    assert inputs_lock["split"] == {
        "counts": {"train": 2, "validation": 1, "test": 1},
        "training_source": TRAIN_SPLIT,
    }
    assert inputs_lock["training"] == {
        "batch_mode": "random_complete_traces",
        "replacement": False,
        "batch_size": 10,
        "total_updates": 2,
        "point_evaluations": 20,
        "traces_per_update": 2,
        "correlation_weight": 0.1,
        "correlation_eps": 1.0e-4,
        "loss_semantics": "mse_plus_trace_correlation",
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

    assert run_metadata["study_id"] == "study_009_full_ffid_trace_batch_correlation"
    assert run_metadata["condition"] == "tracebatch2_corr0p1_trace2"
    assert run_metadata["git_commit"] == git_commit
    assert run_metadata["status"] == "success"
    assert run_metadata["device"] == "cpu"
    assert run_metadata["random_seed"] == 5
    assert run_metadata["batch_mode"] == "random_complete_traces"
    assert run_metadata["batch_size"] == 10
    assert run_metadata["point_evaluations"] == 20
    assert run_metadata["updates_completed"] == 2
    assert run_metadata["correlation_weight"] == 0.1
    assert run_metadata["correlation_eps"] == 1.0e-4
    assert run_metadata["loss_semantics"] == "mse_plus_trace_correlation"
    assert run_metadata["python_version"]
    assert run_metadata["torch_version"]
    assert run_metadata["started_at_utc"].endswith("Z")
    assert run_metadata["finished_at_utc"].endswith("Z")

    summary_run = summary["runs"][0]
    assert summary["study_id"] == "study_009_full_ffid_trace_batch_correlation"
    assert summary["point_evaluations"] == 20
    assert summary["decision"] == pipeline.full_ffid_summary_decision(metrics["classification"])
    assert summary["correlation_weight"] == 0.1
    assert summary["correlation_eps"] == 1.0e-4
    assert summary["loss_semantics"] == "mse_plus_trace_correlation"
    assert summary_run["correlation_weight"] == 0.1
    assert summary_run["correlation_eps"] == 1.0e-4
    assert summary_run["loss_semantics"] == "mse_plus_trace_correlation"

    for path in output_root.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".yaml"}:
            assert str(tmp_path) not in path.read_text(encoding="utf-8")

    original_files = {path: path.read_bytes() for path in output_root.rglob("*") if path.is_file()}
    with pytest.raises(FileExistsError, match="already exist"):
        pipeline.run_full_ffid_trace_batch_correlation(
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
