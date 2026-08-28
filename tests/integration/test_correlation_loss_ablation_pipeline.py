from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from seis_interp.configuration import (
    REPOSITORY_ROOT,
    get_required_config_value,
    load_resolved_config,
)
from seis_interp.data.file_checksums import file_sha256
from seis_interp.pipelines import correlation_loss_ablation as pipeline
from seis_interp.processing.trace_splits import TRAIN_SPLIT
from tests.integration.test_domain_scaling_pipeline import build_experiment_fixture


def test_study_005_config_locks_the_fixed_conditions() -> None:
    config = load_resolved_config(
        REPOSITORY_ROOT / "studies" / "study_005_correlation_loss_ablation" / "config.yaml",
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
        "training.validation_batch_size": 65536,
        "training.device": "cuda:0",
        "experiment.trace_count": 8,
        "experiment.total_updates": 50000,
        "experiment.report_interval": 500,
        "experiment.full_batch": True,
        "experiment.correlation_eps": 1.0e-4,
        "experiment.conditions": [
            {"label": "mse_control", "correlation_weight": 0.0},
            {"label": "mse_corr_0p1", "correlation_weight": 0.1},
        ],
    }

    assert get_required_config_value(config, "study.status") in {"active", "completed"}
    assert {path: get_required_config_value(config, path) for path in expected} == expected


def _build_ablation_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    config, interim, processed = build_experiment_fixture(tmp_path)
    config_data = yaml.safe_load(config.read_text(encoding="utf-8"))
    config_data["experiment"] = {
        "trace_count": 2,
        "total_updates": 2,
        "report_interval": 1,
        "full_batch": True,
        "correlation_eps": 1.0e-4,
        "conditions": [
            {"label": "mse_control", "correlation_weight": 0.0},
            {"label": "mse_corr_0p1", "correlation_weight": 0.1},
        ],
    }
    config.write_text(yaml.safe_dump(config_data, sort_keys=False), encoding="utf-8")
    return config, interim, processed


def test_ablation_writes_paired_immutable_training_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, interim, processed = _build_ablation_fixture(tmp_path)
    output_root = tmp_path / "runs" / "study_005_correlation_loss_ablation"
    monkeypatch.setattr(pipeline, "_run_id_timestamp", lambda: "20260826T010203Z")

    actual_build_model = pipeline._build_model
    models: list[torch.nn.Module] = []
    initial_states: list[dict[str, torch.Tensor]] = []
    forward_input_shapes: list[tuple[int, ...]] = []

    def record_forward_input_shape(
        _model: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
    ) -> None:
        forward_input_shapes.append(tuple(inputs[0].shape))

    def recording_build_model(*args: Any, **kwargs: Any) -> torch.nn.Module:
        model = actual_build_model(*args, **kwargs)
        models.append(model)
        initial_states.append(
            {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        )
        model.register_forward_pre_hook(record_forward_input_shape)
        return model

    actual_build_trace_points = pipeline.build_trace_points
    selected_rows_seen: list[np.ndarray] = []

    def recording_build_trace_points(*args: Any, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        selected_rows_seen.append(np.asarray(args[3]).copy())
        return actual_build_trace_points(*args, **kwargs)

    monkeypatch.setattr(pipeline, "_build_model", recording_build_model)
    monkeypatch.setattr(pipeline, "build_trace_points", recording_build_trace_points)

    summary = pipeline.run_correlation_loss_ablation(
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
    assert len(models) == len(initial_states) == 2
    assert models[0] is not models[1]
    assert initial_states[0].keys() == initial_states[1].keys()
    for name in initial_states[0]:
        assert torch.equal(initial_states[0][name], initial_states[1][name]), name

    split_table = pd.read_parquet(processed / "trace_split.parquet")
    training_rows = set(split_table.loc[split_table["split"] == TRAIN_SPLIT, "array_row"])
    assert len(selected_rows_seen) == 1
    assert all(set(rows.tolist()) == training_rows for rows in selected_rows_seen)
    assert forward_input_shapes
    assert all(shape == (10, 6) for shape in forward_input_shapes)

    required_history_keys = {
        "step",
        "mse_loss",
        "correlation_loss",
        "total_loss",
        "training_median_trace_snr_db",
        "training_global_snr_db",
        "training_median_trace_correlation",
        "training_prediction_target_rms_ratio",
    }
    metrics_by_label: dict[str, dict[str, object]] = {}
    saved_configs: dict[str, dict[str, object]] = {}
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
        metrics = json.loads((run_directory / "metrics.json").read_text(encoding="utf-8"))
        inputs_lock = json.loads((run_directory / "inputs.lock.json").read_text(encoding="utf-8"))
        run_metadata = json.loads((run_directory / "run.json").read_text(encoding="utf-8"))
        label = metrics["label"]
        metrics_by_label[label] = metrics
        saved_configs[label] = saved_config

        assert saved_config["training"]["device"] == "cpu"
        assert saved_config["experiment"]["active_condition"] == {
            "label": label,
            "correlation_weight": metrics["correlation_weight"],
        }
        assert metrics["selected_array_rows"] == inputs_lock["selection"]["selected_array_rows"]
        assert set(metrics["selected_array_rows"]) == training_rows
        assert metrics["updates_completed"] == 2
        assert [row["step"] for row in metrics["history"]] == [1, 2]
        assert all(set(row) == required_history_keys for row in metrics["history"])
        for row in metrics["history"]:
            assert row["total_loss"] == pytest.approx(
                row["mse_loss"] + metrics["correlation_weight"] * row["correlation_loss"]
            )

        best_row = max(metrics["history"], key=lambda row: row["training_median_trace_snr_db"])
        assert metrics["best_step"] == best_row["step"]
        final_row = metrics["history"][-1]
        for metric_name in (
            "training_median_trace_snr_db",
            "training_global_snr_db",
            "training_median_trace_correlation",
            "training_prediction_target_rms_ratio",
        ):
            assert metrics[f"best_{metric_name}"] == best_row[metric_name]
            assert metrics[f"final_{metric_name}"] == final_row[metric_name]

        assert inputs_lock["selection"]["trace_count"] == 2
        assert inputs_lock["selection"]["sample_count"] == 5
        assert inputs_lock["selection"]["full_batch"] is True
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
        assert {
            name: record["sha256"] for name, record in inputs_lock["interim_files"].items()
        } == {name: file_sha256(interim / name) for name in inputs_lock["interim_files"]}
        assert {
            name: record["sha256"] for name, record in inputs_lock["processed_files"].items()
        } == {name: file_sha256(processed / name) for name in inputs_lock["processed_files"]}
        assert run_metadata["study_id"] == "study_005_correlation_loss_ablation"
        assert run_metadata["condition"] == label
        assert run_metadata["correlation_weight"] == metrics["correlation_weight"]
        assert run_metadata["trace_count"] == 2
        assert run_metadata["sample_count"] == 5
        assert run_metadata["updates_completed"] == 2
        assert run_metadata["status"] == "success"
        assert run_metadata["device"] == "cpu"
        assert run_metadata["python_version"] == platform.python_version()
        assert run_metadata["torch_version"] == str(torch.__version__)
        assert run_metadata["random_seed"] == 5
        assert (
            run_metadata["git_commit"]
            == subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=REPOSITORY_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        started_at = datetime.fromisoformat(run_metadata["started_at_utc"].replace("Z", "+00:00"))
        finished_at = datetime.fromisoformat(run_metadata["finished_at_utc"].replace("Z", "+00:00"))
        assert started_at.tzinfo == timezone.utc
        assert finished_at.tzinfo == timezone.utc
        assert started_at <= finished_at
        for path in run_directory.iterdir():
            assert str(tmp_path) not in path.read_text(encoding="utf-8")

    assert set(metrics_by_label) == {"mse_control", "mse_corr_0p1"}
    assert metrics_by_label["mse_control"]["correlation_weight"] == 0.0
    assert metrics_by_label["mse_corr_0p1"]["correlation_weight"] == 0.1
    assert (
        metrics_by_label["mse_control"]["selected_array_rows"]
        == metrics_by_label["mse_corr_0p1"]["selected_array_rows"]
    )
    control_config = saved_configs["mse_control"]
    correlation_config = saved_configs["mse_corr_0p1"]
    control_active = control_config["experiment"].pop("active_condition")
    correlation_active = correlation_config["experiment"].pop("active_condition")
    assert control_config == correlation_config
    assert control_active == {"label": "mse_control", "correlation_weight": 0.0}
    assert correlation_active == {"label": "mse_corr_0p1", "correlation_weight": 0.1}
    assert {run["run_id"] for run in summary["runs"]} == {path.name for path in run_directories}
    assert summary["decision"] in {
        "full_batch_control_succeeds",
        "correlation_loss_promising",
        "correlation_loss_not_effective",
        "correlation_loss_inflates_amplitude_without_alignment",
    }
    assert str(tmp_path) not in summary_paths[0].read_text(encoding="utf-8")

    original_files = {path: path.read_bytes() for path in output_root.rglob("*") if path.is_file()}
    initial_state_count = len(initial_states)
    with pytest.raises(FileExistsError, match="already exist"):
        pipeline.run_correlation_loss_ablation(
            config_path=config,
            interim_dir=interim,
            processed_dir=processed,
            output_root=output_root,
            device_override="cpu",
        )
    assert len(initial_states) == initial_state_count
    assert {
        path: path.read_bytes() for path in output_root.rglob("*") if path.is_file()
    } == original_files
