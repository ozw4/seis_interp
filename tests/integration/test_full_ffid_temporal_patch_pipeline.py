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


def test_study_010_config_locks_shared_temporal_patch_batches_with_pure_mse() -> None:
    config = load_resolved_config(
        REPOSITORY_ROOT / "studies" / "study_010_full_ffid_temporal_patches" / "config.yaml",
        repository_root=REPOSITORY_ROOT,
    )
    expected_patch_starts = [*range(0, 545, 32), 561]
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
        "training.batch_size": 4992,
        "training.total_updates": 50000,
        "training.report_interval": 500,
        "training.prediction_batch_size": 65536,
        "training.device": "cuda:0",
        "experiment.trace_count": 435,
        "experiment.batch_mode": "random_shared_temporal_patch",
        "experiment.traces_per_update": 78,
        "experiment.samples_per_trace": 64,
        "experiment.temporal_patch_overlap_fraction": 0.5,
        "experiment.patch_starts": expected_patch_starts,
        "experiment.shared_temporal_patch": True,
        "experiment.replacement": False,
    }

    assert get_required_config_value(config, "study.status") in {"active", "completed"}
    assert {path: get_required_config_value(config, path) for path in expected} == expected
    assert expected["training.batch_size"] == (
        expected["experiment.traces_per_update"] * expected["experiment.samples_per_trace"]
    )
    assert expected["training.total_updates"] // expected["training.report_interval"] == 100
    assert "correlation_weight" not in config["training"]
    assert "correlation_weight" not in config["experiment"]
    assert "correlation_eps" not in config["experiment"]


def _build_temporal_patch_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    config, interim, processed = build_experiment_fixture(tmp_path)
    config_data = yaml.safe_load(config.read_text(encoding="utf-8"))
    config_data["training"].update(
        {
            "batch_size": 8,
            "total_updates": 2,
            "report_interval": 1,
            "prediction_batch_size": 3,
        }
    )
    config_data["experiment"] = {
        "trace_count": 2,
        "batch_mode": "random_shared_temporal_patch",
        "traces_per_update": 2,
        "samples_per_trace": 4,
        "temporal_patch_overlap_fraction": 0.5,
        "patch_starts": [0, 1],
        "shared_temporal_patch": True,
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


def test_temporal_patch_probe_writes_one_split_isolated_immutable_cpu_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, interim, processed = _build_temporal_patch_fixture(tmp_path)
    output_root = tmp_path / "runs" / "study_010_full_ffid_temporal_patches"
    monkeypatch.setattr(pipeline, "_run_id_timestamp", lambda: "20260826T010203Z")

    actual_sampler = pipeline.RandomTracePatchSampler
    actual_build_trace_points = pipeline.build_trace_points
    actual_predict_points = pipeline.predict_points
    sampler_rows: list[np.ndarray] = []
    evaluation_rows: list[np.ndarray] = []
    sampled_trace_counts: list[int] = []
    sampled_rows: list[list[int]] = []
    sampled_patch_starts: list[int] = []
    sampler_patch_contracts: list[tuple[int, list[int]]] = []
    prediction_batch_sizes: list[int] = []

    class RecordingSampler:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._time = np.asarray(args[0])
            self._spatial = np.asarray(args[1])
            self._allowed_rows = np.asarray(args[3]).copy()
            self._patch_size = int(kwargs["patch_size"])
            self._patch_starts = np.asarray(kwargs["patch_starts"], dtype=np.int64)
            sampler_rows.append(self._allowed_rows.copy())
            sampler_patch_contracts.append(
                (self._patch_size, [int(value) for value in self._patch_starts])
            )
            self._sampler = actual_sampler(*args, **kwargs)

        def sample(self, traces_per_update: int) -> tuple[np.ndarray, np.ndarray]:
            sampled_trace_counts.append(traces_per_update)
            coordinates, targets = self._sampler.sample(traces_per_update)
            trace_coordinates = coordinates.reshape(traces_per_update, self._patch_size, 6)
            np.testing.assert_array_equal(
                trace_coordinates[:, :, 0],
                np.tile(trace_coordinates[0, :, 0], (traces_per_update, 1)),
            )
            matching_starts = [
                int(start)
                for start in self._patch_starts
                if np.array_equal(
                    trace_coordinates[0, :, 0],
                    self._time[start : start + self._patch_size],
                )
            ]
            assert len(matching_starts) == 1
            sampled_patch_starts.append(matching_starts[0])

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

    monkeypatch.setattr(pipeline, "RandomTracePatchSampler", RecordingSampler)
    monkeypatch.setattr(pipeline, "build_trace_points", recording_build_trace_points)
    monkeypatch.setattr(pipeline, "predict_points", recording_predict_points)

    summary = pipeline.run_full_ffid_temporal_patches(
        config_path=config,
        interim_dir=interim,
        processed_dir=processed,
        output_root=output_root,
        device_override="cpu",
    )

    git_commit = _git_head()
    expected_run_id = f"20260826T010203Z_{git_commit[:7]}_patch4_trace2_trace2"
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
    assert all(set(rows) <= training_rows for rows in sampled_rows)
    assert all(set(rows).isdisjoint(held_out_rows) for rows in sampled_rows)
    assert sampler_patch_contracts == [(4, [0, 1])]
    assert len(sampled_patch_starts) == 2
    assert set(sampled_patch_starts) <= {0, 1}
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
    assert saved_config["training"]["batch_size"] == 8
    assert saved_config["training"]["total_updates"] == 2
    assert saved_config["training"]["report_interval"] == 1
    assert saved_config["training"]["prediction_batch_size"] == 3
    assert saved_config["experiment"] == {
        "trace_count": 2,
        "batch_mode": "random_shared_temporal_patch",
        "traces_per_update": 2,
        "samples_per_trace": 4,
        "temporal_patch_overlap_fraction": 0.5,
        "patch_starts": [0, 1],
        "shared_temporal_patch": True,
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
    patch_provenance = {
        "traces_per_update": 2,
        "samples_per_trace": 4,
        "temporal_patch_overlap_fraction": 0.5,
        "patch_starts": [0, 1],
        "shared_temporal_patch": True,
    }
    assert metrics["condition"] == "patch4_trace2_trace2"
    assert metrics["batch_mode"] == "random_shared_temporal_patch"
    assert metrics["full_batch"] is False
    assert metrics["replacement"] is False
    assert metrics["trace_count"] == 2
    assert metrics["sample_count"] == 5
    assert metrics["point_count"] == 10
    assert metrics["batch_size"] == 8
    assert metrics["point_evaluations"] == 16
    assert metrics["updates_completed"] == 2
    assert {key: metrics[key] for key in patch_provenance} == patch_provenance
    assert [row["step"] for row in metrics["history"]] == [1, 2]
    assert len(metrics["history"]) == 2
    assert all(set(row) == required_history_keys for row in metrics["history"])
    assert all(math.isfinite(float(value)) for row in metrics["history"] for value in row.values())
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
        "batch_mode": "random_shared_temporal_patch",
        "replacement": False,
        "batch_size": 8,
        "total_updates": 2,
        "point_evaluations": 16,
        **patch_provenance,
    }
    for file_name, record in inputs_lock["interim_files"].items():
        assert record["sha256"] == file_sha256(interim / file_name)
    for file_name, record in inputs_lock["processed_files"].items():
        assert record["sha256"] == file_sha256(processed / file_name)

    assert run_metadata["study_id"] == "study_010_full_ffid_temporal_patches"
    assert run_metadata["condition"] == "patch4_trace2_trace2"
    assert run_metadata["git_commit"] == git_commit
    assert run_metadata["status"] == "success"
    assert run_metadata["device"] == "cpu"
    assert run_metadata["random_seed"] == 5
    assert run_metadata["batch_mode"] == "random_shared_temporal_patch"
    assert run_metadata["full_batch"] is False
    assert run_metadata["replacement"] is False
    assert run_metadata["batch_size"] == 8
    assert run_metadata["trace_count"] == 2
    assert run_metadata["sample_count"] == 5
    assert run_metadata["point_count"] == 10
    assert run_metadata["point_evaluations"] == 16
    assert run_metadata["updates_completed"] == 2
    assert {key: run_metadata[key] for key in patch_provenance} == patch_provenance
    assert run_metadata["python_version"]
    assert run_metadata["torch_version"]
    assert run_metadata["started_at_utc"].endswith("Z")
    assert run_metadata["finished_at_utc"].endswith("Z")

    summary_run = summary["runs"][0]
    assert summary["study_id"] == "study_010_full_ffid_temporal_patches"
    assert summary["point_evaluations"] == 16
    assert summary["decision"] == pipeline.full_ffid_summary_decision(metrics["classification"])
    assert summary_run["condition"] == metrics["condition"]
    assert summary_run["batch_size"] == 8
    assert summary_run["point_evaluations"] == 16

    correlation_keys = {
        "correlation_weight",
        "correlation_eps",
        "loss_semantics",
        "mean_train_mse_loss_since_last_report",
        "mean_train_correlation_loss_since_last_report",
    }
    for payload in (
        saved_config["training"],
        saved_config["experiment"],
        metrics,
        inputs_lock["training"],
        run_metadata,
        summary,
        summary_run,
    ):
        assert correlation_keys.isdisjoint(payload)

    for path in output_root.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".yaml"}:
            assert str(tmp_path) not in path.read_text(encoding="utf-8")

    original_files = {path: path.read_bytes() for path in output_root.rglob("*") if path.is_file()}
    with pytest.raises(FileExistsError, match="already exist"):
        pipeline.run_full_ffid_temporal_patches(
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


@pytest.mark.parametrize(
    ("invalid_contract", "message"),
    [
        (
            "patch_starts",
            "experiment.patch_starts must equal the predefined overlapping starts",
        ),
        (
            "non_integral_stride",
            r"patch_size \* \(1 - overlap_fraction\) must be a positive integer",
        ),
        (
            "batch_size",
            r"training\.batch_size must equal experiment\.traces_per_update "
            r"\* experiment\.samples_per_trace \(8\)",
        ),
        (
            "correlation_loss",
            "pure-MSE full-FFID experiments must not define correlation_weight or correlation_eps",
        ),
    ],
)
def test_temporal_patch_wrapper_rejects_invalid_contract_before_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_contract: str,
    message: str,
) -> None:
    config, interim, processed = _build_temporal_patch_fixture(tmp_path)
    config_data = yaml.safe_load(config.read_text(encoding="utf-8"))
    if invalid_contract == "patch_starts":
        config_data["experiment"]["patch_starts"] = [0]
    elif invalid_contract == "non_integral_stride":
        config_data["experiment"]["temporal_patch_overlap_fraction"] = 0.4
    elif invalid_contract == "batch_size":
        config_data["training"]["batch_size"] = 7
    elif invalid_contract == "correlation_loss":
        config_data["experiment"].update(
            {
                "correlation_weight": 0.1,
                "correlation_eps": 1.0e-4,
            }
        )
    else:  # pragma: no cover - parametrization is fixed above
        raise AssertionError(f"unknown invalid contract: {invalid_contract}")
    config.write_text(yaml.safe_dump(config_data, sort_keys=False), encoding="utf-8")
    output_root = tmp_path / "runs" / invalid_contract

    def unexpected_model_build(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("invalid wrapper configuration must fail before training")

    monkeypatch.setattr(pipeline, "_build_model", unexpected_model_build)

    with pytest.raises(ValueError, match=message):
        pipeline.run_full_ffid_temporal_patches(
            config_path=config,
            interim_dir=interim,
            processed_dir=processed,
            output_root=output_root,
            device_override="cpu",
        )

    assert not output_root.exists()
