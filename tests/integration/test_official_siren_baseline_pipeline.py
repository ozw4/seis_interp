from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from seis_interp.configuration import REPOSITORY_ROOT
from seis_interp.data.file_checksums import file_sha256
from seis_interp.pipelines import batching_ablation as pipeline
from seis_interp.pipelines.prepare_baseline import prepare_baseline_dataset
from seis_interp.processing.trace_splits import TRAIN_SPLIT
from tests.fixtures.siren_experiment import build_experiment_fixture


def _build_official_siren_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
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
            "omega_0": 300.0,
            "hidden_omega": 1.0,
        }
    )
    config_data["training"].update(
        {
            "batch_size": 4,
            "total_updates": 2,
            "report_interval": 1,
            "prediction_batch_size": 3,
        }
    )
    # Reversed input order demonstrates that execution order does not choose the result.
    config_data["experiment"] = {
        "trace_count": 2,
        "batch_mode": "random_replacement",
        "replacement": True,
        "conditions": [
            {
                "label": "official_siren_30",
                "omega_0": 30.0,
                "hidden_omega": 30.0,
            },
            {
                "label": "legacy_control",
                "omega_0": 300.0,
                "hidden_omega": 1.0,
            },
        ],
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


def _assert_finite_json(value: object) -> None:
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, Mapping):
        for child in value.values():
            _assert_finite_json(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            _assert_finite_json(child)


def test_official_siren_baseline_writes_two_isolated_immutable_cpu_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, interim, processed = _build_official_siren_fixture(tmp_path)
    output_root = tmp_path / "runs" / pipeline.OFFICIAL_SIREN_BASELINE_STUDY_ID
    monkeypatch.setattr(pipeline, "_run_id_timestamp", lambda: "20260827T010203Z")

    actual_build_model = pipeline._build_model
    actual_adam = pipeline.torch.optim.Adam
    actual_sampler = pipeline.RandomPointSampler
    actual_build_trace_points = pipeline.build_trace_points
    actual_predict_points = pipeline.predict_points
    built_models: list[torch.nn.Module] = []
    built_optimizers: list[torch.optim.Optimizer] = []
    model_contracts: list[tuple[float, float]] = []
    numpy_states: list[np.ndarray] = []
    torch_states: list[torch.Tensor] = []
    sampler_rows: list[np.ndarray] = []
    sampler_seeds: list[int] = []
    sampled_batches: list[list[tuple[np.ndarray, np.ndarray]]] = []
    evaluation_rows: list[np.ndarray] = []
    prediction_batch_sizes: list[int] = []

    def recording_build_model(config: Mapping[str, object]) -> torch.nn.Module:
        model_config = config["model"]
        assert isinstance(model_config, Mapping)
        model_contracts.append(
            (float(model_config["omega_0"]), float(model_config["hidden_omega"]))
        )
        numpy_states.append(np.random.get_state()[1].copy())
        torch_states.append(torch.random.get_rng_state().clone())
        model = actual_build_model(config)
        built_models.append(model)
        return model

    def recording_adam(*args: Any, **kwargs: Any) -> torch.optim.Optimizer:
        optimizer = actual_adam(*args, **kwargs)
        built_optimizers.append(optimizer)
        return optimizer

    class RecordingSampler:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            sampler_rows.append(np.asarray(args[3]).copy())
            sampler_seeds.append(int(kwargs["random_seed"]))
            sampled_batches.append([])
            self._batch_record = sampled_batches[-1]
            self._delegate = actual_sampler(*args, **kwargs)

        def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
            coordinates, targets = self._delegate.sample(batch_size)
            self._batch_record.append((coordinates.copy(), targets.copy()))
            return coordinates, targets

    def recording_build_trace_points(*args: Any, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        evaluation_rows.append(np.asarray(args[3]).copy())
        return actual_build_trace_points(*args, **kwargs)

    def recording_predict_points(*args: Any, **kwargs: Any) -> np.ndarray:
        prediction_batch_sizes.append(int(kwargs["batch_size"]))
        return actual_predict_points(*args, **kwargs)

    monkeypatch.setattr(pipeline, "_build_model", recording_build_model)
    monkeypatch.setattr(pipeline.torch.optim, "Adam", recording_adam)
    monkeypatch.setattr(pipeline, "RandomPointSampler", RecordingSampler)
    monkeypatch.setattr(pipeline, "build_trace_points", recording_build_trace_points)
    monkeypatch.setattr(pipeline, "predict_points", recording_predict_points)

    summary = pipeline.run_official_siren_baseline(
        config_path=config,
        interim_dir=interim,
        processed_dir=processed,
        output_root=output_root,
        device_override="cpu",
    )

    git_commit = _git_head()
    prefix = f"20260827T010203Z_{git_commit[:7]}"
    expected_run_ids = [f"{prefix}_legacy_control", f"{prefix}_official_siren_30"]
    run_directories = sorted(path for path in output_root.iterdir() if path.is_dir())
    summary_path = output_root / f"{prefix}_summary.json"
    assert sorted(path.name for path in run_directories) == sorted(expected_run_ids)
    assert json.loads(summary_path.read_text(encoding="utf-8")) == summary
    assert set(summary) == {
        "study_id",
        "git_commit",
        "generated_at_utc",
        "decision",
        "control_validity",
        "point_evaluations_per_condition",
        "conditions",
    }
    assert summary["study_id"] == pipeline.OFFICIAL_SIREN_BASELINE_STUDY_ID
    assert summary["git_commit"] == git_commit
    assert summary["point_evaluations_per_condition"] == 8
    assert len(summary["conditions"]) == 2

    split_table = pd.read_parquet(processed / "trace_split.parquet")
    training_rows = set(split_table.loc[split_table["split"] == TRAIN_SPLIT, "array_row"])
    held_out_rows = set(split_table.loc[split_table["split"] != TRAIN_SPLIT, "array_row"])
    assert len(evaluation_rows) == 1
    assert set(evaluation_rows[0]) == training_rows
    assert set(evaluation_rows[0]).isdisjoint(held_out_rows)
    assert len(sampler_rows) == 2
    assert all(set(rows) == training_rows for rows in sampler_rows)
    assert all(set(rows).isdisjoint(held_out_rows) for rows in sampler_rows)

    assert model_contracts == [(300.0, 1.0), (30.0, 30.0)]
    assert len(built_models) == len(built_optimizers) == 2
    assert built_models[0] is not built_models[1]
    assert built_optimizers[0] is not built_optimizers[1]
    np.testing.assert_array_equal(numpy_states[0], numpy_states[1])
    torch.testing.assert_close(torch_states[0], torch_states[1], rtol=0.0, atol=0.0)
    assert sampler_seeds == [42, 42]
    assert len(sampled_batches) == 2
    assert len(sampled_batches[0]) == len(sampled_batches[1]) == 2
    for legacy_batch, official_batch in zip(sampled_batches[0], sampled_batches[1], strict=True):
        np.testing.assert_array_equal(legacy_batch[0], official_batch[0])
        np.testing.assert_array_equal(legacy_batch[1], official_batch[1])
    assert prediction_batch_sizes == [3, 3, 3, 3]

    summary_by_label = {str(condition["label"]): condition for condition in summary["conditions"]}
    assert list(summary_by_label) == ["legacy_control", "official_siren_30"]
    metrics_by_label: dict[str, dict[str, object]] = {}
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
        label = str(metrics["condition"])
        metrics_by_label[label] = metrics
        expected_omegas = {
            "legacy_control": (300.0, 1.0),
            "official_siren_30": (30.0, 30.0),
        }[label]

        assert saved_config["training"]["batch_size"] == 4
        assert saved_config["training"]["total_updates"] == 2
        assert saved_config["training"]["report_interval"] == 1
        assert saved_config["training"]["prediction_batch_size"] == 3
        assert saved_config["training"]["device"] == "cpu"
        assert saved_config["training"]["loss"] == "l2"
        assert saved_config["model"]["omega_0"] == expected_omegas[0]
        assert saved_config["model"]["hidden_omega"] == expected_omegas[1]
        assert saved_config["experiment"]["active_condition"] == {
            "label": label,
            "omega_0": expected_omegas[0],
            "hidden_omega": expected_omegas[1],
        }

        assert metrics["selected_array_rows"] == sorted(training_rows)
        assert metrics["trace_count"] == 2
        assert metrics["sample_count"] == 5
        assert metrics["point_count"] == 10
        assert metrics["batch_size"] == 4
        assert metrics["point_evaluations"] == 8
        assert metrics["updates_completed"] == 2
        assert metrics["omega_0"] == expected_omegas[0]
        assert metrics["hidden_omega"] == expected_omegas[1]
        assert [row["step"] for row in metrics["history"]] == [1, 2]

        condition_summary = summary_by_label[label]
        assert set(condition_summary) == {
            "label",
            "run_directory",
            "omega_0",
            "hidden_omega",
            "classification",
            "best_report",
            "final_report",
            "updates_completed",
        }
        assert condition_summary["run_directory"] == run_directory.name
        assert condition_summary["classification"] == metrics["classification"]
        assert condition_summary["updates_completed"] == 2
        assert condition_summary["omega_0"] == expected_omegas[0]
        assert condition_summary["hidden_omega"] == expected_omegas[1]
        best_row = max(metrics["history"], key=lambda row: row["training_median_trace_snr_db"])
        assert condition_summary["best_report"] == best_row
        assert condition_summary["final_report"] == metrics["history"][-1]

        assert inputs_lock["selection"]["source_split"] == TRAIN_SPLIT
        assert inputs_lock["selection"]["selected_array_rows"] == sorted(training_rows)
        assert inputs_lock["selection"]["random_seed"] == 42
        assert inputs_lock["preparation"]["random_seed"] == 42
        assert inputs_lock["split"] == {
            "counts": {"train": 2, "validation": 1, "test": 1},
            "training_source": TRAIN_SPLIT,
        }
        assert inputs_lock["training"] == {
            "condition": label,
            "batch_mode": "random_replacement",
            "replacement": True,
            "batch_size": 4,
            "total_updates": 2,
            "point_evaluations": 8,
            "omega_0": expected_omegas[0],
            "hidden_omega": expected_omegas[1],
        }
        correlation_keys = {"correlation_weight", "correlation_eps", "loss_semantics"}
        assert correlation_keys.isdisjoint(metrics)
        assert correlation_keys.isdisjoint(inputs_lock["training"])
        assert correlation_keys.isdisjoint(run_metadata)
        for file_name, record in inputs_lock["interim_files"].items():
            assert record["sha256"] == file_sha256(interim / file_name)
        for file_name, record in inputs_lock["processed_files"].items():
            assert record["sha256"] == file_sha256(processed / file_name)

        assert run_metadata["study_id"] == pipeline.OFFICIAL_SIREN_BASELINE_STUDY_ID
        assert run_metadata["condition"] == label
        assert run_metadata["git_commit"] == git_commit
        assert run_metadata["status"] == "success"
        assert run_metadata["device"] == "cpu"
        assert run_metadata["random_seed"] == 42
        assert run_metadata["point_evaluations"] == 8
        assert run_metadata["updates_completed"] == 2
        assert run_metadata["omega_0"] == expected_omegas[0]
        assert run_metadata["hidden_omega"] == expected_omegas[1]

        for path in run_directory.iterdir():
            assert str(tmp_path) not in path.read_text(encoding="utf-8")

    assert set(metrics_by_label) == {"legacy_control", "official_siren_30"}
    legacy_classification = str(metrics_by_label["legacy_control"]["classification"])
    official_classification = str(metrics_by_label["official_siren_30"]["classification"])
    assert summary["control_validity"] is (legacy_classification == "near_zero")
    assert summary["decision"] == pipeline.official_siren_summary_decision(
        legacy_classification=legacy_classification,
        official_classification=official_classification,
    )
    _assert_finite_json(summary)
    for path in output_root.rglob("*.json"):
        _assert_finite_json(json.loads(path.read_text(encoding="utf-8")))

    assert {path.name for path in output_root.rglob("*") if path.is_file()} == {
        "config.resolved.yaml",
        "inputs.lock.json",
        "metrics.json",
        "run.json",
        summary_path.name,
    }
    original_files = {path: path.read_bytes() for path in output_root.rglob("*") if path.is_file()}
    build_count = len(built_models)
    sampler_count = len(sampler_rows)
    with pytest.raises(FileExistsError, match="already exist"):
        pipeline.run_official_siren_baseline(
            config_path=config,
            interim_dir=interim,
            processed_dir=processed,
            output_root=output_root,
            device_override="cpu",
        )
    assert {path: path.read_bytes() for path in original_files} == original_files
    assert len(built_models) == build_count
    assert len(sampler_rows) == sampler_count
