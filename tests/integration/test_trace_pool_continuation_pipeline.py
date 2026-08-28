from __future__ import annotations

import json
import math
import subprocess
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
from seis_interp.pipelines import batching_ablation as pipeline
from seis_interp.pipelines.domain_scaling import deterministic_nested_trace_subsets
from seis_interp.processing.trace_splits import TRAIN_SPLIT
from tests.integration.test_domain_scaling_pipeline import build_experiment_fixture


def test_study_011_config_locks_the_seven_stage_continuation() -> None:
    config = load_resolved_config(
        REPOSITORY_ROOT / "studies" / "study_011_trace_pool_continuation" / "config.yaml",
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
        "training.updates_per_stage": 50000,
        "training.report_interval": 500,
        "training.prediction_batch_size": 65536,
        "training.device": "cuda:0",
        "experiment.trace_counts": [8, 16, 32, 64, 128, 256, 435],
        "experiment.batch_mode": "random_replacement",
        "experiment.replacement": True,
        "experiment.sampler_seed_policy": "base_seed_plus_stage_index",
        "experiment.carry_model_state": True,
        "experiment.carry_optimizer_state": True,
        "experiment.reset_optimizer_between_stages": False,
        "experiment.rewind_to_best": False,
        "experiment.checkpoint": False,
        "experiment.first_stage_final_min_median_trace_snr_db": 20.0,
    }

    assert get_required_config_value(config, "study.status") in {"active", "completed"}
    assert {path: get_required_config_value(config, path) for path in expected} == expected
    stage_count = len(expected["experiment.trace_counts"])
    assert stage_count == 7
    assert expected["training.updates_per_stage"] * stage_count == 350000
    assert expected["training.batch_size"] * 350000 == 1_750_000_000
    assert "total_updates" not in config["training"]
    assert "correlation_weight" not in config["experiment"]
    assert "correlation_eps" not in config["experiment"]


def _build_continuation_fixture(
    tmp_path: Path,
    *,
    anchor_threshold: float,
) -> tuple[Path, Path, Path]:
    config, interim, processed = build_experiment_fixture(tmp_path)
    config_data = yaml.safe_load(config.read_text(encoding="utf-8"))
    config_data["training"] = {
        "loss": "l2",
        "optimizer": "adam",
        "learning_rate": 1.0e-3,
        "batch_size": 4,
        "updates_per_stage": 2,
        "report_interval": 1,
        "prediction_batch_size": 3,
        "device": "cuda",
    }
    config_data.pop("experiment_a")
    config_data["experiment"] = {
        "trace_counts": [1, 2],
        "batch_mode": "random_replacement",
        "replacement": True,
        "sampler_seed_policy": "base_seed_plus_stage_index",
        "carry_model_state": True,
        "carry_optimizer_state": True,
        "reset_optimizer_between_stages": False,
        "rewind_to_best": False,
        "checkpoint": False,
        "first_stage_final_min_median_trace_snr_db": anchor_threshold,
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


def _optimizer_step(optimizer: torch.optim.Optimizer) -> int:
    steps = [int(state["step"].item()) for state in optimizer.state.values() if "step" in state]
    assert steps
    assert len(set(steps)) == 1
    return steps[0]


def test_continuation_writes_one_immutable_run_with_shared_model_and_adam(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, interim, processed = _build_continuation_fixture(
        tmp_path,
        anchor_threshold=-1.0e9,
    )
    output_root = tmp_path / "runs" / "study_011_trace_pool_continuation"
    monkeypatch.setattr(pipeline, "_run_id_timestamp", lambda: "20260826T010203Z")

    actual_build_model = pipeline._build_model
    actual_adam = pipeline.torch.optim.Adam
    actual_run_condition = pipeline.run_training_fit_condition
    actual_sampler = pipeline.RandomPointSampler
    actual_build_trace_points = pipeline.build_trace_points
    actual_predict_points = pipeline.predict_points
    built_models: list[torch.nn.Module] = []
    built_optimizers: list[torch.optim.Optimizer] = []
    stage_models: list[torch.nn.Module] = []
    stage_optimizers: list[torch.optim.Optimizer] = []
    stage_rows: list[np.ndarray] = []
    stage_seeds: list[int] = []
    sampler_rows: list[np.ndarray] = []
    sampler_seeds: list[int] = []
    evaluation_rows: list[np.ndarray] = []
    prediction_batch_sizes: list[int] = []

    def recording_build_model(*args: Any, **kwargs: Any) -> torch.nn.Module:
        model = actual_build_model(*args, **kwargs)
        built_models.append(model)
        return model

    def recording_adam(*args: Any, **kwargs: Any) -> torch.optim.Optimizer:
        optimizer = actual_adam(*args, **kwargs)
        built_optimizers.append(optimizer)
        return optimizer

    def recording_run_condition(*args: Any, **kwargs: Any) -> dict[str, object]:
        stage_models.append(kwargs["model"])
        stage_optimizers.append(kwargs["optimizer"])
        stage_rows.append(np.asarray(kwargs["selected_array_rows"]).copy())
        stage_seeds.append(int(kwargs["random_seed"]))
        return actual_run_condition(*args, **kwargs)

    def recording_sampler(*args: Any, **kwargs: Any) -> Any:
        sampler_rows.append(np.asarray(args[3]).copy())
        sampler_seeds.append(int(kwargs["random_seed"]))
        return actual_sampler(*args, **kwargs)

    def recording_build_trace_points(*args: Any, **kwargs: Any) -> Any:
        evaluation_rows.append(np.asarray(args[3]).copy())
        return actual_build_trace_points(*args, **kwargs)

    def recording_predict_points(*args: Any, **kwargs: Any) -> Any:
        prediction_batch_sizes.append(int(kwargs["batch_size"]))
        return actual_predict_points(*args, **kwargs)

    monkeypatch.setattr(pipeline, "_build_model", recording_build_model)
    monkeypatch.setattr(pipeline.torch.optim, "Adam", recording_adam)
    monkeypatch.setattr(pipeline, "run_training_fit_condition", recording_run_condition)
    monkeypatch.setattr(pipeline, "RandomPointSampler", recording_sampler)
    monkeypatch.setattr(pipeline, "build_trace_points", recording_build_trace_points)
    monkeypatch.setattr(pipeline, "predict_points", recording_predict_points)

    summary = pipeline.run_trace_pool_continuation(
        config_path=config,
        interim_dir=interim,
        processed_dir=processed,
        output_root=output_root,
        device_override="cpu",
    )

    git_commit = _git_head()
    expected_run_id = f"20260826T010203Z_{git_commit[:7]}_continuation1to2_random4"
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
    training_rows = split_table.loc[
        split_table["split"] == TRAIN_SPLIT,
        "array_row",
    ].to_numpy(dtype=np.int64)
    held_out_rows = set(split_table.loc[split_table["split"] != TRAIN_SPLIT, "array_row"])
    expected_subsets = deterministic_nested_trace_subsets(
        training_rows,
        (1, 2),
        random_seed=5,
    )
    assert len(built_models) == len(built_optimizers) == 1
    assert len(stage_models) == len(stage_optimizers) == 2
    assert all(model is built_models[0] for model in stage_models)
    assert all(optimizer is built_optimizers[0] for optimizer in stage_optimizers)
    assert _optimizer_step(built_optimizers[0]) == 4
    assert stage_seeds == sampler_seeds == [5, 6]
    assert len(stage_rows) == len(sampler_rows) == len(evaluation_rows) == 2
    for index, trace_count in enumerate((1, 2)):
        np.testing.assert_array_equal(stage_rows[index], expected_subsets[trace_count])
        np.testing.assert_array_equal(sampler_rows[index], expected_subsets[trace_count])
        np.testing.assert_array_equal(evaluation_rows[index], expected_subsets[trace_count])
        assert set(stage_rows[index]).isdisjoint(held_out_rows)
    assert set(stage_rows[0]) <= set(stage_rows[1])
    assert prediction_batch_sizes == [3, 3, 3, 3, 3, 3]

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
    metrics = json.loads((run_directory / "metrics.json").read_text(encoding="utf-8"))
    inputs_lock = json.loads((run_directory / "inputs.lock.json").read_text(encoding="utf-8"))
    run_metadata = json.loads((run_directory / "run.json").read_text(encoding="utf-8"))
    assert saved_config["training"]["device"] == "cpu"
    assert saved_config["training"]["batch_size"] == 4
    assert saved_config["training"]["updates_per_stage"] == 2
    assert saved_config["training"]["report_interval"] == 1
    assert saved_config["experiment"]["trace_counts"] == [1, 2]

    assert metrics["condition"] == "continuation1to2_random4"
    assert metrics["planned_stage_trace_counts"] == [1, 2]
    assert metrics["completed_stage_trace_counts"] == [1, 2]
    assert metrics["updates_per_stage"] == 2
    assert metrics["planned_total_updates"] == 4
    assert metrics["updates_completed"] == 4
    assert metrics["planned_point_evaluations"] == 16
    assert metrics["point_evaluations"] == 16
    assert metrics["anchor_reproduced"] is True
    assert metrics["classification_scope"] == "final_full_ffid_stage"
    assert metrics["final_full_ffid_classification"] == metrics["classification"]
    assert metrics["trace_count"] == 2
    assert metrics["sample_count"] == 5
    assert metrics["point_count"] == 10
    assert len(metrics["stages"]) == 2
    assert metrics["classification"] in {
        "strong_fit",
        "escaped_zero_predictor",
        "near_zero",
    }
    entry_keys = {
        "entry_training_median_trace_snr_db",
        "entry_training_global_snr_db",
        "entry_training_median_trace_correlation",
        "entry_training_prediction_target_rms_ratio",
    }
    for stage_index, stage in enumerate(metrics["stages"]):
        assert stage["stage_index"] == stage_index + 1
        assert stage["trace_count"] == stage_index + 1
        assert stage["sampler_seed"] == 5 + stage_index
        assert stage["updates_completed"] == 2
        assert stage["point_evaluations"] == 8
        assert entry_keys <= set(stage)
        assert all(math.isfinite(float(stage[key])) for key in entry_keys)
        assert len(stage["history"]) == 2
        expected_steps = [stage_index * 2 + 1, stage_index * 2 + 2]
        assert [row["step"] for row in stage["history"]] == expected_steps
        assert [row["cumulative_step"] for row in stage["history"]] == expected_steps
        assert [row["stage_step"] for row in stage["history"]] == [1, 2]
        assert stage["best_step"] in expected_steps
        assert stage["best_stage_step"] in {1, 2}

    expected_nested_rows = {
        str(trace_count): [int(value) for value in expected_subsets[trace_count]]
        for trace_count in (1, 2)
    }
    assert inputs_lock["selection"]["source_split"] == TRAIN_SPLIT
    assert inputs_lock["selection"]["planned_trace_counts"] == [1, 2]
    assert inputs_lock["selection"]["planned_nested_selected_array_rows"] == expected_nested_rows
    assert inputs_lock["selection"]["selected_array_rows"] == expected_nested_rows["2"]
    expected_state_contract = {
        "carry_model_state": True,
        "carry_optimizer_state": True,
        "reset_optimizer_between_stages": False,
        "rewind_to_best": False,
        "checkpoint": False,
    }
    assert inputs_lock["training"] == {
        "batch_mode": "random_replacement",
        "replacement": True,
        "batch_size": 4,
        "updates_per_stage": 2,
        "report_interval": 1,
        "prediction_batch_size": 3,
        "planned_stage_trace_counts": [1, 2],
        "completed_stage_trace_counts": [1, 2],
        "sampler_seed_policy": "base_seed_plus_stage_index",
        "planned_sampler_seeds": [5, 6],
        "completed_sampler_seeds": [5, 6],
        **expected_state_contract,
        "first_stage_final_min_median_trace_snr_db": -1.0e9,
        "planned_total_updates": 4,
        "planned_point_evaluations": 16,
        "updates_completed": 4,
        "point_evaluations": 16,
    }

    assert run_metadata["study_id"] == "study_011_trace_pool_continuation"
    assert run_metadata["condition"] == "continuation1to2_random4"
    assert run_metadata["git_commit"] == git_commit
    assert run_metadata["status"] == "success"
    assert run_metadata["device"] == "cpu"
    assert run_metadata["random_seed"] == 5
    assert run_metadata["planned_stage_trace_counts"] == [1, 2]
    assert run_metadata["completed_stage_trace_counts"] == [1, 2]
    assert run_metadata["stages_completed"] == 2
    assert run_metadata["planned_sampler_seeds"] == [5, 6]
    assert run_metadata["completed_sampler_seeds"] == [5, 6]
    assert run_metadata["updates_per_stage"] == 2
    assert run_metadata["planned_total_updates"] == 4
    assert run_metadata["updates_completed"] == 4
    assert run_metadata["planned_point_evaluations"] == 16
    assert run_metadata["point_evaluations"] == 16
    assert run_metadata["anchor_reproduced"] is True
    assert run_metadata["classification_scope"] == "final_full_ffid_stage"
    assert run_metadata["final_full_ffid_classification"] == metrics["classification"]
    assert run_metadata["first_stage_final_min_median_trace_snr_db"] == -1.0e9
    assert {key: run_metadata[key] for key in expected_state_contract} == expected_state_contract
    assert run_metadata["python_version"]
    assert run_metadata["torch_version"]

    assert summary["study_id"] == "study_011_trace_pool_continuation"
    assert summary["planned_total_updates"] == 4
    assert summary["updates_completed"] == 4
    assert summary["planned_point_evaluations"] == 16
    assert summary["point_evaluations"] == 16
    assert summary["anchor_reproduced"] is True
    assert summary["first_stage_final_min_median_trace_snr_db"] == -1.0e9
    assert summary["final_full_ffid_classification"] == metrics["classification"]
    assert summary["decision"] == pipeline.full_ffid_summary_decision(metrics["classification"])
    assert summary["runs"][0]["anchor_reproduced"] is True
    assert summary["runs"][0]["stages_completed"] == 2

    correlation_keys = {"correlation_weight", "correlation_eps", "loss_semantics"}
    for payload in (
        saved_config["experiment"],
        metrics,
        inputs_lock["training"],
        run_metadata,
        summary,
        summary["runs"][0],
    ):
        assert correlation_keys.isdisjoint(payload)
    for stage in metrics["stages"]:
        assert correlation_keys.isdisjoint(stage)

    for path in output_root.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".yaml"}:
            assert str(tmp_path) not in path.read_text(encoding="utf-8")

    original_files = {path: path.read_bytes() for path in output_root.rglob("*") if path.is_file()}
    original_stage_call_count = len(stage_models)
    with pytest.raises(FileExistsError, match="already exist"):
        pipeline.run_trace_pool_continuation(
            config_path=config,
            interim_dir=interim,
            processed_dir=processed,
            output_root=output_root,
            device_override="cpu",
        )
    assert {
        path: path.read_bytes() for path in output_root.rglob("*") if path.is_file()
    } == original_files
    assert len(stage_models) == original_stage_call_count


def test_continuation_stops_successfully_when_the_first_stage_anchor_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, interim, processed = _build_continuation_fixture(
        tmp_path,
        anchor_threshold=1.0e9,
    )
    output_root = tmp_path / "runs" / "anchor_failure"
    monkeypatch.setattr(pipeline, "_run_id_timestamp", lambda: "20260826T020304Z")

    summary = pipeline.run_trace_pool_continuation(
        config_path=config,
        interim_dir=interim,
        processed_dir=processed,
        output_root=output_root,
        device_override="cpu",
    )

    run_directory = next(path for path in output_root.iterdir() if path.is_dir())
    metrics = json.loads((run_directory / "metrics.json").read_text(encoding="utf-8"))
    run_metadata = json.loads((run_directory / "run.json").read_text(encoding="utf-8"))
    inputs_lock = json.loads((run_directory / "inputs.lock.json").read_text(encoding="utf-8"))
    assert summary["decision"] == "stage8_anchor_failed"
    assert summary["planned_total_updates"] == 4
    assert summary["updates_completed"] == 2
    assert summary["planned_point_evaluations"] == 16
    assert summary["point_evaluations"] == 8
    assert metrics["anchor_reproduced"] is False
    assert metrics["classification_scope"] == "completed_anchor_stage"
    assert metrics["final_full_ffid_classification"] is None
    assert metrics["planned_total_updates"] == 4
    assert metrics["updates_completed"] == 2
    assert metrics["planned_point_evaluations"] == 16
    assert metrics["point_evaluations"] == 8
    assert metrics["planned_stage_trace_counts"] == [1, 2]
    assert metrics["completed_stage_trace_counts"] == [1]
    assert len(metrics["stages"]) == 1
    assert metrics["stages"][0]["trace_count"] == 1
    assert run_metadata["status"] == "success"
    assert run_metadata["anchor_reproduced"] is False
    assert run_metadata["classification_scope"] == "completed_anchor_stage"
    assert run_metadata["final_full_ffid_classification"] is None
    assert run_metadata["planned_stage_trace_counts"] == [1, 2]
    assert run_metadata["completed_stage_trace_counts"] == [1]
    assert run_metadata["stages_completed"] == 1
    assert run_metadata["planned_sampler_seeds"] == [5, 6]
    assert run_metadata["completed_sampler_seeds"] == [5]
    assert run_metadata["updates_completed"] == 2
    assert run_metadata["point_evaluations"] == 8
    assert inputs_lock["selection"]["trace_count"] == 1
    assert inputs_lock["selection"]["point_count"] == 5
    assert inputs_lock["training"]["planned_sampler_seeds"] == [5, 6]
    assert inputs_lock["training"]["completed_sampler_seeds"] == [5]
    assert inputs_lock["training"]["updates_completed"] == 2
    assert inputs_lock["training"]["point_evaluations"] == 8
    assert summary["anchor_reproduced"] is False
    assert summary["final_full_ffid_classification"] is None
    assert summary["runs"][0]["classification_scope"] == "completed_anchor_stage"
    assert sorted(path.name for path in run_directory.iterdir()) == [
        "config.resolved.yaml",
        "inputs.lock.json",
        "metrics.json",
        "run.json",
    ]


def test_pure_mse_continuation_rejects_correlation_configuration_before_output(
    tmp_path: Path,
) -> None:
    config, interim, processed = _build_continuation_fixture(
        tmp_path,
        anchor_threshold=-1.0e9,
    )
    config_data = yaml.safe_load(config.read_text(encoding="utf-8"))
    config_data["experiment"].update(
        {
            "correlation_weight": 0.1,
            "correlation_eps": 1.0e-4,
        }
    )
    config.write_text(yaml.safe_dump(config_data, sort_keys=False), encoding="utf-8")
    output_root = tmp_path / "runs" / "invalid_correlation"

    with pytest.raises(
        pipeline.ConfigurationError,
        match="pure-MSE continuation must not define correlation-loss keys",
    ):
        pipeline.run_trace_pool_continuation(
            config_path=config,
            interim_dir=interim,
            processed_dir=processed,
            output_root=output_root,
            device_override="cpu",
        )

    assert not output_root.exists()
