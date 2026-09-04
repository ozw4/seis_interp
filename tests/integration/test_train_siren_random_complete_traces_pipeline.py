from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from seis_interp.cli import main
from seis_interp.configuration import ConfigurationError
from seis_interp.pipelines.train_siren import train_siren_run
from seis_interp.training.checkpoints import load_siren_checkpoint
from tests.fixtures.full_ffid_siren import prepare_full_ffid_siren_fixture


def test_random_complete_traces_filters_ffids_and_selects_by_streamed_global_snr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, interim, processed = prepare_full_ffid_siren_fixture(tmp_path)
    config_payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    config_payload["training"].update(
        {
            "batch_mode": "random_complete_traces",
            "amplitude_scaling": "per_trace_rms",
            "ffid_range": [20, 30],
            "traces_per_update": 2,
            "steps_per_epoch": 3,
            "max_epochs": 2,
            "early_stopping_patience": 2,
            "evaluate_training_snr": True,
        }
    )
    config.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")

    trace_table = pd.read_parquet(interim / "traces.parquet")
    split_table = pd.read_parquet(processed / "trace_split.parquet")
    ffid_by_row = np.empty(len(trace_table), dtype=np.int64)
    ffid_by_row[trace_table["array_row"].to_numpy(dtype=np.int64)] = trace_table["ffid"].to_numpy(
        dtype=np.int64
    )
    selected_mask = split_table["array_row"].map(lambda row: 20 <= ffid_by_row[int(row)] <= 30)
    expected_rows = {
        split: set(split_table.loc[selected_mask & split_table["split"].eq(split), "array_row"])
        for split in ("train", "validation", "test")
    }

    import seis_interp.pipelines.train_siren as pipeline

    actual_load = pipeline.load_interim_trace_dataset
    actual_sampler = pipeline.RandomCompleteTraceBatchSampler
    memory_map_flags: list[bool] = []
    sampled_trace_counts: list[int] = []
    received_training_rows: set[int] = set()
    evaluated_splits: list[str] = []
    forwarded_scaling: list[str] = []
    training_scores = iter([10.0, 11.0])
    validation_scores = iter([1.0, 3.0])

    def recording_load(directory: Path, *, memory_map_amplitudes: bool = False):
        memory_map_flags.append(memory_map_amplitudes)
        return actual_load(directory, memory_map_amplitudes=memory_map_amplitudes)

    class RecordingSampler:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            received_training_rows.update(int(row) for row in args[3])
            self._sampler = actual_sampler(*args, **kwargs)

        @property
        def training_trace_count(self) -> int:
            return self._sampler.training_trace_count

        @property
        def amplitude_scaling(self) -> str:
            return self._sampler.amplitude_scaling

        def sample(self, traces_per_update: int) -> tuple[np.ndarray, np.ndarray]:
            sampled_trace_counts.append(traces_per_update)
            coordinates, targets = self._sampler.sample(traces_per_update)
            target_traces = targets.reshape(traces_per_update, -1).astype(np.float64)
            np.testing.assert_allclose(
                np.sqrt(np.mean(np.square(target_traces), axis=1)),
                np.ones(traces_per_update),
                rtol=1.0e-7,
            )
            return coordinates, targets

    def controlled_global_snr(*_args: Any, **kwargs: Any) -> float:
        rows = {int(row) for group in kwargs["rows_by_ffid"].values() for row in group}
        forwarded_scaling.append(kwargs["amplitude_scaling"])
        if rows == expected_rows["train"]:
            evaluated_splits.append("train")
            return next(training_scores)
        if rows == expected_rows["validation"]:
            evaluated_splits.append("validation")
            return next(validation_scores)
        pytest.fail(f"unexpected streamed evaluation rows: {sorted(rows)}")

    monkeypatch.setattr(pipeline, "load_interim_trace_dataset", recording_load)
    monkeypatch.setattr(pipeline, "RandomCompleteTraceBatchSampler", RecordingSampler)
    monkeypatch.setattr(pipeline, "evaluate_model_global_snr_by_ffid", controlled_global_snr)
    monkeypatch.setattr(
        pipeline,
        "normalize_amplitudes",
        lambda *_args, **_kwargs: pytest.fail("must not normalize the survey-wide array"),
    )
    monkeypatch.setattr(
        pipeline,
        "per_trace_rms_scaled_rows",
        lambda *_args, **_kwargs: pytest.fail("must scale only sampled trace rows"),
    )
    monkeypatch.setattr(
        pipeline,
        "build_trace_points",
        lambda *_args, **_kwargs: pytest.fail("must stream validation by FFID"),
    )

    output = tmp_path / "run"
    metrics = train_siren_run(
        config_path=config,
        interim_dir=interim,
        processed_dir=processed,
        output_dir=output,
        progress_reporter=lambda _message: None,
    )

    assert memory_map_flags == [True]
    assert received_training_rows == expected_rows["train"]
    assert sampled_trace_counts == [2] * 6
    assert evaluated_splits == ["train", "validation", "train", "validation"]
    assert forwarded_scaling == ["per_trace_rms"] * 4
    assert expected_rows["test"].isdisjoint(received_training_rows)
    assert metrics["batch_mode"] == "random_complete_traces"
    assert metrics["best_epoch"] == 2
    assert metrics["best_validation_global_snr_db"] == 3.0
    assert "best_validation_median_trace_snr_db" not in metrics
    assert metrics["training_global_snr_db_at_best_epoch"] == 11.0
    assert [row["training_global_snr_db"] for row in metrics["history"]] == [10.0, 11.0]
    assert metrics["selected_ffid_range"] == [20, 30]
    assert metrics["selected_ffid_count"] == 2
    assert metrics["configured_ffid_range"] == [20, 30]
    assert metrics["training_ffid_count"] == 2
    assert metrics["training_trace_count"] == len(expected_rows["train"])
    assert metrics["traces_per_update"] == 2
    assert metrics["points_per_update"] == 8
    assert metrics["effective_steps_per_epoch"] == 3
    assert metrics["validation_metric_domain"] == "oracle_per_trace_unit_rms"
    assert metrics["training_metric_domain"] == "oracle_per_trace_unit_rms"
    assert "learning_rate_schedule" not in metrics
    assert all("learning_rate" not in row for row in metrics["history"])

    checkpoint = load_siren_checkpoint(output / "artifacts" / "best.pt")
    assert checkpoint.epoch == 2
    assert checkpoint.global_step == 6
    assert checkpoint.validation_median_trace_snr_db is None
    assert checkpoint.validation_global_snr_db == 3.0
    assert checkpoint.validation_metric_domain == "oracle_per_trace_unit_rms"

    inputs_lock = json.loads((output / "inputs.lock.json").read_text(encoding="utf-8"))
    training_lock = inputs_lock["training"]
    assert training_lock["selected_ffid_range"] == [20, 30]
    assert training_lock["selected_ffid_count"] == 2
    assert training_lock["configured_ffid_range"] == [20, 30]
    assert training_lock["training_trace_count"] == len(expected_rows["train"])
    assert training_lock["validation_trace_count"] == len(expected_rows["validation"])
    assert training_lock["test_trace_count"] == len(expected_rows["test"])
    assert training_lock["validation"] == "all_validation_traces_streamed_per_trace_rms"
    assert training_lock["training_evaluation"] == "all_training_traces_streamed"
    assert training_lock["training_metric_domain"] == "oracle_per_trace_unit_rms"
    assert "learning_rate_schedule" not in training_lock

    run_metadata = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert run_metadata["selected_ffid_range"] == [20, 30]
    assert run_metadata["selected_ffid_count"] == 2
    assert run_metadata["configured_ffid_range"] == [20, 30]
    assert run_metadata["batch_mode"] == "random_complete_traces"
    assert "learning_rate_schedule" not in run_metadata


def test_random_complete_traces_cosine_schedule_records_the_full_horizon_contract(
    tmp_path: Path,
) -> None:
    config, interim, processed = prepare_full_ffid_siren_fixture(tmp_path)
    config_payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    config_payload["training"].update(
        {
            "batch_mode": "random_complete_traces",
            "ffid_range": [20, 20],
            "traces_per_update": 2,
            "steps_per_epoch": 2,
            "max_epochs": 2,
            "early_stopping_patience": 2,
            "learning_rate_schedule": "cosine",
            "minimum_learning_rate": 1.0e-4,
        }
    )
    config.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")

    output = tmp_path / "run"
    metrics = train_siren_run(
        config_path=config,
        interim_dir=interim,
        processed_dir=processed,
        output_dir=output,
        progress_reporter=lambda _message: None,
    )

    expected_schedule_contract = {
        "learning_rate_schedule": "cosine",
        "initial_learning_rate": 1.0e-3,
        "minimum_learning_rate": 1.0e-4,
        "learning_rate_schedule_step_unit": "optimizer_update",
        "learning_rate_schedule_total_updates": 4,
    }
    assert [row["learning_rate"] for row in metrics["history"]] == pytest.approx([5.5e-4, 1.0e-4])
    assert {key: metrics[key] for key in expected_schedule_contract} == (expected_schedule_contract)

    inputs_lock = json.loads((output / "inputs.lock.json").read_text(encoding="utf-8"))
    assert {
        key: inputs_lock["training"][key] for key in expected_schedule_contract
    } == expected_schedule_contract
    run_metadata = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert {key: run_metadata[key] for key in expected_schedule_contract} == (
        expected_schedule_contract
    )
    resolved_config = yaml.safe_load((output / "config.resolved.yaml").read_text(encoding="utf-8"))
    assert resolved_config["training"]["learning_rate_schedule"] == "cosine"
    assert resolved_config["training"]["minimum_learning_rate"] == 1.0e-4


@pytest.mark.parametrize("batch_mode", ["random_points", "full_ffid_epoch"])
def test_cosine_schedule_is_rejected_outside_random_complete_traces(
    tmp_path: Path,
    batch_mode: str,
) -> None:
    config, interim, processed = prepare_full_ffid_siren_fixture(tmp_path)
    config_payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    config_payload["training"].update(
        {
            "batch_mode": batch_mode,
            "learning_rate_schedule": "cosine",
            "minimum_learning_rate": 1.0e-4,
        }
    )
    config.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="supported only by batch_mode"):
        train_siren_run(
            config_path=config,
            interim_dir=interim,
            processed_dir=processed,
            output_dir=tmp_path / "run",
            progress_reporter=lambda _message: None,
        )


def test_random_complete_traces_rejects_an_empty_ffid_range(tmp_path: Path) -> None:
    config, interim, processed = prepare_full_ffid_siren_fixture(tmp_path)
    config_payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    config_payload["training"].update(
        {
            "batch_mode": "random_complete_traces",
            "ffid_range": [100, 200],
            "traces_per_update": 1,
            "steps_per_epoch": 1,
        }
    )
    config.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="selects no eligible FFIDs"):
        train_siren_run(
            config_path=config,
            interim_dir=interim,
            processed_dir=processed,
            output_dir=tmp_path / "run",
            progress_reporter=lambda _message: None,
        )


def test_random_complete_traces_cli_reports_only_the_global_selection_metric(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config, interim, processed = prepare_full_ffid_siren_fixture(tmp_path)
    config_payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    config_payload["training"].update(
        {
            "batch_mode": "random_complete_traces",
            "amplitude_scaling": "per_trace_rms",
            "ffid_range": [20, 20],
            "traces_per_update": 1,
            "steps_per_epoch": 1,
            "max_epochs": 1,
            "early_stopping_patience": 1,
        }
    )
    config.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")

    exit_code = main(
        [
            "train",
            "siren",
            "--config",
            str(config),
            "--interim",
            str(interim),
            "--processed",
            str(processed),
            "--output",
            str(tmp_path / "run"),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Batch mode: random_complete_traces" in captured.out
    assert "Best oracle-normalized validation global S/N:" in captured.out
    assert "median trace S/N" not in captured.out


def test_full_ffid_epoch_can_apply_the_same_optional_ffid_range(tmp_path: Path) -> None:
    config, interim, processed = prepare_full_ffid_siren_fixture(tmp_path)
    config_payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    config_payload["training"].update(
        {
            "ffid_range": [20, 30],
            "max_epochs": 1,
            "early_stopping_patience": 1,
        }
    )
    config.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")

    output = tmp_path / "run"
    metrics = train_siren_run(
        config_path=config,
        interim_dir=interim,
        processed_dir=processed,
        output_dir=output,
        progress_reporter=lambda _message: None,
    )

    assert metrics["batch_mode"] == "full_ffid_epoch"
    assert metrics["selected_ffid_range"] == [20, 30]
    assert metrics["selected_ffid_count"] == 2
    assert metrics["configured_ffid_range"] == [20, 30]
    assert metrics["training_ffid_count"] == 2
    assert metrics["effective_steps_per_epoch"] == 2
    assert metrics["global_steps"] == 2
    inputs_lock = json.loads((output / "inputs.lock.json").read_text(encoding="utf-8"))
    assert inputs_lock["training"]["selected_ffid_count"] == 2
    run_metadata = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert run_metadata["selected_ffid_range"] == [20, 30]
