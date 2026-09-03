from __future__ import annotations

import hashlib
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

from seis_interp.configuration import REPOSITORY_ROOT
from seis_interp.pipelines.train_siren import train_siren_run
from seis_interp.processing.trace_splits import TRAIN_SPLIT, VALIDATION_SPLIT
from seis_interp.training.checkpoints import load_siren_checkpoint
from tests.fixtures.siren_training import prepare_siren_training_fixture


def test_pipeline_trains_on_cpu_and_writes_minimal_run(tmp_path: Path) -> None:
    config, interim, processed = prepare_siren_training_fixture(tmp_path)
    output = tmp_path / "run"

    metrics = train_siren_run(
        config_path=config,
        interim_dir=interim,
        processed_dir=processed,
        output_dir=output,
    )

    assert sorted(
        path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()
    ) == [
        "artifacts/best.pt",
        "config.resolved.yaml",
        "inputs.lock.json",
        "metrics.json",
        "run.json",
    ]
    assert json.loads((output / "metrics.json").read_text(encoding="utf-8")) == metrics
    assert (
        yaml.safe_load((output / "config.resolved.yaml").read_text(encoding="utf-8"))["model"][
            "input_features"
        ]
        == 6
    )
    assert (
        yaml.safe_load((output / "config.resolved.yaml").read_text(encoding="utf-8"))["training"][
            "amplitude_scaling"
        ]
        == "train_global_rms"
    )
    inputs_lock_text = (output / "inputs.lock.json").read_text(encoding="utf-8")
    inputs_lock = json.loads(inputs_lock_text)
    assert str(tmp_path) not in inputs_lock_text
    assert inputs_lock == {
        "interim_files": {
            file_name: {"sha256": hashlib.sha256((interim / file_name).read_bytes()).hexdigest()}
            for file_name in (
                "traces.parquet",
                "amplitudes.npy",
                "time_s.npy",
                "dataset.json",
            )
        },
        "processed_files": {
            file_name: {"sha256": hashlib.sha256((processed / file_name).read_bytes()).hexdigest()}
            for file_name in (
                "trace_split.parquet",
                "normalization.json",
                "preparation.json",
            )
        },
        "preparation": {
            "random_seed": 5,
            "holdout_fraction": 0.4,
            "validation_fraction_of_holdout": 0.5,
            "normalization": {
                "coordinates": "train_minmax_linear_plus_azimuth_sin_cos",
                "amplitude": "train_global_rms",
            },
        },
    }
    assert metrics["best_validation_median_trace_snr_db"] == pytest.approx(
        metrics["history"][metrics["best_epoch"] - 1]["validation_median_trace_snr_db"]
    )
    assert metrics["best_validation_global_snr_db"] == pytest.approx(
        metrics["history"][metrics["best_epoch"] - 1]["validation_global_snr_db"]
    )
    assert "best_validation_snr_db" not in metrics
    assert all("validation_snr_db" not in item for item in metrics["history"])
    loaded = load_siren_checkpoint(output / "artifacts" / "best.pt")
    assert loaded.validation_median_trace_snr_db == pytest.approx(
        metrics["best_validation_median_trace_snr_db"]
    )
    assert loaded.validation_global_snr_db == pytest.approx(
        metrics["best_validation_global_snr_db"]
    )
    assert loaded.model.input_features == 6
    assert loaded.normalization.coordinate_min[-2:] == (-1.0, -1.0)
    assert loaded.normalization.coordinate_max[-2:] == (1.0, 1.0)
    run_metadata = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert run_metadata == {
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "started_at_utc": run_metadata["started_at_utc"],
        "finished_at_utc": run_metadata["finished_at_utc"],
        "status": "success",
        "device": "cpu",
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "random_seed": 5,
    }
    started_at = datetime.fromisoformat(run_metadata["started_at_utc"].replace("Z", "+00:00"))
    finished_at = datetime.fromisoformat(run_metadata["finished_at_utc"].replace("Z", "+00:00"))
    assert started_at.tzinfo == timezone.utc
    assert finished_at.tzinfo == timezone.utc
    assert started_at <= finished_at


def test_random_points_pipeline_supports_per_trace_rms_target_scaling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, interim, processed = prepare_siren_training_fixture(tmp_path)
    config_payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    config_payload["training"]["amplitude_scaling"] = "per_trace_rms"
    config.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")
    output = tmp_path / "run"
    split_table = pd.read_parquet(processed / "trace_split.parquet")
    expected_scaled_rows = set(
        split_table.loc[split_table["split"].isin(["train", "validation"]), "array_row"]
    )
    received_scaled_rows: set[int] = set()
    from seis_interp.training.amplitude_scaling import (
        per_trace_rms_scaled_rows as actual_scale_rows,
    )

    def recording_scale_rows(amplitudes: np.ndarray, array_rows: np.ndarray) -> np.ndarray:
        received_scaled_rows.update(int(row) for row in array_rows)
        scaled = actual_scale_rows(amplitudes, array_rows)
        selected = scaled[array_rows].astype(np.float64)
        np.testing.assert_allclose(
            np.sqrt(np.mean(np.square(selected), axis=1)),
            np.ones(len(array_rows)),
            rtol=1.0e-7,
        )
        return scaled

    monkeypatch.setattr(
        "seis_interp.pipelines.train_siren.per_trace_rms_scaled_rows",
        recording_scale_rows,
    )

    metrics = train_siren_run(
        config_path=config,
        interim_dir=interim,
        processed_dir=processed,
        output_dir=output,
    )

    assert metrics["amplitude_scaling"] == "per_trace_rms"
    assert metrics["validation_metric_domain"] == "oracle_per_trace_unit_rms"
    assert received_scaled_rows == expected_scaled_rows
    assert metrics["validation_scale_source"] == "validation_trace_target_rms"
    checkpoint = load_siren_checkpoint(output / "artifacts" / "best.pt")
    assert checkpoint.amplitude_scaling == "per_trace_rms"
    assert checkpoint.validation_metric_domain == "oracle_per_trace_unit_rms"
    inputs_lock = json.loads((output / "inputs.lock.json").read_text(encoding="utf-8"))
    assert inputs_lock["preparation"]["normalization"]["amplitude"] == "train_global_rms"
    assert inputs_lock["training"] == {
        "batch_mode": "random_points",
        "amplitude_scaling": "per_trace_rms",
        "validation": "all_validation_traces_materialized_per_trace_rms",
        "validation_metric_domain": "oracle_per_trace_unit_rms",
        "validation_scale_source": "validation_trace_target_rms",
    }
    run_metadata = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert run_metadata["amplitude_scaling"] == "per_trace_rms"
    assert run_metadata["validation_metric_domain"] == "oracle_per_trace_unit_rms"
    assert run_metadata["validation_scale_source"] == "validation_trace_target_rms"


def test_pipeline_routes_only_train_and_validation_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, interim, processed = prepare_siren_training_fixture(tmp_path)
    split_table = pd.read_parquet(processed / "trace_split.parquet")
    expected_train = set(split_table.loc[split_table["split"] == TRAIN_SPLIT, "array_row"])
    expected_validation = list(
        split_table.loc[split_table["split"] == VALIDATION_SPLIT, "array_row"]
    )
    received: dict[str, Any] = {}
    from seis_interp.training.point_sampler import RandomPointSampler as ActualSampler
    from seis_interp.training.point_sampler import build_trace_points as actual_build

    def recording_sampler(*args: Any, **kwargs: Any) -> ActualSampler:
        received["train"] = set(args[3].tolist())
        return ActualSampler(*args, **kwargs)

    def recording_build(*args: Any, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        received["validation"] = args[3].tolist()
        return actual_build(*args, **kwargs)

    monkeypatch.setattr("seis_interp.pipelines.train_siren.RandomPointSampler", recording_sampler)
    monkeypatch.setattr("seis_interp.pipelines.train_siren.build_trace_points", recording_build)

    train_siren_run(
        config_path=config,
        interim_dir=interim,
        processed_dir=processed,
        output_dir=tmp_path / "run",
    )

    assert received == {"train": expected_train, "validation": expected_validation}


def test_random_points_pipeline_does_not_reset_the_global_numpy_rng(tmp_path: Path) -> None:
    config, interim, processed = prepare_siren_training_fixture(tmp_path)
    original_state = np.random.get_state()
    try:
        np.random.seed(1234)
        expected_next_value = np.random.random()
        np.random.seed(1234)

        train_siren_run(
            config_path=config,
            interim_dir=interim,
            processed_dir=processed,
            output_dir=tmp_path / "run",
        )

        assert np.random.random() == expected_next_value
    finally:
        np.random.set_state(original_state)


def test_pipeline_passes_the_common_trace_sample_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, interim, processed = prepare_siren_training_fixture(tmp_path)
    expected_sample_count = len(np.load(interim / "time_s.npy", allow_pickle=False))
    received: dict[str, Any] = {}
    from seis_interp.training.trainer import train_siren as actual_train

    def recording_train(*args: Any, **kwargs: Any) -> Any:
        received["validation_samples_per_trace"] = kwargs["validation_samples_per_trace"]
        return actual_train(*args, **kwargs)

    monkeypatch.setattr("seis_interp.pipelines.train_siren.train_siren", recording_train)

    train_siren_run(
        config_path=config,
        interim_dir=interim,
        processed_dir=processed,
        output_dir=tmp_path / "run",
    )

    assert received == {"validation_samples_per_trace": expected_sample_count}


def test_pipeline_rejects_an_existing_run_directory(tmp_path: Path) -> None:
    config, interim, processed = prepare_siren_training_fixture(tmp_path)
    output = tmp_path / "run"
    output.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        train_siren_run(
            config_path=config,
            interim_dir=interim,
            processed_dir=processed,
            output_dir=output,
        )


def test_pipeline_rejects_interim_file_changed_after_preparation(tmp_path: Path) -> None:
    config, interim, processed = prepare_siren_training_fixture(tmp_path)
    amplitude_path = interim / "amplitudes.npy"
    amplitudes = np.load(amplitude_path, allow_pickle=False)
    amplitudes[0, 0] += 1.0
    np.save(amplitude_path, amplitudes)

    with pytest.raises(ValueError, match="interim file checksums"):
        train_siren_run(
            config_path=config,
            interim_dir=interim,
            processed_dir=processed,
            output_dir=tmp_path / "run",
        )


def test_pipeline_rejects_an_unsupported_loss(tmp_path: Path) -> None:
    config, interim, processed = prepare_siren_training_fixture(tmp_path)
    config_data = yaml.safe_load(config.read_text(encoding="utf-8"))
    config_data["training"]["loss"] = "huber"
    config.write_text(yaml.safe_dump(config_data, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported loss: huber"):
        train_siren_run(
            config_path=config,
            interim_dir=interim,
            processed_dir=processed,
            output_dir=tmp_path / "run",
        )


@pytest.mark.parametrize(
    ("section", "key", "changed_value", "mismatched_field"),
    [
        ("project", "random_seed", 6, "random_seed"),
        ("sampling", "random_trace_holdout_fraction", 0.3, "holdout_fraction"),
        (
            "sampling",
            "validation_fraction_of_holdout",
            0.25,
            "validation_fraction_of_holdout",
        ),
        ("normalization", "coordinates", "changed_coordinates", "normalization"),
        ("normalization", "amplitude", "changed_amplitude", "normalization"),
    ],
)
def test_pipeline_rejects_config_that_does_not_match_preparation(
    tmp_path: Path,
    section: str,
    key: str,
    changed_value: object,
    mismatched_field: str,
) -> None:
    config, interim, processed = prepare_siren_training_fixture(tmp_path)
    config_data = yaml.safe_load(config.read_text(encoding="utf-8"))
    config_data[section][key] = changed_value
    config.write_text(yaml.safe_dump(config_data, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=mismatched_field):
        train_siren_run(
            config_path=config,
            interim_dir=interim,
            processed_dir=processed,
            output_dir=tmp_path / "run",
        )
