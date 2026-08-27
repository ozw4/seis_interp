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
from seis_interp.data.trace_store import write_interim_trace_dataset
from seis_interp.pipelines.prepare_baseline import prepare_baseline_dataset
from seis_interp.pipelines.train_siren import train_siren_run
from seis_interp.processing.trace_splits import TRAIN_SPLIT, VALIDATION_SPLIT
from seis_interp.training.checkpoints import load_siren_checkpoint


def _build_training_fixture(tmp_path: Path, *, configured_device: str = "cpu") -> tuple[Path, ...]:
    source = tmp_path / "source.sgy"
    source.write_bytes(b"synthetic seismic source")
    interim = tmp_path / "interim"
    trace_count = 10
    sample_count = 5
    indices = np.arange(trace_count)
    trace_table = pd.DataFrame(
        {
            "trace_index": indices,
            "ffid": np.full(trace_count, 2348),
            "cmp_x_m": indices.astype(np.float64),
            "cmp_y_m": indices.astype(np.float64) * 2.0,
            "offset_m": 100.0 + indices.astype(np.float64),
            "azimuth_deg": indices.astype(np.float64) * 30.0,
            "sample_interval_s": np.full(trace_count, 0.008),
        }
    )
    time_s = np.arange(sample_count, dtype=np.float64) * 0.008
    amplitudes = (np.sin(indices[:, np.newaxis] * 0.2 + time_s[np.newaxis, :] * 10.0) + 1.5).astype(
        np.float32
    )
    write_interim_trace_dataset(
        interim,
        trace_table,
        amplitudes,
        time_s,
        source,
        "synthetic",
        selection={"ffid": 2348, "expected_trace_count": trace_count},
    )
    processed = tmp_path / "processed"
    prepare_baseline_dataset(
        interim,
        processed,
        holdout_fraction=0.4,
        validation_fraction_of_holdout=0.5,
        random_seed=5,
        config_source="studies/synthetic/config.yaml",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "project": {"random_seed": 5},
                "sampling": {
                    "random_trace_holdout_fraction": 0.4,
                    "validation_fraction_of_holdout": 0.5,
                },
                "normalization": {
                    "coordinates": "train_minmax_linear_plus_azimuth_sin_cos",
                    "amplitude": "train_global_rms",
                },
                "model": {
                    "name": "siren",
                    "input_features": 6,
                    "hidden_width": 8,
                    "hidden_layers": 1,
                    "omega_0": 10.0,
                    "hidden_omega": 1.0,
                },
                "training": {
                    "loss": "l2",
                    "optimizer": "adam",
                    "learning_rate": 1e-3,
                    "batch_size": 8,
                    "steps_per_epoch": 2,
                    "max_epochs": 2,
                    "early_stopping_patience": 2,
                    "validation_batch_size": 4,
                    "device": configured_device,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config, interim, processed


def test_pipeline_trains_on_cpu_and_writes_minimal_run(tmp_path: Path) -> None:
    config, interim, processed = _build_training_fixture(tmp_path)
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


def test_pipeline_routes_only_train_and_validation_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, interim, processed = _build_training_fixture(tmp_path)
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
    config, interim, processed = _build_training_fixture(tmp_path)
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
    config, interim, processed = _build_training_fixture(tmp_path)
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
    config, interim, processed = _build_training_fixture(tmp_path)
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
    config, interim, processed = _build_training_fixture(tmp_path)
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
    config, interim, processed = _build_training_fixture(tmp_path)
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
    config, interim, processed = _build_training_fixture(tmp_path)
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
