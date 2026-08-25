from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

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
                "model": {
                    "name": "siren",
                    "input_features": 6,
                    "hidden_width": 8,
                    "hidden_layers": 1,
                    "omega_0": 10.0,
                },
                "training": {
                    "loss": "l1",
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
    assert inputs_lock["split_counts"] == {"train": 6, "validation": 2, "test": 2}
    loaded = load_siren_checkpoint(output / "artifacts" / "best.pt")
    assert loaded.model.input_features == 6
    assert loaded.normalization.coordinate_min[-2:] == (-1.0, -1.0)
    assert loaded.normalization.coordinate_max[-2:] == (1.0, 1.0)


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


def test_pipeline_rejects_nonempty_output_without_overwrite(tmp_path: Path) -> None:
    config, interim, processed = _build_training_fixture(tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    (output / "marker").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        train_siren_run(
            config_path=config,
            interim_dir=interim,
            processed_dir=processed,
            output_dir=output,
        )
