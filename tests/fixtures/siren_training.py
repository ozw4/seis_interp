"""Synthetic single-FFID SIREN training setup shared by pipeline and CLI tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from seis_interp.data.trace_store import write_interim_trace_dataset
from seis_interp.pipelines.prepare_baseline import prepare_baseline_dataset


def prepare_siren_training_fixture(
    tmp_path: Path,
    *,
    configured_device: str = "cpu",
) -> tuple[Path, Path, Path]:
    """Write a 10-trace interim + processed dataset and a minimal SIREN config."""
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
