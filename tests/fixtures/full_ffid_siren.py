"""Synthetic 3-FFID per-FFID-split SIREN setup shared by full-FFID pipeline tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from seis_interp.data.trace_store import write_interim_trace_dataset
from seis_interp.pipelines.prepare_baseline import prepare_baseline_dataset


def prepare_full_ffid_siren_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Write a two-source 30-trace interim + processed dataset and a full-FFID config."""
    source_a = tmp_path / "source_a.sgy"
    source_b = tmp_path / "source_b.sgy"
    source_a.write_bytes(b"synthetic source A")
    source_b.write_bytes(b"synthetic source B")
    interim = tmp_path / "interim"
    trace_count = 30
    sample_count = 4
    array_indices = np.arange(trace_count)
    cmp_x_m = array_indices.astype(np.float64)
    cmp_y_m = array_indices.astype(np.float64) * 2.0
    offset_m = 100.0 + array_indices.astype(np.float64)
    azimuth_deg = array_indices.astype(np.float64) * 7.0
    azimuth_rad = np.deg2rad(azimuth_deg)
    half_offset_x_m = 0.5 * offset_m * np.sin(azimuth_rad)
    half_offset_y_m = 0.5 * offset_m * np.cos(azimuth_rad)
    trace_table = pd.DataFrame(
        {
            "source_file": np.repeat([source_a.name, source_b.name], [20, 10]),
            "trace_index": np.concatenate(
                [np.arange(20, dtype=np.int64), np.arange(10, dtype=np.int64)]
            ),
            "ffid": np.repeat([10, 20, 30], 10),
            "source_x_m": cmp_x_m + half_offset_x_m,
            "source_y_m": cmp_y_m + half_offset_y_m,
            "receiver_x_m": cmp_x_m - half_offset_x_m,
            "receiver_y_m": cmp_y_m - half_offset_y_m,
            "cmp_x_m": cmp_x_m,
            "cmp_y_m": cmp_y_m,
            "offset_m": offset_m,
            "azimuth_deg": azimuth_deg,
            "sample_interval_s": np.full(trace_count, 0.008),
        }
    )
    time_s = np.arange(sample_count, dtype=np.float64) * 0.008
    amplitudes = (
        np.sin(array_indices[:, np.newaxis] * 0.2 + time_s[np.newaxis, :] * 10.0) + 1.5
    ).astype(np.float32)
    write_interim_trace_dataset(
        interim,
        trace_table,
        amplitudes,
        time_s,
        source_a,
        "synthetic",
        selection={"ffid_scope": "all", "include_incomplete_ffids": True},
    )
    metadata_path = interim / "dataset.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("source_file")
    metadata.pop("source_sha256")
    metadata["source_files"] = [
        {"name": source.name, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        for source in (source_a, source_b)
    ]
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    processed = tmp_path / "processed"
    prepare_baseline_dataset(
        interim,
        processed,
        holdout_fraction=0.3,
        validation_fraction_of_holdout=0.5,
        random_seed=7,
        split_scope="per_ffid",
        config_source="studies/synthetic/config.yaml",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "project": {"random_seed": 7},
                "sampling": {
                    "random_trace_holdout_fraction": 0.3,
                    "validation_fraction_of_holdout": 0.5,
                    "split_scope": "per_ffid",
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
                    "batch_mode": "full_ffid_epoch",
                    "loss": "l2",
                    "optimizer": "adam",
                    "learning_rate": 1.0e-3,
                    "max_epochs": 2,
                    "early_stopping_patience": 2,
                    "validation_batch_size": 5,
                    "device": "cpu",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config, interim, processed
