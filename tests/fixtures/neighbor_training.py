"""Synthetic 3-FFID neighbor-inpainter training setup shared by integration tests."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from seis_interp.data.trace_store import write_interim_trace_dataset
from seis_interp.pipelines.prepare_baseline import prepare_baseline_dataset
from seis_interp.processing.trace_amplitude_filter import TraceAmplitudeFilterConfig


def _neighbor_trace_data(
    *,
    include_excluded_trace: bool,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rows: list[dict[str, object]] = []
    amplitudes: list[np.ndarray] = []
    time_s = np.arange(17, dtype=np.float64) * 0.008
    trace_index = 0
    for ffid_index, ffid in enumerate((10, 11, 12)):
        source_x = 4000.0
        source_y = 1000.0 + ffid_index * 80.0
        for relative_x in (-140.0, -100.0):
            for relative_y in (-120.0, -80.0, -40.0, 0.0):
                receiver_x = source_x + relative_x
                receiver_y = source_y + relative_y
                midpoint_x = (source_x + receiver_x) / 2.0
                midpoint_y = (source_y + receiver_y) / 2.0
                offset = math.hypot(relative_x, relative_y)
                azimuth = math.degrees(math.atan2(-relative_x, -relative_y)) % 360.0
                rows.append(
                    {
                        "trace_index": trace_index,
                        "ffid": ffid,
                        "source_x_m": source_x,
                        "source_y_m": source_y,
                        "receiver_x_m": receiver_x,
                        "receiver_y_m": receiver_y,
                        "cmp_x_m": midpoint_x,
                        "cmp_y_m": midpoint_y,
                        "offset_m": offset,
                        "azimuth_deg": azimuth,
                        "sample_interval_s": 0.008,
                        "trace_count_in_ffid": 8,
                        "is_complete_ffid": False,
                    }
                )
                phase = ffid_index * 0.2 + relative_x * 0.003 + relative_y * 0.002
                trace = np.sin(np.arange(17) * 0.35 + phase)
                trace += 0.25 * np.cos(np.arange(17) * 0.11 - phase)
                amplitudes.append(trace.astype(np.float32))
                trace_index += 1
    if include_excluded_trace:
        amplitudes[0] = np.zeros_like(amplitudes[0])
    return pd.DataFrame(rows), np.stack(amplitudes), time_s


def _neighbor_config(
    *,
    configured_device: str,
    ffid_range: list[int] | None,
    formal_candidate: bool,
    split_scope: str,
    trace_filter: TraceAmplitudeFilterConfig,
) -> dict[str, object]:
    training: dict[str, object] = {
        "amplitude_scaling": "per_trace_rms",
        "loss": "l2_plus_first_difference",
        "optimizer": "adamw",
        "learning_rate": 1.0e-3,
        "weight_decay": 1.0e-5,
        "learning_rate_schedule": "cosine",
        "minimum_learning_rate": 3.0e-5,
        "total_steps": 2,
        "batch_size": 4,
        "exclude_target_ffid_neighbors": split_scope == "whole_ffid",
        "neighbor_dropout": 0.05,
        "derivative_weight": 0.1,
        "gradient_clip_norm": 1.0,
        "evaluation_interval_steps": 1,
        "validation_batch_size": 4,
        "training_audit_count": 4,
        "mixed_precision": "bfloat16",
        "device": configured_device,
    }
    if not formal_candidate:
        training["ffid_range"] = ffid_range or [10, 12]
    evaluation: dict[str, object] = {
        "primary_metric": "oracle_per_trace_unit_rms_global_snr_db",
        "success_threshold_db": 15.0,
        "comparison": "strictly_greater_than",
        "required_eligible_ffid_count": 4780,
        "required_sample_count": 625,
        "required_effective_split_counts": {
            "train": 1842090,
            "validation": 114490,
            "test": 346885,
        },
        "required_fully_excluded_ffids": [1746],
    }
    if split_scope == "whole_ffid":
        evaluation["required_ffid_split_counts"] = {
            "train": 1,
            "validation": 1,
            "test": 1,
        }
    return {
        "project": {"random_seed": 5},
        "sampling": {
            (
                "random_ffid_holdout_fraction"
                if split_scope == "whole_ffid"
                else "random_trace_holdout_fraction"
            ): 0.5,
            "validation_fraction_of_holdout": 0.5,
            "split_scope": split_scope,
            "trace_amplitude_filter": trace_filter.to_dict(),
            "duplicate_physical_coordinate_policy": "keep_lowest_array_row",
        },
        "normalization": {
            "coordinates": "train_minmax_linear_plus_azimuth_sin_cos",
            "amplitude": "train_global_rms",
        },
        "model": {
            "name": "neighbor_trace_inpainter",
            "hidden_width": 8,
            "target_coordinates": [
                "relative_receiver_x_m",
                "source_y_m",
                "relative_receiver_y_m",
            ],
            "target_coordinate_scaling": "train_minmax",
            "stem_kernel_size": 15,
            "residual_kernel_size": 7,
            "temporal_dilations": [1, 2, 4, 8, 16, 32, 16, 8, 4, 2, 1],
            "neighborhood": {
                "relative_receiver_x_radius": 1,
                "source_shot_radius": 2,
                "relative_receiver_y_radius": 3,
                "relative_receiver_spacing_m": 40.0,
                "source_shot_spacing_m": 80.0,
                "same_source_x_only": True,
            },
        },
        "training": training,
        "evaluation": evaluation,
    }


def prepare_neighbor_training_fixture(
    tmp_path: Path,
    *,
    configured_device: str = "cpu",
    ffid_range: list[int] | None = None,
    include_excluded_trace: bool = False,
    formal_candidate: bool = False,
    split_scope: str = "per_ffid",
) -> tuple[Path, Path, Path]:
    """Write the interim + processed neighbor datasets and a minimal config."""
    source = tmp_path / "source.sgy"
    source.write_bytes(b"synthetic neighbor-inpainter source")
    trace_table, amplitudes, time_s = _neighbor_trace_data(
        include_excluded_trace=include_excluded_trace,
    )
    interim = tmp_path / "interim"
    write_interim_trace_dataset(
        interim,
        trace_table,
        amplitudes,
        time_s,
        source,
        "synthetic_neighbor_grid",
        selection={"ffid_min": 10, "ffid_max": 12},
    )
    trace_filter = TraceAmplitudeFilterConfig(
        exclude_all_zero=True,
        max_abs_amplitude=10000.0,
    )
    processed = tmp_path / "processed"
    prepare_baseline_dataset(
        interim,
        processed,
        holdout_fraction=0.5,
        validation_fraction_of_holdout=0.5,
        random_seed=5,
        split_scope=split_scope,
        trace_amplitude_filter=trace_filter,
        config_source="studies/synthetic_neighbor/config.yaml",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            _neighbor_config(
                configured_device=configured_device,
                ffid_range=ffid_range,
                formal_candidate=formal_candidate,
                split_scope=split_scope,
                trace_filter=trace_filter,
            ),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config, interim, processed
