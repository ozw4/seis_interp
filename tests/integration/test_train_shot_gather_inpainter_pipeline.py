from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from seis_interp.cli import main
from seis_interp.data.trace_store import write_interim_trace_dataset
from seis_interp.pipelines.prepare_baseline import prepare_baseline_dataset
from seis_interp.processing.trace_amplitude_filter import TraceAmplitudeFilterConfig
from seis_interp.training.shot_gather_inpainter_checkpoints import (
    load_shot_gather_inpainter_checkpoint,
)


def _build_shot_gather_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "source.sgy"
    source.write_bytes(b"synthetic whole-shot source")
    receiver_x_offsets = np.arange(-140.0, 180.0, 40.0)
    receiver_y_offsets = np.arange(-2680.0, 40.0, 40.0)
    time_s = np.arange(5, dtype=np.float64) * 0.008
    rows: list[dict[str, object]] = []
    amplitudes: list[np.ndarray] = []
    trace_index = 0
    for ffid_index, ffid in enumerate(range(10, 18)):
        source_x = 4000.0 + (ffid_index // 4) * 160.0
        source_y = 1000.0 + (ffid_index % 4) * 80.0
        for receiver_x_offset in receiver_x_offsets:
            for receiver_y_offset in receiver_y_offsets:
                receiver_x = source_x + receiver_x_offset
                receiver_y = source_y + receiver_y_offset
                offset = math.hypot(receiver_x_offset, receiver_y_offset)
                rows.append(
                    {
                        "trace_index": trace_index,
                        "ffid": ffid,
                        "source_x_m": source_x,
                        "source_y_m": source_y,
                        "receiver_x_m": receiver_x,
                        "receiver_y_m": receiver_y,
                        "cmp_x_m": (source_x + receiver_x) / 2.0,
                        "cmp_y_m": (source_y + receiver_y) / 2.0,
                        "offset_m": offset,
                        "azimuth_deg": math.degrees(
                            math.atan2(-receiver_x_offset, -receiver_y_offset)
                        )
                        % 360.0,
                        "sample_interval_s": 0.008,
                        "trace_count_in_ffid": 544,
                        "is_complete_ffid": True,
                    }
                )
                phase = ffid_index * 0.1 + receiver_x_offset * 0.001
                trace = np.sin(np.arange(5) * 0.4 + phase)
                trace += 0.2 * np.cos(np.arange(5) * 0.2 + receiver_y_offset * 0.001)
                amplitudes.append(trace.astype(np.float32))
                trace_index += 1
    interim = tmp_path / "interim"
    write_interim_trace_dataset(
        interim,
        pd.DataFrame(rows),
        np.stack(amplitudes),
        time_s,
        source,
        "synthetic_shot_gather_grid",
        selection={"ffid_min": 10, "ffid_max": 17},
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
        split_scope="whole_ffid",
        trace_amplitude_filter=trace_filter,
        config_source="studies/synthetic_shot_gather/config.yaml",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "project": {"random_seed": 5},
                "sampling": {
                    "random_ffid_holdout_fraction": 0.5,
                    "validation_fraction_of_holdout": 0.5,
                    "split_scope": "whole_ffid",
                    "trace_amplitude_filter": trace_filter.to_dict(),
                    "duplicate_physical_coordinate_policy": "keep_lowest_array_row",
                },
                "normalization": {
                    "coordinates": "train_minmax_linear_plus_azimuth_sin_cos",
                    "amplitude": "train_global_rms",
                },
                "model": {
                    "name": "shot_gather_inpainter",
                    "hidden_width": 8,
                    "target_coordinates": ["source_x_m", "source_y_m"],
                    "target_coordinate_scaling": "train_minmax",
                    "stem_kernel_size": 3,
                    "residual_kernel_size": 3,
                    "temporal_dilations": [1],
                    "distance_epsilon": 1.0e-6,
                    "neighborhood": {
                        "type": "nearest_train_source_gathers",
                        "distance": "euclidean_source_xy_m",
                        "source_gather_count": 2,
                    },
                },
                "training": {
                    "amplitude_scaling": "per_trace_rms",
                    "loss": "l2_plus_first_difference",
                    "optimizer": "adamw",
                    "learning_rate": 1.0e-3,
                    "weight_decay": 1.0e-5,
                    "learning_rate_schedule": "cosine",
                    "minimum_learning_rate": 3.0e-5,
                    "total_steps": 1,
                    "batch_size": 1,
                    "target_sampling": "epoch_without_replacement",
                    "exclude_target_ffid_neighbors": True,
                    "neighbor_dropout": 0.0,
                    "derivative_weight": 0.1,
                    "gradient_clip_norm": 1.0,
                    "evaluation_interval_steps": 1,
                    "validation_batch_size": 1,
                    "training_audit_count": 4,
                    "mixed_precision": "bfloat16",
                    "device": "cuda:0",
                    "ffid_range": [10, 17],
                },
                "evaluation": {
                    "primary_metric": "oracle_per_trace_unit_rms_global_snr_db",
                    "success_threshold_db": 25.0,
                    "comparison": "strictly_greater_than",
                    "required_eligible_ffid_count": 4780,
                    "required_sample_count": 625,
                    "required_effective_split_counts": {
                        "train": 578685,
                        "validation": 437087,
                        "test": 1287693,
                    },
                    "required_ffid_split_counts": {
                        "train": 1195,
                        "validation": 896,
                        "test": 2689,
                    },
                    "required_fully_excluded_ffids": [],
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config, interim, processed


def test_cli_runs_leakage_safe_whole_shot_pipeline(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config, interim, processed = _build_shot_gather_fixture(tmp_path)
    output = tmp_path / "run"

    exit_code = main(
        [
            "train",
            "shot-gather-inpainter",
            "--config",
            str(config),
            "--interim",
            str(interim),
            "--processed",
            str(processed),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--json",
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    metrics = json.loads(captured.out)
    assert metrics == json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert "shot_gather_inpainter 0/1" in captured.err
    assert "shot_gather_inpainter 1/1" in captured.err
    assert metrics["validation_metric_domain"] == "oracle_per_trace_unit_rms"
    assert metrics["training_audit_trace_count"] == 4
    assert metrics["formal_success_scope"]["checks"]["target_ffid_neighbor_entries_zero"]
    assert metrics["formal_success_scope"]["checks"]["neighbor_sources_train_only"]
    inputs_lock = json.loads((output / "inputs.lock.json").read_text(encoding="utf-8"))
    assert inputs_lock["preparation"]["split_scope"] == "whole_ffid"
    assert inputs_lock["amplitude_access"]["value_rows_materialized_by_split"] == {
        "excluded": False,
        "test": False,
        "train": True,
        "validation": True,
    }
    assert inputs_lock["model"]["input_feature_schema_version"] == 1
    assert inputs_lock["model"]["input_feature_names"]
    assert inputs_lock["training"]["target_sampling_rng_independent_of_neighbor_dropout"]
    assert (
        inputs_lock["training"]["target_sampling_seed"]
        != inputs_lock["training"]["neighbor_dropout_seed"]
    )
    resolved = yaml.safe_load((output / "config.resolved.yaml").read_text(encoding="utf-8"))
    assert resolved["training"]["device"] == "cpu"
    checkpoint = load_shot_gather_inpainter_checkpoint(output / "artifacts/best.pt")
    assert checkpoint.best_step == metrics["best_step"]
    assert (
        checkpoint.input_feature_schema_version
        == inputs_lock["model"]["input_feature_schema_version"]
    )
    assert list(checkpoint.input_feature_names) == inputs_lock["model"]["input_feature_names"]
    assert (
        checkpoint.best_validation_global_snr_db
        == metrics["oracle_per_trace_unit_rms_global_snr_db"]
    )
