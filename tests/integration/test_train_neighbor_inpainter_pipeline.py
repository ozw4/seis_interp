from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from seis_interp.configuration import REPOSITORY_ROOT, ConfigurationError, load_resolved_config
from seis_interp.data.file_checksums import file_sha256
from seis_interp.data.trace_store import write_interim_trace_dataset
from seis_interp.models import NeighborTraceInpainter, SharedOffsetAttentionInpainter
from seis_interp.models.neighbor_trace_inpainter import (
    SAME_LINE_EXACT_RECEIVER_LINEAR_BRACKETING_CHANNELS_REFERENCE,
    SAME_LINE_EXACT_RECEIVER_LINEAR_BRACKETING_REFERENCE,
)
from seis_interp.pipelines.prepare_baseline import prepare_baseline_dataset
from seis_interp.pipelines.train_neighbor_inpainter import (
    _validated_settings,
    train_neighbor_inpainter_run,
)
from seis_interp.processing.trace_amplitude_filter import TraceAmplitudeFilterConfig
from seis_interp.training.neighbor_inpainter_checkpoints import (
    load_neighbor_inpainter_checkpoint,
)


def _build_neighbor_training_fixture(
    tmp_path: Path,
    *,
    configured_device: str = "cpu",
    ffid_range: list[int] | None = None,
    include_excluded_trace: bool = False,
    formal_candidate: bool = False,
    split_scope: str = "per_ffid",
) -> tuple[Path, Path, Path]:
    source = tmp_path / "source.sgy"
    source.write_bytes(b"synthetic neighbor-inpainter source")
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
    interim = tmp_path / "interim"
    write_interim_trace_dataset(
        interim,
        pd.DataFrame(rows),
        np.stack(amplitudes),
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
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
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
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config, interim, processed


def test_pipeline_supports_disjoint_whole_ffid_splits(tmp_path: Path) -> None:
    config, interim, processed = _build_neighbor_training_fixture(
        tmp_path,
        split_scope="whole_ffid",
    )
    output = tmp_path / "run"

    metrics = train_neighbor_inpainter_run(
        config_path=config,
        interim_dir=interim,
        processed_dir=processed,
        output_dir=output,
    )

    trace_table = pd.read_parquet(interim / "traces.parquet")
    split_table = pd.read_parquet(processed / "trace_split.parquet")
    joined = trace_table[["array_row", "ffid"]].merge(split_table, on="array_row")
    assert joined.groupby("ffid")["split"].nunique().eq(1).all()
    assert metrics["split_counts"] == {
        "train": 8,
        "validation": 8,
        "test": 8,
        "excluded": 0,
    }
    inputs_lock = json.loads((output / "inputs.lock.json").read_text(encoding="utf-8"))
    assert inputs_lock["preparation"]["split_scope"] == "whole_ffid"
    assert inputs_lock["preparation"]["ffid_split_counts"] == {
        "train": 1,
        "validation": 1,
        "test": 1,
    }
    assert inputs_lock["training"]["exclude_target_ffid_neighbors"] is True
    assert metrics["formal_success_scope"]["checks"]["target_ffid_context_matches"] is True
    assert metrics["formal_success_scope"]["checks"]["target_ffid_neighbor_entries_zero"] is True
    assert metrics["neighbor_availability"]["train"]["target_ffid_neighbor_entries"] == 0
    assert metrics["neighbor_availability"]["validation"]["target_ffid_neighbor_entries"] == 0


def test_pipeline_writes_reproducible_train_only_neighbor_run(tmp_path: Path) -> None:
    config, interim, processed = _build_neighbor_training_fixture(tmp_path)
    output = tmp_path / "run"

    metrics = train_neighbor_inpainter_run(
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
    assert metrics == json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["validation_metric_domain"] == "oracle_per_trace_unit_rms"
    assert (
        metrics["oracle_per_trace_unit_rms_global_snr_db"]
        == (metrics["best_validation_global_snr_db"])
    )
    assert metrics["success_threshold_db"] == 15.0
    assert metrics["metric_success"] is (metrics["oracle_per_trace_unit_rms_global_snr_db"] > 15.0)
    assert metrics["scope_success"] is False
    assert metrics["success"] is False
    assert metrics["prediction_reference"] == "none"
    assert metrics["split_counts"] == {
        "train": 12,
        "validation": 6,
        "test": 6,
        "excluded": 0,
    }
    assert metrics["training_trace_count"] == 12
    assert metrics["training_audit_trace_count"] == 4
    assert metrics["best_validation_signal_energy"] == pytest.approx(6 * 17)
    assert metrics["best_validation_error_energy"] > 0.0
    assert metrics["best_validation_error_mean_square"] == pytest.approx(
        metrics["best_validation_error_energy"] / (6 * 17)
    )
    assert metrics["clean_validation_trace_count"] == 6
    assert (
        metrics["clean_validation_raw_global_snr_db"]
        == metrics["oracle_per_trace_unit_rms_global_snr_db"]
    )
    assert metrics["duplicate_physical_coordinates"]["removed_trace_count"] == 0
    assert metrics["collision_audit"] == {
        "canonical_remaining_duplicate_physical_cells": 0,
        "train_coordinate_collision_rows": 0,
        "train_coordinate_collision_cells": 0,
        "train_validation_coordinate_overlap_rows": 0,
        "train_validation_exact_unit_amplitude_duplicate_rows": 0,
    }
    checkpoint = load_neighbor_inpainter_checkpoint(output / "artifacts" / "best.pt")
    assert checkpoint.best_step == metrics["best_step"]
    assert checkpoint.best_validation_global_snr_db == (metrics["best_validation_global_snr_db"])
    assert checkpoint.model.neighbor_count == 104
    assert checkpoint.model.width == 8
    assert checkpoint.model.neighbor_gating == "none"
    assert checkpoint.model.neighbor_alignment_kernel_size == 1
    assert checkpoint.model.neighbor_alignment is None
    assert checkpoint.model.prediction_reference == "none"

    inputs_lock_text = (output / "inputs.lock.json").read_text(encoding="utf-8")
    inputs_lock = json.loads(inputs_lock_text)
    assert str(tmp_path) not in inputs_lock_text
    assert inputs_lock["amplitude_access"] == metrics["amplitude_access"]
    assert inputs_lock["amplitude_access"]["value_rows_materialized_by_split"] == {
        "train": True,
        "validation": True,
        "test": False,
        "excluded": False,
    }
    assert inputs_lock["amplitude_access"]["full_file_bytes_hashed"] is True
    assert inputs_lock["neighborhood"]["neighbor_count"] == 104
    assert inputs_lock["neighborhood"]["same_source_x_only"] is True
    assert inputs_lock["target_coordinates"]["fit_split"] == "train"
    assert inputs_lock["target_coordinates"]["scaling"] == "train_minmax"
    assert inputs_lock["model"]["neighbor_gating"] == "none"
    assert inputs_lock["model"]["neighbor_alignment_kernel_size"] == 1
    assert inputs_lock["model"]["neighbor_alignment"]["enabled"] is False
    assert inputs_lock["model"]["prediction_reference"] == "none"
    assert "source_bracketing" not in inputs_lock
    assert inputs_lock["split_counts"] == metrics["split_counts"]
    assert inputs_lock["training"]["minimum_learning_rate_factor"] == 0.03
    assert inputs_lock["training"]["gradient_clip_norm"] == 1.0
    assert inputs_lock["training"]["target_sampling"] == "with_replacement"
    assert inputs_lock["training"]["target_sampling_seed"] == 6
    assert inputs_lock["training"]["neighbor_dropout_seed"] == 6
    assert inputs_lock["training"]["target_sampling_rng_independent_of_neighbor_dropout"] is False
    assert inputs_lock["interim_files"] == {
        name: {"sha256": hashlib.sha256((interim / name).read_bytes()).hexdigest()}
        for name in ("traces.parquet", "amplitudes.npy", "time_s.npy", "dataset.json")
    }
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert (
        run["git_commit"]
        == subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    assert run["random_seed"] == 5
    assert run["validation_metric_domain"] == "oracle_per_trace_unit_rms"
    assert run["training"]["effective_bfloat16"] is False
    assert run["training"]["target_sampling"] == "with_replacement"
    assert run["training"]["target_sampling_seed"] == 6
    assert run["training"]["neighbor_dropout_seed"] == 6
    assert run["model"]["neighbor_gating"] == "none"
    assert run["model"]["neighbor_alignment_kernel_size"] == 1
    assert run["model"]["prediction_reference"] == "none"
    assert "source_bracketing" not in run
    assert run["checkpoint"]["revalidation_matches"] is True
    assert "source_bracketing" not in metrics


def test_pipeline_appends_train_only_source_bracketing_reference(tmp_path: Path) -> None:
    config, interim, processed = _build_neighbor_training_fixture(tmp_path)
    configured = yaml.safe_load(config.read_text(encoding="utf-8"))
    configured["model"]["prediction_reference"] = "same_line_exact_receiver_linear_bracketing"
    config.write_text(yaml.safe_dump(configured, sort_keys=False), encoding="utf-8")
    output = tmp_path / "bracketing-run"

    metrics = train_neighbor_inpainter_run(
        config_path=config,
        interim_dir=interim,
        processed_dir=processed,
        output_dir=output,
    )

    inputs_lock = json.loads((output / "inputs.lock.json").read_text(encoding="utf-8"))
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    checkpoint = load_neighbor_inpainter_checkpoint(output / "artifacts" / "best.pt")
    assert metrics["prediction_reference"] == ("same_line_exact_receiver_linear_bracketing")
    assert checkpoint.model.neighbor_count == 105
    assert checkpoint.model.local_neighbor_count == 104
    assert inputs_lock["model"]["neighbor_count"] == 105
    assert inputs_lock["model"]["local_neighbor_count"] == 104
    assert inputs_lock["model"]["reference_neighbor_count"] == 1
    assert inputs_lock["neighborhood"]["neighbor_count"] == 104
    assert inputs_lock["source_bracketing"] == metrics["source_bracketing"]
    assert run["source_bracketing"] == metrics["source_bracketing"]
    for split in ("train", "validation"):
        audit = metrics["source_bracketing"][split]
        assert audit["source_split_counts"]["non_train"] == 0
        assert audit["target_ffid_reference_entries"] == 0
        assert audit["same_source_y_reference_entries"] == 0
        assert (
            audit["bracketed_rows"] + audit["one_sided_rows"] + audit["unresolved_rows"]
            == audit["row_count"]
        )
    checks = metrics["formal_success_scope"]["checks"]
    assert checks["source_bracketing_target_ffid_entries_zero"] is True
    assert checks["source_bracketing_same_source_y_entries_zero"] is True
    assert checks["source_bracketing_sources_train_only"] is True


def test_pipeline_appends_two_train_only_weighted_source_bracketing_channels(
    tmp_path: Path,
) -> None:
    config, interim, processed = _build_neighbor_training_fixture(tmp_path)
    configured = yaml.safe_load(config.read_text(encoding="utf-8"))
    configured["model"]["prediction_reference"] = (
        SAME_LINE_EXACT_RECEIVER_LINEAR_BRACKETING_CHANNELS_REFERENCE
    )
    config.write_text(yaml.safe_dump(configured, sort_keys=False), encoding="utf-8")
    output = tmp_path / "bracketing-channels-run"

    metrics = train_neighbor_inpainter_run(
        config_path=config,
        interim_dir=interim,
        processed_dir=processed,
        output_dir=output,
    )

    inputs_lock = json.loads((output / "inputs.lock.json").read_text(encoding="utf-8"))
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    checkpoint = load_neighbor_inpainter_checkpoint(output / "artifacts" / "best.pt")
    assert metrics["prediction_reference"] == (
        SAME_LINE_EXACT_RECEIVER_LINEAR_BRACKETING_CHANNELS_REFERENCE
    )
    assert checkpoint.model.prediction_reference == (
        SAME_LINE_EXACT_RECEIVER_LINEAR_BRACKETING_CHANNELS_REFERENCE
    )
    assert checkpoint.model.neighbor_count == 106
    assert checkpoint.model.reference_neighbor_count == 2
    assert checkpoint.model.local_neighbor_count == 104
    assert checkpoint.model.input_channels == 216
    assert inputs_lock["neighborhood"]["neighbor_count"] == 104
    assert inputs_lock["model"]["neighbor_count"] == 106
    assert inputs_lock["model"]["local_neighbor_count"] == 104
    assert inputs_lock["model"]["reference_neighbor_count"] == 2
    assert inputs_lock["model"]["reference_channel_indices"] == [104, 105]
    assert inputs_lock["model"]["reference_channel_order"] == [
        "strict_lower_source_y",
        "strict_upper_source_y",
    ]
    assert inputs_lock["model"]["reference_weight_source"] == ("last_two_availability_channels")
    assert inputs_lock["source_bracketing"] == metrics["source_bracketing"]
    assert run["source_bracketing"] == metrics["source_bracketing"]
    assert metrics["source_bracketing"]["type"] == (
        SAME_LINE_EXACT_RECEIVER_LINEAR_BRACKETING_CHANNELS_REFERENCE
    )
    assert metrics["source_bracketing"]["reference_channel_order"] == [
        "strict_lower_source_y",
        "strict_upper_source_y",
    ]
    assert metrics["source_bracketing"]["reference_trace_values"] == ("raw_train_amplitudes")
    assert metrics["source_bracketing"]["availability_values"] == ("linear_interpolation_weights")
    assert metrics["source_bracketing"]["neighbor_dropout_applied"] is False
    for split in ("train", "validation"):
        audit = metrics["source_bracketing"][split]
        assert audit["source_split_counts"]["non_train"] == 0
        assert audit["target_ffid_reference_entries"] == 0
        assert audit["same_source_y_reference_entries"] == 0
    checks = metrics["formal_success_scope"]["checks"]
    assert checks["source_bracketing_unresolved_rows_zero"] is all(
        metrics["source_bracketing"][split]["unresolved_rows"] == 0
        for split in ("train", "validation")
    )
    assert checks["source_bracketing_target_ffid_entries_zero"] is True
    assert checks["source_bracketing_same_source_y_entries_zero"] is True
    assert checks["source_bracketing_sources_train_only"] is True
    assert run["checkpoint"]["revalidation_matches"] is True


def test_pipeline_trains_shared_offset_attention_with_exact_geometry_offsets(
    tmp_path: Path,
) -> None:
    config, interim, processed = _build_neighbor_training_fixture(tmp_path)
    configured = yaml.safe_load(config.read_text(encoding="utf-8"))
    configured["model"].update(
        {
            "name": "shared_offset_attention_inpainter",
            "hidden_width": 8,
            "neighbor_feature_width": 4,
            "attention_width": 4,
            "target_coordinates": [
                "source_x_m",
                "source_y_m",
                "relative_receiver_x_m",
                "relative_receiver_y_m",
            ],
            "coordinate_conditioning": "film",
            "neighbor_gating": "offset_target_time_masked_softmax",
            "neighbor_alignment_kernel_size": 1,
            "prediction_reference": "distance_prior_shifted_neighbor_mean",
            "coarse_shift_samples_per_relative_receiver_y_index": 2,
            "attention_geometry_prior_scale": 0.5,
            "stem_kernel_size": 5,
            "residual_kernel_size": 3,
            "temporal_dilations": [1],
            "neighborhood": {
                "type": "multiline_staggered_source",
                "relative_receiver_x_radius": 1,
                "source_x_line_radius": 0,
                "source_y_half_shot_radius": 2,
                "relative_receiver_y_radius": 1,
                "relative_receiver_spacing_m": 40.0,
                "source_x_line_spacing_m": 160.0,
                "source_y_half_shot_spacing_m": 40.0,
            },
        }
    )
    configured["training"]["total_steps"] = 1
    config.write_text(yaml.safe_dump(configured, sort_keys=False), encoding="utf-8")
    output = tmp_path / "shared-offset-attention-run"

    metrics = train_neighbor_inpainter_run(
        config_path=config,
        interim_dir=interim,
        processed_dir=processed,
        output_dir=output,
    )

    checkpoint = load_neighbor_inpainter_checkpoint(output / "artifacts" / "best.pt")
    inputs_lock = json.loads((output / "inputs.lock.json").read_text(encoding="utf-8"))
    assert isinstance(checkpoint.model, SharedOffsetAttentionInpainter)
    assert checkpoint.model.neighbor_count == 26
    checkpoint_offsets = checkpoint.model.neighbor_offsets.tolist()
    assert checkpoint_offsets == inputs_lock["neighborhood"]["offset_order"]
    assert checkpoint_offsets == inputs_lock["model"]["offset_order"]
    assert inputs_lock["model"]["name"] == "shared_offset_attention_inpainter"
    assert inputs_lock["model"]["offset_order_source"] == "pipeline_geometry_exact"
    assert inputs_lock["model"]["attention"]["complexity"] == "O(B*K*T)"
    assert inputs_lock["model"]["coarse_alignment"] == {
        "type": "zero_padded_integer_shift",
        "samples_per_relative_receiver_y_index": 2,
        "source_sample_index": "output_sample_index_minus_shift",
        "circular_wrap": False,
    }
    assert metrics["prediction_reference"] == "distance_prior_shifted_neighbor_mean"


def test_pipeline_trains_legacy_inpainter_with_exact_coarse_alignment_offsets(
    tmp_path: Path,
) -> None:
    config, interim, processed = _build_neighbor_training_fixture(tmp_path)
    configured = yaml.safe_load(config.read_text(encoding="utf-8"))
    configured["model"].update(
        {
            "target_coordinates": [
                "source_x_m",
                "source_y_m",
                "relative_receiver_x_m",
                "relative_receiver_y_m",
            ],
            "coordinate_conditioning": "film",
            "neighbor_gating": "target_coordinate_masked_softmax",
            "neighbor_alignment_kernel_size": 3,
            "coarse_shift_samples_per_relative_receiver_y_index": 2,
            "stem_kernel_size": 5,
            "residual_kernel_size": 3,
            "temporal_dilations": [1],
            "neighborhood": {
                "type": "multiline_staggered_source",
                "relative_receiver_x_radius": 1,
                "source_x_line_radius": 0,
                "source_y_half_shot_radius": 2,
                "relative_receiver_y_radius": 1,
                "relative_receiver_spacing_m": 40.0,
                "source_x_line_spacing_m": 160.0,
                "source_y_half_shot_spacing_m": 40.0,
            },
        }
    )
    configured["training"]["total_steps"] = 1
    config.write_text(yaml.safe_dump(configured, sort_keys=False), encoding="utf-8")
    output = tmp_path / "legacy-coarse-alignment-run"

    train_neighbor_inpainter_run(
        config_path=config,
        interim_dir=interim,
        processed_dir=processed,
        output_dir=output,
    )

    checkpoint = load_neighbor_inpainter_checkpoint(output / "artifacts" / "best.pt")
    inputs_lock = json.loads((output / "inputs.lock.json").read_text(encoding="utf-8"))
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert isinstance(checkpoint.model, NeighborTraceInpainter)
    assert checkpoint.model.neighbor_offsets is not None
    assert checkpoint.model.coarse_sample_shifts is not None
    checkpoint_offsets = checkpoint.model.neighbor_offsets.tolist()
    expected_shifts = [2 * offset[3] for offset in checkpoint_offsets]
    assert checkpoint_offsets == inputs_lock["neighborhood"]["offset_order"]
    assert checkpoint_offsets == inputs_lock["model"]["coarse_alignment"]["offset_order"]
    assert checkpoint.model.coarse_sample_shifts.tolist() == expected_shifts
    assert inputs_lock["model"]["coarse_alignment"] == {
        "type": "zero_padded_integer_shift",
        "offset_order": checkpoint_offsets,
        "offset_order_axes": [
            "relative_receiver_x_index",
            "source_x_line_index",
            "source_y_half_shot_index",
            "relative_receiver_y_index",
        ],
        "offset_order_source": "pipeline_geometry_exact",
        "samples_per_relative_receiver_y_index": 2,
        "sample_shifts": expected_shifts,
        "source_sample_index": "output_sample_index_minus_shift",
        "circular_wrap": False,
        "valid_sample_availability_channels": "time_dependent",
        "applied_before_target_gate_fir_and_stem": True,
    }
    assert inputs_lock["model"] == run["model"]
    assert inputs_lock["model"]["neighbor_alignment"] == {
        "enabled": True,
        "type": "depthwise_fir",
        "kernel_size": 3,
        "groups": 26,
        "bias": False,
        "initialization": "identity_center_tap",
        "applied_after_time_invariant_neighbor_gating": False,
        "unavailable_channels_zeroed_before_fir": True,
        "applied_after_time_dependent_target_gating": True,
        "coarse_alignment_applied_before_fir": True,
    }


def test_pipeline_persists_target_coordinate_neighbor_gating(tmp_path: Path) -> None:
    config, interim, processed = _build_neighbor_training_fixture(tmp_path)
    configured = yaml.safe_load(config.read_text(encoding="utf-8"))
    configured["model"]["neighbor_gating"] = "target_coordinate_masked_softmax"
    configured["training"]["total_steps"] = 1
    config.write_text(yaml.safe_dump(configured, sort_keys=False), encoding="utf-8")
    output = tmp_path / "gated-run"

    train_neighbor_inpainter_run(
        config_path=config,
        interim_dir=interim,
        processed_dir=processed,
        output_dir=output,
    )

    checkpoint = load_neighbor_inpainter_checkpoint(output / "artifacts/best.pt")
    inputs_lock = json.loads((output / "inputs.lock.json").read_text(encoding="utf-8"))
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert checkpoint.model.neighbor_gating == "target_coordinate_masked_softmax"
    assert checkpoint.model.neighbor_gate_projection is not None
    assert inputs_lock["model"]["neighbor_gating"] == "target_coordinate_masked_softmax"
    assert run["model"]["neighbor_gating"] == "target_coordinate_masked_softmax"


def test_pipeline_persists_depthwise_neighbor_alignment_contract(tmp_path: Path) -> None:
    config, interim, processed = _build_neighbor_training_fixture(tmp_path)
    configured = yaml.safe_load(config.read_text(encoding="utf-8"))
    configured["model"]["neighbor_alignment_kernel_size"] = 3
    configured["training"]["total_steps"] = 1
    config.write_text(yaml.safe_dump(configured, sort_keys=False), encoding="utf-8")
    output = tmp_path / "alignment-run"

    train_neighbor_inpainter_run(
        config_path=config,
        interim_dir=interim,
        processed_dir=processed,
        output_dir=output,
    )

    checkpoint = load_neighbor_inpainter_checkpoint(output / "artifacts/best.pt")
    inputs_lock = json.loads((output / "inputs.lock.json").read_text(encoding="utf-8"))
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert checkpoint.model.neighbor_alignment_kernel_size == 3
    assert checkpoint.model.neighbor_alignment is not None
    expected_contract = {
        "enabled": True,
        "type": "depthwise_fir",
        "kernel_size": 3,
        "groups": 104,
        "bias": False,
        "initialization": "identity_center_tap",
        "applied_after_time_invariant_neighbor_gating": True,
        "unavailable_channels_zeroed_before_fir": True,
    }
    assert inputs_lock["model"]["neighbor_alignment_kernel_size"] == 3
    assert inputs_lock["model"]["neighbor_alignment"] == expected_contract
    assert run["model"]["neighbor_alignment_kernel_size"] == 3
    assert run["model"]["neighbor_alignment"] == expected_contract


def test_pipeline_persists_masked_aligned_neighbor_mean_reference(tmp_path: Path) -> None:
    config, interim, processed = _build_neighbor_training_fixture(tmp_path)
    configured = yaml.safe_load(config.read_text(encoding="utf-8"))
    configured["model"]["neighbor_gating"] = "target_coordinate_masked_softmax"
    configured["model"]["neighbor_alignment_kernel_size"] = 3
    configured["model"]["prediction_reference"] = "masked_aligned_neighbor_mean"
    configured["training"]["total_steps"] = 1
    config.write_text(yaml.safe_dump(configured, sort_keys=False), encoding="utf-8")
    output = tmp_path / "prediction-reference-run"

    metrics = train_neighbor_inpainter_run(
        config_path=config,
        interim_dir=interim,
        processed_dir=processed,
        output_dir=output,
    )

    checkpoint = load_neighbor_inpainter_checkpoint(output / "artifacts/best.pt")
    inputs_lock = json.loads((output / "inputs.lock.json").read_text(encoding="utf-8"))
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert checkpoint.model.prediction_reference == "masked_aligned_neighbor_mean"
    assert inputs_lock["model"]["prediction_reference"] == "masked_aligned_neighbor_mean"
    assert run["model"]["prediction_reference"] == "masked_aligned_neighbor_mean"
    assert metrics["prediction_reference"] == "masked_aligned_neighbor_mean"


def test_pipeline_never_materializes_test_or_excluded_amplitude_values(tmp_path: Path) -> None:
    config, interim, processed = _build_neighbor_training_fixture(
        tmp_path,
        include_excluded_trace=True,
    )
    split_table = pd.read_parquet(processed / "trace_split.parquet")
    test_rows = split_table.loc[split_table["split"].eq("test"), "array_row"].to_numpy(
        dtype=np.int64
    )
    excluded_rows = split_table.loc[split_table["split"].eq("excluded"), "array_row"].to_numpy(
        dtype=np.int64
    )
    assert len(excluded_rows) == 1
    amplitudes_path = interim / "amplitudes.npy"
    amplitudes = np.load(amplitudes_path, allow_pickle=False)
    amplitudes[test_rows] = np.nan
    amplitudes[excluded_rows] = np.nan
    np.save(amplitudes_path, amplitudes)
    preparation_path = processed / "preparation.json"
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    preparation["input_files"]["amplitudes.npy"]["sha256"] = file_sha256(amplitudes_path)
    preparation_path.write_text(json.dumps(preparation, indent=2, sort_keys=True) + "\n")

    metrics = train_neighbor_inpainter_run(
        config_path=config,
        interim_dir=interim,
        processed_dir=processed,
        output_dir=tmp_path / "run",
    )

    assert metrics["split_counts"]["test"] == len(test_rows)
    assert metrics["split_counts"]["excluded"] == len(excluded_rows)
    assert metrics["amplitude_access"]["value_rows_materialized_by_split"]["test"] is False
    assert metrics["amplitude_access"]["value_rows_materialized_by_split"]["excluded"] is False


def test_pipeline_canonicalizes_cross_split_physical_duplicates_before_selection(
    tmp_path: Path,
) -> None:
    config, interim, processed = _build_neighbor_training_fixture(tmp_path)
    split_table = pd.read_parquet(processed / "trace_split.parquet").set_index("array_row")
    train_row = int(split_table.index[split_table["split"].eq("train")][0])
    validation_row = int(split_table.index[split_table["split"].eq("validation")][0])
    kept_row, removed_row = sorted((train_row, validation_row))
    kept_split = str(split_table.loc[kept_row, "split"])
    removed_split = str(split_table.loc[removed_row, "split"])

    traces_path = interim / "traces.parquet"
    traces = pd.read_parquet(traces_path)
    coordinate_columns = [
        "source_x_m",
        "source_y_m",
        "receiver_x_m",
        "receiver_y_m",
    ]
    traces.loc[removed_row, coordinate_columns] = traces.loc[
        kept_row, coordinate_columns
    ].to_numpy()
    traces.to_parquet(traces_path, index=False)
    preparation_path = processed / "preparation.json"
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    preparation["input_files"]["traces.parquet"]["sha256"] = file_sha256(traces_path)
    preparation_path.write_text(json.dumps(preparation, indent=2, sort_keys=True) + "\n")

    metrics = train_neighbor_inpainter_run(
        config_path=config,
        interim_dir=interim,
        processed_dir=processed,
        output_dir=tmp_path / "run",
    )

    audit = metrics["duplicate_physical_coordinates"]
    assert audit["physical_coordinate_key"] == coordinate_columns
    assert audit["duplicate_physical_cell_count"] == 1
    assert audit["duplicate_physical_row_count"] == 2
    assert audit["removed_trace_count"] == 1
    assert audit["removed_counts_by_split"][removed_split] == 1
    assert audit["removed_rows"] == [
        {
            "array_row": removed_row,
            "ffid": int(traces.loc[removed_row, "ffid"]),
            "split": removed_split,
            "kept_array_row": kept_row,
            "kept_ffid": int(traces.loc[kept_row, "ffid"]),
            "kept_split": kept_split,
        }
    ]
    assert audit["winner_selection_uses_split"] is False
    assert audit["winner_selection_uses_amplitude"] is False
    assert audit["remaining_duplicate_physical_cell_count"] == 0
    assert metrics["collision_audit"]["train_coordinate_collision_cells"] == 0
    assert metrics["collision_audit"]["train_validation_coordinate_overlap_rows"] == 0


def test_pipeline_supports_inclusive_ffid_range(tmp_path: Path) -> None:
    config, interim, processed = _build_neighbor_training_fixture(
        tmp_path,
        ffid_range=[10, 11],
    )

    metrics = train_neighbor_inpainter_run(
        config_path=config,
        interim_dir=interim,
        processed_dir=processed,
        output_dir=tmp_path / "run",
    )

    assert metrics["split_counts"] == {
        "train": 8,
        "validation": 4,
        "test": 4,
        "excluded": 0,
    }
    inputs_lock = json.loads((tmp_path / "run/inputs.lock.json").read_text(encoding="utf-8"))
    assert inputs_lock["selection"]["configured_ffid_range"] == [10, 11]
    assert inputs_lock["selection"]["selected_ffids"] == [10, 11]
    assert metrics["scope_success"] is False
    assert metrics["success"] is False
    assert metrics["formal_success_scope"]["checks"]["ffid_range_not_configured"] is False


def test_formal_candidate_refuses_scope_drift_before_training(tmp_path: Path) -> None:
    config, interim, processed = _build_neighbor_training_fixture(
        tmp_path,
        formal_candidate=True,
    )

    with pytest.raises(ValueError, match="does not match its required survey scope"):
        train_neighbor_inpainter_run(
            config_path=config,
            interim_dir=interim,
            processed_dir=processed,
            output_dir=tmp_path / "run",
        )


def test_pipeline_rejects_existing_output_before_reading_inputs(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        train_neighbor_inpainter_run(
            config_path=tmp_path / "missing-config.yaml",
            interim_dir=tmp_path / "missing-interim",
            processed_dir=tmp_path / "missing-processed",
            output_dir=output,
        )


def test_study_017_config_resolves_the_implemented_contract() -> None:
    config = load_resolved_config(
        REPOSITORY_ROOT / "studies/study_017_all_ffid_neighbor_inpainter/config.yaml"
    )

    settings = _validated_settings(config, device_override=None)

    assert settings.model_name == "neighbor_trace_inpainter"
    assert settings.hidden_width == 128
    assert settings.total_steps == 2500
    assert settings.batch_size == 96
    assert settings.evaluation_interval_steps == 500
    assert settings.validation_batch_size == 2048
    assert settings.training_audit_count == 114492
    assert settings.neighbor_gating == "none"
    assert settings.neighbor_alignment_kernel_size == 1
    assert settings.prediction_reference == "none"
    assert settings.coarse_shift_samples_per_relative_receiver_y_index == 0
    assert settings.minimum_learning_rate == 1.5e-5
    assert settings.ffid_range is None
    assert settings.required_eligible_ffid_count == 4780
    assert settings.required_sample_count == 625
    assert dict(settings.required_effective_split_counts) == {
        "train": 1842090,
        "validation": 114490,
        "test": 346885,
    }
    assert settings.required_fully_excluded_ffids == (1746,)


def test_shared_offset_attention_config_resolves_model_specific_settings() -> None:
    config = load_resolved_config(
        REPOSITORY_ROOT / "studies/study_019_all_ffid_25pct_neighbor_inpainter/config.yaml"
    )
    config["model"]["name"] = "shared_offset_attention_inpainter"
    config["model"]["neighbor_gating"] = "offset_target_time_masked_softmax"
    config["model"].update(
        {
            "neighbor_feature_width": 8,
            "attention_width": 16,
            "coarse_shift_samples_per_relative_receiver_y_index": 3,
            "attention_geometry_prior_scale": 1.0,
            "neighbor_alignment_kernel_size": 1,
            "prediction_reference": "distance_prior_shifted_neighbor_mean",
        }
    )

    settings = _validated_settings(config, device_override="cpu")

    assert settings.model_name == "shared_offset_attention_inpainter"
    assert settings.neighbor_feature_width == 8
    assert settings.attention_width == 16
    assert settings.coarse_shift_samples_per_relative_receiver_y_index == 3
    assert settings.attention_geometry_prior_scale == 1.0
    assert settings.prediction_reference == "distance_prior_shifted_neighbor_mean"


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("neighbor_feature_width", 0, "neighbor_feature_width"),
        ("attention_width", True, "attention_width"),
        (
            "coarse_shift_samples_per_relative_receiver_y_index",
            -1,
            "coarse_shift_samples",
        ),
        ("attention_geometry_prior_scale", -0.1, "attention_geometry_prior_scale"),
        ("prediction_reference", "none", "prediction_reference"),
    ],
)
def test_shared_offset_attention_config_rejects_invalid_model_fields(
    key: str,
    value: object,
    message: str,
) -> None:
    config = load_resolved_config(
        REPOSITORY_ROOT / "studies/study_019_all_ffid_25pct_neighbor_inpainter/config.yaml"
    )
    config["model"]["name"] = "shared_offset_attention_inpainter"
    config["model"]["neighbor_gating"] = "offset_target_time_masked_softmax"
    config["model"][key] = value

    with pytest.raises(ConfigurationError, match=message):
        _validated_settings(config, device_override="cpu")


@pytest.mark.parametrize(
    "prediction_reference",
    [
        SAME_LINE_EXACT_RECEIVER_LINEAR_BRACKETING_REFERENCE,
        SAME_LINE_EXACT_RECEIVER_LINEAR_BRACKETING_CHANNELS_REFERENCE,
    ],
)
def test_legacy_model_accepts_same_line_bracketing_reference(
    prediction_reference: str,
) -> None:
    config = load_resolved_config(
        REPOSITORY_ROOT
        / "studies/study_020_all_ffid_25pct_whole_ffid_neighbor_inpainter/config.yaml"
    )
    config["model"]["prediction_reference"] = prediction_reference

    settings = _validated_settings(config, device_override="cpu")

    assert settings.prediction_reference == prediction_reference


@pytest.mark.parametrize(
    "prediction_reference",
    [
        SAME_LINE_EXACT_RECEIVER_LINEAR_BRACKETING_REFERENCE,
        SAME_LINE_EXACT_RECEIVER_LINEAR_BRACKETING_CHANNELS_REFERENCE,
    ],
)
def test_bracketing_reference_rejects_legacy_coarse_alignment(
    prediction_reference: str,
) -> None:
    config = load_resolved_config(
        REPOSITORY_ROOT
        / "studies/study_020_all_ffid_25pct_whole_ffid_neighbor_inpainter/config.yaml"
    )
    config["model"]["prediction_reference"] = prediction_reference
    config["model"]["coarse_shift_samples_per_relative_receiver_y_index"] = 3

    with pytest.raises(ConfigurationError, match="bracketing cannot be combined"):
        _validated_settings(config, device_override="cpu")


@pytest.mark.parametrize(
    "prediction_reference",
    [
        SAME_LINE_EXACT_RECEIVER_LINEAR_BRACKETING_REFERENCE,
        SAME_LINE_EXACT_RECEIVER_LINEAR_BRACKETING_CHANNELS_REFERENCE,
    ],
)
def test_shared_attention_rejects_same_line_bracketing_reference(
    prediction_reference: str,
) -> None:
    config = load_resolved_config(
        REPOSITORY_ROOT / "studies/study_019_all_ffid_25pct_neighbor_inpainter/config.yaml"
    )
    config["model"]["name"] = "shared_offset_attention_inpainter"
    config["model"]["neighbor_gating"] = "offset_target_time_masked_softmax"
    config["model"]["neighbor_alignment_kernel_size"] = 1
    config["model"]["prediction_reference"] = prediction_reference

    with pytest.raises(ConfigurationError, match="prediction_reference"):
        _validated_settings(config, device_override="cpu")


def test_legacy_neighbor_model_rejects_offset_time_attention_gating() -> None:
    config = load_resolved_config(
        REPOSITORY_ROOT / "studies/study_019_all_ffid_25pct_neighbor_inpainter/config.yaml"
    )
    config["model"]["neighbor_gating"] = "offset_target_time_masked_softmax"

    with pytest.raises(ConfigurationError, match="requires model.name"):
        _validated_settings(config, device_override="cpu")


def test_legacy_neighbor_model_rejects_shared_attention_only_fields() -> None:
    config = load_resolved_config(
        REPOSITORY_ROOT / "studies/study_019_all_ffid_25pct_neighbor_inpainter/config.yaml"
    )
    config["model"]["attention_width"] = 16

    with pytest.raises(ConfigurationError, match="shared offset attention fields require"):
        _validated_settings(config, device_override="cpu")


def test_legacy_coarse_alignment_requires_multiline_geometry(tmp_path: Path) -> None:
    config_path, _interim, _processed = _build_neighbor_training_fixture(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["model"]["coarse_shift_samples_per_relative_receiver_y_index"] = 2

    with pytest.raises(ConfigurationError, match="requires model.neighborhood.type"):
        _validated_settings(config, device_override="cpu")


@pytest.mark.parametrize(
    ("section", "key", "drifted_value", "message"),
    [
        ("sampling", "duplicate_physical_coordinate_policy", "keep_last", "duplicate"),
        ("model", "stem_kernel_size", 14, "stem_kernel_size.*odd"),
        ("model", "neighbor_gating", "sigmoid", "neighbor_gating"),
        ("model", "prediction_reference", "mean", "prediction_reference"),
        (
            "model",
            "neighbor_alignment_kernel_size",
            4,
            "neighbor_alignment_kernel_size.*odd",
        ),
        ("training", "optimizer", "sgd", "optimizer"),
        ("evaluation", "comparison", "greater_than_or_equal", "comparison"),
        ("evaluation", "required_eligible_ffid_count", 0, "positive integer"),
    ],
)
def test_pipeline_rejects_fixed_contract_drift(
    tmp_path: Path,
    section: str,
    key: str,
    drifted_value: object,
    message: str,
) -> None:
    config_path, _interim, _processed = _build_neighbor_training_fixture(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config[section][key] = drifted_value

    with pytest.raises(ConfigurationError, match=message):
        _validated_settings(config, device_override="cpu")
