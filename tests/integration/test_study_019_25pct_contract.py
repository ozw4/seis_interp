from __future__ import annotations

from pathlib import Path

import yaml

from seis_interp.configuration import REPOSITORY_ROOT, load_resolved_config
from seis_interp.pipelines.train_neighbor_inpainter import _validated_settings

STUDY_DIRECTORY = REPOSITORY_ROOT / "studies" / "study_019_all_ffid_25pct_neighbor_inpainter"


def test_study_019_locks_the_25pct_per_ffid_split_contract() -> None:
    config = load_resolved_config(STUDY_DIRECTORY / "config.yaml")
    settings = _validated_settings(config, device_override="cpu")

    assert config["sampling"] == {
        "random_trace_holdout_fraction": 0.75,
        "validation_fraction_of_holdout": 0.25,
        "split_scope": "per_ffid",
        "trace_amplitude_filter": {
            "exclude_all_zero": True,
            "max_abs_amplitude": 10000.0,
        },
        "duplicate_physical_coordinate_policy": "keep_lowest_array_row",
    }
    assert settings.success_threshold_db == 25.0
    assert settings.required_eligible_ffid_count == 4780
    assert dict(settings.required_effective_split_counts) == {
        "train": 575864,
        "validation": 431887,
        "test": 1295714,
    }
    assert settings.required_fully_excluded_ffids == (1746,)


def test_study_019_starts_from_the_study_018_formal_model() -> None:
    config = load_resolved_config(STUDY_DIRECTORY / "config.yaml")
    settings = _validated_settings(config, device_override="cpu")

    assert settings.hidden_width == 384
    assert settings.target_coordinates == (
        "source_x_m",
        "source_y_m",
        "relative_receiver_x_m",
        "relative_receiver_y_m",
    )
    assert settings.coordinate_conditioning == "film"
    assert settings.neighbor_gating == "target_coordinate_masked_softmax"
    assert settings.neighbor_alignment_kernel_size == 31
    assert settings.coarse_shift_samples_per_relative_receiver_y_index == 0
    assert settings.temporal_dilations == (1, 2, 4, 8, 16, 32, 64, 32, 16, 8, 4, 2, 1)
    assert settings.relative_receiver_x_radius == 2
    assert settings.source_x_line_radius == 0
    assert settings.source_y_half_shot_radius == 4
    assert settings.relative_receiver_y_radius == 5


def test_stage01_changes_only_the_training_horizon_from_the_base_candidate() -> None:
    config = load_resolved_config(
        STUDY_DIRECTORY / "variants" / "stage01_study018_formal_2500_steps.yaml"
    )
    settings = _validated_settings(config, device_override="cpu")

    assert settings.total_steps == 2500
    assert settings.evaluation_interval_steps == 500
    assert settings.batch_size == 96
    assert settings.training_audit_count == 287933


def test_stage02_adds_only_the_masked_aligned_neighbor_reference() -> None:
    config = load_resolved_config(
        STUDY_DIRECTORY / "variants" / "stage02_residual_neighbor_reference.yaml"
    )
    settings = _validated_settings(config, device_override="cpu")

    assert settings.prediction_reference == "masked_aligned_neighbor_mean"
    assert settings.total_steps == 2500
    assert settings.hidden_width == 384
    assert settings.neighbor_gating == "target_coordinate_masked_softmax"
    assert settings.neighbor_alignment_kernel_size == 31


def test_stage03_uses_k274_shared_offset_attention() -> None:
    config = load_resolved_config(
        STUDY_DIRECTORY / "variants" / "stage03_shared_offset_attention.yaml"
    )
    settings = _validated_settings(config, device_override="cpu")

    assert settings.model_name == "shared_offset_attention_inpainter"
    assert settings.neighbor_feature_width == 8
    assert settings.attention_width == 16
    assert settings.neighbor_gating == "offset_target_time_masked_softmax"
    assert settings.neighbor_alignment_kernel_size == 1
    assert settings.prediction_reference == "distance_prior_shifted_neighbor_mean"
    assert settings.coarse_shift_samples_per_relative_receiver_y_index == 3
    assert settings.attention_geometry_prior_scale == 0.1
    assert settings.relative_receiver_x_radius == 2
    assert settings.source_x_line_radius == 0
    assert settings.source_y_half_shot_radius == 4
    assert settings.relative_receiver_y_radius == 5
    assert settings.total_steps == 2500


def test_stage03a_smoke_limits_only_scope_and_diagnostic_budget() -> None:
    config = load_resolved_config(
        STUDY_DIRECTORY / "variants" / "stage03a_shared_attention_ffid2348_2363_smoke.yaml"
    )
    settings = _validated_settings(config, device_override="cpu")

    assert settings.model_name == "shared_offset_attention_inpainter"
    assert settings.ffid_range == (2348, 2363)
    assert settings.total_steps == 100
    assert settings.evaluation_interval_steps == 50
    assert settings.validation_batch_size == 128
    assert settings.training_audit_count == 1000


def test_stage04_changes_only_the_aperture_and_diagnostic_audit_schedule() -> None:
    config = load_resolved_config(STUDY_DIRECTORY / "variants" / "stage04_same_line_k734.yaml")
    settings = _validated_settings(config, device_override="cpu")

    assert settings.model_name == "neighbor_trace_inpainter"
    assert settings.hidden_width == 384
    assert settings.relative_receiver_x_radius == 3
    assert settings.source_x_line_radius == 0
    assert settings.source_y_half_shot_radius == 6
    assert settings.relative_receiver_y_radius == 7
    assert settings.neighbor_alignment_kernel_size == 31
    assert settings.evaluation_interval_steps == 2500
    assert settings.training_audit_count == 10000


def test_stage05_adds_only_deterministic_coarse_shift_and_diagnostic_schedule() -> None:
    config = load_resolved_config(
        STUDY_DIRECTORY / "variants" / "stage05_legacy_coarse_shift_k274.yaml"
    )
    settings = _validated_settings(config, device_override="cpu")

    assert settings.model_name == "neighbor_trace_inpainter"
    assert settings.hidden_width == 384
    assert settings.relative_receiver_x_radius == 2
    assert settings.source_x_line_radius == 0
    assert settings.source_y_half_shot_radius == 4
    assert settings.relative_receiver_y_radius == 5
    assert settings.neighbor_gating == "target_coordinate_masked_softmax"
    assert settings.neighbor_alignment_kernel_size == 31
    assert settings.prediction_reference == "none"
    assert settings.coarse_shift_samples_per_relative_receiver_y_index == 3
    assert settings.total_steps == 2500
    assert settings.evaluation_interval_steps == 2500
    assert settings.training_audit_count == 10000


def test_study_019_inputs_lock_split_and_canonical_counts() -> None:
    inputs = yaml.safe_load((STUDY_DIRECTORY / "inputs.yaml").read_text(encoding="utf-8"))
    (dataset,) = inputs["datasets"]

    assert dataset["subset"]["split"] == {
        "scope": "per_ffid",
        "random_seed": 42,
        "train_fraction": 0.25,
        "validation_fraction": 0.1875,
        "test_fraction": 0.5625,
        "prepared_counts": {
            "train": 575870,
            "validation": 431890,
            "test": 1295720,
        },
    }
    canonical = dataset["subset"]["physical_coordinate_canonicalization"]
    assert canonical["removed_split_counts"] == {
        "train": 6,
        "validation": 3,
        "test": 6,
    }
    assert canonical["effective_split_counts"] == {
        "train": 575864,
        "validation": 431887,
        "test": 1295714,
    }
    for key in ("manifest", "interim", "processed"):
        path = Path(dataset[key])
        assert not path.is_absolute()
        assert (STUDY_DIRECTORY / path).resolve().is_relative_to(REPOSITORY_ROOT)
