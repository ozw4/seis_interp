from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from seis_interp.configuration import REPOSITORY_ROOT, ConfigurationError, load_resolved_config
from seis_interp.pipelines.train_neighbor_inpainter import _validated_settings
from seis_interp.pipelines.train_shot_gather_inpainter import (
    _validated_settings as _validated_shot_gather_settings,
)

STUDY_DIRECTORY = (
    REPOSITORY_ROOT / "studies" / "study_020_all_ffid_25pct_whole_ffid_neighbor_inpainter"
)


def test_study_020_locks_the_25pct_whole_ffid_split_contract() -> None:
    config = load_resolved_config(STUDY_DIRECTORY / "config.yaml")
    settings = _validated_settings(config, device_override="cpu")

    assert config["study"]["status"] == "running"
    assert config["sampling"] == {
        "random_ffid_holdout_fraction": 0.75,
        "validation_fraction_of_holdout": 0.25,
        "split_scope": "whole_ffid",
        "trace_amplitude_filter": {
            "exclude_all_zero": True,
            "max_abs_amplitude": 10000.0,
        },
        "duplicate_physical_coordinate_policy": "keep_lowest_array_row",
    }
    assert settings.success_threshold_db == 25.0
    assert settings.required_eligible_ffid_count == 4780
    assert dict(settings.required_ffid_split_counts or {}) == {
        "train": 1195,
        "validation": 896,
        "test": 2689,
    }
    assert dict(settings.required_effective_split_counts) == {
        "train": 578685,
        "validation": 437087,
        "test": 1287693,
    }
    assert settings.required_fully_excluded_ffids == (1746,)
    assert settings.exclude_target_ffid_neighbors is True
    assert settings.total_steps == 2500


def test_stage01_locks_the_matched_2500_step_baseline() -> None:
    config = load_resolved_config(
        STUDY_DIRECTORY / "variants" / "stage01_study018_formal_k274_2500_steps.yaml"
    )
    settings = _validated_settings(config, device_override="cpu")

    assert settings.total_steps == 2500
    assert settings.evaluation_interval_steps == 2500
    assert settings.training_audit_count == 10000
    assert settings.hidden_width == 384
    assert settings.relative_receiver_x_radius == 2
    assert settings.source_x_line_radius == 0
    assert settings.source_y_half_shot_radius == 4
    assert settings.relative_receiver_y_radius == 5


def test_stage02_changes_only_crossline_source_support() -> None:
    config = load_resolved_config(STUDY_DIRECTORY / "variants" / "stage02_crossline_k714.yaml")
    settings = _validated_settings(config, device_override="cpu")

    assert settings.total_steps == 2500
    assert settings.hidden_width == 384
    assert settings.relative_receiver_x_radius == 2
    assert settings.source_x_line_radius == 1
    assert settings.source_y_half_shot_radius == 4
    assert settings.relative_receiver_y_radius == 5
    assert settings.evaluation_interval_steps == 2500


def test_stage03_changes_only_source_y_radius_and_validation_batch() -> None:
    config = load_resolved_config(STUDY_DIRECTORY / "variants" / "stage03_crossline_k1374.yaml")
    settings = _validated_settings(config, device_override="cpu")

    assert settings.total_steps == 2500
    assert settings.hidden_width == 384
    assert settings.relative_receiver_x_radius == 2
    assert settings.source_x_line_radius == 1
    assert settings.source_y_half_shot_radius == 8
    assert settings.relative_receiver_y_radius == 5
    assert settings.validation_batch_size == 512


def test_stage04_changes_only_the_prediction_reference_from_stage01() -> None:
    config = load_resolved_config(
        STUDY_DIRECTORY / "variants" / "stage04_k274_source_bracketing_residual.yaml"
    )
    settings = _validated_settings(config, device_override="cpu")

    assert settings.total_steps == 2500
    assert settings.hidden_width == 384
    assert settings.relative_receiver_x_radius == 2
    assert settings.source_x_line_radius == 0
    assert settings.source_y_half_shot_radius == 4
    assert settings.relative_receiver_y_radius == 5
    assert settings.prediction_reference == ("same_line_exact_receiver_linear_bracketing")


def test_stage05_combines_only_the_promoted_stage03_and_stage04_changes() -> None:
    config = load_resolved_config(
        STUDY_DIRECTORY / "variants" / "stage05_crossline_k1374_source_bracketing_residual.yaml"
    )
    settings = _validated_settings(config, device_override="cpu")

    assert settings.total_steps == 2500
    assert settings.hidden_width == 384
    assert settings.relative_receiver_x_radius == 2
    assert settings.source_x_line_radius == 1
    assert settings.source_y_half_shot_radius == 8
    assert settings.relative_receiver_y_radius == 5
    assert settings.validation_batch_size == 512
    assert settings.prediction_reference == ("same_line_exact_receiver_linear_bracketing")


def test_stage06_changes_only_target_sampling_from_stage03() -> None:
    config = load_resolved_config(
        STUDY_DIRECTORY / "variants" / "stage06_epoch_sampling_k1374.yaml"
    )
    settings = _validated_settings(config, device_override="cpu")

    assert settings.total_steps == 2500
    assert settings.batch_size == 96
    assert settings.target_sampling == "epoch_without_replacement"
    assert settings.source_x_line_radius == 1
    assert settings.source_y_half_shot_radius == 8
    assert settings.validation_batch_size == 512


def test_stage07_extends_stage06_to_one_complete_train_sweep() -> None:
    config = load_resolved_config(
        STUDY_DIRECTORY / "variants" / "stage07_full_train_sweep_k1374.yaml"
    )
    settings = _validated_settings(config, device_override="cpu")

    assert settings.total_steps == 6030
    assert settings.batch_size == 96
    assert settings.total_steps * settings.batch_size >= 578685
    assert settings.target_sampling == "epoch_without_replacement"
    assert settings.evaluation_interval_steps == 3015
    assert settings.source_x_line_radius == 1
    assert settings.source_y_half_shot_radius == 8


def test_stage08_changes_only_the_stage03_bracketing_representation() -> None:
    config = load_resolved_config(
        STUDY_DIRECTORY / "variants" / "stage08_crossline_k1374_bracketing_channels.yaml"
    )
    settings = _validated_settings(config, device_override="cpu")

    assert settings.total_steps == 2500
    assert settings.batch_size == 96
    assert settings.target_sampling == "with_replacement"
    assert settings.prediction_reference == ("same_line_exact_receiver_linear_bracketing_channels")
    assert settings.source_x_line_radius == 1
    assert settings.source_y_half_shot_radius == 8
    assert settings.validation_batch_size == 512


def test_stage09_is_an_explicit_memory_bounded_joint_shot_gather_condition() -> None:
    config = load_resolved_config(
        STUDY_DIRECTORY / "variants" / "stage09_joint_shot_gather_k8.yaml"
    )
    settings = _validated_shot_gather_settings(config, device_override="cpu")

    assert config["model"] == {
        "name": "shot_gather_inpainter",
        "hidden_width": 32,
        "target_coordinates": ["source_x_m", "source_y_m"],
        "target_coordinate_scaling": "train_minmax",
        "stem_kernel_size": 7,
        "residual_kernel_size": 3,
        "temporal_dilations": [1, 2, 4, 8, 4, 2, 1],
        "distance_epsilon": 1.0e-6,
        "neighborhood": {
            "type": "nearest_train_source_gathers",
            "distance": "euclidean_source_xy_m",
            "source_gather_count": 8,
        },
    }
    assert settings.hidden_width == 32
    assert settings.source_gather_count == 8
    assert settings.batch_size == 1
    assert settings.validation_batch_size == 4
    assert settings.total_steps == 2500
    assert settings.target_sampling == "epoch_without_replacement"
    assert settings.required_eligible_ffid_count == 4780
    assert dict(settings.required_ffid_split_counts) == {
        "train": 1195,
        "validation": 896,
        "test": 2689,
    }
    assert dict(settings.required_effective_split_counts) == {
        "train": 578685,
        "validation": 437087,
        "test": 1287693,
    }


def test_study_020_inputs_lock_whole_ffid_and_trace_counts() -> None:
    inputs = yaml.safe_load((STUDY_DIRECTORY / "inputs.yaml").read_text(encoding="utf-8"))
    (dataset,) = inputs["datasets"]
    split = dataset["subset"]["split"]
    canonical = dataset["subset"]["physical_coordinate_canonicalization"]

    assert split["scope"] == "whole_ffid"
    assert split["prepared_ffid_counts"] == {
        "train": 1195,
        "validation": 896,
        "test": 2689,
    }
    assert split["prepared_trace_counts"] == {
        "train": 578688,
        "validation": 437088,
        "test": 1287704,
    }
    assert canonical["removed_split_counts"] == {
        "train": 3,
        "validation": 1,
        "test": 11,
    }
    assert canonical["effective_split_counts"] == {
        "train": 578685,
        "validation": 437087,
        "test": 1287693,
    }
    for key in ("manifest", "interim", "processed"):
        path = Path(dataset[key])
        assert not path.is_absolute()
        assert (STUDY_DIRECTORY / path).resolve().is_relative_to(REPOSITORY_ROOT)


def test_whole_ffid_config_requires_exact_ffid_counts() -> None:
    config = load_resolved_config(STUDY_DIRECTORY / "config.yaml")
    del config["evaluation"]["required_ffid_split_counts"]

    with pytest.raises(ConfigurationError, match="required_ffid_split_counts"):
        _validated_settings(config, device_override="cpu")


def test_whole_ffid_config_requires_target_ffid_mask() -> None:
    config = load_resolved_config(STUDY_DIRECTORY / "config.yaml")
    config["training"]["exclude_target_ffid_neighbors"] = False

    with pytest.raises(ConfigurationError, match="exclude_target_ffid_neighbors"):
        _validated_settings(config, device_override="cpu")
