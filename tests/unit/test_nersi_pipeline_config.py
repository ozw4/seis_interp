from __future__ import annotations

from copy import deepcopy

import pytest

from seis_interp.configuration import ConfigurationError
from seis_interp.nersi_config import validate_nersi_poc_config


@pytest.mark.parametrize("optimizer", ["adam", "adamw"])
def test_optimizer_choice(optimizer):
    config = _config()
    config["training"]["optimizer"] = optimizer
    assert validate_nersi_poc_config(config).training.optimizer == optimizer


def test_weight_decay_is_optional_and_requires_adamw():
    config = _config()
    assert validate_nersi_poc_config(config).training.weight_decay == 0.0
    config["training"].update(optimizer="adamw", weight_decay=0.0001)
    assert validate_nersi_poc_config(config).training.weight_decay == 0.0001
    config["training"]["optimizer"] = "adam"
    with pytest.raises(ConfigurationError, match="requires AdamW"):
        validate_nersi_poc_config(config)


def _config() -> dict[str, object]:
    return {
        "project": {"random_seed": 142},
        "model": {
            "name": "nersi",
            "coordinate_order": [
                "source_line",
                "shot_in_line",
                "relative_receiver_x",
            ],
            "fourier_components": 40,
            "frequency_schedule": "exponential",
            "frequency_base": 1.25,
            "encoder_width": 16,
            "latent_channels": 8,
            "decoder_channels": [8, 4, 2],
            "upsample_scales": [2, 2, 2],
            "kernel_size": 3,
            "activation": "gelu",
            "output_activation": "linear",
        },
        "training": {
            "model_initialization_seed": 20260908,
            "sampling_seed": 201,
            "optimizer": "adam",
            "loss": "masked_trace_relative_mse",
            "amplitude_scaling": "observed_volume_global_rms",
            "learning_rate": 1e-3,
            "profiles_per_step": 4,
            "max_steps": 10,
            "report_interval": 5,
            "device": "cpu",
        },
        "prediction": {"batch_size": 8},
        "evaluation": {
            "primary_metric": "physical_amplitude_global_snr_db",
            "domain": "evaluation_target",
        },
    }


def test_gradient_accumulation_is_optional_and_strict():
    config = _config()
    assert validate_nersi_poc_config(config).training.gradient_accumulation_steps == 1
    config["training"]["gradient_accumulation_steps"] = 8
    assert validate_nersi_poc_config(config).training.gradient_accumulation_steps == 8
    for value in [0, -1, True, 1.5]:
        config["training"]["gradient_accumulation_steps"] = value
        with pytest.raises(ConfigurationError, match="gradient_accumulation_steps"):
            validate_nersi_poc_config(config)


def test_trace_rms_idw_requires_explicit_scaling_and_valid_parameters():
    config = _config()
    assert validate_nersi_poc_config(config).trace_rms_idw is None
    config["trace_rms_idw"] = {"radius": 2, "power": 2, "axis_scales": [1] * 4}
    with pytest.raises(ConfigurationError, match="requires"):
        validate_nersi_poc_config(config)
    config["training"]["amplitude_scaling"] = "observed_trace_rms_idw"
    assert validate_nersi_poc_config(config).trace_rms_idw == config["trace_rms_idw"]
    for key, value in (("radius", 0), ("power", float("inf")), ("axis_scales", [1, 0, 1, 1])):
        invalid = deepcopy(config)
        invalid["trace_rms_idw"][key] = value
        with pytest.raises(ConfigurationError):
            validate_nersi_poc_config(invalid)


def test_valid_config_preserves_separate_mask_and_training_seeds() -> None:
    config = _config()
    settings = validate_nersi_poc_config(config)

    assert config["project"]["random_seed"] == 142
    assert settings.training.model_initialization_seed == 20260908
    assert settings.model_constructor_config((384, 32)) == {
        "input_features": 3,
        "fourier_components": 40,
        "frequency_base": 1.25,
        "encoder_width": 16,
        "latent_channels": 8,
        "decoder_channels": (8, 4, 2),
        "profile_shape": (384, 32),
        "upsample_scales": (2, 2, 2),
        "kernel_size": 3,
        "activation": "gelu",
        "output_activation": "linear",
    }
    assert settings.prediction_batch_size == 8


def test_shot_profile_axis_requires_matching_coordinates_and_no_receiver_y_options() -> None:
    config = _config()
    config["profile_axis"] = "shot_in_line"
    config["model"]["coordinate_order"] = [
        "source_line",
        "relative_receiver_x",
        "relative_receiver_y",
    ]

    settings = validate_nersi_poc_config(config)

    assert settings.profile_axis == "shot_in_line"
    for section, value in (
        (
            "time_alignment",
            {"receiver_y_shift_samples_per_cell": 3, "boundary": "circular"},
        ),
        ("augmentation", {"coordinate_jitter_cells": 0.1, "random_seed": 501}),
        ("fourier_bandlimit", {"nyquist_fraction": [1.0, 1.0, 1.0]}),
    ):
        invalid = deepcopy(config)
        invalid[section] = value
        with pytest.raises(ConfigurationError, match="profiles"):
            validate_nersi_poc_config(invalid)


def test_optional_depthwise_separable_decoder_is_preserved() -> None:
    config = _config()
    config["model"]["decoder_convolution"] = "depthwise_separable"

    settings = validate_nersi_poc_config(config)

    assert settings.model["decoder_convolution"] == "depthwise_separable"
    config["model"]["decoder_convolution"] = "grouped"
    with pytest.raises(ConfigurationError, match="decoder_convolution"):
        validate_nersi_poc_config(config)


def test_axis_specific_frequency_bases_are_preserved() -> None:
    config = _config()
    config["model"]["frequency_base"] = [1.07, 1.09, 1.05]

    settings = validate_nersi_poc_config(config)

    assert settings.model["frequency_base"] == (1.07, 1.09, 1.05)
    for value in ([1.1, 1.2], [1.1, 1.0, 1.2], [1.1, True, 1.2]):
        invalid = deepcopy(config)
        invalid["model"]["frequency_base"] = value
        with pytest.raises(ConfigurationError, match="frequency_base"):
            validate_nersi_poc_config(invalid)


def test_profile_embedding_derives_fixed_grid_from_resolved_selection() -> None:
    config = _config()
    config["benchmark_volume"] = {
        "selection": {
            "time": [0, 384],
            "source_line": [25, 41],
            "shot_in_line": [18, 50],
            "relative_receiver_x": [0, 8],
            "relative_receiver_y": [18, 50],
        }
    }
    config["model"]["profile_embedding_channels"] = 32

    settings = validate_nersi_poc_config(config)

    assert settings.model["profile_embedding_channels"] == 32
    assert settings.model["profile_grid_shape"] == (16, 32, 8)


def test_profile_embedding_rejects_missing_selection_and_coordinate_augmentation() -> None:
    config = _config()
    config["model"]["profile_embedding_channels"] = 32
    with pytest.raises(ConfigurationError, match="selection"):
        validate_nersi_poc_config(config)

    config["benchmark_volume"] = {
        "selection": {
            "source_line": [25, 41],
            "shot_in_line": [18, 50],
            "relative_receiver_x": [0, 8],
        }
    }
    config["augmentation"] = {"coordinate_jitter_cells": 0.1, "random_seed": 501}
    with pytest.raises(ConfigurationError, match="without coordinate augmentation"):
        validate_nersi_poc_config(config)


def test_optional_rank_two_spatial_latent_is_preserved() -> None:
    config = _config()
    config["model"]["latent_spatial_rank"] = 2

    settings = validate_nersi_poc_config(config)

    assert settings.model["latent_spatial_rank"] == 2
    assert settings.model_constructor_config((384, 32))["latent_spatial_rank"] == 2
    config["model"]["latent_spatial_rank"] = 0
    with pytest.raises(ConfigurationError, match="latent_spatial_rank"):
        validate_nersi_poc_config(config)


def test_optional_temporal_basis_components_are_preserved() -> None:
    config = _config()
    config["model"]["temporal_basis_components"] = 128

    settings = validate_nersi_poc_config(config)

    assert settings.model["temporal_basis_components"] == 128
    assert settings.model_constructor_config((384, 32))["temporal_basis_components"] == 128
    config["model"]["temporal_basis_components"] = 0
    with pytest.raises(ConfigurationError, match="temporal_basis_components"):
        validate_nersi_poc_config(config)


@pytest.mark.parametrize("profile_axis", ["relative_receiver_x", "unknown", 1])
def test_unsupported_profile_axis_is_rejected(profile_axis) -> None:
    config = _config()
    config["profile_axis"] = profile_axis
    with pytest.raises(ConfigurationError, match="profile_axis"):
        validate_nersi_poc_config(config)


@pytest.mark.parametrize("section", ["model", "training", "prediction", "evaluation"])
def test_method_sections_reject_unexpected_keys(section: str) -> None:
    config = _config()
    config[section]["unexpected"] = True

    with pytest.raises(ConfigurationError, match="exactly"):
        validate_nersi_poc_config(config)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("model", "name", "siren"),
        ("model", "coordinate_order", ["shot_in_line", "source_line", "relative_receiver_x"]),
        ("model", "frequency_schedule", "linear"),
        ("model", "frequency_base", 1.0),
        ("model", "decoder_channels", [8, 4]),
        ("model", "upsample_scales", [2, 2, 1]),
        ("model", "kernel_size", 2),
        ("model", "activation", "relu"),
        ("model", "output_activation", "relu"),
        ("training", "optimizer", "sgd"),
        ("training", "model_initialization_seed", True),
        ("training", "sampling_seed", -1),
        ("training", "loss", "l2"),
        ("training", "loss", "observed_masked_mse"),
        ("training", "amplitude_scaling", "per_trace_rms"),
        ("training", "profiles_per_step", 0),
        ("prediction", "batch_size", 0),
        ("evaluation", "domain", "observed"),
    ],
)
def test_invalid_literal_shape_and_positive_values_are_rejected(
    section: str, key: str, value: object
) -> None:
    config = deepcopy(_config())
    config[section][key] = value

    with pytest.raises(ConfigurationError):
        validate_nersi_poc_config(config)


@pytest.mark.parametrize("shape", [(7, 8), (8, 9), (0, 8), [8, 8]])
def test_profile_shape_must_fit_three_two_x_upsampling_stages(shape: object) -> None:
    settings = validate_nersi_poc_config(_config())

    with pytest.raises(ConfigurationError, match="profile"):
        settings.model_constructor_config(shape)  # type: ignore[arg-type]


def test_optional_augmentation_is_explicit_and_keeps_its_own_seed():
    config = _config()
    assert validate_nersi_poc_config(config).augmentation is None
    config["augmentation"] = {"coordinate_jitter_cells": 0.1, "random_seed": 501}
    assert validate_nersi_poc_config(config).augmentation == config["augmentation"]
    config["augmentation"] = {"profile_mixup_max_fraction": 0.2, "random_seed": 501}
    assert validate_nersi_poc_config(config).augmentation == config["augmentation"]


@pytest.mark.parametrize(
    "augmentation",
    [
        {},
        {"coordinate_jitter_cells": 0.1},
        {"coordinate_jitter_cells": 0, "random_seed": 501},
        {"coordinate_jitter_cells": 0.51, "random_seed": 501},
        {"coordinate_jitter_cells": float("nan"), "random_seed": 501},
        {"coordinate_jitter_cells": 0.1, "random_seed": -1},
        {"coordinate_jitter_cells": 0.1, "random_seed": True},
        {"coordinate_jitter_cells": 0.1, "random_seed": 501, "flip": True},
        {"profile_mixup_max_fraction": 0.2, "coordinate_jitter_cells": 0.1, "random_seed": 501},
        {"profile_mixup_max_fraction": 0.51, "random_seed": 501},
        {"profile_mixup_max_fraction": 0, "random_seed": 501},
    ],
)
def test_invalid_augmentation_is_rejected(augmentation):
    config = _config()
    config["augmentation"] = augmentation
    with pytest.raises(ConfigurationError):
        validate_nersi_poc_config(config)


def test_time_alignment_requires_explicit_integer_shift_and_boundary():
    config = _config()
    assert validate_nersi_poc_config(config).time_alignment is None
    config["time_alignment"] = {"receiver_y_shift_samples_per_cell": 3, "boundary": "circular"}
    assert validate_nersi_poc_config(config).time_alignment == config["time_alignment"]
    for value in (
        {},
        {"receiver_y_shift_samples_per_cell": 0, "boundary": "circular"},
        {"receiver_y_shift_samples_per_cell": 3, "boundary": "zero"},
    ):
        config["time_alignment"] = value
        with pytest.raises(ConfigurationError):
            validate_nersi_poc_config(config)
