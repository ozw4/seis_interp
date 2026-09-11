"""Validate the PoC configuration for profile-wise NeRSI runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from seis_interp import config_values
from seis_interp.configuration import ConfigurationError
from seis_interp.training.c3_volume_nersi_data import PROFILE_COORDINATE_ORDER


@dataclass(frozen=True)
class NersiTrainingSettings:
    """Fixed-step observed-profile training settings."""

    random_seed: int
    learning_rate: float
    profiles_per_step: int
    max_steps: int
    report_interval: int
    device: str


@dataclass(frozen=True)
class NersiPocSettings:
    """Validated model, training, and prediction values for one PoC run."""

    model: dict[str, object]
    training: NersiTrainingSettings
    prediction_batch_size: int

    def model_constructor_config(self, profile_shape: tuple[int, int]) -> dict[str, object]:
        """Return complete constructor metadata after the input profile shape is known."""
        shape = validate_nersi_profile_shape(profile_shape)
        return {**self.model, "profile_shape": shape}


def validate_nersi_poc_config(config: Mapping[str, object]) -> NersiPocSettings:
    """Validate the strict NeRSI PoC method sections."""
    model = config_values.exact_section(
        config,
        "model",
        {
            "name",
            "coordinate_order",
            "fourier_components",
            "frequency_schedule",
            "frequency_base",
            "encoder_width",
            "latent_channels",
            "decoder_channels",
            "upsample_scales",
            "kernel_size",
            "activation",
            "output_activation",
        },
    )
    config_values.require_exact(config, "model.name", "nersi")
    _require_literal(model, "frequency_schedule", "exponential")
    _require_literal(model, "activation", "gelu")
    _require_literal(model, "output_activation", "linear")
    coordinate_order = model["coordinate_order"]
    if (
        not isinstance(coordinate_order, list)
        or tuple(coordinate_order) != PROFILE_COORDINATE_ORDER
    ):
        raise ConfigurationError(
            f"model.coordinate_order must be exactly {list(PROFILE_COORDINATE_ORDER)!r}"
        )
    frequency_base = config_values.positive_float(model["frequency_base"], "model.frequency_base")
    if frequency_base <= 1.0:
        raise ConfigurationError("model.frequency_base must be greater than 1")
    decoder_channels = config_values.validated_positive_integer_list(
        model["decoder_channels"], "model.decoder_channels"
    )
    if len(decoder_channels) != 3:
        raise ConfigurationError("model.decoder_channels must contain exactly three values")
    upsample_scales = config_values.validated_positive_integer_list(
        model["upsample_scales"], "model.upsample_scales"
    )
    if upsample_scales != (2, 2, 2):
        raise ConfigurationError("model.upsample_scales must be exactly [2, 2, 2]")
    model_config: dict[str, object] = {
        "input_features": 3,
        "fourier_components": config_values.positive_integer(
            model["fourier_components"], "model.fourier_components"
        ),
        "frequency_base": frequency_base,
        "encoder_width": config_values.positive_integer(
            model["encoder_width"], "model.encoder_width"
        ),
        "latent_channels": config_values.positive_integer(
            model["latent_channels"], "model.latent_channels"
        ),
        "decoder_channels": decoder_channels,
        "upsample_scales": upsample_scales,
        "kernel_size": config_values.odd_positive_integer(
            model["kernel_size"], "model.kernel_size"
        ),
        "activation": "gelu",
        "output_activation": "linear",
    }

    training = config_values.exact_section(
        config,
        "training",
        {
            "random_seed",
            "optimizer",
            "loss",
            "amplitude_scaling",
            "learning_rate",
            "profiles_per_step",
            "max_steps",
            "report_interval",
            "device",
        },
    )
    _require_literal(training, "optimizer", "adam")
    _require_literal(training, "loss", "masked_trace_relative_mse")
    _require_literal(training, "amplitude_scaling", "observed_volume_global_rms")
    device = training["device"]
    if not isinstance(device, str) or not device.strip():
        raise ConfigurationError("training.device must be a non-empty string")
    training_settings = NersiTrainingSettings(
        random_seed=config_values.nonnegative_integer(
            training["random_seed"], "training.random_seed"
        ),
        learning_rate=config_values.positive_float(
            training["learning_rate"], "training.learning_rate"
        ),
        profiles_per_step=config_values.positive_integer(
            training["profiles_per_step"], "training.profiles_per_step"
        ),
        max_steps=config_values.positive_integer(training["max_steps"], "training.max_steps"),
        report_interval=config_values.positive_integer(
            training["report_interval"], "training.report_interval"
        ),
        device=device,
    )

    prediction = config_values.exact_section(config, "prediction", {"batch_size"})
    evaluation = config_values.exact_section(config, "evaluation", {"primary_metric", "domain"})
    if evaluation != {
        "primary_metric": "physical_amplitude_global_snr_db",
        "domain": "evaluation_target",
    }:
        raise ConfigurationError("evaluation must use target-only physical-amplitude global S/N")
    return NersiPocSettings(
        model=model_config,
        training=training_settings,
        prediction_batch_size=config_values.positive_integer(
            prediction["batch_size"], "prediction.batch_size"
        ),
    )


def validate_nersi_profile_shape(value: object) -> tuple[int, int]:
    """Require a positive two-axis profile exactly compatible with three 2x blocks."""
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value)
    ):
        raise ConfigurationError("NeRSI profile shape must contain two positive integers")
    shape = tuple(int(item) for item in value)
    if any(item % 8 for item in shape):
        raise ConfigurationError("NeRSI profile time and receiver-y axes must be divisible by 8")
    return shape  # type: ignore[return-value]


def _require_literal(section: Mapping[str, object], key: str, expected: object) -> None:
    if section[key] != expected:
        raise ConfigurationError(f"{key} must be {expected!r}, got {section[key]!r}")
