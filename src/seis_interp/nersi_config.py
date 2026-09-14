"""Validate the PoC configuration for profile-wise NeRSI runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from seis_interp import config_values
from seis_interp.configuration import ConfigurationError
from seis_interp.processing.trace_time_alignment import validate_time_alignment
from seis_interp.training.c3_volume_nersi_data import PROFILE_COORDINATE_ORDER
from seis_interp.training.nersi_optimization import validate_nersi_optimization
from seis_interp.training.trace_relative_loss import POC_TRACE_LOSSES


@dataclass(frozen=True)
class NersiTrainingSettings:
    """Fixed-step observed-profile training settings."""

    model_initialization_seed: int
    sampling_seed: int
    learning_rate: float
    profiles_per_step: int
    max_steps: int
    report_interval: int
    device: str
    loss: str
    gradient_accumulation_steps: int = 1


@dataclass(frozen=True)
class NersiPocSettings:
    """Validated model, training, and prediction values for one PoC run."""

    model: dict[str, object]
    training: NersiTrainingSettings
    prediction_batch_size: int
    augmentation: dict[str, float | int] | None = None
    trace_rms_idw: dict[str, object] | None = None
    time_alignment: dict[str, object] | None = None
    optimization: dict | None = None
    cartesian_profile_coordinates: bool = False
    nyquist_fractions: tuple[float, ...] | None = None

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

    training_keys = {
        "model_initialization_seed",
        "sampling_seed",
        "optimizer",
        "loss",
        "amplitude_scaling",
        "learning_rate",
        "profiles_per_step",
        "max_steps",
        "report_interval",
        "device",
    }
    if (
        isinstance(config.get("training"), Mapping)
        and "gradient_accumulation_steps" in config["training"]
    ):
        training_keys.add("gradient_accumulation_steps")
    training = config_values.exact_section(config, "training", training_keys)
    _require_literal(training, "optimizer", "adam")
    if training["loss"] not in POC_TRACE_LOSSES:
        raise ConfigurationError(f"training.loss must be one of {POC_TRACE_LOSSES!r}")
    scaling = training["amplitude_scaling"]
    if scaling not in ("observed_volume_global_rms", "observed_trace_rms_idw"):
        raise ConfigurationError(
            "amplitude_scaling must be observed_volume_global_rms or observed_trace_rms_idw"
        )
    idw = None
    if scaling == "observed_trace_rms_idw":
        section = config_values.exact_section(
            config, "trace_rms_idw", {"radius", "power", "axis_scales"}
        )
        scales = section["axis_scales"]
        if not isinstance(scales, list) or len(scales) != 4:
            raise ConfigurationError("trace_rms_idw.axis_scales must contain four values")
        idw = {
            "radius": config_values.positive_integer(section["radius"], "trace_rms_idw.radius"),
            "power": config_values.positive_float(section["power"], "trace_rms_idw.power"),
            "axis_scales": [
                config_values.positive_float(v, "trace_rms_idw.axis_scales") for v in scales
            ],
        }
    elif "trace_rms_idw" in config:
        raise ConfigurationError("trace_rms_idw requires observed_trace_rms_idw amplitude_scaling")
    device = training["device"]
    if not isinstance(device, str) or not device.strip():
        raise ConfigurationError("training.device must be a non-empty string")
    training_settings = NersiTrainingSettings(
        loss=training["loss"],
        model_initialization_seed=config_values.nonnegative_integer(
            training["model_initialization_seed"], "training.model_initialization_seed"
        ),
        sampling_seed=config_values.nonnegative_integer(
            training["sampling_seed"], "training.sampling_seed"
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
        gradient_accumulation_steps=config_values.positive_integer(
            training.get("gradient_accumulation_steps", 1), "training.gradient_accumulation_steps"
        ),
    )

    prediction = config_values.exact_section(config, "prediction", {"batch_size"})
    evaluation = config_values.exact_section(config, "evaluation", {"primary_metric", "domain"})
    if evaluation["domain"] != "evaluation_target" or evaluation["primary_metric"] not in (
        "physical_amplitude_global_snr_db",
        "physical_amplitude_mean_trace_snr_db",
    ):
        raise ConfigurationError("evaluation must use target-only physical-amplitude global S/N")
    augmentation = None
    if "augmentation" in config:
        raw_augmentation = config["augmentation"]
        field = (
            "profile_mixup_max_fraction"
            if isinstance(raw_augmentation, Mapping)
            and "profile_mixup_max_fraction" in raw_augmentation
            else "coordinate_jitter_cells"
        )
        section = config_values.exact_section(config, "augmentation", {field, "random_seed"})
        radius = config_values.positive_float(section[field], f"augmentation.{field}")
        if radius > 0.5:
            raise ConfigurationError(f"augmentation.{field} must not exceed 0.5")
        augmentation = {
            field: radius,
            "random_seed": config_values.nonnegative_integer(
                section["random_seed"], "augmentation.random_seed"
            ),
        }
    alignment = None
    if "time_alignment" in config:
        try:
            alignment = validate_time_alignment(config["time_alignment"])
        except ValueError as error:
            raise ConfigurationError(str(error)) from error
    cartesian = config.get("profile_coordinates", "index")
    if cartesian not in ("index", "cartesian_cmp_half_offset"):
        raise ConfigurationError("profile_coordinates must be index or cartesian_cmp_half_offset")
    fractions = None
    if "fourier_bandlimit" in config:
        band = config_values.exact_section(config, "fourier_bandlimit", {"nyquist_fraction"})
        values = band["nyquist_fraction"]
        if not isinstance(values, list) or len(values) != 3:
            raise ConfigurationError("nyquist_fraction requires three profile-axis values")
        fractions = tuple(config_values.positive_float(v, "nyquist_fraction") for v in values)
        if any(v > 1 for v in fractions):
            raise ConfigurationError("nyquist_fraction must not exceed one")
    if cartesian != "index" and augmentation is not None:
        raise ConfigurationError(
            "Cartesian profile candidates do not support index-space augmentation"
        )
    return NersiPocSettings(
        cartesian_profile_coordinates=cartesian != "index",
        nyquist_fractions=fractions,
        model=model_config,
        training=training_settings,
        augmentation=augmentation,
        trace_rms_idw=idw,
        time_alignment=alignment,
        optimization=(
            validate_nersi_optimization(config["optimization"], training_settings.learning_rate)
            if "optimization" in config
            else None
        ),
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
