"""Validate fixed observed-only CCNet-5D PoC settings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from seis_interp import config_values
from seis_interp.configuration import ConfigurationError
from seis_interp.processing.ccnet5d_tiles import validate_ccnet5d_shape
from seis_interp.training.ccnet5d_poc_checkpoints import CCNET5D_POC_LOSS


@dataclass(frozen=True)
class CCNet5DPocTrainingSettings:
    """Fixed-step optimizer and device settings."""

    random_seed: int
    learning_rate: float
    max_steps: int
    report_interval: int
    device: str


@dataclass(frozen=True)
class CCNet5DPocPatchSettings:
    """Random observed patch and pseudo-mask settings."""

    shape: tuple[int, ...]
    inner_mask_fraction: float
    random_seed: int


@dataclass(frozen=True)
class CCNet5DPocSettings:
    """All model-specific values needed by one end-to-end PoC run."""

    model: dict[str, object]
    patches: CCNet5DPocPatchSettings
    training: CCNet5DPocTrainingSettings
    prediction_core_shape: tuple[int, ...]


def validate_ccnet5d_poc_config(config: Mapping[str, object]) -> CCNet5DPocSettings:
    """Require the single current observed-only CCNet-5D configuration."""
    model = config_values.exact_section(
        config,
        "model",
        {
            "name",
            "hidden_channels",
            "intermediate_channels",
            "kernel_size",
            "output_activation",
        },
    )
    config_values.require_exact(config, "model.name", "ccnet5d")
    output_activation = model["output_activation"]
    if not isinstance(output_activation, str) or output_activation not in {"linear", "relu"}:
        raise ConfigurationError("model.output_activation must be 'linear' or 'relu'")
    model_config: dict[str, object] = {
        "hidden_channels": config_values.positive_integer(
            model["hidden_channels"], "model.hidden_channels"
        ),
        "intermediate_channels": config_values.positive_integer(
            model["intermediate_channels"], "model.intermediate_channels"
        ),
        "kernel_size": config_values.odd_positive_integer(
            model["kernel_size"], "model.kernel_size"
        ),
        "output_activation": output_activation,
    }

    patches = config_values.exact_section(
        config,
        "patches",
        {"shape", "inner_mask_fraction", "mask_kind", "random_seed"},
    )
    if patches["mask_kind"] != "random_trace":
        raise ConfigurationError("patches.mask_kind must be 'random_trace'")
    fraction = config_values.positive_float(
        patches["inner_mask_fraction"], "patches.inner_mask_fraction"
    )
    if fraction >= 1.0:
        raise ConfigurationError("patches.inner_mask_fraction must be strictly less than 1")
    try:
        patch_shape = validate_ccnet5d_shape(patches["shape"], "patches.shape")
    except ValueError as error:
        raise ConfigurationError(str(error)) from error
    patch_settings = CCNet5DPocPatchSettings(
        shape=patch_shape,
        inner_mask_fraction=fraction,
        random_seed=config_values.nonnegative_integer(
            patches["random_seed"], "patches.random_seed"
        ),
    )

    training = config_values.exact_section(
        config,
        "training",
        {
            "random_seed",
            "optimizer",
            "loss",
            "amplitude_scaling",
            "learning_rate",
            "max_steps",
            "report_interval",
            "device",
        },
    )
    _require_literal(training, "optimizer", "adam")
    _require_literal(training, "loss", CCNET5D_POC_LOSS)
    _require_literal(training, "amplitude_scaling", "observed_volume_global_rms")
    device = training["device"]
    if not isinstance(device, str) or not device.strip():
        raise ConfigurationError("training.device must be a non-empty string")
    training_settings = CCNet5DPocTrainingSettings(
        random_seed=config_values.nonnegative_integer(
            training["random_seed"], "training.random_seed"
        ),
        learning_rate=config_values.positive_float(
            training["learning_rate"], "training.learning_rate"
        ),
        max_steps=config_values.positive_integer(training["max_steps"], "training.max_steps"),
        report_interval=config_values.positive_integer(
            training["report_interval"], "training.report_interval"
        ),
        device=device,
    )

    prediction = config_values.exact_section(config, "prediction", {"core_shape"})
    try:
        core_shape = validate_ccnet5d_shape(prediction["core_shape"], "prediction.core_shape")
    except ValueError as error:
        raise ConfigurationError(str(error)) from error
    evaluation = config_values.exact_section(config, "evaluation", {"primary_metric", "domain"})
    if evaluation != {
        "primary_metric": "physical_amplitude_global_snr_db",
        "domain": "evaluation_target",
    }:
        raise ConfigurationError("evaluation must use target-only physical-amplitude global S/N")
    return CCNet5DPocSettings(
        model=model_config,
        patches=patch_settings,
        training=training_settings,
        prediction_core_shape=core_shape,
    )


def _require_literal(section: Mapping[str, object], key: str, expected: object) -> None:
    if section[key] != expected:
        raise ConfigurationError(f"{key} must be {expected!r}, got {section[key]!r}")
