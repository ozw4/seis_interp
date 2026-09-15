"""Validate fixed observed-only CCNet-5D PoC settings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from seis_interp import config_values
from seis_interp.configuration import ConfigurationError
from seis_interp.processing.ccnet5d_tiles import validate_ccnet5d_shape
from seis_interp.training.ccnet5d_ema import validate_ccnet5d_ema_decay
from seis_interp.training.trace_relative_loss import POC_TRACE_LOSSES


@dataclass(frozen=True)
class CCNet5DPocTrainingSettings:
    """Fixed-step optimizer and device settings."""

    model_initialization_seed: int
    learning_rate: float
    max_steps: int
    report_interval: int
    device: str
    loss: str
    ema_decay: float | None = None
    optimizer: str = "adam"
    weight_decay: float = 0.0
    supervised_traces_per_update: int | None = None


@dataclass(frozen=True)
class CCNet5DPocPatchSettings:
    """Random observed patch and pseudo-mask settings."""

    shape: tuple[int, ...]
    inner_mask_fraction: float
    placement_seed: int
    inner_mask_seed: int


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
    _require_literal(model, "output_activation", "linear")
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
        "output_activation": model["output_activation"],
    }

    patches = config_values.exact_section(
        config,
        "patches",
        {"shape", "inner_mask_fraction", "mask_kind", "placement_seed", "inner_mask_seed"},
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
        placement_seed=config_values.nonnegative_integer(
            patches["placement_seed"], "patches.placement_seed"
        ),
        inner_mask_seed=config_values.nonnegative_integer(
            patches["inner_mask_seed"], "patches.inner_mask_seed"
        ),
    )

    training = config_values.exact_section(
        config,
        "training",
        {
            "model_initialization_seed",
            "optimizer",
            "loss",
            "amplitude_scaling",
            "learning_rate",
            "max_steps",
            "report_interval",
            "device",
        }
        | (
            {"ema_decay", "weight_decay", "supervised_traces_per_update"}.intersection(
                config["training"]
            )
            if isinstance(config.get("training"), Mapping)
            else set()
        ),
    )
    if training["optimizer"] not in ("adam", "adamw"):
        raise ConfigurationError("optimizer must be adam or adamw")
    if training["loss"] not in POC_TRACE_LOSSES:
        raise ConfigurationError(f"training.loss must be one of {POC_TRACE_LOSSES!r}")
    _require_literal(training, "amplitude_scaling", "observed_volume_global_rms")
    device = training["device"]
    if not isinstance(device, str) or not device.strip():
        raise ConfigurationError("training.device must be a non-empty string")
    try:
        ema_decay = validate_ccnet5d_ema_decay(training.get("ema_decay"))
    except ValueError as error:
        raise ConfigurationError(str(error)) from error
    training_settings = CCNet5DPocTrainingSettings(
        optimizer=training["optimizer"],
        weight_decay=config_values.nonnegative_float(
            training.get("weight_decay", 0.0), "training.weight_decay"
        ),
        supervised_traces_per_update=(
            config_values.positive_integer(
                training["supervised_traces_per_update"], "training.supervised_traces_per_update"
            )
            if "supervised_traces_per_update" in training
            else None
        ),
        ema_decay=ema_decay,
        loss=training["loss"],
        model_initialization_seed=config_values.nonnegative_integer(
            training["model_initialization_seed"], "training.model_initialization_seed"
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
    if evaluation["domain"] != "evaluation_target" or evaluation["primary_metric"] not in (
        "physical_amplitude_global_snr_db",
        "physical_amplitude_mean_trace_snr_db",
    ):
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
