"""Save and restore SIREN checkpoints with target-scaling metadata."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path

import torch

from seis_interp.models.siren import Siren
from seis_interp.processing.normalization import NormalizationParameters
from seis_interp.processing.training_coordinates import ModelCoordinateParameters
from seis_interp.training.amplitude_scaling import (
    TRAIN_GLOBAL_RMS_SCALING,
    validated_amplitude_scaling,
    validation_metric_domain_for_scaling,
)

FIXED_STEP_FINAL_CHECKPOINT_ROLE = "fixed_step_final"
VOLUME_SIREN_METHOD_VARIANT = "per_volume_internal_learning_fixed_steps"


@dataclass(frozen=True)
class LoadedFixedStepSirenCheckpoint:
    """The final observed-only fit and the transforms needed for prediction."""

    model: Siren
    normalization: NormalizationParameters
    model_coordinates: ModelCoordinateParameters
    global_step: int
    final_batch_loss: float


@dataclass(frozen=True)
class LoadedSirenCheckpoint:
    """A restored SIREN and its training-time coordinate and target metadata.

    A per-trace-RMS model still needs an externally supplied scale to recover
    physical amplitudes at a query trace.
    """

    model: Siren
    normalization: NormalizationParameters
    model_coordinates: ModelCoordinateParameters | None
    amplitude_scaling: str
    validation_metric_domain: str
    epoch: int
    global_step: int
    validation_median_trace_snr_db: float | None
    validation_global_snr_db: float

    @property
    def time_coordinate_scale(self) -> float:
        """Return the post-normalization temporal scale needed for inference."""
        if self.model_coordinates is None:
            return 1.0
        return self.model_coordinates.time_coordinate_scale


def save_siren_checkpoint(
    path: Path,
    model: Siren,
    normalization: NormalizationParameters,
    *,
    model_coordinates: ModelCoordinateParameters | None = None,
    amplitude_scaling: str = TRAIN_GLOBAL_RMS_SCALING,
    epoch: int,
    global_step: int,
    validation_median_trace_snr_db: float | None,
    validation_global_snr_db: float,
) -> None:
    """Save constructor values, CPU weights, feature/target scaling, and best metadata.

    ``validation_median_trace_snr_db`` is ``None`` for training modes whose
    model-selection contract defines only a global validation metric. Existing
    checkpoints and the random-point training path continue to store a float.
    ``per_trace_rms`` records the unit-RMS target domain but cannot embed the
    unknown physical scale of an unseen trace.
    """
    checkpoint_path = Path(path)
    if model_coordinates is not None and model_coordinates.input_features != model.input_features:
        raise ValueError(
            "model coordinate width must match model.input_features: "
            f"{model_coordinates.input_features} != {model.input_features}"
        )
    stored_amplitude_scaling = validated_amplitude_scaling(amplitude_scaling)
    validation_metric_domain = validation_metric_domain_for_scaling(stored_amplitude_scaling)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_type": "siren",
        "model_config": _siren_model_config(model),
        "model_state_dict": _cpu_state_dict(model),
        "normalization": normalization.to_dict(),
        "amplitude_scaling": stored_amplitude_scaling,
        "validation_metric_domain": validation_metric_domain,
        "training": {
            "epoch": epoch,
            "global_step": global_step,
            "validation_median_trace_snr_db": validation_median_trace_snr_db,
            "validation_global_snr_db": validation_global_snr_db,
        },
    }
    if model_coordinates is not None:
        payload["model_coordinates"] = model_coordinates.to_dict()
    torch.save(payload, checkpoint_path)


def load_siren_checkpoint(
    path: Path,
    *,
    device: torch.device | str = "cpu",
) -> LoadedSirenCheckpoint:
    """Rebuild a SIREN from its saved constructor values and strict weights."""
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if payload.get("model_type") != "siren":
        raise ValueError("checkpoint model_type must be 'siren'")
    model = Siren(**payload["model_config"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device)
    normalization = NormalizationParameters.from_dict(payload["normalization"])
    raw_model_coordinates = payload.get("model_coordinates")
    model_coordinates = (
        ModelCoordinateParameters.from_dict(raw_model_coordinates)
        if raw_model_coordinates is not None
        else None
    )
    if model_coordinates is not None and model_coordinates.input_features != model.input_features:
        raise ValueError(
            "checkpoint model coordinate width does not match model.input_features: "
            f"{model_coordinates.input_features} != {model.input_features}"
        )
    amplitude_scaling = validated_amplitude_scaling(
        payload.get("amplitude_scaling", TRAIN_GLOBAL_RMS_SCALING)
    )
    expected_validation_metric_domain = validation_metric_domain_for_scaling(amplitude_scaling)
    validation_metric_domain = payload.get(
        "validation_metric_domain",
        expected_validation_metric_domain,
    )
    if validation_metric_domain != expected_validation_metric_domain:
        raise ValueError(
            "checkpoint validation_metric_domain does not match amplitude_scaling: "
            f"{validation_metric_domain!r} != {expected_validation_metric_domain!r}"
        )
    training = payload["training"]
    return LoadedSirenCheckpoint(
        model=model,
        normalization=normalization,
        model_coordinates=model_coordinates,
        amplitude_scaling=amplitude_scaling,
        validation_metric_domain=validation_metric_domain,
        epoch=training["epoch"],
        global_step=training["global_step"],
        validation_median_trace_snr_db=training["validation_median_trace_snr_db"],
        validation_global_snr_db=training["validation_global_snr_db"],
    )


def save_fixed_step_siren_checkpoint(
    path: Path,
    model: Siren,
    normalization: NormalizationParameters,
    model_coordinates: ModelCoordinateParameters,
    *,
    global_step: int,
    final_batch_loss: float,
) -> None:
    """Save the final fit, without validation selection or resumable state.

    The caller owns creation of the artifact directory.
    """
    if model_coordinates.input_features != model.input_features:
        raise ValueError("model coordinate width must match model.input_features")
    step, loss = _fixed_step_training_values(global_step, final_batch_loss)
    payload = {
        "model_type": "siren",
        "checkpoint_role": FIXED_STEP_FINAL_CHECKPOINT_ROLE,
        "method_variant": VOLUME_SIREN_METHOD_VARIANT,
        "model_config": {
            **_siren_model_config(model),
            "layer_omega_schedule": model.layer_omega_schedule,
            "skip_connections": model.skip_connections,
        },
        "model_state_dict": _cpu_state_dict(model),
        "normalization": normalization.to_dict(),
        "model_coordinates": model_coordinates.to_dict(),
        "amplitude_scaling": TRAIN_GLOBAL_RMS_SCALING,
        "training_domain": "benchmark_observed_samples",
        "training": {"global_step": step, "final_batch_loss": loss},
    }
    torch.save(payload, Path(path))


def load_fixed_step_siren_checkpoint(
    path: Path,
    *,
    device: torch.device | str = "cpu",
) -> LoadedFixedStepSirenCheckpoint:
    """Restore the fixed-final function and reject incomplete model metadata."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    required = {
        "model_type",
        "checkpoint_role",
        "method_variant",
        "model_config",
        "model_state_dict",
        "normalization",
        "model_coordinates",
        "amplitude_scaling",
        "training_domain",
        "training",
    }
    if not isinstance(payload, Mapping) or not required.issubset(payload):
        raise ValueError("fixed-step checkpoint is missing required fields")
    for key, expected in (
        ("model_type", "siren"),
        ("checkpoint_role", FIXED_STEP_FINAL_CHECKPOINT_ROLE),
        ("method_variant", VOLUME_SIREN_METHOD_VARIANT),
        ("amplitude_scaling", TRAIN_GLOBAL_RMS_SCALING),
        ("training_domain", "benchmark_observed_samples"),
    ):
        if payload[key] != expected:
            raise ValueError(f"checkpoint {key} must be {expected!r}")
    model_config = payload["model_config"]
    required_model = {
        "input_features",
        "hidden_width",
        "hidden_layers",
        "output_features",
        "omega_0",
        "hidden_omega",
        "layer_omega_schedule",
        "skip_connections",
    }
    if not isinstance(model_config, Mapping) or not required_model.issubset(model_config):
        raise ValueError("checkpoint model_config is missing required constructor fields")
    try:
        model = Siren(**model_config)
    except TypeError as error:
        raise ValueError("checkpoint model_config contains invalid constructor fields") from error
    model.load_state_dict(payload["model_state_dict"], strict=True)
    normalization = NormalizationParameters.from_dict(payload["normalization"])
    coordinates = ModelCoordinateParameters.from_dict(payload["model_coordinates"])
    if coordinates.input_features != model.input_features:
        raise ValueError("checkpoint model coordinate width must match model.input_features")
    training = payload["training"]
    if not isinstance(training, Mapping) or not {"global_step", "final_batch_loss"}.issubset(
        training
    ):
        raise ValueError("checkpoint training is missing required fields")
    step, loss = _fixed_step_training_values(training["global_step"], training["final_batch_loss"])
    model.to(device)
    return LoadedFixedStepSirenCheckpoint(model, normalization, coordinates, step, loss)


def _fixed_step_training_values(step: object, loss: object) -> tuple[int, float]:
    if isinstance(step, bool) or not isinstance(step, Integral) or step <= 0:
        raise ValueError("global_step must be a positive integer")
    if isinstance(loss, bool) or not isinstance(loss, Real) or not math.isfinite(loss) or loss < 0:
        raise ValueError("final_batch_loss must be non-negative and finite")
    return int(step), float(loss)


def _siren_model_config(model: Siren) -> dict[str, object]:
    config: dict[str, object] = {
        "input_features": model.input_features,
        "hidden_width": model.hidden_width,
        "hidden_layers": model.hidden_layers,
        "output_features": model.output_features,
        "omega_0": model.omega_0,
        "hidden_omega": model.hidden_omega,
    }
    if model.layer_omega_schedule is not None:
        config["layer_omega_schedule"] = model.layer_omega_schedule
    if model.skip_connections is not None:
        config["skip_connections"] = model.skip_connections
    return config


def _cpu_state_dict(model: Siren) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
