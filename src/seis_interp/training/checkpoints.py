"""Save and restore SIREN checkpoints with target-scaling metadata."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import torch

from seis_interp.models.siren import Siren
from seis_interp.processing.normalization import NormalizationParameters
from seis_interp.processing.training_coordinates import ModelCoordinateParameters
from seis_interp.training.amplitude_scaling import (
    PER_TRACE_RMS_SCALING,
    TRAIN_GLOBAL_RMS_SCALING,
    validated_amplitude_scaling,
    validation_metric_domain_for_scaling,
)

FIXED_STEP_FINAL_CHECKPOINT_ROLE = "fixed_step_final"
VOLUME_SIREN_METHOD_VARIANT = "per_volume_internal_learning_fixed_steps"


@dataclass(frozen=True)
class LoadedFixedStepSirenCheckpoint:
    """The final observed-only fit and the transforms needed for prediction.

    Per-trace physical restoration requires the stored scales aligned by
    ``trace_array_rows``. In that mode, ``normalization.amplitude_rms`` is only
    the observed-volume diagnostic and must not restore physical amplitudes.
    """

    model: Siren
    normalization: NormalizationParameters
    model_coordinates: ModelCoordinateParameters
    global_step: int
    final_batch_loss: float
    amplitude_scaling: str = TRAIN_GLOBAL_RMS_SCALING
    trace_amplitude_scales: np.ndarray | None = None
    trace_array_rows: np.ndarray | None = None
    scale_interpolation: dict[str, object] | None = None


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
    amplitude_scaling: str = TRAIN_GLOBAL_RMS_SCALING,
    trace_amplitude_scales: np.ndarray | None = None,
    trace_array_rows: np.ndarray | None = None,
    scale_interpolation: dict[str, object] | None = None,
) -> None:
    """Save the final fit, without validation selection or resumable state.

    The caller owns creation of the artifact directory.
    """
    if model_coordinates.input_features != model.input_features:
        raise ValueError("model coordinate width must match model.input_features")
    step, loss = _fixed_step_training_values(global_step, final_batch_loss)
    scaling = validated_amplitude_scaling(amplitude_scaling)
    trace_scaling = None
    if scaling == PER_TRACE_RMS_SCALING:
        scales, rows, interpolation = _fixed_step_trace_scaling_values(
            trace_amplitude_scales, trace_array_rows, scale_interpolation
        )
        trace_scaling = {
            "amplitude_scales": torch.from_numpy(scales.copy()),
            "array_rows": torch.from_numpy(rows.copy()),
            "interpolation": _trace_interpolation_metadata(interpolation),
        }
    elif any(
        value is not None
        for value in (trace_amplitude_scales, trace_array_rows, scale_interpolation)
    ):
        raise ValueError("train_global_rms must not contain trace scaling metadata")
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
        "amplitude_scaling": scaling,
        "training_domain": "benchmark_observed_samples",
        "training": {"global_step": step, "final_batch_loss": loss},
    }
    if trace_scaling is not None:
        payload["trace_scaling"] = trace_scaling
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
        ("training_domain", "benchmark_observed_samples"),
    ):
        if payload[key] != expected:
            raise ValueError(f"checkpoint {key} must be {expected!r}")
    scaling = validated_amplitude_scaling(
        payload["amplitude_scaling"], name="checkpoint amplitude_scaling"
    )
    scales, rows, interpolation = _load_fixed_step_trace_scaling(payload, scaling)
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
    return LoadedFixedStepSirenCheckpoint(
        model, normalization, coordinates, step, loss, scaling, scales, rows, interpolation
    )


def _fixed_step_trace_scaling_values(
    scales: object, rows: object, interpolation: object
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if (
        not isinstance(scales, np.ndarray)
        or scales.dtype != np.float64
        or scales.ndim != 1
        or scales.size == 0
        or not np.all(np.isfinite(scales))
        or np.any(scales < 0)
    ):
        raise ValueError("trace_amplitude_scales must be nonempty flat finite nonnegative float64")
    if (
        not isinstance(rows, np.ndarray)
        or rows.dtype != np.int64
        or rows.shape != scales.shape
        or np.any(rows < 0)
        or np.unique(rows).size != rows.size
    ):
        raise ValueError("trace_array_rows must be aligned flat unique nonnegative int64")
    required = {"neighbors", "power", "distance_scales_m"}
    if not isinstance(interpolation, Mapping) or set(interpolation) != required:
        raise ValueError("scale_interpolation requires exactly neighbors, power, distance_scales_m")
    neighbors = interpolation["neighbors"]
    if isinstance(neighbors, bool) or not isinstance(neighbors, Integral) or neighbors <= 0:
        raise ValueError("scale_interpolation neighbors must be a positive integer")
    power = _positive_finite_interpolation_value(interpolation["power"], "power")
    distance_scales = interpolation["distance_scales_m"]
    if not isinstance(distance_scales, list) or len(distance_scales) != 4:
        raise ValueError("scale_interpolation distance_scales_m must contain four positive values")
    return (
        scales,
        rows,
        {
            "neighbors": int(neighbors),
            "power": power,
            "distance_scales_m": [
                _positive_finite_interpolation_value(value, "distance_scales_m")
                for value in distance_scales
            ],
        },
    )


def _positive_finite_interpolation_value(value: object, name: str) -> float:
    if not isinstance(value, bool) and isinstance(value, Real):
        try:
            number = float(value)
        except (OverflowError, ValueError):
            pass
        else:
            if math.isfinite(number) and number > 0:
                return number
    raise ValueError(f"scale_interpolation {name} must be positive and finite")


def _trace_interpolation_metadata(interpolation: Mapping[str, object]) -> dict[str, object]:
    return {
        "method": "inverse_distance_weighting",
        "coordinate_order": [
            "source_x_m",
            "source_y_m",
            "relative_receiver_x_m",
            "relative_receiver_y_m",
        ],
        "coordinate_units": "m",
        "distance_metric": "euclidean_after_dividing_coordinates_by_distance_scales_m",
        **interpolation,
    }


def _load_fixed_step_trace_scaling(
    payload: Mapping[str, object], scaling: str
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, object] | None]:
    if scaling == TRAIN_GLOBAL_RMS_SCALING:
        if "trace_scaling" in payload:
            raise ValueError("train_global_rms must not contain trace scaling metadata")
        return None, None, None
    trace_scaling = payload.get("trace_scaling")
    if not isinstance(trace_scaling, Mapping) or set(trace_scaling) != {
        "amplitude_scales",
        "array_rows",
        "interpolation",
    }:
        raise ValueError("per_trace_rms checkpoint requires complete trace_scaling metadata")
    scales, rows = trace_scaling["amplitude_scales"], trace_scaling["array_rows"]
    if not isinstance(scales, torch.Tensor) or scales.dtype != torch.float64:
        raise ValueError("checkpoint trace_amplitude_scales must be a float64 tensor")
    if not isinstance(rows, torch.Tensor) or rows.dtype != torch.int64:
        raise ValueError("checkpoint trace_array_rows must be an int64 tensor")
    metadata = trace_scaling["interpolation"]
    if not isinstance(metadata, Mapping):
        raise ValueError("checkpoint scale_interpolation must contain fixed IDW metadata")
    scales, rows, interpolation = _fixed_step_trace_scaling_values(
        scales.numpy().copy(),
        rows.numpy().copy(),
        {name: metadata.get(name) for name in ("neighbors", "power", "distance_scales_m")},
    )
    if metadata != _trace_interpolation_metadata(interpolation):
        raise ValueError("checkpoint scale_interpolation must match fixed IDW coordinate metadata")
    return scales, rows, interpolation


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
