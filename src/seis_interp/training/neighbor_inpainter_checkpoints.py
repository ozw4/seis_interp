"""Save and restore neighbor-trace inpainter checkpoints."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path

import torch

from seis_interp.models.neighbor_trace_inpainter import (
    DEFAULT_COORDINATE_CONDITIONING,
    DEFAULT_NEIGHBOR_ALIGNMENT_KERNEL_SIZE,
    DEFAULT_NEIGHBOR_GATING,
    DEFAULT_PREDICTION_REFERENCE,
    DEFAULT_RESIDUAL_KERNEL_SIZE,
    DEFAULT_STEM_KERNEL_SIZE,
    DEFAULT_TARGET_COORDINATE_COUNT,
    TEMPORAL_DILATIONS,
    NeighborTraceInpainter,
)
from seis_interp.training.amplitude_scaling import (
    ORACLE_PER_TRACE_RMS_VALIDATION_DOMAIN,
    PER_TRACE_RMS_SCALING,
)


@dataclass(frozen=True)
class LoadedNeighborInpainterCheckpoint:
    """A restored inpainter and its model-selection metadata."""

    model: NeighborTraceInpainter
    amplitude_scaling: str
    validation_metric_domain: str
    best_step: int
    best_validation_global_snr_db: float


def save_neighbor_inpainter_checkpoint(
    path: Path,
    model: NeighborTraceInpainter,
    *,
    best_step: int,
    best_validation_global_snr_db: float,
) -> None:
    """Save constructor values, CPU weights, and the raw validation optimum."""
    step = _positive_integer(best_step, "best_step")
    validation_snr = _finite_float(
        best_validation_global_snr_db,
        "best_validation_global_snr_db",
    )
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {
        name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
    }
    torch.save(
        {
            "model_type": "neighbor_trace_inpainter",
            "model_config": {
                "neighbor_count": model.neighbor_count,
                "width": model.width,
                "target_coordinate_count": model.target_coordinate_count,
                "stem_kernel_size": model.stem_kernel_size,
                "residual_kernel_size": model.residual_kernel_size,
                "temporal_dilations": list(model.temporal_dilations),
                "coordinate_conditioning": model.coordinate_conditioning,
                "neighbor_gating": model.neighbor_gating,
                "neighbor_alignment_kernel_size": model.neighbor_alignment_kernel_size,
                "prediction_reference": model.prediction_reference,
            },
            "model_state_dict": state_dict,
            "amplitude_scaling": PER_TRACE_RMS_SCALING,
            "validation_metric_domain": ORACLE_PER_TRACE_RMS_VALIDATION_DOMAIN,
            "training": {
                "best_step": step,
                "best_validation_global_snr_db": validation_snr,
            },
        },
        checkpoint_path,
    )


def load_neighbor_inpainter_checkpoint(
    path: Path,
    *,
    device: torch.device | str = "cpu",
) -> LoadedNeighborInpainterCheckpoint:
    """Rebuild an inpainter on ``device`` and load its weights strictly."""
    requested_device = torch.device(device)
    payload = torch.load(Path(path), map_location=requested_device, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("neighbor inpainter checkpoint must contain a mapping")
    if payload.get("model_type") != "neighbor_trace_inpainter":
        raise ValueError("checkpoint model_type must be 'neighbor_trace_inpainter'")
    if payload.get("amplitude_scaling") != PER_TRACE_RMS_SCALING:
        raise ValueError("neighbor inpainter checkpoint amplitude_scaling must be 'per_trace_rms'")
    if payload.get("validation_metric_domain") != ORACLE_PER_TRACE_RMS_VALIDATION_DOMAIN:
        raise ValueError(
            "neighbor inpainter checkpoint validation_metric_domain must be "
            f"'{ORACLE_PER_TRACE_RMS_VALIDATION_DOMAIN}'"
        )

    model_config = payload.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("neighbor inpainter checkpoint model_config must be a mapping")
    try:
        model = NeighborTraceInpainter(
            neighbor_count=model_config["neighbor_count"],
            width=model_config["width"],
            target_coordinate_count=model_config.get(
                "target_coordinate_count", DEFAULT_TARGET_COORDINATE_COUNT
            ),
            stem_kernel_size=model_config.get("stem_kernel_size", DEFAULT_STEM_KERNEL_SIZE),
            residual_kernel_size=model_config.get(
                "residual_kernel_size", DEFAULT_RESIDUAL_KERNEL_SIZE
            ),
            temporal_dilations=model_config.get("temporal_dilations", TEMPORAL_DILATIONS),
            coordinate_conditioning=model_config.get(
                "coordinate_conditioning", DEFAULT_COORDINATE_CONDITIONING
            ),
            neighbor_gating=model_config.get("neighbor_gating", DEFAULT_NEIGHBOR_GATING),
            neighbor_alignment_kernel_size=model_config.get(
                "neighbor_alignment_kernel_size",
                DEFAULT_NEIGHBOR_ALIGNMENT_KERNEL_SIZE,
            ),
            prediction_reference=model_config.get(
                "prediction_reference",
                DEFAULT_PREDICTION_REFERENCE,
            ),
        )
    except KeyError as error:
        raise ValueError(
            f"neighbor inpainter checkpoint model_config is missing {error.args[0]!r}"
        ) from error

    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("neighbor inpainter checkpoint model_state_dict must be a mapping")
    model.to(requested_device)
    model.load_state_dict(state_dict, strict=True)

    training = payload.get("training")
    if not isinstance(training, dict):
        raise ValueError("neighbor inpainter checkpoint training metadata must be a mapping")
    try:
        best_step = _positive_integer(training["best_step"], "best_step")
        best_validation = _finite_float(
            training["best_validation_global_snr_db"],
            "best_validation_global_snr_db",
        )
    except KeyError as error:
        raise ValueError(
            f"neighbor inpainter checkpoint training metadata is missing {error.args[0]!r}"
        ) from error

    return LoadedNeighborInpainterCheckpoint(
        model=model,
        amplitude_scaling=PER_TRACE_RMS_SCALING,
        validation_metric_domain=ORACLE_PER_TRACE_RMS_VALIDATION_DOMAIN,
        best_step=best_step,
        best_validation_global_snr_db=best_validation,
    )


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be a finite number")
    return converted
