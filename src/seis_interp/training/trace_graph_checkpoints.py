"""Save and restore trace-graph interpolator checkpoints."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path

import torch

from seis_interp.models.trace_graph_interpolator import (
    POOLED_ATTENTION_TIME_RESOLUTION,
    TraceGraphInterpolator,
)
from seis_interp.training.amplitude_scaling import (
    ORACLE_PER_TRACE_RMS_VALIDATION_DOMAIN,
    PER_TRACE_RMS_SCALING,
)

MODEL_TYPE = "trace_graph_interpolator"


@dataclass(frozen=True)
class LoadedTraceGraphCheckpoint:
    """A restored trace-graph model and its selection metadata."""

    model: TraceGraphInterpolator
    amplitude_scaling: str
    validation_metric_domain: str
    graph_mode: str
    best_step: int
    best_validation_global_snr_db: float


def save_trace_graph_checkpoint(
    path: Path,
    model: TraceGraphInterpolator,
    *,
    best_step: int,
    best_validation_global_snr_db: float,
) -> None:
    """Save constructor values, CPU weights, and the validation optimum."""
    if not isinstance(model, TraceGraphInterpolator):
        raise TypeError("model must be a TraceGraphInterpolator")
    step = _nonnegative_integer(best_step, "best_step")
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
            "model_type": MODEL_TYPE,
            "model_config": {
                "width": model.width,
                "graph_mode": model.graph_mode,
                "message_passing_rounds": model.message_passing_rounds,
                "time_downsample_factor": model.time_downsample_factor,
                "stem_kernel_size": model.stem_kernel_size,
                "temporal_kernel_size": model.temporal_kernel_size,
                "temporal_dilations": list(model.temporal_dilations),
                "spatial_kernel_size": model.spatial_kernel_size,
                "attention_width": model.attention_width,
                "attention_time_resolution": model.attention_time_resolution,
                "distance_epsilon": model.distance_epsilon,
                "use_gradient_checkpointing": model.use_gradient_checkpointing,
                "refinement_passes": model.refinement_passes,
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


def load_trace_graph_checkpoint(
    path: Path,
    *,
    device: torch.device | str = "cpu",
) -> LoadedTraceGraphCheckpoint:
    """Rebuild a trace-graph model on ``device`` and load weights strictly."""
    requested_device = torch.device(device)
    payload = torch.load(Path(path), map_location=requested_device, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("trace graph checkpoint must contain a mapping")
    if payload.get("model_type") != MODEL_TYPE:
        raise ValueError(f"checkpoint model_type must be {MODEL_TYPE!r}")
    if payload.get("amplitude_scaling") != PER_TRACE_RMS_SCALING:
        raise ValueError("checkpoint amplitude_scaling must be 'per_trace_rms'")
    if payload.get("validation_metric_domain") != ORACLE_PER_TRACE_RMS_VALIDATION_DOMAIN:
        raise ValueError(
            "checkpoint validation_metric_domain must be "
            f"{ORACLE_PER_TRACE_RMS_VALIDATION_DOMAIN!r}"
        )
    model_config = payload.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("checkpoint model_config must be a mapping")
    try:
        model = TraceGraphInterpolator(
            width=model_config["width"],
            graph_mode=model_config["graph_mode"],
            message_passing_rounds=model_config["message_passing_rounds"],
            time_downsample_factor=model_config["time_downsample_factor"],
            stem_kernel_size=model_config["stem_kernel_size"],
            temporal_kernel_size=model_config["temporal_kernel_size"],
            temporal_dilations=model_config["temporal_dilations"],
            spatial_kernel_size=model_config["spatial_kernel_size"],
            attention_width=model_config["attention_width"],
            attention_time_resolution=model_config.get(
                "attention_time_resolution",
                POOLED_ATTENTION_TIME_RESOLUTION,
            ),
            distance_epsilon=model_config["distance_epsilon"],
            use_gradient_checkpointing=bool(model_config.get("use_gradient_checkpointing", False)),
            refinement_passes=model_config.get("refinement_passes", 1),
        )
    except KeyError as error:
        raise ValueError(f"checkpoint model_config is missing {error.args[0]!r}") from error
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint model_state_dict must be a mapping")
    model.to(requested_device)
    model.load_state_dict(state_dict, strict=True)

    training = payload.get("training")
    if not isinstance(training, dict):
        raise ValueError("checkpoint training metadata must be a mapping")
    try:
        best_step = _nonnegative_integer(training["best_step"], "best_step")
        best_validation = _finite_float(
            training["best_validation_global_snr_db"],
            "best_validation_global_snr_db",
        )
    except KeyError as error:
        raise ValueError(f"checkpoint training metadata is missing {error.args[0]!r}") from error
    return LoadedTraceGraphCheckpoint(
        model=model,
        amplitude_scaling=PER_TRACE_RMS_SCALING,
        validation_metric_domain=ORACLE_PER_TRACE_RMS_VALIDATION_DOMAIN,
        graph_mode=model.graph_mode,
        best_step=best_step,
        best_validation_global_snr_db=best_validation,
    )


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be a finite number")
    return converted
