"""Save and restore whole-shot gather inpainter checkpoints."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path

import torch

from seis_interp.models.shot_gather_inpainter import ShotGatherInpainter
from seis_interp.training.amplitude_scaling import (
    ORACLE_PER_TRACE_RMS_VALIDATION_DOMAIN,
    PER_TRACE_RMS_SCALING,
)

MODEL_TYPE = "shot_gather_inpainter"
INPUT_FEATURE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class LoadedShotGatherInpainterCheckpoint:
    """A restored whole-shot model and its selection metadata."""

    model: ShotGatherInpainter
    amplitude_scaling: str
    validation_metric_domain: str
    input_feature_schema_version: int
    input_feature_names: tuple[str, ...]
    best_step: int
    best_validation_global_snr_db: float


def save_shot_gather_inpainter_checkpoint(
    path: Path,
    model: ShotGatherInpainter,
    *,
    best_step: int,
    best_validation_global_snr_db: float,
) -> None:
    """Save constructor values, CPU weights, and the validation optimum."""
    if not isinstance(model, ShotGatherInpainter):
        raise TypeError("model must be a ShotGatherInpainter")
    step = _nonnegative_integer(best_step, "best_step")
    validation_snr = _finite_float(
        best_validation_global_snr_db,
        "best_validation_global_snr_db",
    )
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    input_feature_names = _validated_model_input_feature_names(model)
    state_dict = {
        name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
    }
    torch.save(
        {
            "model_type": MODEL_TYPE,
            "model_config": {
                "width": model.width,
                "temporal_dilations": list(model.temporal_dilations),
                "spatial_y_dilations": list(model.spatial_y_dilations),
                "stem_kernel_size": model.stem_kernel_size,
                "residual_kernel_size": model.residual_kernel_size,
                "distance_epsilon": model.distance_epsilon,
            },
            "model_state_dict": state_dict,
            "amplitude_scaling": PER_TRACE_RMS_SCALING,
            "validation_metric_domain": ORACLE_PER_TRACE_RMS_VALIDATION_DOMAIN,
            "input_feature_schema": {
                "version": INPUT_FEATURE_SCHEMA_VERSION,
                "names": list(input_feature_names),
            },
            "training": {
                "best_step": step,
                "best_validation_global_snr_db": validation_snr,
            },
        },
        checkpoint_path,
    )


def load_shot_gather_inpainter_checkpoint(
    path: Path,
    *,
    device: torch.device | str = "cpu",
) -> LoadedShotGatherInpainterCheckpoint:
    """Rebuild a whole-shot model on ``device`` and load weights strictly."""
    requested_device = torch.device(device)
    payload = torch.load(Path(path), map_location=requested_device, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("shot gather inpainter checkpoint must contain a mapping")
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
        model = ShotGatherInpainter(
            width=model_config["width"],
            temporal_dilations=model_config["temporal_dilations"],
            spatial_y_dilations=model_config.get("spatial_y_dilations"),
            stem_kernel_size=model_config["stem_kernel_size"],
            residual_kernel_size=model_config["residual_kernel_size"],
            distance_epsilon=model_config["distance_epsilon"],
        )
    except KeyError as error:
        raise ValueError(f"checkpoint model_config is missing {error.args[0]!r}") from error
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint model_state_dict must be a mapping")
    model.to(requested_device)
    model.load_state_dict(state_dict, strict=True)

    input_feature_schema = payload.get("input_feature_schema")
    if not isinstance(input_feature_schema, dict):
        raise ValueError("checkpoint input_feature_schema must be a mapping")
    schema_version = input_feature_schema.get("version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, Integral)
        or int(schema_version) != INPUT_FEATURE_SCHEMA_VERSION
    ):
        raise ValueError(
            f"checkpoint input feature schema version must be {INPUT_FEATURE_SCHEMA_VERSION}"
        )
    stored_input_feature_names = _validated_feature_names(
        input_feature_schema.get("names"),
        "checkpoint input_feature_schema.names",
    )
    expected_input_feature_names = _validated_model_input_feature_names(model)
    if stored_input_feature_names != expected_input_feature_names:
        raise ValueError(
            "checkpoint input feature names do not match the current model feature order"
        )

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
    return LoadedShotGatherInpainterCheckpoint(
        model=model,
        amplitude_scaling=PER_TRACE_RMS_SCALING,
        validation_metric_domain=ORACLE_PER_TRACE_RMS_VALIDATION_DOMAIN,
        input_feature_schema_version=INPUT_FEATURE_SCHEMA_VERSION,
        input_feature_names=stored_input_feature_names,
        best_step=best_step,
        best_validation_global_snr_db=best_validation,
    )


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _validated_model_input_feature_names(model: ShotGatherInpainter) -> tuple[str, ...]:
    names = _validated_feature_names(
        model.input_feature_names,
        "model.input_feature_names",
    )
    if len(names) != model.input_channels:
        raise ValueError("model.input_feature_names length must equal model.input_channels")
    return names


def _validated_feature_names(value: object, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a non-empty sequence of feature names")
    names = tuple(value)
    if not names or any(not isinstance(item, str) or not item for item in names):
        raise ValueError(f"{name} must be a non-empty sequence of feature names")
    if len(set(names)) != len(names):
        raise ValueError(f"{name} must contain unique feature names")
    return names


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be a finite number")
    return converted
