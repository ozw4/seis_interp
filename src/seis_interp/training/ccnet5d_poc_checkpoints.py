"""Final checkpoints for per-volume observed-only CCNet-5D PoC runs."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path

import torch

from seis_interp.models.ccnet5d import CCNet5D
from seis_interp.processing.c3_volume_index import VOLUME_AXIS_ORDER
from seis_interp.processing.ccnet5d_tiles import validate_ccnet5d_shape
from seis_interp.training.trace_relative_loss import POC_TRACE_LOSSES

CCNET5D_POC_METHOD_VARIANT = "common_protocol_masked_observed_training"
CCNET5D_POC_CHECKPOINT_ROLE = "final"
_MODEL_CONFIG_FIELDS = {
    "hidden_channels",
    "intermediate_channels",
    "kernel_size",
    "output_activation",
}


@dataclass(frozen=True)
class LoadedCCNet5DPocCheckpoint:
    """A final model and the observed-only training constants bound to it."""

    model: CCNet5D
    amplitude_scale: float
    patch_shape: tuple[int, ...]
    inner_mask_fraction: float
    placement_seed: int
    inner_mask_seed: int
    model_initialization_seed: int
    optimizer_updates: int
    loss: str


def save_ccnet5d_poc_checkpoint(
    path: Path,
    model: CCNet5D,
    *,
    amplitude_scale: float,
    patch_shape: tuple[int, ...],
    inner_mask_fraction: float,
    placement_seed: int,
    inner_mask_seed: int,
    model_initialization_seed: int,
    optimizer_updates: int,
    loss: str = "masked_trace_relative_mse",
) -> None:
    """Save one non-resumable final CPU snapshot without optimizer state."""
    if not isinstance(model, CCNet5D):
        raise TypeError("model must be a CCNet5D")
    if loss not in POC_TRACE_LOSSES:
        raise ValueError(f"checkpoint loss must be one of {POC_TRACE_LOSSES!r}")
    scale = _positive_finite_float(amplitude_scale, "amplitude_scale")
    shape = validate_ccnet5d_shape(patch_shape, "patch_shape")
    fraction = _fraction(inner_mask_fraction)
    placement = _nonnegative_integer(placement_seed, "placement_seed")
    mask = _nonnegative_integer(inner_mask_seed, "inner_mask_seed")
    initialization = _nonnegative_integer(model_initialization_seed, "model_initialization_seed")
    updates = _positive_integer(optimizer_updates, "optimizer_updates")
    payload = {
        "model_type": "ccnet5d",
        "method_variant": CCNET5D_POC_METHOD_VARIANT,
        "model_config": model.constructor_config(),
        "state_dict": _cpu_state_dict(model.state_dict()),
        "axis_order": list(VOLUME_AXIS_ORDER),
        "normalization": {
            "type": "global_rms",
            "source": "O_only",
            "scale": scale,
        },
        "patches": {
            "shape": list(shape),
            "inner_mask_fraction": fraction,
            "placement_seed": placement,
            "inner_mask_seed": mask,
        },
        "model_initialization_seed": initialization,
        "optimizer_updates": updates,
        "loss": loss,
        "checkpoint_role": CCNET5D_POC_CHECKPOINT_ROLE,
    }
    torch.save(payload, Path(path))


def load_ccnet5d_poc_checkpoint(
    path: Path,
    *,
    device: torch.device | str = "cpu",
) -> LoadedCCNet5DPocCheckpoint:
    """Strictly restore one final observed-only CCNet-5D checkpoint."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    expected_fields = {
        "model_type",
        "method_variant",
        "model_config",
        "state_dict",
        "axis_order",
        "normalization",
        "patches",
        "optimizer_updates",
        "model_initialization_seed",
        "loss",
        "checkpoint_role",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_fields:
        raise ValueError("CCNet-5D PoC checkpoint must contain exactly the required fields")
    if payload["model_type"] != "ccnet5d":
        raise ValueError("checkpoint model_type must be 'ccnet5d'")
    if payload["method_variant"] != CCNET5D_POC_METHOD_VARIANT:
        raise ValueError("checkpoint method_variant does not match the observed-only PoC")
    if payload["axis_order"] != list(VOLUME_AXIS_ORDER):
        raise ValueError("checkpoint axis_order must match the C3 volume axis order")
    if payload["loss"] not in POC_TRACE_LOSSES:
        raise ValueError(f"checkpoint loss must be one of {POC_TRACE_LOSSES!r}")
    if payload["checkpoint_role"] != CCNET5D_POC_CHECKPOINT_ROLE:
        raise ValueError("checkpoint checkpoint_role must be 'final'")

    normalization = payload["normalization"]
    if not isinstance(normalization, Mapping) or set(normalization) != {
        "type",
        "source",
        "scale",
    }:
        raise ValueError("checkpoint normalization must contain type, source, and scale")
    if normalization["type"] != "global_rms" or normalization["source"] != "O_only":
        raise ValueError("checkpoint normalization must be O-only global RMS")
    scale = _positive_finite_float(normalization["scale"], "normalization.scale")

    patches = payload["patches"]
    if not isinstance(patches, Mapping) or set(patches) != {
        "shape",
        "inner_mask_fraction",
        "placement_seed",
        "inner_mask_seed",
    }:
        raise ValueError(
            "checkpoint patches must contain shape, fraction, placement and inner mask seeds"
        )
    shape = validate_ccnet5d_shape(patches["shape"], "patches.shape")
    fraction = _fraction(patches["inner_mask_fraction"])
    placement = _nonnegative_integer(patches["placement_seed"], "patches.placement_seed")
    mask = _nonnegative_integer(patches["inner_mask_seed"], "patches.inner_mask_seed")
    initialization = _nonnegative_integer(
        payload["model_initialization_seed"], "model_initialization_seed"
    )
    updates = _positive_integer(payload["optimizer_updates"], "optimizer_updates")
    model = _model_from_payload(payload["model_config"], payload["state_dict"])
    model.to(device)
    return LoadedCCNet5DPocCheckpoint(
        loss=payload["loss"],
        model=model,
        amplitude_scale=scale,
        patch_shape=shape,
        inner_mask_fraction=fraction,
        placement_seed=placement,
        inner_mask_seed=mask,
        model_initialization_seed=initialization,
        optimizer_updates=updates,
    )


def _model_from_payload(model_config: object, state_dict: object) -> CCNet5D:
    if not isinstance(model_config, Mapping) or set(model_config) != _MODEL_CONFIG_FIELDS:
        raise ValueError("checkpoint model_config must contain every CCNet-5D constructor field")
    if not isinstance(state_dict, Mapping) or not all(
        isinstance(value, torch.Tensor) for value in state_dict.values()
    ):
        raise ValueError("checkpoint state_dict must map parameter names to tensors")
    try:
        model = CCNet5D(**dict(model_config))
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint model_config contains invalid values") from error
    model.load_state_dict(state_dict, strict=True)
    return model


def _cpu_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().to(device="cpu").clone() for name, tensor in state_dict.items()}


def _fraction(value: object) -> float:
    converted = _positive_finite_float(value, "inner_mask_fraction")
    if converted >= 1.0:
        raise ValueError("inner_mask_fraction must be strictly less than 1")
    return converted


def _positive_finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be positive and finite")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return converted


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)
