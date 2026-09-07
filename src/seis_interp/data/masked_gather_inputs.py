"""Named tensor contract for masked target and context gathers."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from seis_interp.processing.c3_receiver_grid import RECEIVER_X_COUNT, RECEIVER_Y_COUNT


@dataclass(frozen=True)
class MaskedGatherInputs:
    """Leakage-safe target observations and neighboring context gathers."""

    target_observed: torch.Tensor
    target_observation_mask: torch.Tensor
    context_gathers: torch.Tensor
    context_availability: torch.Tensor
    source_deltas_m: torch.Tensor
    target_coordinates: torch.Tensor


def validate_masked_gather_inputs(inputs: MaskedGatherInputs) -> MaskedGatherInputs:
    """Validate and return one masked-gather input object unchanged."""
    if not isinstance(inputs, MaskedGatherInputs):
        raise TypeError("inputs must be a MaskedGatherInputs object")

    fields = (
        ("target_observed", inputs.target_observed),
        ("target_observation_mask", inputs.target_observation_mask),
        ("context_gathers", inputs.context_gathers),
        ("context_availability", inputs.context_availability),
        ("source_deltas_m", inputs.source_deltas_m),
        ("target_coordinates", inputs.target_coordinates),
    )
    for name, value in fields:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")

    target = inputs.target_observed
    if target.ndim != 4:
        raise ValueError("target_observed must have shape (batch, 8, 68, time)")
    batch_size, receiver_x, receiver_y, time_count = target.shape
    if batch_size <= 0:
        raise ValueError("target_observed batch dimension must be positive")
    if (receiver_x, receiver_y) != (RECEIVER_X_COUNT, RECEIVER_Y_COUNT):
        raise ValueError("target_observed must use the fixed 8 x 68 receiver grid")
    if time_count < 2:
        raise ValueError("target_observed time dimension must contain at least two samples")

    target_mask_shape = (batch_size, RECEIVER_X_COUNT, RECEIVER_Y_COUNT)
    if inputs.target_observation_mask.shape != target_mask_shape:
        raise ValueError("target_observation_mask must match target batch and receiver dimensions")

    contexts = inputs.context_gathers
    if contexts.ndim != 5:
        raise ValueError("context_gathers must have shape (batch, contexts, 8, 68, time)")
    context_batch, context_count, context_x, context_y, context_time = contexts.shape
    if context_batch != batch_size:
        raise ValueError("context_gathers batch dimension must match target_observed")
    if context_count <= 0:
        raise ValueError("context_gathers context dimension must be positive")
    if (context_x, context_y) != (RECEIVER_X_COUNT, RECEIVER_Y_COUNT):
        raise ValueError("context_gathers must use the fixed 8 x 68 receiver grid")
    if context_time != time_count:
        raise ValueError("context_gathers time dimension must match target_observed")

    context_mask_shape = (
        batch_size,
        context_count,
        RECEIVER_X_COUNT,
        RECEIVER_Y_COUNT,
    )
    if inputs.context_availability.shape != context_mask_shape:
        raise ValueError(
            "context_availability must match context batch, source, and receiver dimensions"
        )
    if inputs.source_deltas_m.shape != (batch_size, context_count, 2):
        raise ValueError("source_deltas_m must have shape (batch, contexts, 2)")
    if inputs.target_coordinates.shape != (batch_size, 2):
        raise ValueError("target_coordinates must have shape (batch, 2)")

    floating_fields = (
        ("target_observed", inputs.target_observed),
        ("context_gathers", inputs.context_gathers),
        ("source_deltas_m", inputs.source_deltas_m),
        ("target_coordinates", inputs.target_coordinates),
    )
    for name, value in floating_fields:
        if not value.is_floating_point():
            raise TypeError(f"{name} must have a floating-point dtype")
    floating_dtypes = {value.dtype for _, value in floating_fields}
    if len(floating_dtypes) != 1:
        raise TypeError("floating input fields must have the same dtype")
    if (
        inputs.target_observation_mask.dtype != torch.bool
        or inputs.context_availability.dtype != torch.bool
    ):
        raise TypeError("observation masks must have dtype torch.bool")

    devices = {value.device for _, value in fields}
    if len(devices) != 1:
        raise ValueError("all input fields must be on the same device")

    for name, value in floating_fields:
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must contain only finite values")
    coordinates = inputs.target_coordinates
    if not bool(((coordinates >= 0.0) & (coordinates <= 1.0)).all()):
        raise ValueError("target_coordinates must be within the inclusive range [0, 1]")
    if not bool(torch.linalg.vector_norm(inputs.source_deltas_m, dim=-1).gt(0.0).all()):
        raise ValueError("every context source delta must have positive Euclidean norm")

    target_mask = inputs.target_observation_mask
    if not bool((~target_mask).flatten(start_dim=1).any(dim=1).all()):
        raise ValueError("every target sample must contain at least one unobserved receiver cell")
    if bool((inputs.target_observed[~target_mask] != 0).any()):
        raise ValueError("target_observed must be exactly zero at unobserved receiver cells")

    context_mask = inputs.context_availability
    if not bool(context_mask.flatten(start_dim=2).any(dim=2).all()):
        raise ValueError("every context shot must contain at least one available receiver cell")
    if bool((inputs.context_gathers[~context_mask] != 0).any()):
        raise ValueError("context_gathers must be exactly zero at unavailable receiver cells")
    return inputs
