"""Save trace-group gradients from a full-time CCNet patch batch before updates."""

import math
from contextlib import nullcontext
from dataclasses import dataclass

import torch

from seis_interp.training.trace_relative_loss import masked_trace_loss


@dataclass(frozen=True)
class CCNet5DPatchGradient:
    loss: float
    gradients: tuple[torch.Tensor, ...]
    supervised_traces: int


def precompute_ccnet5d_patch_gradients(
    model, batches, *, device, autocast_dtype=None, loss_name="masked_trace_mse", reporter=None
):
    """Forward each independent patch once and save gradients at the same weights.

    The caller applies the returned groups in order. No optimizer step or EMA
    update occurs here. The last backward releases the shared forward graph.
    Small patches can be stacked to supply a complete supervised trace group.
    """
    if not batches:
        raise ValueError("cached patch gradients require at least one trace group")
    input_values, target_values, _ = batches[0]
    if input_values.shape[0] < 1:
        raise ValueError("cached patch gradients require at least one patch")
    if any(x is not input_values or y is not target_values for x, y, _ in batches):
        raise ValueError("cached gradient groups must share the same patch input and targets")
    parameter = next(model.parameters())
    inputs = torch.as_tensor(input_values, dtype=parameter.dtype, device=device)[:, None]
    targets = torch.as_tensor(target_values, dtype=parameter.dtype, device=device).movedim(1, -1)
    with (
        torch.autocast(device_type="cuda", dtype=autocast_dtype)
        if autocast_dtype is not None
        else nullcontext()
    ):
        prediction = model(inputs)
    if prediction.shape != inputs.shape:
        raise ValueError("CCNet-5D output shape must match the pseudo-mask input patch")
    if autocast_dtype is not None:
        prediction = prediction.float()
    prediction = prediction[:, 0].movedim(1, -1)
    if reporter is not None:
        reporter(
            f"ccnet5d cached patch: one forward, time_samples={input_values.shape[1]}, "
            f"gradient_groups={len(batches)}, patches={input_values.shape[0]}"
        )
    parameters = tuple(model.parameters())
    saved = []
    for index, (_, _, mask_values) in enumerate(batches):
        mask = torch.as_tensor(mask_values, dtype=torch.bool, device=device)
        objective = masked_trace_loss(prediction[mask], targets[mask], loss_name=loss_name)
        loss = float(objective.detach().cpu())
        if not math.isfinite(loss):
            raise RuntimeError("non-finite CCNet-5D cached patch loss")
        gradients = torch.autograd.grad(
            objective, parameters, retain_graph=index < len(batches) - 1
        )
        if not bool(torch.stack([gradient.isfinite().all() for gradient in gradients]).all()):
            raise RuntimeError("non-finite CCNet-5D cached patch gradients")
        saved.append(
            CCNet5DPatchGradient(
                loss,
                tuple(gradient.detach().clone() for gradient in gradients),
                int(mask_values.sum()),
            )
        )
        if reporter is not None:
            reporter(f"ccnet5d cached patch: saved gradient group {index + 1}/{len(batches)}")
    return tuple(saved)
