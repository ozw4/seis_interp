"""Bound CCNet activation memory while backpropagating one complete-trace objective."""

from contextlib import nullcontext

import numpy as np
import torch

from seis_interp.models.ccnet5d_trace_queries import (
    forward_ccnet5d_trace_queries,
    prepare_ccnet5d_trace_queries,
)
from seis_interp.training.trace_relative_loss import masked_trace_loss


def backward_ccnet5d_trace_queries(
    model,
    inputs,
    targets,
    trace_mask,
    *,
    device,
    spatial_chunk_size=64,
    autocast_dtype=None,
    loss_name="masked_trace_mse",
):
    """Backpropagate selected complete traces without splitting the time axis.

    Only the caller performs the optimizer step. Spatial chunk checkpointing
    releases gathered convolution columns and recomputes them during backward.
    """
    queries = prepare_ccnet5d_trace_queries(model, trace_mask, device=device)
    time = inputs.shape[1]
    flat_inputs = np.moveaxis(inputs, 1, -1).reshape(-1, time)
    flat_targets = np.moveaxis(targets, 1, -1).reshape(-1, time)
    parameter = next(model.parameters())
    values = torch.as_tensor(
        np.ascontiguousarray(flat_inputs[queries.input_cells]), dtype=parameter.dtype, device=device
    )[:, None]
    target = torch.as_tensor(
        np.ascontiguousarray(flat_targets[queries.query_cells]),
        dtype=parameter.dtype,
        device=device,
    )
    with (
        torch.autocast(device_type="cuda", dtype=autocast_dtype)
        if autocast_dtype is not None
        else nullcontext()
    ):
        prediction = forward_ccnet5d_trace_queries(
            model, values, queries, spatial_chunk_size=spatial_chunk_size
        )
    if autocast_dtype is not None:
        prediction = prediction.float()
    objective = masked_trace_loss(prediction, target, loss_name=loss_name)
    if not torch.isfinite(objective):
        raise RuntimeError("non-finite CCNet-5D training loss")
    objective.backward()
    return float(objective.detach().cpu())
