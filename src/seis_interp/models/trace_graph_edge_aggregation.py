"""Chunked edge-message aggregation that keeps no per-edge waveform for backward."""

from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

# One chunk of gathered edge values holds at most this many elements (256 MiB in FP32).
_CHUNK_ELEMENTS = 1 << 26


def aggregate_edge_messages(
    values: torch.Tensor,
    coefficients: torch.Tensor,
    sender: torch.Tensor,
    group: torch.Tensor,
    group_count: int,
) -> torch.Tensor:
    """Return ``out[g] = sum_{e: group[e] = g} coefficients[e, :, None] * values[sender[e]]``.

    ``values[N,C,F]`` and ``coefficients[E,C]`` are floating tensors; ``sender[E]``
    and ``group[E]`` are int64 indices. The result ``[group_count,C,F]`` has the
    promoted dtype of both inputs, like the unfused gather, multiply and
    ``index_add_``. Edge products are evaluated in chunks and recomputed in
    backward, so no ``[E,C,F]`` tensor is saved. The value gradient is
    accumulated in the promoted dtype and cast to the values' dtype once.
    """
    return _GatherScaleScatter.apply(values, coefficients, sender, group, group_count)


class _GatherScaleScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, coefficients, sender, group, group_count):
        if values.ndim != 3 or coefficients.shape != (sender.numel(), values.shape[1]):
            raise ValueError("values must be [N,C,F] and coefficients [E,C]")
        if sender.shape != group.shape or sender.ndim != 1:
            raise ValueError("sender and group must be matching [E] index vectors")
        output = values.new_zeros(
            (group_count, *values.shape[1:]), dtype=torch.result_type(coefficients, values)
        )
        for rows in _edge_chunks(sender.numel(), values):
            output.index_add_(0, group[rows], coefficients[rows, :, None] * values[sender[rows]])
        ctx.save_for_backward(values, coefficients, sender, group)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        values, coefficients, sender, group = ctx.saved_tensors
        need_values, need_coefficients = ctx.needs_input_grad[:2]
        grad_values = grad_output.new_zeros(values.shape) if need_values else None
        grad_coefficients = torch.empty_like(coefficients) if need_coefficients else None
        for rows in _edge_chunks(sender.numel(), values):
            upstream = grad_output[group[rows]]
            if need_coefficients:
                grad_coefficients[rows] = (upstream * values[sender[rows]]).sum(dim=-1)
            if need_values:
                grad_values.index_add_(0, sender[rows], upstream * coefficients[rows, :, None])
        if grad_values is not None:
            grad_values = grad_values.to(values.dtype)
        return grad_values, grad_coefficients, None, None, None


def _edge_chunks(edge_count: int, values: torch.Tensor):
    step = max(1, _CHUNK_ELEMENTS // max(1, values.shape[1] * values.shape[2]))
    for start in range(0, edge_count, step):
        yield slice(start, start + step)
