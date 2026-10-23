"""Chunked gather-scale-scatter of edge messages against the unfused expression."""

from __future__ import annotations

import pytest
import torch

from seis_interp.models import trace_graph_edge_aggregation
from seis_interp.models.trace_graph_edge_aggregation import aggregate_edge_messages


def _edges(nodes=5, edges=11, channels=3, frames=4, groups=7, dtype=torch.float64, seed=0):
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(nodes, channels, frames, generator=generator, dtype=dtype)
    coefficients = torch.randn(edges, channels, generator=generator, dtype=dtype)
    sender = torch.randint(0, nodes, (edges,), generator=generator)
    group = torch.randint(0, groups, (edges,), generator=generator)
    return values, coefficients, sender, group, groups


def _unfused(values, coefficients, sender, group, group_count):
    weighted = coefficients[:, :, None] * values[sender]
    output = weighted.new_zeros(group_count, *values.shape[1:])
    output.index_add_(0, group, weighted)
    return output


def test_gradcheck_float64() -> None:
    values, coefficients, sender, group, groups = _edges()
    values.requires_grad_(True)
    coefficients.requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda v, c: aggregate_edge_messages(v, c, sender, group, groups),
        (values, coefficients),
    )


@pytest.mark.parametrize("chunk_elements", [1, 12, 25, 1 << 26])
def test_matches_unfused_values_and_gradients_for_any_chunking(monkeypatch, chunk_elements):
    monkeypatch.setattr(trace_graph_edge_aggregation, "_CHUNK_ELEMENTS", chunk_elements)
    values, coefficients, sender, group, groups = _edges()
    upstream = torch.randn(groups, 3, 4, dtype=torch.float64)
    results = []
    for function in (aggregate_edge_messages, _unfused):
        v = values.clone().requires_grad_(True)
        c = coefficients.clone().requires_grad_(True)
        output = function(v, c, sender, group, groups)
        (output * upstream).sum().backward()
        results.append((output.detach(), v.grad, c.grad))
    for fused, reference in zip(*results, strict=True):
        torch.testing.assert_close(fused, reference, rtol=1e-12, atol=1e-14)


def test_empty_edges_give_zero_messages_and_gradients() -> None:
    values = torch.randn(3, 2, 5, dtype=torch.float64, requires_grad=True)
    coefficients = torch.empty(0, 2, dtype=torch.float64, requires_grad=True)
    index = torch.empty(0, dtype=torch.int64)
    output = aggregate_edge_messages(values, coefficients, index, index, 8)
    assert torch.equal(output, torch.zeros(8, 2, 5, dtype=torch.float64))
    output.sum().backward()
    assert torch.equal(values.grad, torch.zeros_like(values))
    assert coefficients.grad.shape == (0, 2)


def test_low_precision_values_follow_unfused_type_promotion() -> None:
    values, coefficients, sender, group, groups = _edges(dtype=torch.float32)
    values = values.bfloat16().requires_grad_(True)
    coefficients.requires_grad_(True)
    output = aggregate_edge_messages(values, coefficients, sender, group, groups)
    assert output.dtype == torch.float32
    output.square().sum().backward()
    assert values.grad.dtype == torch.bfloat16
    assert coefficients.grad.dtype == torch.float32
    reference = _unfused(values.detach(), coefficients.detach(), sender, group, groups)
    torch.testing.assert_close(output.detach(), reference, rtol=1e-6, atol=1e-6)


def test_rejects_mismatched_shapes() -> None:
    values, coefficients, sender, group, groups = _edges()
    with pytest.raises(ValueError, match="coefficients"):
        aggregate_edge_messages(values, coefficients[:, :2], sender, group, groups)
    with pytest.raises(ValueError, match="sender and group"):
        aggregate_edge_messages(values, coefficients, sender, group[:-1], groups)
