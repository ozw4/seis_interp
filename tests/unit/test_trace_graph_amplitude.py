"""Observed-only amplitude factors and direct-neighbor gain interpolation."""

from __future__ import annotations

import pytest
import torch

from seis_interp.models.trace_graph_amplitude import (
    interpolate_query_trace_rms,
    normalize_observed_trace_rms,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_unit_rms_zero_and_hidden_poison_are_handled_without_mutation(dtype) -> None:
    values = torch.tensor([[3.0, -3.0], [0.0, 0.0], [float("nan"), 1e30]], dtype=dtype)
    observed = torch.tensor([True, True, False])
    unit, rms = normalize_observed_trace_rms(values, observed)
    torch.testing.assert_close(rms, torch.tensor([3.0, 0.0, 0.0], dtype=dtype))
    assert torch.equal(unit, torch.tensor([[1.0, -1.0], [0.0, 0.0], [0.0, 0.0]], dtype=dtype))
    assert torch.isnan(values[2, 0]) and values[0, 0] == 3


@pytest.mark.parametrize("factor", [0.1, 0.3, 1.0, 3.0, 10.0, 1e-30, 1e30])
def test_rms_factorization_is_homogeneous_without_squaring_overflow(factor) -> None:
    values = torch.tensor([[1.0, -2.0, 3.0], [0.0, 0.0, 0.0]])
    observed = torch.tensor([True, False])
    unit, rms = normalize_observed_trace_rms(values, observed)
    scaled_unit, scaled_rms = normalize_observed_trace_rms(values * factor, observed)
    torch.testing.assert_close(scaled_unit, unit, rtol=2e-6, atol=1e-7)
    torch.testing.assert_close(scaled_rms, rms * factor, rtol=2e-6, atol=0)


def test_rms_and_unit_gradients_remain_finite_including_observed_zero_trace() -> None:
    values = torch.tensor([[1.0, -2.0, 3.0], [0.0, 0.0, 0.0], [50.0, 50.0, 50.0]])
    values.requires_grad_()
    unit, rms = normalize_observed_trace_rms(values, torch.tensor([True, True, False]))
    (unit.square().sum() + rms.sum()).backward()
    assert torch.isfinite(values.grad).all()
    assert values.grad[0].abs().sum() > 0
    assert torch.equal(values.grad[2], torch.zeros(3))


def _gain_example():
    # Sender 0 occurs in two relations into query 3. Sender 2 is only a
    # second-hop dependency for query 3, and a direct sender into query 4.
    rms = torch.tensor([10.0, 30.0, 1000.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    edges = torch.tensor([[0, 1, 0, 2, 2], [3, 3, 3, 0, 4]])
    distance = torch.tensor([1.0, 2.0, 1.0, 0.1, 2.0], dtype=torch.float64)
    return rms, edges, distance


def test_direct_sender_union_excludes_second_hop_and_other_query_support() -> None:
    rms, edges, distance = _gain_example()
    actual = interpolate_query_trace_rms(rms, edges, torch.tensor([3, 4, 5]), distance)
    # (10 / 1**2 + 30 / 2**2) / (1 / 1**2 + 1 / 2**2) = 14.
    assert torch.equal(actual, torch.tensor([14.0, 1000.0, 0.0], dtype=torch.float64))
    without_duplicate = torch.tensor([0, 1, 3, 4])
    deduplicated = interpolate_query_trace_rms(
        rms, edges[:, without_duplicate], torch.tensor([3, 4, 5]), distance[without_duplicate]
    )
    assert torch.equal(actual, deduplicated)


def test_query_order_edge_order_and_batching_preserve_gains() -> None:
    rms, edges, distance = _gain_example()
    joint = interpolate_query_trace_rms(rms, edges, torch.tensor([4, 5, 3]), distance)
    edge_order = torch.tensor([4, 3, 2, 1, 0])
    reordered = interpolate_query_trace_rms(
        rms, edges[:, edge_order], torch.tensor([4, 5, 3]), distance[edge_order]
    )
    separate = torch.cat(
        [interpolate_query_trace_rms(rms, edges, torch.tensor([q]), distance) for q in [4, 5, 3]]
    )
    assert torch.equal(joint, reordered) and torch.equal(joint, separate)


def test_zero_distance_floor_includes_zero_rms_sender() -> None:
    actual = interpolate_query_trace_rms(
        torch.tensor([0.0, 8.0, 0.0], dtype=torch.float64),
        torch.tensor([[0, 1], [2, 2]]),
        torch.tensor([2]),
        torch.tensor([0.0, 1e-7], dtype=torch.float64),
    )
    assert torch.equal(actual, torch.tensor([4.0], dtype=torch.float64))


def test_large_finite_rms_is_not_multiplied_by_unnormalized_idw_weight() -> None:
    actual = interpolate_query_trace_rms(
        torch.tensor([1e30, 3e30, 0.0]),
        torch.tensor([[0, 1], [2, 2]]),
        torch.tensor([2]),
        torch.tensor([0.0, 0.0]),
    )
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, torch.tensor([2e30]), rtol=1e-6, atol=0)


def test_large_finite_distances_do_not_underflow_every_weight_to_zero() -> None:
    actual = interpolate_query_trace_rms(
        torch.tensor([10.0, 30.0, 0.0]),
        torch.tensor([[0, 1], [2, 2]]),
        torch.tensor([2]),
        torch.tensor([1e30, 2e30]),
    )
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, torch.tensor([14.0]), rtol=1e-6, atol=0)


@pytest.mark.parametrize("queries", [[], [3], [3, 4]])
def test_no_edges_returns_zero_gains_in_query_order(queries) -> None:
    result = interpolate_query_trace_rms(
        torch.ones(5),
        torch.empty((2, 0), dtype=torch.int64),
        torch.tensor(queries, dtype=torch.int64),
        torch.empty(0),
    )
    assert torch.equal(result, torch.zeros(len(queries)))


def test_nonquery_edges_do_not_provide_context() -> None:
    rms, edges, distance = _gain_example()
    result = interpolate_query_trace_rms(rms, edges, torch.tensor([5]), distance)
    assert torch.equal(result, torch.zeros(1, dtype=torch.float64))


def test_query_gain_gradient_counts_each_sender_once_and_excludes_other_support() -> None:
    rms, edges, distance = _gain_example()
    rms.requires_grad_()
    gain = interpolate_query_trace_rms(rms, edges, torch.tensor([3]), distance)
    gain.sum().backward()
    torch.testing.assert_close(
        rms.grad, torch.tensor([0.8, 0.2, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    )
