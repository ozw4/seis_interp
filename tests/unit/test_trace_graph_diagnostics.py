"""Geometry diagnostics count query neighborhoods and undefined set overlap."""

from __future__ import annotations

import json

import numpy as np
import pytest

from seis_interp.processing.trace_graph_diagnostics import (
    summarize_trace_graph_queries,
    trace_graph_resource_measurements,
)
from seis_interp.processing.trace_graph_geometry import compute_trace_graph_geometry
from seis_interp.processing.trace_graph_subgraphs import build_trace_graph_subgraph


def _geometry(source, receiver=None):
    source = np.asarray(source, dtype=np.float64).reshape(-1, 2)
    receiver = source - [0, 1000] if receiver is None else receiver
    return compute_trace_graph_geometry(source, receiver, azimuth_min_offset_m=1)


@pytest.mark.parametrize("distinct", [False, True])
def test_query_degree_empty_and_jaccard_distinguish_distinct_relations(distinct):
    candidates = (
        _geometry(
            [[0, 0], [300, 0], [250, 0], [300, 0]],
            [[300, -1000], [0, -1000], [-250, -1000], [300, -1000]],
        )
        if distinct
        else _geometry([[1, 0], [2, 0], [3, 0], [4, 0]])
    )
    plan = build_trace_graph_subgraph(
        _geometry([[0, 0], [1e6, 0]]),
        [100, 101],
        candidates,
        [10, 20, 30, 40],
        np.ones(4, dtype=bool),
        rounds=1,
        relation_scales_m=np.array([[160, 640], [640, 160], [160, 640], [640, 160]]),
        neighbors_per_relation=1,
    )
    assert np.count_nonzero(plan.depth == 1) == (4 if distinct else 1)
    assert not np.any(np.isin(plan.edge_index[1], np.flatnonzero(plan.depth == 1)))
    summary = summarize_trace_graph_queries(plan)
    assert summary["query_count"] == 2
    assert summary["no_context_query_count"] == 1
    assert summary["no_context_fraction"] == 0.5
    assert summary["typed_edge_count"] == 4
    assert summary["unique_pair_count"] == (4 if distinct else 1)
    for relation in summary["relations"]:
        assert relation["degree_sum"] == 1
        assert relation["empty_query_count"] == 1
        assert relation["empty_fraction"] == 0.5
        assert relation["nearest_distance_count"] == 1
    for pair in summary["relation_jaccard"]:
        assert pair["mean_jaccard"] == (0.0 if distinct else 1.0)
        assert pair["defined_query_count"] == pair["both_empty_query_count"] == 1
    json.dumps(summary, allow_nan=False)


def test_all_empty_jaccard_is_null_and_spread_uses_physical_axes():
    candidates = _geometry([[-1, 2], [3, 5]])
    plan = build_trace_graph_subgraph(
        _geometry([[0, 0]]),
        [100],
        candidates,
        [1, 2],
        np.ones(2, dtype=bool),
        rounds=2,
        relation_scales_m=np.full((4, 2), 1000.0),
        neighbors_per_relation=2,
    )
    summary = summarize_trace_graph_queries(plan)
    for relation in summary["relations"]:
        for spread in relation["sender_spread_m"].values():
            assert spread["mean_xy"] == [4.0, 3.0]
            assert spread["max_xy"] == [4.0, 3.0]
    empty = build_trace_graph_subgraph(
        _geometry([[0, 0]]),
        [100],
        candidates,
        [1, 2],
        np.zeros(2, dtype=bool),
        rounds=2,
        relation_scales_m=np.ones((4, 2)),
    )
    for pair in summarize_trace_graph_queries(empty)["relation_jaccard"]:
        assert pair["mean_jaccard"] is None
        assert pair["both_empty_query_count"] == 1
        assert pair["defined_query_count"] == 0


def test_cpu_memory_is_measured_and_cuda_is_explicitly_unmeasured():
    measurements = trace_graph_resource_measurements("cpu")
    assert measurements["process_max_rss_bytes"] > 0
    assert measurements["cpu_memory_scope"] == "process_lifetime_peak"
    assert measurements["cuda_max_memory_allocated_bytes"] is None
    assert measurements["cuda_max_memory_reserved_bytes"] is None
    assert measurements["cuda_memory_scope"] == "unmeasured"
    json.dumps(measurements, allow_nan=False)
