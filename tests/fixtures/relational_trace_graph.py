"""Small irregular trace domains and dependency plans for input tests."""

from __future__ import annotations

import numpy as np

from seis_interp.data.trace_graph_domain import TraceGraphDomain, build_trace_graph_domain
from seis_interp.processing.trace_graph_geometry import compute_trace_graph_geometry
from seis_interp.processing.trace_graph_subgraphs import TraceGraphPlan, build_trace_graph_subgraph


def make_relational_trace_domains() -> tuple[TraceGraphDomain, TraceGraphDomain, np.ndarray]:
    source = np.column_stack(([0.0, 0.5, 1.0, 1.5, 2.0, 4.0, 50.0], np.zeros(7)))
    receiver = source + [0.0, -2.0]
    ids = np.array([40, 10, 60, 20, 30, 50, 70], dtype=np.int64)
    rows = np.array([4, 0, 5, 1, 2, 3, 6], dtype=np.int64)
    ffids = np.array([2, 2, 5, 7, 7, 8, 9], dtype=np.int64)
    time_s = np.arange(1, 6, dtype=np.float64) * 0.01
    amplitudes = np.arange(1, 8, dtype=np.float32)[:, None] * np.arange(1, 8, dtype=np.float32)
    domain = build_trace_graph_domain(
        trace_ids=ids,
        source_xy_m=source,
        receiver_xy_m=receiver,
        ffids=ffids,
        array_rows=rows,
        observed_mask=np.array([True, True, True, False, False, True, True]),
        time_s=time_s,
        time_samples=(1, 6),
        inputs_lock={"partition": "test", "fixture": "irregular_traces"},
    )
    training = build_trace_graph_domain(
        trace_ids=ids[:3],
        source_xy_m=source[:3],
        receiver_xy_m=receiver[:3],
        ffids=ffids[:3],
        array_rows=rows[:3],
        observed_mask=np.ones(3, dtype=bool),
        time_s=time_s,
        time_samples=(1, 6),
        pool="mask_observed",
        inputs_lock={"partition": "train", "fixture": "training_traces"},
    )
    return domain, training, amplitudes


def make_relational_trace_plan(
    domain: TraceGraphDomain, query_ids: np.ndarray | None = None
) -> TraceGraphPlan:
    if query_ids is None:
        query_ids = domain.trace_ids[domain.query_mask]
    positions = {int(trace_id): index for index, trace_id in enumerate(domain.trace_ids)}
    query_rows = [positions[int(trace_id)] for trace_id in query_ids]
    geometry = compute_trace_graph_geometry(
        domain.source_xy_m, domain.receiver_xy_m, azimuth_min_offset_m=0.1
    )
    queries = compute_trace_graph_geometry(
        domain.source_xy_m[query_rows], domain.receiver_xy_m[query_rows], azimuth_min_offset_m=0.1
    )
    return build_trace_graph_subgraph(
        queries,
        query_ids,
        geometry,
        domain.trace_ids,
        domain.observed_mask,
        rounds=2,
        relation_scales_m=np.ones((4, 2)),
        neighbors_per_relation=2,
        candidate_chunk_size=2,
    )
