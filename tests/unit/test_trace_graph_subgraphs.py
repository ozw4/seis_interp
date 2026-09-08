"""Dependency and scalar synchronous-update oracles for trace graph closure."""

from __future__ import annotations

import math

import numpy as np
import pytest

from seis_interp.processing.trace_graph_geometry import compute_trace_graph_geometry
from seis_interp.processing.trace_graph_neighbors import select_trace_graph_neighbors
from seis_interp.processing.trace_graph_subgraphs import build_trace_graph_subgraph

SCALES = np.array([[1.4, 1.4], [1.4, 1.4], [1.0, 1.0], [1.0, 1.0]])


def _geometry(x):
    source = np.column_stack((x, np.zeros(len(x))))
    return compute_trace_graph_geometry(source, source - [0.0, 10.0], azimuth_min_offset_m=0.0)


def _build(query_x=(0.0,), query_ids=(100,), rounds=2, **kwargs):
    candidates = _geometry([0.8, 1.6, 2.4, 3.2, 4.0])
    return build_trace_graph_subgraph(
        _geometry(query_x),
        query_ids,
        candidates,
        np.arange(1, 6),
        np.ones(5, dtype=bool),
        rounds=rounds,
        relation_scales_m=SCALES,
        neighbors_per_relation=2,
        **kwargs,
    )


def _full_graph(query_x, query_ids):
    candidate_x = np.array([0.8, 1.6, 2.4, 3.2, 4.0])
    node_ids = np.concatenate((query_ids, np.arange(1, 6)))
    edges = select_trace_graph_neighbors(
        _geometry(np.concatenate((query_x, candidate_x))),
        node_ids,
        _geometry(candidate_x),
        np.arange(1, 6),
        np.ones(5, dtype=bool),
        relation_scales_m=SCALES,
        neighbors_per_relation=2,
    )
    records = list(
        zip(edges.sender_ids, edges.destination_ids, edges.edge_type, edges.distances, strict=True)
    )
    return node_ids, records


def _plan_records(plan):
    return list(
        zip(
            plan.trace_ids[plan.edge_index[0]],
            plan.trace_ids[plan.edge_index[1]],
            plan.edge_type,
            plan.edge_distances,
            strict=True,
        )
    )


def _reverse_closure(query_ids, records, rounds):
    depths = dict.fromkeys(query_ids, 0)
    for depth in range(rounds):
        current = {trace_id for trace_id, value in depths.items() if value == depth}
        for sender, destination, _, _ in records:
            if destination in current and sender not in depths:
                depths[sender] = depth + 1
    return depths


def _synchronous_scalar(node_ids, query_ids, records, rounds):
    """Independent per-node recurrence using old states and complete coverage."""
    states = {
        trace_id: 0.0 if trace_id in query_ids else 0.2 + 0.07 * trace_id for trace_id in node_ids
    }
    for _ in range(rounds):
        next_states = {}
        for destination in node_ids:
            total = 0.4 * states[destination]
            for relation in range(4):
                incoming = [
                    (sender, distance)
                    for sender, receiver, edge_type, distance in records
                    if receiver == destination and edge_type == relation
                ]
                if incoming:
                    total += (
                        (relation + 1)
                        * sum(
                            states[sender] * (1.0 - 0.1 * distance) for sender, distance in incoming
                        )
                        / (10.0 * len(incoming))
                    )
                    total += 0.03 * len(incoming) / 2 - 0.02 * min(d for _, d in incoming)
            next_states[destination] = math.tanh(total)
        states = next_states
    return states


@pytest.mark.parametrize("rounds", [1, 2, 3])
def test_closure_matches_full_domain_reverse_reachability_and_stops_at_leaves(rounds):
    plan = _build(rounds=rounds)
    _, full_records = _full_graph([0.0], [100])
    expected_depths = _reverse_closure([100], full_records, rounds)

    assert plan.dependency_rounds == rounds
    assert dict(zip(plan.trace_ids, plan.depth, strict=True)) == expected_depths
    assert set(plan.trace_ids) == {100, *range(1, rounds + 1)}
    assert np.all(plan.observed_mask[plan.edge_index[0]])
    assert np.all(plan.depth[plan.edge_index[1]] < rounds)
    for trace_id, depth in expected_depths.items():
        actual = [record for record in _plan_records(plan) if record[1] == trace_id]
        expected = [record for record in full_records if record[1] == trace_id]
        assert actual == (expected if depth < rounds else [])


@pytest.mark.parametrize("rounds", [1, 2, 3])
def test_synchronous_scalar_query_result_equals_full_graph_with_nonzero_messages(rounds):
    plan = _build(rounds=rounds)
    full_ids, full_records = _full_graph([0.0], [100])
    full_values = _synchronous_scalar(full_ids, [100], full_records, rounds)
    local_values = _synchronous_scalar(plan.trace_ids, [100], _plan_records(plan), rounds)

    assert full_values[100] > 0.05
    assert local_values[100] == pytest.approx(full_values[100], abs=1e-14)
    leaf_id = plan.trace_ids[plan.depth == rounds][0]
    assert any(record[1] == leaf_id for record in full_records)
    assert not any(record[1] == leaf_id for record in _plan_records(plan))


def test_query_addition_order_and_split_preserve_dependencies_and_scalar_values():
    separate_first = _build(query_x=[0.0], query_ids=[100])
    separate_second = _build(query_x=[2.0], query_ids=[200])
    together = _build(query_x=[0.0, 2.0], query_ids=[100, 200])
    reordered = _build(query_x=[2.0, 0.0], query_ids=[200, 100])
    full_ids, full_records = _full_graph([0.0, 2.0], [100, 200])
    full_values = _synchronous_scalar(full_ids, [100, 200], full_records, 2)

    # Node 2 is a boundary leaf for query 100, but depth 1 from query 200.
    assert separate_first.depth[np.flatnonzero(separate_first.trace_ids == 2)[0]] == 2
    assert together.depth[np.flatnonzero(together.trace_ids == 2)[0]] == 1
    assert not any(record[1] == 2 for record in _plan_records(separate_first))
    assert any(record[1] == 2 for record in _plan_records(together))
    for plan in (separate_first, separate_second, together, reordered):
        query_ids = plan.trace_ids[plan.query_indices]
        values = _synchronous_scalar(plan.trace_ids, query_ids, _plan_records(plan), 2)
        for query_id in query_ids:
            assert values[query_id] == pytest.approx(full_values[query_id], abs=1e-14)
    for field in ("trace_ids", "edge_index", "edge_type", "edge_distances", "degree", "depth"):
        np.testing.assert_array_equal(getattr(together, field), getattr(reordered, field))
    np.testing.assert_array_equal(together.trace_ids[together.query_indices], [100, 200])
    np.testing.assert_array_equal(reordered.trace_ids[reordered.query_indices], [200, 100])


def test_degree_minimum_coverage_and_diagnostics_use_full_selected_incoming_edges():
    plan = _build(rounds=3)
    _, full_records = _full_graph([0.0], [100])
    for row, trace_id in enumerate(plan.trace_ids):
        for relation in range(4):
            distances = [
                distance
                for _, destination, edge_type, distance in full_records
                if destination == trace_id and edge_type == relation and plan.depth[row] < 3
            ]
            assert plan.degree[row, relation] == len(distances)
            assert plan.min_distance[row, relation] == (min(distances) if distances else 0.0)
    np.testing.assert_allclose(plan.coverage[:, :, 0], plan.degree / 2)
    np.testing.assert_allclose(plan.coverage[:, :, 1], plan.min_distance)
    assert plan.coverage.dtype == np.float32
    assert plan.diagnostics == {
        "query_count": 1,
        "support_node_count": 3,
        "typed_edge_count": 16,
        "unique_pair_count": 4,
        "max_depth": 3,
    }


def test_invisible_ffid_and_disallowed_candidates_never_return_at_any_hop():
    ids = np.arange(1, 10)
    x = np.arange(1, 10) * 0.4
    ffid = np.array([9, 1, 9, 2, 9, 3, 9, 4, 9])
    allowed = ids != 8
    plan = build_trace_graph_subgraph(
        _geometry([0.0]),
        [100],
        _geometry(x),
        ids,
        ffid != 9,
        allowed_mask=allowed,
        rounds=5,
        relation_scales_m=SCALES,
        neighbors_per_relation=2,
        candidate_chunk_size=2,
    )

    assert set(plan.trace_ids) == {100, 2, 4, 6}
    assert set(plan.trace_ids[plan.edge_index[0]]) == {2, 4, 6}
    assert plan.diagnostics["max_depth"] == 3
    assert plan.dependency_rounds == 5
    assert np.all(plan.depth < 5)  # Observed cycles terminate after exhausting the domain.


@pytest.mark.parametrize("empty_queries", [False, True])
def test_empty_context_or_query_set_has_well_shaped_empty_edges(empty_queries):
    plan = build_trace_graph_subgraph(
        _geometry([] if empty_queries else [0.0]),
        [] if empty_queries else [100],
        _geometry([100.0]),
        [1],
        np.array([False]),
        rounds=3,
        relation_scales_m=SCALES,
    )

    count = 0 if empty_queries else 1
    assert plan.trace_ids.shape == (count,)
    assert plan.geometry.source_xy_m.shape == (count, 2)
    assert plan.edge_index.shape == (2, 0)
    assert plan.coverage.shape == (count, 4, 2)
    assert plan.diagnostics["unique_pair_count"] == 0
    assert plan.diagnostics["max_depth"] == 0
    assert plan.dependency_rounds == 3
    np.testing.assert_array_equal(plan.degree, np.zeros((count, 4)))
    np.testing.assert_array_equal(plan.min_distance, np.zeros((count, 4)))


def test_hidden_candidate_can_be_query_but_visible_query_is_rejected():
    kwargs = {
        "rounds": 2,
        "relation_scales_m": SCALES,
    }
    plan = build_trace_graph_subgraph(
        _geometry([0.0]),
        [1],
        _geometry([0.0, 0.8]),
        [1, 2],
        np.array([False, True]),
        **kwargs,
    )
    assert set(plan.trace_ids[plan.edge_index[0]]) == {2}
    for observed, query_x, message in (
        ([True, True], 0.0, "query IDs"),
        ([False, True], 0.1, "identical geometry"),
    ):
        with pytest.raises(ValueError, match=message):
            build_trace_graph_subgraph(
                _geometry([query_x]),
                [1],
                _geometry([0.0, 0.8]),
                [1, 2],
                np.array(observed),
                **kwargs,
            )


@pytest.mark.parametrize("rounds", [0, -1, 1.5, True])
def test_invalid_round_count_is_rejected(rounds):
    with pytest.raises(ValueError, match="rounds"):
        _build(rounds=rounds)
