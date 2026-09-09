"""An owned spatial index preserves exact graph edges and visibility boundaries."""

from collections import Counter
from dataclasses import fields

import numpy as np
import pytest

from seis_interp.processing import trace_graph_neighbors as neighbors_module
from seis_interp.processing.trace_graph_geometry import (
    TraceGraphGeometry,
    compute_trace_graph_geometry,
)
from seis_interp.processing.trace_graph_neighbors import (
    FixedTraceGraphNeighborIndex,
    select_trace_graph_neighbors,
)
from seis_interp.processing.trace_graph_subgraphs import (
    FixedTraceGraphSubgraphBuilder,
    build_trace_graph_subgraph,
)


def _geometry(source, receiver=None):
    source = np.asarray(source, dtype=np.float64).reshape(-1, 2)
    receiver = source - [0.0, 80.0] if receiver is None else np.asarray(receiver, dtype=np.float64)
    return compute_trace_graph_geometry(source, receiver, azimuth_min_offset_m=1.0)


def _select(geometry, rows):
    return TraceGraphGeometry(
        **{field.name: getattr(geometry, field.name)[rows] for field in fields(geometry)}
    )


def _case():
    rng = np.random.default_rng(927)
    source = rng.integers(-5, 6, size=(113, 2)) * [160.0, 80.0]
    receiver = source + rng.integers(-6, 7, size=(113, 2)) * 40.0
    geometry = _geometry(source, receiver)
    ids = rng.permutation(np.arange(1000, 1113, dtype=np.int64))
    observed = rng.random(113) < 0.75
    allowed = rng.random(113) < 0.9
    observed[[2, 17, 56]] = False
    queries = _select(geometry, [2, 17, 56])
    settings = {
        "relation_scales_m": [[160.0, 640.0], [640.0, 160.0], [160.0, 640.0], [640.0, 160.0]],
        "neighbors_per_relation": 3,
        "allowed_mask": allowed,
        "radius": 1.0,
        "common_distance_scales_m": [320.0, 320.0],
    }
    return geometry, ids, observed, queries, ids[[2, 17, 56]], settings


def _assert_edges(actual, expected):
    for field in fields(expected):
        np.testing.assert_array_equal(getattr(actual, field.name), getattr(expected, field.name))


def _assert_plan(actual, expected):
    for field in fields(expected):
        left, right = getattr(actual, field.name), getattr(expected, field.name)
        if isinstance(right, TraceGraphGeometry):
            _assert_edges(left, right)
        elif isinstance(right, np.ndarray):
            np.testing.assert_array_equal(left, right)
        else:
            assert left == right
    np.testing.assert_array_equal(actual.coverage, expected.coverage)


@pytest.mark.parametrize("chunk_size", [1, 7, 4096])
@pytest.mark.parametrize(
    "variant",
    [{}, {"excluded_relation": "receiver"}, {"topology": "single_4d", "single_4d_neighbors": 5}],
)
def test_index_matches_all_brute_arrays_with_self_hidden_allowed_and_offgrid(chunk_size, variant):
    geometry, ids, observed, queries, query_ids, settings = _case()
    settings.update(candidate_chunk_size=chunk_size, **variant)
    # Direct neighbor selection permits a visible destination, but never self edges.
    rows = [2, 17, 56, int(np.flatnonzero(observed & settings["allowed_mask"])[0])]
    queries = _select(geometry, rows)
    query_ids = ids[rows].copy()
    query_ids[0] = -17  # An off-grid ID needs no amplitude row correspondence.
    index = FixedTraceGraphNeighborIndex(geometry, ids, observed, **settings)
    actual = index.select(queries, query_ids)
    expected = select_trace_graph_neighbors(queries, query_ids, geometry, ids, observed, **settings)
    _assert_edges(actual, expected)
    assert not np.any(actual.sender_ids == actual.destination_ids)
    assert np.all(np.isin(actual.sender_ids, ids[observed & settings["allowed_mask"]]))


@pytest.mark.parametrize("rounds", [1, 2, 3])
@pytest.mark.parametrize(
    "variant",
    [{}, {"excluded_relation": "cmp"}, {"topology": "single_4d", "single_4d_neighbors": 5}],
)
def test_builder_matches_every_plan_array_in_both_query_orders_and_splits(rounds, variant):
    geometry, ids, observed, queries, query_ids, settings = _case()
    settings.update(variant)
    builder = FixedTraceGraphSubgraphBuilder(geometry, ids, observed, **settings)
    for order in ([0, 1, 2], [2, 1, 0], [0], [1, 2]):
        selected = _select(queries, order)
        wanted = build_trace_graph_subgraph(
            selected, query_ids[order], geometry, ids, observed, rounds=rounds, **settings
        )
        actual = builder.build(selected, query_ids[order], rounds=rounds)
        _assert_plan(actual, wanted)


def test_inclusive_radius_ties_and_nextafter_values_match_brute():
    query = _geometry([[0.0, 0.0]], [[0.0, -1000.0]])
    x = np.array([640.0, np.nextafter(640.0, 0.0), np.nextafter(640.0, np.inf), -640.0])
    geometry = _geometry(np.zeros((4, 2)), np.column_stack((x, np.full(4, -1000.0))))
    ids = np.array([40, 30, 20, 10])
    settings = {"relation_scales_m": [[160.0, 640.0]] * 4, "neighbors_per_relation": 4}
    index = FixedTraceGraphNeighborIndex(geometry, ids, np.ones(4, bool), **settings)
    actual = index.select(query, [-1])
    expected = select_trace_graph_neighbors(
        query, [-1], geometry, ids, np.ones(4, bool), **settings
    )
    _assert_edges(actual, expected)
    relation = actual.sender_ids[actual.edge_type == 0]
    assert 40 in relation and 10 in relation and 20 not in relation


@pytest.mark.parametrize("scale", [1e-170, 1e-150, 1.0, 1e160])
@pytest.mark.parametrize("radius", [1e-200, 1.0, 1e200])
def test_extreme_finite_scales_retain_brute_rounding_and_underflow_behavior(scale, radius):
    geometry = _geometry([[0, 0], [1e-160, 0], [1, 0], [1e150, 0]])
    query = _geometry([[0, 0]])
    settings = {"relation_scales_m": [[scale, scale]] * 4, "radius": radius}
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        actual = FixedTraceGraphNeighborIndex(
            geometry, np.arange(4), np.ones(4, bool), **settings
        ).select(query, [-1])
        expected = select_trace_graph_neighbors(
            query, [-1], geometry, np.arange(4), np.ones(4, bool), **settings
        )
    _assert_edges(actual, expected)


def test_builder_owns_geometry_ids_masks_and_settings_and_new_mask_changes_edges():
    geometry, ids, observed, queries, query_ids, settings = _case()
    builder = FixedTraceGraphSubgraphBuilder(geometry, ids, observed, **settings)
    expected = build_trace_graph_subgraph(
        queries, query_ids, geometry, ids, observed, rounds=2, **settings
    )
    next_observed = observed.copy()
    removed = expected.trace_ids[expected.observed_mask]
    next_observed[np.isin(ids, removed)] = False
    next_builder = FixedTraceGraphSubgraphBuilder(geometry, ids, next_observed, **settings)
    next_expected = build_trace_graph_subgraph(
        queries, query_ids, geometry, ids, next_observed, rounds=2, **settings
    )
    assert not np.array_equal(expected.edge_index, next_expected.edge_index)
    settings["relation_scales_m"][0][0] = 1e6
    settings["allowed_mask"][:] = False
    observed[:] = False
    ids[:] += 10000
    geometry.source_xy_m[:] += 1e6
    _assert_plan(builder.build(queries, query_ids, rounds=2), expected)
    _assert_plan(next_builder.build(queries, query_ids, rounds=2), next_expected)


def test_scaled_squared_distance_underflow_does_not_remove_a_brute_neighbor():
    query = _geometry([[0, 0]], [[0, -80]])
    geometry = _geometry([[1e-20, 0]], [[0, -80]])
    settings = {"relation_scales_m": [[1e150, 1e150]] * 4, "radius": 1e-200}
    actual = FixedTraceGraphNeighborIndex(geometry, [10], [True], **settings).select(query, [-1])
    expected = select_trace_graph_neighbors(query, [-1], geometry, [10], [True], **settings)
    _assert_edges(actual, expected)
    np.testing.assert_array_equal(actual.sender_ids, [10] * 4)


def test_builder_offgrid_no_context_and_empty_candidates_match_brute():
    for candidates in (_geometry([]), _geometry([[0, 0], [160, 0]])):
        ids = np.arange(len(candidates.offset_m))
        visible = np.ones(len(ids), bool)
        settings = {"relation_scales_m": [[160.0, 640.0]] * 4}
        query = _geometry([[1e6, 1e6]])
        builder = FixedTraceGraphSubgraphBuilder(candidates, ids, visible, **settings)
        _assert_plan(
            builder.build(query, [-1], rounds=2),
            build_trace_graph_subgraph(query, [-1], candidates, ids, visible, rounds=2, **settings),
        )


@pytest.mark.parametrize("rounds", [0, -1, True, 1.5])
def test_builder_rejects_invalid_rounds(rounds):
    geometry, ids, observed, queries, query_ids, settings = _case()
    builder = FixedTraceGraphSubgraphBuilder(geometry, ids, observed, **settings)
    with pytest.raises(ValueError, match="rounds must be a positive integer"):
        builder.build(queries, query_ids, rounds=rounds)


def test_builder_rejects_visible_query_and_changed_geometry_for_shared_id():
    geometry, ids, observed, queries, query_ids, settings = _case()
    builder = FixedTraceGraphSubgraphBuilder(geometry, ids, observed, **settings)
    row = int(np.flatnonzero(observed & settings["allowed_mask"])[0])
    with pytest.raises(ValueError, match="must not be eligible observed"):
        builder.build(_select(geometry, [row]), ids[[row]], rounds=2)
    changed = _geometry(queries.source_xy_m + 1.0, queries.receiver_xy_m)
    with pytest.raises(ValueError, match="must have identical geometry"):
        builder.build(changed, query_ids, rounds=2)


def test_index_prunes_distant_candidates_before_exact_distance_work(monkeypatch):
    geometry = _geometry(np.column_stack((np.arange(1000) * 100.0, np.zeros(1000))))
    settings = {"relation_scales_m": [[160.0, 640.0]] * 4, "candidate_chunk_size": 4096}
    index = FixedTraceGraphNeighborIndex(geometry, np.arange(1000), np.ones(1000, bool), **settings)
    counts = []
    original = neighbors_module._relation_distances

    def capture(destination, row, candidates, rows, scales):
        counts.append(len(rows))
        return original(destination, row, candidates, rows, scales)

    monkeypatch.setattr(neighbors_module, "_relation_distances", capture)
    index.select(_geometry([[50, 0]]), [-1])
    assert counts and max(counts) < 10


@pytest.mark.parametrize("rounds", [2, 3])
@pytest.mark.parametrize(
    "variant",
    [{}, {"excluded_relation": "cmp"}, {"topology": "single_4d", "single_4d_neighbors": 5}],
)
def test_observed_cache_reuses_searches_across_batches_without_caching_queries(
    monkeypatch, rounds, variant
):
    geometry, ids, observed, queries, query_ids, settings = _case()
    settings.update(variant)
    eligible_ids = set(ids[observed & settings["allowed_mask"]].tolist())
    observed_searches, query_searches = Counter(), Counter()
    original = FixedTraceGraphNeighborIndex.select

    def capture(index, selected_geometry, selected_ids):
        for trace_id in selected_ids:
            counter = observed_searches if int(trace_id) in eligible_ids else query_searches
            counter[int(trace_id)] += 1
        return original(index, selected_geometry, selected_ids)

    monkeypatch.setattr(FixedTraceGraphNeighborIndex, "select", capture)
    builder = FixedTraceGraphSubgraphBuilder(geometry, ids, observed, **settings)
    assert builder.observed_neighbor_cache_info()["cached_sender_count"] == 0
    expected_query_calls = Counter()
    for order in ([0, 1], [1, 2], [2, 1, 0], [0, 1, 2]):
        selected_ids = query_ids[order]
        expected_query_calls.update(selected_ids.tolist())
        actual = builder.build(_select(queries, order), selected_ids, rounds=rounds)
        expected = build_trace_graph_subgraph(
            _select(queries, order),
            selected_ids,
            geometry,
            ids,
            observed,
            rounds=rounds,
            **settings,
        )
        _assert_plan(actual, expected)
    assert observed_searches and all(count == 1 for count in observed_searches.values())
    assert query_searches == expected_query_calls
    info = builder.observed_neighbor_cache_info()
    assert info["cached_sender_count"] == len(observed_searches) <= len(eligible_ids)
    assert info["cached_edge_count"] <= info["edge_capacity"]
    assert info["cached_array_bytes"] == info["cached_edge_count"] * 24


def test_cached_plans_and_offgrid_queries_do_not_share_mutable_arrays(monkeypatch):
    geometry, ids, observed, queries, _, settings = _case()
    builder = FixedTraceGraphSubgraphBuilder(geometry, ids, observed, **settings)
    query_ids = np.array([-11], dtype=np.int64)
    searches = []
    original = FixedTraceGraphNeighborIndex.select

    def capture(index, selected_geometry, selected_ids):
        if -11 in selected_ids:
            searches.append(selected_ids.copy())
        return original(index, selected_geometry, selected_ids)

    monkeypatch.setattr(FixedTraceGraphNeighborIndex, "select", capture)
    first_geometry = _select(queries, [0])
    first = builder.build(first_geometry, query_ids, rounds=3)
    assert builder.observed_neighbor_cache_info()["cached_sender_count"] > 0
    for field in fields(first):
        value = getattr(first, field.name)
        if isinstance(value, np.ndarray):
            value[...] = 0
        elif isinstance(value, TraceGraphGeometry):
            for child in fields(value):
                getattr(value, child.name)[...] = 0
    for selected in (first_geometry, _select(queries, [1]), first_geometry):
        actual = builder.build(selected, query_ids, rounds=3)
        expected = build_trace_graph_subgraph(
            selected, query_ids, geometry, ids, observed, rounds=3, **settings
        )
        _assert_plan(actual, expected)
    assert len(searches) == 4


def test_new_episode_builder_does_not_reuse_cached_previous_visibility():
    geometry, ids, observed, queries, query_ids, settings = _case()
    first = FixedTraceGraphSubgraphBuilder(geometry, ids, observed, **settings)
    plan = first.build(queries, query_ids, rounds=3)
    assert first.observed_neighbor_cache_info()["cached_sender_count"] > 0
    changed = observed.copy()
    changed[np.isin(ids, plan.trace_ids[plan.observed_mask])] = False
    second = FixedTraceGraphSubgraphBuilder(geometry, ids, changed, **settings)
    assert second.observed_neighbor_cache_info()["cached_sender_count"] == 0
    for builder, mask in ((first, observed), (second, changed), (first, observed)):
        actual = builder.build(queries, query_ids, rounds=3)
        expected = build_trace_graph_subgraph(
            queries, query_ids, geometry, ids, mask, rounds=3, **settings
        )
        _assert_plan(actual, expected)


def test_observed_sender_with_no_incoming_edges_is_cached(monkeypatch):
    geometry = _geometry([[0, 0]])
    query = _geometry([[0.1, 0]])
    settings = {"relation_scales_m": [[1.0, 1.0]] * 4}
    builder = FixedTraceGraphSubgraphBuilder(geometry, [10], [True], **settings)
    searched = []
    original = FixedTraceGraphNeighborIndex.select

    def capture(index, selected_geometry, selected_ids):
        searched.extend(selected_ids.tolist())
        return original(index, selected_geometry, selected_ids)

    monkeypatch.setattr(FixedTraceGraphNeighborIndex, "select", capture)
    for _ in range(2):
        actual = builder.build(query, np.array([-1]), rounds=3)
        expected = build_trace_graph_subgraph(
            query, [-1], geometry, [10], [True], rounds=3, **settings
        )
        _assert_plan(actual, expected)
    assert searched.count(10) == 1 and searched.count(-1) == 2
    assert builder.observed_neighbor_cache_info() == {
        "cached_sender_count": 1,
        "cached_edge_count": 0,
        "cached_array_bytes": 0,
        "eligible_sender_count": 1,
        "edge_capacity": 32,
    }
