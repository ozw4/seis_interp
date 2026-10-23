"""The precomputed neighbor table reproduces exact dependency plans bit for bit."""

from dataclasses import fields

import numpy as np
import pytest

from seis_interp.processing.trace_graph_geometry import compute_trace_graph_geometry
from seis_interp.processing.trace_graph_neighbor_table import TraceGraphNeighborTable
from seis_interp.processing.trace_graph_neighbors import TraceGraphSpatialIndex
from seis_interp.processing.trace_graph_subgraphs import FixedTraceGraphSubgraphBuilder

_SCALES = np.array([[4.0, 9.0], [9.0, 4.0], [4.0, 9.0], [9.0, 4.0]])


def _survey(count: int = 360, seed: int = 0):
    rng = np.random.default_rng(seed)
    # Integer coordinates create many equal distances, so the ID tie-break matters.
    source = np.round(rng.uniform(0, 24, (count, 2)))
    receiver = source + np.round(rng.uniform(-6, 6, (count, 2)))
    trace_ids = rng.permutation(np.arange(10, 10 + 3 * count, 3)).astype(np.int64)
    return source, receiver, trace_ids


def _geometry(source, receiver):
    return compute_trace_graph_geometry(source, receiver, azimuth_min_offset_m=0.5)


def _assert_same_plan(actual, expected):
    for field in fields(expected):
        value, reference = getattr(actual, field.name), getattr(expected, field.name)
        if field.name == "geometry":
            for part in fields(reference):
                left, right = getattr(value, part.name), getattr(reference, part.name)
                assert left.dtype == right.dtype and left.tobytes() == right.tobytes(), part.name
        elif isinstance(reference, np.ndarray):
            assert value.dtype == reference.dtype and value.shape == reference.shape, field.name
            assert value.tobytes() == reference.tobytes(), field.name
        else:
            assert value == reference, field.name


@pytest.mark.parametrize("neighbors", [1, 2, 3, 4, 5, 6])
def test_table_plans_equal_exact_plans_for_many_episodes(neighbors, monkeypatch):
    source, receiver, trace_ids = _survey()
    geometry = _geometry(source, receiver)
    index = TraceGraphSpatialIndex(
        geometry,
        trace_ids,
        relation_scales_m=_SCALES,
        neighbors_per_relation=neighbors,
        candidate_chunk_size=64,
    )
    # A one-entry margin truncates lists often enough to exercise the exact fallback.
    table = TraceGraphNeighborTable(index, margin=1, destination_chunk_size=97)
    fallbacks = []
    original_select = index._select

    def counted_select(destination_geometry, destination_ids, eligible):
        fallbacks.append(len(destination_ids))
        return original_select(destination_geometry, destination_ids, eligible)

    monkeypatch.setattr(index, "_select", counted_select)
    rng = np.random.default_rng(neighbors)
    fallback_calls = 0
    for fraction in (0.05, 0.3, 0.7):
        for _ in range(3):
            hidden = rng.choice(len(trace_ids), size=int(len(trace_ids) * fraction), replace=False)
            visible = np.ones(len(trace_ids), dtype=bool)
            visible[hidden] = False
            exact = FixedTraceGraphSubgraphBuilder.from_spatial_index(index, visible)
            tabled = FixedTraceGraphSubgraphBuilder.from_spatial_index(
                index, visible, neighbor_table=table
            )
            for rounds in (1, 2, 3):
                queries = rng.permutation(hidden)[:13]
                query_geometry = _geometry(source[queries], receiver[queries])
                expected = exact.build(query_geometry, trace_ids[queries], rounds=rounds)
                searched = len(fallbacks)
                actual = tabled.build(query_geometry, trace_ids[queries], rounds=rounds)
                fallback_calls += len(fallbacks) - searched
                _assert_same_plan(actual, expected)
                options = {"fanout_per_relation": max(1, neighbors - 1)}
                _assert_same_plan(
                    tabled.build(
                        query_geometry,
                        trace_ids[queries],
                        rounds=rounds,
                        rng=np.random.default_rng(5),
                        **options,
                    ),
                    exact.build(
                        query_geometry,
                        trace_ids[queries],
                        rounds=rounds,
                        rng=np.random.default_rng(5),
                        **options,
                    ),
                )
    assert table.neighbors == neighbors
    assert fallback_calls > 0


def test_fallback_search_runs_when_visible_lists_may_be_truncated(monkeypatch):
    source, receiver, trace_ids = _survey(seed=4)
    index = TraceGraphSpatialIndex(
        _geometry(source, receiver), trace_ids, relation_scales_m=_SCALES, neighbors_per_relation=3
    )
    table = TraceGraphNeighborTable(index, margin=1)
    searched = []
    original_select = index._select

    def counted_select(destination_geometry, destination_ids, eligible):
        searched.append(len(destination_ids))
        return original_select(destination_geometry, destination_ids, eligible)

    visible = np.zeros(len(trace_ids), dtype=bool)
    visible[::2] = True
    queries = np.flatnonzero(~visible)[:20]
    query_geometry = _geometry(source[queries], receiver[queries])
    expected = FixedTraceGraphSubgraphBuilder.from_spatial_index(index, visible).build(
        query_geometry, trace_ids[queries], rounds=2
    )
    monkeypatch.setattr(index, "_select", counted_select)
    actual = FixedTraceGraphSubgraphBuilder.from_spatial_index(
        index, visible, neighbor_table=table
    ).build(query_geometry, trace_ids[queries], rounds=2)
    _assert_same_plan(actual, expected)
    # Some destinations needed the exact search, the others were served by the table.
    destinations = len(queries) + np.count_nonzero(actual.depth == 1)
    assert 0 < sum(searched) < destinations


def test_table_over_observed_traces_serves_a_larger_prediction_domain():
    source, receiver, trace_ids = _survey(seed=7)
    observed = np.random.default_rng(7).random(len(trace_ids)) < 0.6
    settings = {"relation_scales_m": _SCALES, "neighbors_per_relation": 2}
    observed_index = TraceGraphSpatialIndex(
        _geometry(source[observed], receiver[observed]), trace_ids[observed], **settings
    )
    table = TraceGraphNeighborTable(observed_index)
    candidate_geometry = _geometry(source, receiver)
    exact = FixedTraceGraphSubgraphBuilder(candidate_geometry, trace_ids, observed, **settings)
    tabled = FixedTraceGraphSubgraphBuilder(
        candidate_geometry, trace_ids, observed, neighbor_table=table, **settings
    )
    targets = np.flatnonzero(~observed)
    for start in range(0, len(targets), 31):
        queries = targets[start : start + 31]
        query_geometry = _geometry(source[queries], receiver[queries])
        _assert_same_plan(
            tabled.build(query_geometry, trace_ids[queries], rounds=2),
            exact.build(query_geometry, trace_ids[queries], rounds=2),
        )
    # Off-domain queries take the general path with the same result.
    off_grid = _geometry(np.array([[3.5, 4.5]]), np.array([[5.5, 2.5]]))
    _assert_same_plan(
        tabled.build(off_grid, np.array([99999]), rounds=2),
        exact.build(off_grid, np.array([99999]), rounds=2),
    )


def test_table_refuses_views_that_could_change_neighbors():
    source, receiver, trace_ids = _survey(count=80, seed=2)
    geometry = _geometry(source, receiver)
    settings = {"relation_scales_m": _SCALES, "neighbors_per_relation": 2}
    index = TraceGraphSpatialIndex(geometry, trace_ids, **settings)
    table = TraceGraphNeighborTable(index)
    everyone = np.ones(len(trace_ids), dtype=bool)
    other_k = TraceGraphSpatialIndex(
        geometry, trace_ids, **{**settings, "neighbors_per_relation": 3}
    )
    with pytest.raises(ValueError, match="search settings"):
        table.view(other_k, everyone)
    moved = _geometry(source + 0.25, receiver)
    with pytest.raises(ValueError, match="geometry"):
        table.view(TraceGraphSpatialIndex(moved, trace_ids, **settings), everyone)
    subset = TraceGraphNeighborTable(
        TraceGraphSpatialIndex(_geometry(source[:40], receiver[:40]), trace_ids[:40], **settings)
    )
    with pytest.raises(ValueError, match="every eligible candidate"):
        subset.view(index, everyone)
    with pytest.raises(ValueError, match="positive integer"):
        TraceGraphNeighborTable(index, margin=0)
