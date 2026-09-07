"""Independent scalar-distance checks for exact typed incoming selection."""

from __future__ import annotations

import math

import numpy as np
import pytest

from seis_interp.processing.trace_graph_geometry import compute_trace_graph_geometry
from seis_interp.processing.trace_graph_neighbors import select_trace_graph_neighbors

SCALES = np.array([[160.0, 640.0], [640.0, 160.0], [160.0, 640.0], [640.0, 160.0]])


def _geometry(source, receiver=None):
    source = np.asarray(source, dtype=np.float64).reshape(-1, 2)
    receiver = source - [0.0, 1000.0] if receiver is None else receiver
    return compute_trace_graph_geometry(source, receiver, azimuth_min_offset_m=1.0)


def _oracle(destinations, destination_ids, candidates, candidate_ids, observed, allowed, k, radius):
    result = []
    for row, destination_id in enumerate(destination_ids):
        for relation in range(4):
            selected = []
            for other, candidate_id in enumerate(candidate_ids):
                if not observed[other] or not allowed[other] or candidate_id == destination_id:
                    continue
                if relation < 2:
                    first = "source_xy_m"
                    second = "receiver_xy_m"
                else:
                    first = "midpoint_xy_m"
                    second = "offset_xy_m"
                distance_squared = 0.0
                for name, scale in zip((first, second), SCALES[relation], strict=True):
                    for axis in range(2):
                        delta = float(getattr(candidates, name)[other, axis]) - float(
                            getattr(destinations, name)[row, axis]
                        )
                        distance_squared += delta**2 / float(scale) ** 2
                distance = math.sqrt(distance_squared)
                if distance <= radius:
                    selected.append((distance, int(candidate_id)))
            for distance, candidate_id in sorted(selected)[:k]:
                result.append((candidate_id, destination_id, relation, distance))
    return result


def _rows(edges):
    return list(
        zip(edges.sender_ids, edges.destination_ids, edges.edge_type, edges.distances, strict=True)
    )


def test_four_relations_select_only_their_intended_candidate() -> None:
    destination = _geometry([[0.0, 0.0]])
    candidates = _geometry(
        [[0.0, 0.0], [300.0, 0.0], [250.0, 0.0], [300.0, 0.0]],
        [[300.0, -1000.0], [0.0, -1000.0], [-250.0, -1000.0], [300.0, -1000.0]],
    )
    edges = select_trace_graph_neighbors(
        destination,
        [100],
        candidates,
        [10, 20, 30, 40],
        np.ones(4, dtype=bool),
        relation_scales_m=SCALES,
    )

    np.testing.assert_array_equal(edges.sender_ids, [10, 20, 30, 40])
    np.testing.assert_array_equal(edges.edge_type, [0, 1, 2, 3])
    np.testing.assert_array_equal(edges.destination_ids, [100] * 4)
    np.testing.assert_allclose(edges.distances, [300 / 640, 300 / 640, 500 / 640, 300 / 640])


def test_visibility_and_allowed_domain_are_applied_before_top_k() -> None:
    edges = select_trace_graph_neighbors(
        _geometry([[0.0, 0.0]]),
        [100],
        _geometry([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]),
        [10, 20, 30, 40],
        np.array([False, True, True, True]),
        allowed_mask=np.array([True, False, True, True]),
        relation_scales_m=SCALES,
        neighbors_per_relation=2,
        candidate_chunk_size=1,
    )

    np.testing.assert_array_equal(edges.sender_ids, [30, 40] * 4)
    np.testing.assert_array_equal(edges.edge_type, np.repeat(np.arange(4), 2))


@pytest.mark.parametrize("permutation", [[0, 1, 2, 3], [2, 0, 3, 1], [3, 2, 1, 0]])
@pytest.mark.parametrize("chunk_size", [1, 2, 17])
def test_ties_follow_stable_id_independent_of_storage_and_chunk_size(permutation, chunk_size):
    source = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    ids = np.array([50, 10, 30, 20])
    edges = select_trace_graph_neighbors(
        _geometry([[0.0, 0.0]]),
        [-1],
        _geometry(source[permutation]),
        ids[permutation],
        np.ones(4, dtype=bool),
        relation_scales_m=SCALES,
        neighbors_per_relation=3,
        candidate_chunk_size=chunk_size,
    )

    np.testing.assert_array_equal(edges.sender_ids, [10, 20, 30] * 4)


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 8, 100])
def test_exact_search_matches_independent_scalar_oracle(chunk_size):
    source = np.array([[0, 0], [64, 0], [160, 0], [0, 320], [-128, 64], [512, 256]])
    receiver = np.array([[0, -1000], [128, -1000], [0, -936], [0, -872], [64, -744], [512, -488]])
    candidates = _geometry(source, receiver)
    destinations = _geometry([[0, 0], [128, 64], [160, 0]], [[0, -1000], [64, -808], [0, -936]])
    candidate_ids = np.array([9, 3, 8, 1, 7, 2])
    destination_ids = np.array([-1, -2, 8])
    observed = np.array([True, False, True, True, True, True])
    allowed = np.array([True, True, True, True, True, False])
    edges = select_trace_graph_neighbors(
        destinations,
        destination_ids,
        candidates,
        candidate_ids,
        observed,
        allowed_mask=allowed,
        relation_scales_m=SCALES,
        neighbors_per_relation=2,
        candidate_chunk_size=chunk_size,
    )
    expected = _oracle(
        destinations, destination_ids, candidates, candidate_ids, observed, allowed, 2, 1.0
    )

    assert len(_rows(edges)) == len(expected)
    for actual, wanted in zip(_rows(edges), expected, strict=True):
        assert actual[:3] == wanted[:3]
        assert actual[3] == pytest.approx(wanted[3], abs=1e-15)
    assert edges.distances.dtype == np.float64


def test_radius_is_inclusive_and_has_no_outside_fallback() -> None:
    edges = select_trace_graph_neighbors(
        _geometry([[0.0, 0.0]]),
        [100],
        _geometry([[0.0, 0.0], [0.0, 0.0]], [[640.0, -1000.0], [640.001, -1000.0]]),
        [1, 2],
        np.ones(2, dtype=bool),
        relation_scales_m=SCALES,
    )

    assert _rows(edges) == [(1, 100, 0, 1.0)]


def test_self_edges_are_excluded_and_relations_retain_their_shared_pair() -> None:
    geometry = _geometry([[0, 0], [1, 0], [3, 0]])
    edges = select_trace_graph_neighbors(
        geometry,
        [1, 2, 3],
        geometry,
        [1, 2, 3],
        np.ones(3, dtype=bool),
        relation_scales_m=SCALES,
        neighbors_per_relation=1,
    )

    pairs = list(zip(edges.sender_ids, edges.destination_ids, strict=True))
    assert pairs == [(2, 1)] * 4 + [(1, 2)] * 4 + [(2, 3)] * 4
    assert (3, 2) not in pairs
    assert (
        len(set(zip(edges.sender_ids, edges.destination_ids, edges.edge_type, strict=True))) == 12
    )


@pytest.mark.parametrize("kind", ["no_candidates", "no_destinations", "all_invisible", "far_away"])
def test_empty_search_has_typed_empty_arrays(kind):
    destinations = _geometry([] if kind == "no_destinations" else [[0, 0]])
    candidates = _geometry([] if kind == "no_candidates" else [[10000, 10000]])
    edges = select_trace_graph_neighbors(
        destinations,
        np.arange(len(destinations.offset_m)),
        candidates,
        np.arange(len(candidates.offset_m)) + 10,
        np.full(len(candidates.offset_m), kind != "all_invisible", dtype=bool),
        relation_scales_m=SCALES,
    )

    for name in ("sender_ids", "destination_ids", "edge_type", "distances"):
        assert getattr(edges, name).shape == (0,)
    assert (
        edges.sender_ids.dtype == edges.destination_ids.dtype == edges.edge_type.dtype == np.int64
    )
    assert edges.distances.dtype == np.float64


@pytest.mark.parametrize(
    "settings",
    [
        {"relation_scales_m": np.ones((3, 2))},
        {"relation_scales_m": np.zeros((4, 2))},
        {"relation_scales_m": np.full((4, 2), np.nan)},
        {"neighbors_per_relation": 0},
        {"neighbors_per_relation": 1.5},
        {"neighbors_per_relation": True},
        {"candidate_chunk_size": 0},
        {"radius": 0},
        {"radius": np.inf},
        {"allowed_mask": np.array([1])},
    ],
)
def test_invalid_search_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        select_trace_graph_neighbors(
            _geometry([[0, 0]]),
            [1],
            _geometry([[1, 0]]),
            [2],
            np.array([True]),
            **{"relation_scales_m": SCALES, **settings},
        )


@pytest.mark.parametrize(
    ("ids", "observed"),
    [([1, 1], [True, True]), ([1.5, 2], [True, True]), ([1], [True, True]), ([1, 2], [1, 1])],
)
def test_invalid_candidate_identity_or_role_is_rejected(ids, observed):
    with pytest.raises(ValueError):
        select_trace_graph_neighbors(
            _geometry([[0, 0]]),
            [100],
            _geometry([[1, 0], [2, 0]]),
            ids,
            np.array(observed),
            relation_scales_m=SCALES,
        )
