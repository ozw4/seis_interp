from __future__ import annotations

import numpy as np
import pytest

from seis_interp.processing.geometry import compute_trace_geometry
from seis_interp.processing.trace_graph_geometry import (
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    RELATION_IDS,
    RELATION_NAMES,
    TraceGraphGeometry,
    build_trace_graph_edge_features,
    build_trace_graph_node_features,
    compute_trace_graph_geometry,
)


def _known_geometry() -> TraceGraphGeometry:
    # Destination faces north with full offset 4; sender faces east with offset 6.
    return compute_trace_graph_geometry(
        np.array([[102.0, 206.0], [108.0, 209.0]]),
        np.array([[102.0, 202.0], [102.0, 209.0]]),
        azimuth_min_offset_m=1.0,
    )


def test_relation_and_feature_names_define_fixed_order() -> None:
    assert RELATION_NAMES == ("source", "receiver", "cmp", "offset_azimuth")
    assert RELATION_IDS == {"source": 0, "receiver": 1, "cmp": 2, "offset_azimuth": 3}
    assert NODE_FEATURE_NAMES == (
        "midpoint_x_scaled",
        "midpoint_y_scaled",
        "offset_x_scaled",
        "offset_y_scaled",
        "offset_length_scaled",
        "azimuth_sin",
        "azimuth_cos",
        "azimuth_valid",
        "observed",
    )
    assert EDGE_FEATURE_NAMES == (
        "delta_source_x_scaled",
        "delta_source_y_scaled",
        "delta_receiver_x_scaled",
        "delta_receiver_y_scaled",
        "delta_midpoint_x_scaled",
        "delta_midpoint_y_scaled",
        "delta_offset_x_scaled",
        "delta_offset_y_scaled",
        "destination_offset_length_scaled",
        "sender_offset_length_scaled",
        "delta_offset_length_scaled",
        "azimuth_delta_cos",
        "azimuth_delta_sin",
        "azimuth_pair_valid",
        "relation_distance",
    )


def test_geometry_uses_full_offset_and_round_trips_absolute_coordinates() -> None:
    source = np.array([[4.0, 8.0], [-3.0, 6.0], [10.5, -4.0]])
    receiver = np.array([[1.0, 4.0], [2.0, -6.0], [-1.5, -9.0]])

    geometry = compute_trace_graph_geometry(source, receiver, azimuth_min_offset_m=0.0)

    np.testing.assert_allclose(geometry.midpoint_xy_m, [[2.5, 6.0], [-0.5, 0.0], [4.5, -6.5]])
    np.testing.assert_allclose(geometry.offset_xy_m, [[3.0, 4.0], [-5.0, 12.0], [12.0, 5.0]])
    np.testing.assert_allclose(geometry.offset_m, [5.0, 13.0, 13.0])
    np.testing.assert_allclose(geometry.midpoint_xy_m + geometry.offset_xy_m / 2.0, source)
    np.testing.assert_allclose(geometry.midpoint_xy_m - geometry.offset_xy_m / 2.0, receiver)
    for array in (
        geometry.source_xy_m,
        geometry.receiver_xy_m,
        geometry.midpoint_xy_m,
        geometry.offset_xy_m,
        geometry.offset_m,
        geometry.azimuth_sin,
        geometry.azimuth_cos,
    ):
        assert array.dtype == np.float64
    assert geometry.azimuth_valid.dtype == np.bool_


def test_geometry_matches_existing_coordinate_convention() -> None:
    source = np.array([[1.0, 1.0], [-1.0, 1.0], [0.0, -2.0], [-3.0, 0.0]])
    receiver = np.zeros_like(source)
    cmp_x, cmp_y, offset, azimuth = compute_trace_geometry(
        source[:, 0], source[:, 1], receiver[:, 0], receiver[:, 1]
    )

    geometry = compute_trace_graph_geometry(source, receiver, azimuth_min_offset_m=0.0)

    np.testing.assert_array_equal(geometry.midpoint_xy_m, np.column_stack((cmp_x, cmp_y)))
    np.testing.assert_array_equal(geometry.offset_m, offset)
    np.testing.assert_allclose(geometry.azimuth_sin, np.sin(np.deg2rad(azimuth)), atol=1e-15)
    np.testing.assert_allclose(geometry.azimuth_cos, np.cos(np.deg2rad(azimuth)), atol=1e-15)


def test_node_features_match_hand_calculated_order_and_dtype() -> None:
    features = build_trace_graph_node_features(
        _known_geometry(),
        midpoint_origin_m=np.array([100.0, 200.0]),
        position_scale_m=2.0,
        offset_scale_m=4.0,
        observed_mask=np.array([False, True]),
    )

    assert features.shape == (2, 9)
    assert features.dtype == np.float32
    np.testing.assert_array_equal(
        features,
        [
            [1.0, 2.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            [2.5, 4.5, 1.5, 0.0, 1.5, 1.0, 0.0, 1.0, 1.0],
        ],
    )


def test_edge_features_match_hand_calculated_order_and_sender_sign() -> None:
    features = build_trace_graph_edge_features(
        _known_geometry(),
        sender_indices=np.array([1, 0]),
        destination_indices=np.array([0, 1]),
        relation_distances=np.array([0.75, 0.75]),
        position_scale_m=2.0,
        offset_scale_m=4.0,
    )

    assert features.shape == (2, 15)
    assert features.dtype == np.float32
    np.testing.assert_array_equal(
        features,
        [
            [3.0, 1.5, 0.0, 3.5, 1.5, 2.5, 1.5, -1.0, 1.0, 1.5, 0.5, 0.0, 1.0, 1.0, 0.75],
            [-3.0, -1.5, 0.0, -3.5, -1.5, -2.5, -1.5, 1.0, 1.5, 1.0, -0.5, 0.0, -1.0, 1.0, 0.75],
        ],
    )


def test_azimuth_wraparound_is_continuous() -> None:
    angles = np.deg2rad([359.0, 1.0])
    source = 10.0 * np.column_stack((np.sin(angles), np.cos(angles)))
    geometry = compute_trace_graph_geometry(source, np.zeros_like(source), azimuth_min_offset_m=1.0)
    nodes = build_trace_graph_node_features(
        geometry,
        midpoint_origin_m=np.zeros(2),
        position_scale_m=1.0,
        offset_scale_m=1.0,
        observed_mask=np.array([False, True]),
    )
    edges = build_trace_graph_edge_features(
        geometry,
        sender_indices=np.array([1, 0]),
        destination_indices=np.array([0, 1]),
        relation_distances=np.zeros(2),
        position_scale_m=1.0,
        offset_scale_m=1.0,
    )

    np.testing.assert_allclose(
        nodes[:, 5:7], [[-0.017452406, 0.999847695], [0.017452406, 0.999847695]]
    )
    np.testing.assert_allclose(
        edges[:, 11:14], [[0.999390827, 0.034899497, 1.0], [0.999390827, -0.034899497, 1.0]]
    )


def test_small_offsets_including_threshold_have_invalid_azimuth_but_keep_vector() -> None:
    source = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [0.0, 2.0]])
    geometry = compute_trace_graph_geometry(source, np.zeros_like(source), azimuth_min_offset_m=1.0)
    nodes = build_trace_graph_node_features(
        geometry,
        midpoint_origin_m=np.zeros(2),
        position_scale_m=1.0,
        offset_scale_m=1.0,
        observed_mask=np.array([False, True, True, True]),
    )
    edges = build_trace_graph_edge_features(
        geometry,
        sender_indices=np.array([0, 1, 2, 3, 3, 3, 3]),
        destination_indices=np.array([3, 3, 3, 3, 0, 1, 2]),
        relation_distances=np.zeros(7),
        position_scale_m=1.0,
        offset_scale_m=1.0,
    )

    np.testing.assert_array_equal(geometry.offset_xy_m, source)
    np.testing.assert_array_equal(geometry.azimuth_valid, [False, False, False, True])
    np.testing.assert_array_equal(nodes[:, 2:4], source)
    np.testing.assert_array_equal(nodes[:, 5:8], [[0.0, 0.0, 0.0]] * 3 + [[0.0, 1.0, 1.0]])
    np.testing.assert_array_equal(
        edges[:, 11:14],
        [[0.0, 0.0, 0.0]] * 3 + [[1.0, 0.0, 1.0]] + [[0.0, 0.0, 0.0]] * 3,
    )
    assert np.all(np.isfinite(nodes))
    assert np.all(np.isfinite(edges))


def test_large_shared_origin_preserves_small_differences_before_float32_conversion() -> None:
    origin = np.array([1.0e12, -1.0e12])
    source = origin + [[1.25, 2.5], [1.5, 2.875]]
    receiver = origin + [[0.75, 1.5], [1.0, 1.875]]
    geometry = compute_trace_graph_geometry(source, receiver, azimuth_min_offset_m=0.0)
    nodes = build_trace_graph_node_features(
        geometry,
        midpoint_origin_m=origin,
        position_scale_m=1.0,
        offset_scale_m=2.0,
        observed_mask=np.array([False, True]),
    )
    edges = build_trace_graph_edge_features(
        geometry,
        sender_indices=np.array([1]),
        destination_indices=np.array([0]),
        relation_distances=np.array([0.25]),
        position_scale_m=1.0,
        offset_scale_m=2.0,
    )

    np.testing.assert_array_equal(nodes[:, :4], [[1.0, 2.0, 0.25, 0.5], [1.25, 2.375, 0.25, 0.5]])
    np.testing.assert_array_equal(edges[:, :8], [[0.25, 0.375, 0.25, 0.375, 0.25, 0.375, 0.0, 0.0]])


def test_features_follow_node_and_edge_permutations_with_fixed_origin() -> None:
    source = np.array([[0.2, 1.3], [2.1, -3.6], [-6.2, 4.7], [0.8, 0.1]])
    receiver = np.array([[-0.4, 0.7], [1.1, 8.3], [2.0, 3.8], [0.8, 0.1]])
    observed = np.array([False, True, False, True])
    sender = np.array([1, 3, 3, 1])
    destination = np.array([0, 2, 0, 3])
    distances = np.array([0.1, 0.4, 0.3, 0.2])
    permutation = np.array([2, 0, 3, 1])
    inverse_permutation = np.argsort(permutation)
    edge_permutation = np.array([3, 1, 0, 2])
    geometry = compute_trace_graph_geometry(source, receiver, azimuth_min_offset_m=0.1)
    permuted_geometry = compute_trace_graph_geometry(
        source[permutation], receiver[permutation], azimuth_min_offset_m=0.1
    )
    node_settings = {
        "midpoint_origin_m": np.array([12.3, -45.6]),
        "position_scale_m": 3.0,
        "offset_scale_m": 4.0,
    }
    nodes = build_trace_graph_node_features(geometry, observed_mask=observed, **node_settings)
    permuted_nodes = build_trace_graph_node_features(
        permuted_geometry, observed_mask=observed[permutation], **node_settings
    )
    edges = build_trace_graph_edge_features(
        geometry,
        sender_indices=sender,
        destination_indices=destination,
        relation_distances=distances,
        position_scale_m=3.0,
        offset_scale_m=4.0,
    )
    permuted_edges = build_trace_graph_edge_features(
        permuted_geometry,
        sender_indices=inverse_permutation[sender[edge_permutation]],
        destination_indices=inverse_permutation[destination[edge_permutation]],
        relation_distances=distances[edge_permutation],
        position_scale_m=3.0,
        offset_scale_m=4.0,
    )

    np.testing.assert_array_equal(permuted_nodes, nodes[permutation])
    np.testing.assert_array_equal(permuted_edges, edges[edge_permutation])


def test_empty_geometry_and_features_have_correct_shapes_and_dtypes() -> None:
    geometry = compute_trace_graph_geometry(
        np.empty((0, 2)), np.empty((0, 2)), azimuth_min_offset_m=1.0
    )
    nodes = build_trace_graph_node_features(
        geometry,
        midpoint_origin_m=np.zeros(2),
        position_scale_m=1.0,
        offset_scale_m=1.0,
        observed_mask=np.array([], dtype=bool),
    )
    edges = build_trace_graph_edge_features(
        geometry,
        sender_indices=[],
        destination_indices=[],
        relation_distances=[],
        position_scale_m=1.0,
        offset_scale_m=1.0,
    )

    assert geometry.midpoint_xy_m.shape == (0, 2)
    assert geometry.offset_xy_m.shape == (0, 2)
    assert geometry.offset_m.shape == (0,)
    assert geometry.azimuth_valid.shape == (0,)
    assert nodes.shape == (0, 9)
    assert edges.shape == (0, 15)
    assert nodes.dtype == edges.dtype == np.float32


def test_feature_builders_do_not_modify_inputs() -> None:
    source = np.array([[2.0, 3.0], [-4.0, 5.0]])
    receiver = np.array([[0.0, 0.0], [-1.0, 2.0]])
    origin = np.array([10.0, 20.0])
    observed = np.array([False, True])
    sender = np.array([1])
    destination = np.array([0])
    distances = np.array([0.5])
    arrays = (source, receiver, origin, observed, sender, destination, distances)
    originals = tuple(array.copy() for array in arrays)
    geometry = compute_trace_graph_geometry(source, receiver, azimuth_min_offset_m=0.0)
    build_trace_graph_node_features(
        geometry,
        midpoint_origin_m=origin,
        position_scale_m=10.0,
        offset_scale_m=5.0,
        observed_mask=observed,
    )
    build_trace_graph_edge_features(
        geometry,
        sender_indices=sender,
        destination_indices=destination,
        relation_distances=distances,
        position_scale_m=10.0,
        offset_scale_m=5.0,
    )

    for actual, expected in zip(arrays, originals, strict=True):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("bad_threshold", [-1.0, np.nan, np.inf])
def test_geometry_rejects_invalid_azimuth_threshold(bad_threshold: float) -> None:
    with pytest.raises(ValueError):
        compute_trace_graph_geometry(
            np.zeros((1, 2)), np.zeros((1, 2)), azimuth_min_offset_m=bad_threshold
        )


@pytest.mark.parametrize(
    ("source", "receiver"),
    [
        (np.zeros(2), np.zeros((1, 2))),
        (np.zeros((1, 3)), np.zeros((1, 2))),
        (np.zeros((1, 2)), np.zeros((2, 2))),
        (np.array([[np.nan, 0.0]]), np.zeros((1, 2))),
        (np.zeros((1, 2)), np.array([[0.0, np.inf]])),
    ],
)
def test_geometry_rejects_invalid_coordinates(source: np.ndarray, receiver: np.ndarray) -> None:
    with pytest.raises(ValueError):
        compute_trace_graph_geometry(source, receiver, azimuth_min_offset_m=1.0)


@pytest.mark.parametrize("scale_name", ["position_scale_m", "offset_scale_m"])
@pytest.mark.parametrize("bad_scale", [0.0, -1.0, np.nan, np.inf])
def test_feature_builders_reject_invalid_scales(scale_name: str, bad_scale: float) -> None:
    scales = {"position_scale_m": 1.0, "offset_scale_m": 1.0, scale_name: bad_scale}
    geometry = _known_geometry()
    with pytest.raises(ValueError):
        build_trace_graph_node_features(
            geometry,
            midpoint_origin_m=np.zeros(2),
            observed_mask=np.array([False, True]),
            **scales,
        )
    with pytest.raises(ValueError):
        build_trace_graph_edge_features(
            geometry,
            sender_indices=[1],
            destination_indices=[0],
            relation_distances=[0.5],
            **scales,
        )


@pytest.mark.parametrize("origin", [np.zeros(3), np.zeros((1, 2)), [np.nan, 0.0], [0.0, np.inf]])
def test_node_features_reject_invalid_origin(origin: np.ndarray | list[float]) -> None:
    with pytest.raises(ValueError):
        build_trace_graph_node_features(
            _known_geometry(),
            midpoint_origin_m=origin,
            position_scale_m=1.0,
            offset_scale_m=1.0,
            observed_mask=np.array([False, True]),
        )


@pytest.mark.parametrize("mask", [np.array([True]), np.array([[False, True]]), np.array([0, 1])])
def test_node_features_reject_invalid_observed_mask(mask: np.ndarray) -> None:
    with pytest.raises(ValueError):
        build_trace_graph_node_features(
            _known_geometry(),
            midpoint_origin_m=np.zeros(2),
            position_scale_m=1.0,
            offset_scale_m=1.0,
            observed_mask=mask,
        )


@pytest.mark.parametrize(
    ("sender", "destination", "distances"),
    [
        ([1.5], [0], [0.5]),
        ([-1], [0], [0.5]),
        ([2], [0], [0.5]),
        ([1], [-1], [0.5]),
        ([1], [2], [0.5]),
        ([[1]], [0], [0.5]),
        ([1], [0, 1], [0.5]),
        ([1], [0], []),
        ([1], [0], [[0.5]]),
        ([1], [0], [-0.5]),
        ([1], [0], [np.nan]),
        ([1], [0], [np.inf]),
    ],
)
def test_edge_features_reject_invalid_indices_and_distances(
    sender: list,
    destination: list,
    distances: list,
) -> None:
    with pytest.raises(ValueError):
        build_trace_graph_edge_features(
            _known_geometry(),
            sender_indices=sender,
            destination_indices=destination,
            relation_distances=distances,
            position_scale_m=1.0,
            offset_scale_m=1.0,
        )
