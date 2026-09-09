"""Observed-only scalar interpolation has explicit geometry and tie contracts."""

import numpy as np
import pytest

from seis_interp.processing import trace_rms_interpolation as interpolation
from seis_interp.processing.trace_rms_interpolation import interpolate_trace_rms


def test_two_observations_match_hand_weighted_rms_and_allow_fewer_than_k():
    result = interpolate_trace_rms(
        np.array([[0.0], [4.0]]),
        np.array([2.0, 10.0]),
        np.array([[1.0], [2.0], [3.0]]),
        observed_array_rows=np.array([7, 2]),
        neighbors=8,
        power=2.0,
    )
    np.testing.assert_allclose(result, [2.8, 6.0, 9.2], rtol=1e-15)
    assert result.dtype == np.float64


@pytest.mark.parametrize(
    "neighbors,expected", [(2, 3.2), (3, (2 + 0.8 + 14 / 9) / (1 + 0.1 + 1 / 9))]
)
def test_three_observations_use_euclidean_distances_and_only_nearest_k(neighbors, expected):
    result = interpolate_trace_rms(
        np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]]),
        np.array([2.0, 8.0, 14.0]),
        np.array([[0.0, 1.0]]),
        observed_array_rows=np.array([10, 20, 30]),
        neighbors=neighbors,
        power=2.0,
    )
    assert result[0] == pytest.approx(expected)


@pytest.mark.parametrize("order", [[0, 1, 2, 3], [3, 0, 2, 1]])
def test_neighbor_boundary_ties_are_broken_by_lowest_original_array_row(order):
    coordinates = np.array([[-1.0, 0], [1.0, 0], [0, -1.0], [0, 1.0]])
    result = interpolate_trace_rms(
        coordinates[order],
        np.array([30.0, 20.0, 10.0, 40.0])[order],
        np.array([[0.0, 0.0]]),
        observed_array_rows=np.array([30, 20, 10, 40])[order],
        neighbors=2,
        power=2.0,
    )
    assert result.tolist() == [15.0]


def test_axis_permuted_equal_squared_distances_use_array_row_tie_break():
    # A sequential hypot fold rounds these equal squared distances (11)
    # differently, and would incorrectly choose the second observation.
    result = interpolate_trace_rms(
        np.array([[0.0, 1.0, 3.0, 1.0], [0.0, 1.0, 1.0, 3.0]]),
        np.array([10.0, 20.0]),
        np.zeros((1, 4)),
        observed_array_rows=np.array([1, 2]),
        neighbors=1,
        power=2.0,
    )
    assert result.tolist() == [10.0]


def test_exact_matches_choose_lowest_array_row_and_zero_rms_is_valid():
    result = interpolate_trace_rms(
        np.array([[0.0], [0.0], [2.0]]),
        np.array([90.0, 0.0, 4.0]),
        np.array([[0.0], [2.0]]),
        observed_array_rows=np.array([90, 2, 4]),
        neighbors=1,
        power=2.0,
    )
    np.testing.assert_array_equal(result, [0.0, 4.0])
    mixed = interpolate_trace_rms(
        np.array([[0.0], [2.0]]),
        np.array([0.0, 4.0]),
        np.array([[1.0]]),
        observed_array_rows=np.array([2, 4]),
        neighbors=2,
        power=2.0,
    )
    assert mixed[0] == 2.0
    zeros = interpolate_trace_rms(
        np.array([[0.0], [2.0]]),
        np.zeros(2),
        np.array([[1.0]]),
        observed_array_rows=np.array([2, 4]),
        neighbors=2,
        power=2.0,
    )
    assert zeros[0] == 0.0


def test_tiny_distances_and_large_scales_have_stable_finite_weights():
    result = interpolate_trace_rms(
        np.array([[1e-250], [2e-250]]),
        np.array([1e308, 1.6e308]),
        np.array([[0.0]]),
        observed_array_rows=np.array([2, 4]),
        neighbors=2,
        power=2.0,
    )
    assert np.isfinite(result).all()
    assert result[0] == pytest.approx(1.12e308)


def test_query_chunks_are_bounded_and_chunk_size_does_not_change_result(monkeypatch):
    coordinates = np.array([[0.0, 1.0], [1.0, 2.0], [4.0, 5.0]])
    queries = np.arange(14, dtype=np.float64).reshape(7, 2)
    settings = dict(observed_array_rows=np.array([5, 1, 10]), neighbors=2, power=2.0)
    expected = interpolate_trace_rms(coordinates, np.array([0.0, 2.0, 8.0]), queries, **settings)
    calls = []
    original = interpolation._euclidean_distances

    def record(chunk, observed):
        calls.append(len(chunk))
        return original(chunk, observed)

    monkeypatch.setattr(interpolation, "_euclidean_distances", record)
    actual = interpolate_trace_rms(
        coordinates,
        np.array([0.0, 2.0, 8.0]),
        queries,
        query_chunk_size=3,
        **settings,
    )
    assert calls == [3, 3, 1]
    np.testing.assert_array_equal(actual, expected)


def test_empty_queries_return_empty_float64_vector_without_reading_distances(monkeypatch):
    monkeypatch.setattr(
        interpolation, "_euclidean_distances", lambda *args: pytest.fail("no queries")
    )
    result = interpolate_trace_rms(
        np.array([[0.0]]),
        np.array([2.0]),
        np.empty((0, 1)),
        observed_array_rows=np.array([4]),
        neighbors=8,
        power=2.0,
    )
    assert result.shape == (0,)
    assert result.dtype == np.float64


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"neighbors": 0}, "neighbors"),
        ({"neighbors": True}, "neighbors"),
        ({"neighbors": 2.0}, "neighbors"),
        ({"power": 0}, "power"),
        ({"power": np.inf}, "power"),
        ({"power": True}, "power"),
        ({"power": 10**400}, "power"),
        ({"query_chunk_size": 0}, "query_chunk_size"),
        ({"observed_scales": np.array([-1.0, 2.0])}, "observed_scales"),
        ({"observed_scales": np.array([1.0, np.nan])}, "observed_scales"),
        ({"observed_array_rows": np.array([1, 1])}, "observed_array_rows"),
        ({"observed_array_rows": np.array([-1, 1])}, "observed_array_rows"),
        ({"query_coordinates": np.array([[np.nan]])}, "query_coordinates"),
        ({"query_coordinates": np.zeros((1, 2))}, "coordinate width"),
    ],
)
def test_invalid_interpolation_inputs_are_rejected(changes, match):
    arguments = dict(
        observed_coordinates=np.array([[0.0], [2.0]]),
        observed_scales=np.array([1.0, 2.0]),
        query_coordinates=np.array([[1.0]]),
        observed_array_rows=np.array([0, 1]),
        neighbors=2,
        power=2.0,
    )
    arguments.update(changes)
    with pytest.raises(ValueError, match=match):
        interpolate_trace_rms(**arguments)
