from __future__ import annotations

import numpy as np
import pytest

from seis_interp.processing.geometry import apply_coordinate_scalar, compute_trace_geometry


def test_apply_coordinate_scalar_handles_positive_negative_and_zero() -> None:
    values = np.array([100, 100, 100], dtype=np.int32)
    scalars = np.array([10, -100, 0], dtype=np.int32)

    scaled = apply_coordinate_scalar(values, scalars)

    np.testing.assert_allclose(scaled, [1000.0, 1.0, 100.0])


def test_apply_coordinate_scalar_returns_float64() -> None:
    scaled = apply_coordinate_scalar(np.array([1], dtype=np.int32), np.array([1], dtype=np.int32))

    assert scaled.dtype == np.float64


def test_apply_coordinate_scalar_does_not_modify_inputs() -> None:
    values = np.array([100.0, 200.0])
    scalars = np.array([10, -10], dtype=np.int32)

    apply_coordinate_scalar(values, scalars)

    np.testing.assert_allclose(values, [100.0, 200.0])
    np.testing.assert_array_equal(scalars, [10, -10])


def test_apply_coordinate_scalar_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="share a shape"):
        apply_coordinate_scalar(np.array([1.0, 2.0]), np.array([1]))


def test_apply_coordinate_scalar_rejects_two_dimensional_input() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        apply_coordinate_scalar(np.ones((2, 2)), np.ones((2, 2)))


def test_compute_trace_geometry_matches_known_layout() -> None:
    source_x = np.array([1000.0, 1000.0])
    source_y = np.array([2000.0, 2000.0])
    receiver_x = np.array([1400.0, 1000.0])
    receiver_y = np.array([2000.0, 2600.0])

    cmp_x, cmp_y, offset, azimuth = compute_trace_geometry(
        source_x, source_y, receiver_x, receiver_y
    )

    np.testing.assert_allclose(cmp_x, [1200.0, 1000.0])
    np.testing.assert_allclose(cmp_y, [2000.0, 2300.0])
    np.testing.assert_allclose(offset, [400.0, 600.0])
    assert cmp_x.dtype == np.float64
    assert offset.dtype == np.float64
    np.testing.assert_allclose(azimuth, [270.0, 180.0])


def test_compute_trace_geometry_uses_atan2_dx_dy_wrapped_to_360() -> None:
    # dx = +1, dy = +1 -> 45 deg; dx = -1, dy = +1 -> 315 deg (not -45).
    source_x = np.array([1.0, -1.0])
    source_y = np.array([1.0, 1.0])
    receiver_x = np.zeros(2)
    receiver_y = np.zeros(2)

    _, _, _, azimuth = compute_trace_geometry(source_x, source_y, receiver_x, receiver_y)

    np.testing.assert_allclose(azimuth, [45.0, 315.0])
    assert np.all(azimuth >= 0.0)
    assert np.all(azimuth < 360.0)


def test_compute_trace_geometry_allows_zero_offset() -> None:
    zeros = np.zeros(1)

    _, _, offset, azimuth = compute_trace_geometry(zeros, zeros, zeros, zeros)

    np.testing.assert_allclose(offset, [0.0])
    np.testing.assert_allclose(azimuth, [0.0])


def test_compute_trace_geometry_does_not_modify_inputs() -> None:
    source_x = np.array([10.0, 20.0])
    source_y = np.array([30.0, 40.0])
    receiver_x = np.array([50.0, 60.0])
    receiver_y = np.array([70.0, 80.0])

    compute_trace_geometry(source_x, source_y, receiver_x, receiver_y)

    np.testing.assert_allclose(source_x, [10.0, 20.0])
    np.testing.assert_allclose(source_y, [30.0, 40.0])
    np.testing.assert_allclose(receiver_x, [50.0, 60.0])
    np.testing.assert_allclose(receiver_y, [70.0, 80.0])


def test_compute_trace_geometry_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="share a shape"):
        compute_trace_geometry(
            np.zeros(2),
            np.zeros(2),
            np.zeros(3),
            np.zeros(2),
        )


def test_compute_trace_geometry_rejects_two_dimensional_input() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        compute_trace_geometry(
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            np.zeros((2, 2)),
        )


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_compute_trace_geometry_rejects_non_finite_input(bad_value: float) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        compute_trace_geometry(
            np.array([bad_value, 1.0]),
            np.zeros(2),
            np.zeros(2),
            np.zeros(2),
        )
