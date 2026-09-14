import numpy as np
import pytest

from seis_interp.training.nersi_coordinate_jitter import jitter_profile_coordinates


def test_jitter_is_reproducible_subcell_bounded_and_preserves_singleton_axis():
    coordinates = np.array([[0, 0, 0], [0, 0.5, 0.5], [0, 1, 1]], dtype=np.float32)
    original = coordinates.copy()
    first = jitter_profile_coordinates(
        coordinates, (1, 3, 5, 8), jitter_cells=0.1, rng=np.random.default_rng(501)
    )
    second = jitter_profile_coordinates(
        coordinates, (1, 3, 5, 8), jitter_cells=0.1, rng=np.random.default_rng(501)
    )
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(coordinates, original)
    np.testing.assert_array_equal(first[:, 0], coordinates[:, 0])
    assert first.dtype == np.float32
    assert first.flags.c_contiguous
    assert np.all((first >= 0) & (first <= 1))
    assert np.all(np.abs(first - coordinates) <= np.array([0, 0.05, 0.025]) + 1e-7)
    assert not np.array_equal(first, coordinates)


@pytest.mark.parametrize("radius", [0, -0.1, 0.51, float("nan"), float("inf"), True])
def test_invalid_jitter_radius_is_rejected(radius):
    with pytest.raises(ValueError, match="jitter_cells"):
        jitter_profile_coordinates(
            np.zeros((2, 3)), (1, 3, 5, 8), jitter_cells=radius, rng=np.random.default_rng(1)
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
def test_coordinates_outside_normalized_domain_are_rejected(value):
    with pytest.raises(ValueError, match="coordinates"):
        jitter_profile_coordinates(
            np.full((2, 3), value), (1, 3, 5, 8), jitter_cells=0.1, rng=np.random.default_rng(1)
        )
