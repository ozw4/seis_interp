"""The log-offset axis means the same thing on every dataset."""

import numpy as np
import pandas as pd
import pytest

from seis_interp.processing.trace_field_coordinates import (
    LOCAL_COLUMNS,
    OFFSET_DECADES,
    OFFSET_FLOOR_M,
    field_coordinates,
)

SHAPE = (6, 32, 8, 32)


def _geometry(offsets):
    rows = len(offsets)
    axes = np.array(np.unravel_index(np.arange(rows) % int(np.prod(SHAPE)), SHAPE)).T
    frame = pd.DataFrame(
        {column: axes[:, i].astype(float) for i, column in enumerate(LOCAL_COLUMNS)}
    )
    frame["offset_m"] = np.asarray(offsets, dtype=np.float64)
    return frame


def test_the_four_local_axes_are_normalized_by_the_grid_extent():
    geometry = _geometry(np.full(12, 500.0))
    local = field_coordinates(geometry, SHAPE, 4)
    assert local.shape == (12, 4)
    expected = geometry[list(LOCAL_COLUMNS)].to_numpy() / np.maximum(np.array(SHAPE) - 1, 1)
    np.testing.assert_allclose(local, expected)


def test_the_fifth_axis_is_appended_without_touching_the_others():
    geometry = _geometry(np.linspace(10.0, 2000.0, 12))
    four, five = field_coordinates(geometry, SHAPE, 4), field_coordinates(geometry, SHAPE, 5)
    assert five.shape == (12, 5)
    np.testing.assert_array_equal(five[:, :4], four)


@pytest.mark.parametrize("factor", [1.1, 2.0, 10.0])
def test_a_relative_offset_change_is_one_fixed_distance_everywhere(factor):
    """A logarithm bounds the relative difference, which is what a waveform follows.

    Both pairs stay inside the unclipped span; the clipping boundary has its own test.
    """
    near = field_coordinates(_geometry([100.0, 100.0 * factor]), SHAPE, 5)[:, 4]
    far = field_coordinates(_geometry([500.0, 500.0 * factor]), SHAPE, 5)[:, 4]
    assert np.diff(near)[0] == pytest.approx(np.diff(far)[0])
    assert np.diff(near)[0] == pytest.approx(np.log10(factor) / OFFSET_DECADES)


def test_the_axis_is_floored_and_clipped_into_the_unit_range():
    coordinates = field_coordinates(
        _geometry([0.0, OFFSET_FLOOR_M, 10.0**OFFSET_DECADES]), SHAPE, 5
    )
    assert coordinates[0, 4] == coordinates[1, 4] == 0.0
    assert coordinates[2, 4] == pytest.approx(1.0)
    assert field_coordinates(_geometry([10.0 ** (OFFSET_DECADES + 2)]), SHAPE, 5)[0, 4] == 1.0


@pytest.mark.parametrize(
    "features,shape,offsets,message",
    [
        (3, SHAPE, [100.0], "four or five"),
        (6, SHAPE, [100.0], "four or five"),
        (True, SHAPE, [100.0], "must be an integer"),
        (5, (6, 32, 8), [100.0], "four grid axes"),
        (5, SHAPE, [-1.0], "finite and nonnegative"),
        (5, SHAPE, [np.nan], "finite and nonnegative"),
    ],
)
def test_invalid_arguments_are_rejected(features, shape, offsets, message):
    with pytest.raises(ValueError, match=message):
        field_coordinates(_geometry(offsets), shape, features)


def test_the_production_offsets_never_reach_the_clip():
    """FORGE spans 4.63 to 2266.63 m and C3 720.28 to 1964.99 m, both inside the span."""
    for low, high in ((4.63, 2266.63), (720.28, 1964.99)):
        axis = field_coordinates(_geometry([low, high]), SHAPE, 5)[:, 4]
        assert 0.0 < axis[0] < axis[1] < 1.0
