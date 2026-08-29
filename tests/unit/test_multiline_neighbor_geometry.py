from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seis_interp.processing.multiline_neighbor_geometry import (
    TARGET_COORDINATE_ORDER,
    MultilineNeighborGeometryLookup,
    multiline_neighbor_offsets,
    neighbor_offsets,
)


def _trace_row(
    array_row: int,
    *,
    source_x: float = 3860.0,
    source_y: float = 1020.0,
    relative_rx_index: int = 3,
    relative_ry_index: int = 10,
) -> dict[str, float | int]:
    relative_rx = -140.0 + 40.0 * relative_rx_index
    relative_ry = -2680.0 + 40.0 * relative_ry_index
    return {
        "array_row": array_row,
        "source_x_m": source_x,
        "source_y_m": source_y,
        "receiver_x_m": source_x + relative_rx,
        "receiver_y_m": source_y + relative_ry,
    }


def _table(*rows: dict[str, float | int]) -> pd.DataFrame:
    return pd.DataFrame(rows, index=np.arange(len(rows)) * 11 + 5)


def _lookup(
    trace_table: pd.DataFrame,
    available: np.ndarray,
    *,
    relative_receiver_x_radius: int = 1,
    source_x_line_radius: int = 1,
    source_y_half_shot_radius: int = 2,
    relative_receiver_y_radius: int = 1,
) -> MultilineNeighborGeometryLookup:
    return MultilineNeighborGeometryLookup(
        trace_table,
        available,
        relative_receiver_x_radius=relative_receiver_x_radius,
        source_x_line_radius=source_x_line_radius,
        source_y_half_shot_radius=source_y_half_shot_radius,
        relative_receiver_y_radius=relative_receiver_y_radius,
    )


def test_offsets_are_deterministic_physical_and_exclude_center() -> None:
    offsets = multiline_neighbor_offsets(1, 1, 2, 1)

    assert offsets == neighbor_offsets(1, 1, 2, 1)
    assert len(offsets) == 62
    assert len(set(offsets)) == len(offsets)
    assert (0, 0, 0, 0) not in offsets
    assert offsets[:4] == (
        (-1, 0, 0, 0),
        (0, 0, 0, -1),
        (0, 0, 0, 1),
        (1, 0, 0, 0),
    )
    assert all((delta_x + delta_y) % 2 == 0 for _, delta_x, delta_y, _ in offsets)
    assert (0, 1, 0, 0) not in offsets
    assert (0, 0, 1, 0) not in offsets
    assert (0, 1, 1, 0) in offsets
    assert (0, -1, 1, 0) in offsets


def test_cross_line_and_same_line_neighbors_follow_staggered_parity() -> None:
    trace_table = _table(
        _trace_row(100, source_x=3860.0, source_y=1020.0),
        _trace_row(101, source_x=3860.0, source_y=1100.0),
        _trace_row(102, source_x=4020.0, source_y=1060.0),
        _trace_row(103, source_x=4020.0, source_y=1140.0),
    )
    lookup = _lookup(
        trace_table,
        np.ones(4, dtype=bool),
        relative_receiver_x_radius=0,
        relative_receiver_y_radius=0,
    )

    neighbors = lookup.neighbor_positions(np.array([0, 2], dtype=np.int64))

    assert neighbors[0, lookup.offsets.index((0, 0, 2, 0))] == 1
    assert neighbors[0, lookup.offsets.index((0, 1, 1, 0))] == 2
    assert neighbors[1, lookup.offsets.index((0, 0, 2, 0))] == 3
    assert neighbors[1, lookup.offsets.index((0, -1, -1, 0))] == 0


def test_rejects_geometry_that_mixes_staggered_source_parities() -> None:
    trace_table = _table(
        _trace_row(0, source_x=3860.0, source_y=1020.0),
        _trace_row(1, source_x=3860.0, source_y=1060.0),
    )

    with pytest.raises(ValueError, match="staggered SEG C3 source parity"):
        _lookup(trace_table, np.ones(2, dtype=bool))


def test_row_positions_and_collision_winner_do_not_depend_on_dataframe_order() -> None:
    trace_table = _table(
        _trace_row(100),
        _trace_row(50, source_x=4020.0, source_y=1060.0),
        _trace_row(10, source_x=4020.0, source_y=1060.0),
        _trace_row(75, source_x=4020.0, source_y=1060.0),
    )
    lookup = _lookup(
        trace_table,
        np.ones(4, dtype=bool),
        relative_receiver_x_radius=0,
        source_y_half_shot_radius=1,
        relative_receiver_y_radius=0,
    )

    neighbors = lookup.neighbor_positions(np.array([0]))

    assert neighbors[0, lookup.offsets.index((0, 1, 1, 0))] == 2
    assert lookup.row_count == 4
    assert lookup.available_count == 4
    assert lookup.indexed_available_count == 2
    assert lookup.collision_count == 2
    assert lookup.collision_cell_count == 1


def test_boundaries_missing_cells_and_unavailable_rows_return_minus_one() -> None:
    trace_table = _table(
        _trace_row(0, relative_rx_index=0, relative_ry_index=0),
        _trace_row(1, relative_rx_index=1, relative_ry_index=0),
        _trace_row(2, relative_rx_index=0, relative_ry_index=1),
        _trace_row(3, source_x=4020.0, source_y=1060.0, relative_rx_index=0),
    )
    lookup = _lookup(trace_table, np.array([False, True, False, False]))

    neighbors = lookup.neighbor_positions(np.array([0]))

    assert neighbors.shape == (1, lookup.neighbor_count)
    assert neighbors.dtype == np.int64
    assert neighbors[0, lookup.offsets.index((1, 0, 0, 0))] == 1
    assert neighbors[0, lookup.offsets.index((-1, 0, 0, 0))] == -1
    assert neighbors[0, lookup.offsets.index((0, 0, 0, 1))] == -1
    assert neighbors[0, lookup.offsets.index((0, 1, 1, 0))] == -1
    assert neighbors[0, lookup.offsets.index((0, -1, -1, 0))] == -1
    assert neighbors[0, lookup.offsets.index((0, 0, 2, 0))] == -1


def test_target_coordinate_scaling_is_fitted_on_available_rows_only() -> None:
    trace_table = _table(
        _trace_row(0, source_x=3860.0, source_y=1020.0, relative_rx_index=0, relative_ry_index=0),
        _trace_row(1, source_x=4020.0, source_y=1060.0, relative_rx_index=1, relative_ry_index=2),
        _trace_row(2, source_x=4180.0, source_y=1180.0, relative_rx_index=6, relative_ry_index=66),
        _trace_row(3, source_x=4340.0, source_y=1300.0, relative_rx_index=7, relative_ry_index=67),
    )
    available = np.array([False, True, True, False])
    lookup = _lookup(trace_table, available)

    training_coordinates = lookup.target_coordinates(np.flatnonzero(available))
    held_out_coordinates = lookup.target_coordinates(np.array([0, 3]))

    assert TARGET_COORDINATE_ORDER == (
        "source_x_m",
        "source_y_m",
        "relative_receiver_x_m",
        "relative_receiver_y_m",
    )
    assert lookup.coordinate_min == (4020.0, 1060.0, -100.0, -2600.0)
    assert lookup.coordinate_max == (4180.0, 1180.0, 100.0, -40.0)
    np.testing.assert_allclose(training_coordinates, [[-1.0] * 4, [1.0] * 4])
    assert np.all(held_out_coordinates[0] < -1.0)
    assert np.all(held_out_coordinates[1] > 1.0)


@pytest.mark.parametrize(
    ("radii", "message"),
    [
        ((-1, 0, 0, 0), "relative_receiver_x_radius"),
        ((0, True, 0, 0), "source_x_line_radius"),
        ((0, 0, 1.5, 0), "source_y_half_shot_radius"),
        ((0, 0, 0, -1), "relative_receiver_y_radius"),
    ],
)
def test_validates_configurable_radii(radii: tuple[object, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        multiline_neighbor_offsets(*radii)  # type: ignore[arg-type]


def test_zero_radii_and_empty_target_batch_have_stable_shapes() -> None:
    lookup = _lookup(
        _table(_trace_row(0)),
        np.array([True]),
        relative_receiver_x_radius=0,
        source_x_line_radius=0,
        source_y_half_shot_radius=0,
        relative_receiver_y_radius=0,
    )

    assert lookup.offsets == ()
    assert lookup.neighbor_positions(np.array([0], dtype=np.int64)).shape == (1, 0)
    assert lookup.neighbor_positions(np.array([], dtype=np.int64)).shape == (0, 0)
    assert lookup.target_coordinates(np.array([], dtype=np.int64)).shape == (0, 4)


@pytest.mark.parametrize(
    ("positions", "message"),
    [
        (np.array([0.0]), "integer dtype"),
        (np.array([True]), "integer dtype"),
        (np.array([[0]]), "one-dimensional"),
        (np.array([-1]), "within"),
        (np.array([2]), "within"),
    ],
)
def test_validates_target_positions(positions: np.ndarray, message: str) -> None:
    lookup = _lookup(
        _table(_trace_row(0), _trace_row(1, source_y=1100.0)),
        np.ones(2, dtype=bool),
    )

    with pytest.raises(ValueError, match=message):
        lookup.neighbor_positions(positions)
    with pytest.raises(ValueError, match=message):
        lookup.target_coordinates(positions)
