from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seis_interp.processing.neighbor_geometry import (
    NEIGHBOR_OFFSETS,
    TARGET_COORDINATE_ORDER,
    NeighborGeometryLookup,
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
    return pd.DataFrame(rows, index=np.arange(len(rows)) * 10 + 7)


def _offset_column(offset: tuple[int, int, int]) -> int:
    return NEIGHBOR_OFFSETS.index(offset)


def test_neighbor_offsets_have_exact_count_order_and_exclude_center() -> None:
    expected = [
        (delta_rx, delta_shot, delta_ry)
        for delta_shot in range(-2, 3)
        for delta_rx in range(-1, 2)
        for delta_ry in range(-3, 4)
        if (delta_rx, delta_shot, delta_ry) != (0, 0, 0)
    ]
    expected.sort(
        key=lambda value: (
            4 * value[1] ** 2 + value[0] ** 2 + value[2] ** 2,
            value[1],
            value[0],
            value[2],
        )
    )

    assert neighbor_offsets() is NEIGHBOR_OFFSETS
    assert tuple(expected) == NEIGHBOR_OFFSETS
    assert len(NEIGHBOR_OFFSETS) == 104
    assert len(set(NEIGHBOR_OFFSETS)) == 104
    assert (0, 0, 0) not in NEIGHBOR_OFFSETS
    assert NEIGHBOR_OFFSETS[:4] == ((-1, 0, 0), (0, 0, -1), (0, 0, 1), (1, 0, 0))


def test_alternating_shot_line_y_parity_is_supported_without_crossing_lines() -> None:
    trace_table = _table(
        _trace_row(40, source_x=3860.0, source_y=1020.0),
        _trace_row(41, source_x=3860.0, source_y=1100.0),
        _trace_row(20, source_x=4020.0, source_y=1060.0),
        _trace_row(21, source_x=4020.0, source_y=1140.0),
    )
    lookup = NeighborGeometryLookup(trace_table, np.ones(4, dtype=bool))

    neighbors = lookup.neighbor_positions(np.array([0, 2], dtype=np.int64))

    assert neighbors[:, _offset_column((0, 1, 0))].tolist() == [1, 3]
    source_x = trace_table["source_x_m"].to_numpy()
    for target_position, target_neighbors in zip([0, 2], neighbors, strict=True):
        found = target_neighbors[target_neighbors >= 0]
        assert np.all(source_x[found] == source_x[target_position])


def test_incomplete_cells_and_unavailable_rows_return_minus_one() -> None:
    trace_table = _table(
        _trace_row(0),
        _trace_row(1, relative_ry_index=11),
        _trace_row(2, relative_rx_index=4),
    )
    lookup = NeighborGeometryLookup(
        trace_table,
        np.array([False, False, True], dtype=bool),
    )

    neighbors = lookup.neighbor_positions(np.array([0]))

    assert neighbors.shape == (1, 104)
    assert neighbors.dtype == np.int64
    assert neighbors[0, _offset_column((0, 0, 1))] == -1
    assert neighbors[0, _offset_column((1, 0, 0))] == 2
    assert neighbors[0, _offset_column((-1, 0, 0))] == -1


def test_duplicate_available_cell_prefers_lowest_array_row_and_records_collisions() -> None:
    trace_table = _table(
        _trace_row(100),
        _trace_row(50, relative_rx_index=4),
        _trace_row(10, relative_rx_index=4),
        _trace_row(75, relative_rx_index=4),
    )
    lookup = NeighborGeometryLookup(trace_table, np.ones(4, dtype=bool))

    neighbors = lookup.neighbor_positions(np.array([0]))

    assert neighbors[0, _offset_column((1, 0, 0))] == 2
    assert lookup.available_count == 4
    assert lookup.indexed_available_count == 2
    assert lookup.collision_count == 2
    assert lookup.collision_cell_count == 1


def test_target_coordinates_use_available_rows_only_and_bound_training_values() -> None:
    trace_table = _table(
        _trace_row(0, source_y=1020.0, relative_rx_index=0, relative_ry_index=0),
        _trace_row(1, source_y=1100.0, relative_rx_index=1, relative_ry_index=2),
        _trace_row(2, source_y=1260.0, relative_rx_index=6, relative_ry_index=66),
        _trace_row(3, source_y=1340.0, relative_rx_index=7, relative_ry_index=67),
    )
    available = np.array([False, True, True, False])
    lookup = NeighborGeometryLookup(trace_table, available)

    training_coordinates = lookup.target_coordinates(np.flatnonzero(available))
    held_out_coordinates = lookup.target_coordinates(np.array([0, 3]))

    assert TARGET_COORDINATE_ORDER == (
        "relative_receiver_x_m",
        "source_y_m",
        "relative_receiver_y_m",
    )
    assert lookup.coordinate_min == (-100.0, 1100.0, -2600.0)
    assert lookup.coordinate_max == (100.0, 1260.0, -40.0)
    np.testing.assert_allclose(training_coordinates, [[-1.0] * 3, [1.0] * 3])
    assert np.all(held_out_coordinates[0] < -1.0)
    assert np.all(held_out_coordinates[1] > 1.0)


def test_constant_training_coordinate_dimension_normalizes_to_zero() -> None:
    trace_table = _table(
        _trace_row(0, source_y=1020.0),
        _trace_row(1, source_y=1100.0),
    )
    lookup = NeighborGeometryLookup(trace_table, np.ones(2, dtype=bool))

    coordinates = lookup.target_coordinates(np.array([0, 1]))

    np.testing.assert_allclose(coordinates[:, [0, 2]], 0.0)
    np.testing.assert_allclose(coordinates[:, 1], [-1.0, 1.0])


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda table: table.assign(receiver_x_m=table["receiver_x_m"] + 1.0),
            "40 m SEG C3 lattice",
        ),
        (
            lambda table: table.assign(source_y_m=[1020.0, 1060.0]),
            "80 m shot lattice",
        ),
        (
            lambda table: table.assign(receiver_y_m=[np.nan, table["receiver_y_m"].iloc[1]]),
            "non-finite",
        ),
    ],
)
def test_rejects_non_lattice_and_non_finite_geometry(mutate: object, message: str) -> None:
    trace_table = _table(_trace_row(0), _trace_row(1, source_y=1100.0))

    with pytest.raises(ValueError, match=message):
        NeighborGeometryLookup(
            mutate(trace_table),  # type: ignore[operator]
            np.ones(2, dtype=bool),
        )


@pytest.mark.parametrize(
    ("available", "message"),
    [
        (np.array([1, 0]), "boolean dtype"),
        (np.array([[True, False]]), "shape"),
        (np.array([True]), "shape"),
        (np.array([False, False]), "at least one"),
    ],
)
def test_validates_available_mask(available: np.ndarray, message: str) -> None:
    trace_table = _table(_trace_row(0), _trace_row(1, source_y=1100.0))

    with pytest.raises(ValueError, match=message):
        NeighborGeometryLookup(trace_table, available)


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
    trace_table = _table(_trace_row(0), _trace_row(1, source_y=1100.0))
    lookup = NeighborGeometryLookup(trace_table, np.ones(2, dtype=bool))

    with pytest.raises(ValueError, match=message):
        lookup.neighbor_positions(positions)
    with pytest.raises(ValueError, match=message):
        lookup.target_coordinates(positions)


def test_empty_target_batch_has_stable_shapes() -> None:
    lookup = NeighborGeometryLookup(_table(_trace_row(0)), np.array([True]))

    assert lookup.neighbor_positions(np.array([], dtype=np.int64)).shape == (0, 104)
    assert lookup.target_coordinates(np.array([], dtype=np.int64)).shape == (0, 3)
