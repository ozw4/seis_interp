"""Index fixed SEG C3 trace neighbors from physical acquisition geometry."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_float_dtype, is_integer_dtype

from seis_interp.data.trace_table import validated_array_rows

SOURCE_SHOT_SPACING_M = 80.0
RELATIVE_RECEIVER_X_MIN_M = -140.0
RELATIVE_RECEIVER_X_MAX_M = 140.0
RELATIVE_RECEIVER_Y_MIN_M = -2680.0
RELATIVE_RECEIVER_Y_MAX_M = 0.0
RECEIVER_SPACING_M = 40.0

TARGET_COORDINATE_ORDER = (
    "relative_receiver_x_m",
    "source_y_m",
    "relative_receiver_y_m",
)

_REQUIRED_COLUMNS = (
    "array_row",
    "source_x_m",
    "source_y_m",
    "receiver_x_m",
    "receiver_y_m",
)
_COORDINATE_COLUMNS = _REQUIRED_COLUMNS[1:]
_RELATIVE_RECEIVER_X_COUNT = 8
_RELATIVE_RECEIVER_Y_COUNT = 68
_LATTICE_ATOL_M = 1.0e-6


def _build_neighbor_offsets() -> tuple[tuple[int, int, int], ...]:
    values = [
        (delta_relative_rx, delta_source_shot, delta_relative_ry)
        for delta_source_shot in range(-2, 3)
        for delta_relative_rx in range(-1, 2)
        for delta_relative_ry in range(-3, 4)
        if (delta_relative_rx, delta_source_shot, delta_relative_ry) != (0, 0, 0)
    ]
    values.sort(
        key=lambda value: (
            4 * value[1] ** 2 + value[0] ** 2 + value[2] ** 2,
            value[1],
            value[0],
            value[2],
        )
    )
    return tuple(values)


NEIGHBOR_OFFSETS = _build_neighbor_offsets()
"""The 104 fixed ``(relative_rx, source_shot, relative_ry)`` index offsets."""

_OFFSET_ARRAY = np.asarray(NEIGHBOR_OFFSETS, dtype=np.int64)

__all__ = [
    "NEIGHBOR_OFFSETS",
    "NeighborGeometryLookup",
    "TARGET_COORDINATE_ORDER",
    "neighbor_offsets",
]


def neighbor_offsets() -> tuple[tuple[int, int, int], ...]:
    """Return the deterministic 104-channel SEG C3 neighbor geometry."""
    return NEIGHBOR_OFFSETS


class NeighborGeometryLookup:
    """Map target table positions to available physical SEG C3 neighbors.

    The lookup is constructed from every eligible trace geometry, but only
    rows selected by ``available`` are inserted as possible neighbors. Each
    source-x line owns an independent compressed shot lattice, so lookup never
    crosses source lines and does not allocate storage for absent shots.

    If multiple available rows occupy one physical cell, the row with the
    lowest ``array_row`` is indexed. ``collision_count`` records the number of
    other available rows discarded by that deterministic rule.
    """

    def __init__(self, trace_table: pd.DataFrame, available: np.ndarray) -> None:
        array_rows, coordinates = _validated_trace_table(trace_table)
        available_mask = _validated_available_mask(available, len(trace_table))
        if not np.any(available_mask):
            raise ValueError("available must select at least one trace row")

        source_x = coordinates[:, 0]
        source_y = coordinates[:, 1]
        relative_receiver_x = coordinates[:, 2] - source_x
        relative_receiver_y = coordinates[:, 3] - source_y
        if not (
            np.all(np.isfinite(relative_receiver_x)) and np.all(np.isfinite(relative_receiver_y))
        ):
            raise ValueError("relative receiver coordinates contain non-finite values")

        relative_rx_indices = _fixed_lattice_indices(
            relative_receiver_x,
            origin=RELATIVE_RECEIVER_X_MIN_M,
            spacing=RECEIVER_SPACING_M,
            count=_RELATIVE_RECEIVER_X_COUNT,
            name="receiver_x_m - source_x_m",
        )
        relative_ry_indices = _fixed_lattice_indices(
            relative_receiver_y,
            origin=RELATIVE_RECEIVER_Y_MIN_M,
            spacing=RECEIVER_SPACING_M,
            count=_RELATIVE_RECEIVER_Y_COUNT,
            name="receiver_y_m - source_y_m",
        )

        source_x_lines, line_indices = np.unique(source_x, return_inverse=True)
        source_y_origins = np.empty(len(source_x_lines), dtype=np.float64)
        source_shot_indices = np.empty(len(trace_table), dtype=np.int64)
        line_source_shots: list[np.ndarray] = []
        line_lattices: list[np.ndarray] = []
        collision_count = 0
        collision_cell_count = 0

        lattice_dtype = np.int32 if len(trace_table) <= np.iinfo(np.int32).max else np.int64
        for line_index in range(len(source_x_lines)):
            line_positions = np.flatnonzero(line_indices == line_index)
            line_source_y = source_y[line_positions]
            source_y_origin = float(np.min(line_source_y))
            source_y_origins[line_index] = source_y_origin
            line_shot_indices = _source_shot_indices(line_source_y, source_y_origin)
            source_shot_indices[line_positions] = line_shot_indices

            observed_source_shots = np.unique(line_shot_indices)
            line_source_shots.append(observed_source_shots)
            source_shot_slots = np.searchsorted(observed_source_shots, line_shot_indices)
            line_lattice = np.full(
                (
                    len(observed_source_shots),
                    _RELATIVE_RECEIVER_X_COUNT,
                    _RELATIVE_RECEIVER_Y_COUNT,
                ),
                -1,
                dtype=lattice_dtype,
            )

            line_available = available_mask[line_positions]
            available_positions = line_positions[line_available]
            if len(available_positions) > 0:
                available_slots = source_shot_slots[line_available]
                available_cells = (
                    available_slots * _RELATIVE_RECEIVER_X_COUNT
                    + relative_rx_indices[available_positions]
                ) * _RELATIVE_RECEIVER_Y_COUNT + relative_ry_indices[available_positions]
                order = np.lexsort((array_rows[available_positions], available_cells))
                sorted_cells = available_cells[order]
                sorted_positions = available_positions[order]
                first_in_cell = np.empty(len(sorted_cells), dtype=bool)
                first_in_cell[0] = True
                first_in_cell[1:] = sorted_cells[1:] != sorted_cells[:-1]
                winning_cells = sorted_cells[first_in_cell]
                winning_positions = sorted_positions[first_in_cell]
                collision_count += len(sorted_cells) - len(winning_cells)
                if len(sorted_cells) != len(winning_cells):
                    cell_counts = np.diff(
                        np.append(np.flatnonzero(first_in_cell), len(sorted_cells))
                    )
                    collision_cell_count += int(np.count_nonzero(cell_counts > 1))
                line_lattice.reshape(-1)[winning_cells] = winning_positions
            line_lattices.append(line_lattice)

        self._row_count = len(trace_table)
        self._line_indices = _compact_nonnegative(line_indices)
        self._source_shot_indices = _compact_nonnegative(source_shot_indices)
        self._relative_rx_indices = _compact_nonnegative(relative_rx_indices)
        self._relative_ry_indices = _compact_nonnegative(relative_ry_indices)
        self._source_y_origins = source_y_origins
        self._line_source_shots = tuple(line_source_shots)
        self._line_lattices = tuple(line_lattices)
        self._collision_count = int(collision_count)
        self._collision_cell_count = int(collision_cell_count)
        self._available_count = int(np.count_nonzero(available_mask))

        available_positions = np.flatnonzero(available_mask)
        available_coordinates = self._physical_target_coordinates(available_positions)
        self._coordinate_min = np.min(available_coordinates, axis=0)
        self._coordinate_max = np.max(available_coordinates, axis=0)

    @property
    def row_count(self) -> int:
        """Number of positions in the trace table used to build this lookup."""
        return self._row_count

    @property
    def neighbor_count(self) -> int:
        """Number of fixed neighbor channels returned for each target."""
        return len(NEIGHBOR_OFFSETS)

    @property
    def offsets(self) -> tuple[tuple[int, int, int], ...]:
        """Ordered relative geometry represented by the neighbor columns."""
        return NEIGHBOR_OFFSETS

    @property
    def available_count(self) -> int:
        """Number of input rows marked available, including collisions."""
        return self._available_count

    @property
    def indexed_available_count(self) -> int:
        """Number of distinct available physical cells indexed."""
        return self._available_count - self._collision_count

    @property
    def collision_count(self) -> int:
        """Number of extra available rows discarded from duplicate cells."""
        return self._collision_count

    @property
    def collision_cell_count(self) -> int:
        """Number of physical cells containing multiple available rows."""
        return self._collision_cell_count

    @property
    def coordinate_min(self) -> tuple[float, float, float]:
        """Training-only physical minima in ``TARGET_COORDINATE_ORDER``."""
        return tuple(float(value) for value in self._coordinate_min)

    @property
    def coordinate_max(self) -> tuple[float, float, float]:
        """Training-only physical maxima in ``TARGET_COORDINATE_ORDER``."""
        return tuple(float(value) for value in self._coordinate_max)

    def neighbor_positions(self, target_positions: np.ndarray) -> np.ndarray:
        """Return available table positions for each target and fixed offset.

        Missing, out-of-bounds, held-out, and cross-line cells are returned as
        ``-1``. The result always has signed ``int64`` dtype and shape
        ``[len(target_positions), 104]``.
        """
        positions = _validated_target_positions(target_positions, self._row_count)
        result = np.full((len(positions), len(NEIGHBOR_OFFSETS)), -1, dtype=np.int64)
        if len(positions) == 0:
            return result

        target_lines = self._line_indices[positions].astype(np.int64, copy=False)
        target_shots = self._source_shot_indices[positions].astype(np.int64, copy=False)
        target_rx = self._relative_rx_indices[positions].astype(np.int64, copy=False)
        target_ry = self._relative_ry_indices[positions].astype(np.int64, copy=False)

        neighbor_shot_delta = _OFFSET_ARRAY[:, 1]
        neighbor_rx = target_rx[:, None] + _OFFSET_ARRAY[:, 0]
        neighbor_ry = target_ry[:, None] + _OFFSET_ARRAY[:, 2]
        receiver_in_bounds = (
            (neighbor_rx >= 0)
            & (neighbor_rx < _RELATIVE_RECEIVER_X_COUNT)
            & (neighbor_ry >= 0)
            & (neighbor_ry < _RELATIVE_RECEIVER_Y_COUNT)
        )

        for line_index in np.unique(target_lines):
            batch_rows = np.flatnonzero(target_lines == line_index)
            observed_shots = self._line_source_shots[int(line_index)]
            wanted_shots = target_shots[batch_rows, None] + neighbor_shot_delta
            source_slots = np.searchsorted(observed_shots, wanted_shots)
            source_in_bounds = source_slots < len(observed_shots)
            safe_source_slots = np.minimum(source_slots, len(observed_shots) - 1)
            source_exists = source_in_bounds & (observed_shots[safe_source_slots] == wanted_shots)
            valid = source_exists & receiver_in_bounds[batch_rows]
            if not np.any(valid):
                continue

            batch_grid, offset_grid = np.nonzero(valid)
            lattice_values = self._line_lattices[int(line_index)][
                source_slots[batch_grid, offset_grid],
                neighbor_rx[batch_rows[batch_grid], offset_grid],
                neighbor_ry[batch_rows[batch_grid], offset_grid],
            ]
            result[batch_rows[batch_grid], offset_grid] = lattice_values
        return result

    def target_coordinates(self, target_positions: np.ndarray) -> np.ndarray:
        """Return training-fitted min-max coordinates for target positions.

        Available rows in each varying dimension span ``[-1, 1]``. A constant
        training dimension is represented by zero. Held-out coordinates are
        transformed with the same training-only scale and may extrapolate
        outside ``[-1, 1]``.
        """
        positions = _validated_target_positions(target_positions, self._row_count)
        coordinates = self._physical_target_coordinates(positions)
        coordinate_range = self._coordinate_max - self._coordinate_min
        normalized = np.zeros(coordinates.shape, dtype=np.float64)
        varying = coordinate_range != 0.0
        normalized[:, varying] = (
            2.0
            * (coordinates[:, varying] - self._coordinate_min[varying])
            / coordinate_range[varying]
            - 1.0
        )
        return normalized

    def _physical_target_coordinates(self, positions: np.ndarray) -> np.ndarray:
        line_indices = self._line_indices[positions].astype(np.int64, copy=False)
        source_shots = self._source_shot_indices[positions].astype(np.float64, copy=False)
        relative_rx = (
            RELATIVE_RECEIVER_X_MIN_M
            + self._relative_rx_indices[positions].astype(np.float64, copy=False)
            * RECEIVER_SPACING_M
        )
        source_y = self._source_y_origins[line_indices] + source_shots * SOURCE_SHOT_SPACING_M
        relative_ry = (
            RELATIVE_RECEIVER_Y_MIN_M
            + self._relative_ry_indices[positions].astype(np.float64, copy=False)
            * RECEIVER_SPACING_M
        )
        return np.column_stack((relative_rx, source_y, relative_ry))


def _validated_trace_table(trace_table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(trace_table, pd.DataFrame):
        raise TypeError(f"trace_table must be a pandas DataFrame, got {type(trace_table).__name__}")
    missing = [column for column in _REQUIRED_COLUMNS if column not in trace_table.columns]
    if missing:
        raise ValueError(f"trace table is missing required columns: {missing}")
    array_rows = validated_array_rows(trace_table)

    non_numeric = [
        column
        for column in _COORDINATE_COLUMNS
        if is_bool_dtype(trace_table[column].dtype)
        or not (
            is_integer_dtype(trace_table[column].dtype) or is_float_dtype(trace_table[column].dtype)
        )
    ]
    if non_numeric:
        raise ValueError(f"physical coordinate columns must be numeric: {non_numeric}")
    try:
        coordinates = trace_table[list(_COORDINATE_COLUMNS)].to_numpy(dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("physical coordinates must be numeric") from error
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("physical coordinates contain non-finite values")
    return array_rows, coordinates


def _validated_available_mask(available: np.ndarray, row_count: int) -> np.ndarray:
    mask = np.asarray(available)
    if mask.dtype != np.bool_:
        raise ValueError(f"available must have a boolean dtype, got {mask.dtype}")
    if mask.ndim != 1 or mask.shape != (row_count,):
        raise ValueError(f"available must have shape ({row_count},), got {mask.shape}")
    return mask


def _fixed_lattice_indices(
    values: np.ndarray,
    *,
    origin: float,
    spacing: float,
    count: int,
    name: str,
) -> np.ndarray:
    ratios = (values - origin) / spacing
    nearest = np.rint(ratios)
    aligned = np.abs(values - (origin + nearest * spacing)) <= _LATTICE_ATOL_M
    in_bounds = (nearest >= 0) & (nearest < count)
    if not np.all(aligned & in_bounds):
        bad_value = float(values[np.flatnonzero(~(aligned & in_bounds))[0]])
        maximum = origin + (count - 1) * spacing
        raise ValueError(
            f"{name} must align to the {spacing:g} m SEG C3 lattice in "
            f"[{origin:g}, {maximum:g}], got {bad_value:g}"
        )
    return nearest.astype(np.int64)


def _source_shot_indices(source_y: np.ndarray, origin: float) -> np.ndarray:
    ratios = (source_y - origin) / SOURCE_SHOT_SPACING_M
    nearest = np.rint(ratios)
    aligned = np.abs(source_y - (origin + nearest * SOURCE_SHOT_SPACING_M)) <= _LATTICE_ATOL_M
    if not np.all(aligned):
        bad_value = float(source_y[np.flatnonzero(~aligned)[0]])
        raise ValueError(
            "source_y_m must align to an 80 m shot lattice within each "
            f"source_x_m line, got {bad_value:g}"
        )
    if np.any(nearest > np.iinfo(np.int64).max):
        raise ValueError("source_y_m shot indices must fit in int64")
    return nearest.astype(np.int64)


def _validated_target_positions(target_positions: np.ndarray, row_count: int) -> np.ndarray:
    positions = np.asarray(target_positions)
    if positions.ndim != 1:
        raise ValueError(f"target_positions must be one-dimensional, got shape {positions.shape}")
    if is_bool_dtype(positions.dtype) or not is_integer_dtype(positions.dtype):
        raise ValueError(f"target_positions must have an integer dtype, got {positions.dtype}")
    if positions.size > 0 and (np.any(positions < 0) or np.any(positions >= row_count)):
        raise ValueError(f"target_positions must be within [0, {row_count})")
    return positions.astype(np.int64, copy=False)


def _compact_nonnegative(values: np.ndarray) -> np.ndarray:
    maximum = int(np.max(values))
    for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
        if maximum <= np.iinfo(dtype).max:
            return values.astype(dtype)
    raise AssertionError("no unsigned integer dtype can represent lattice indices")
