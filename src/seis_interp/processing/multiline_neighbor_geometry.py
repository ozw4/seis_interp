"""Index train-only SEG C3 neighbors across staggered source lines."""

from __future__ import annotations

from operator import index

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_float_dtype, is_integer_dtype

from seis_interp.data.trace_table import validated_array_rows
from seis_interp.processing.neighbor_geometry import (
    RECEIVER_SPACING_M,
    RELATIVE_RECEIVER_X_MIN_M,
    RELATIVE_RECEIVER_Y_MIN_M,
)

SOURCE_X_LINE_SPACING_M = 160.0
SOURCE_Y_HALF_SHOT_SPACING_M = 40.0

TARGET_COORDINATE_ORDER = (
    "source_x_m",
    "source_y_m",
    "relative_receiver_x_m",
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

MultilineNeighborOffset = tuple[int, int, int, int]

__all__ = [
    "SOURCE_X_LINE_SPACING_M",
    "SOURCE_Y_HALF_SHOT_SPACING_M",
    "TARGET_COORDINATE_ORDER",
    "MultilineNeighborGeometryLookup",
    "MultilineNeighborOffset",
    "multiline_neighbor_offsets",
    "neighbor_offsets",
]


def multiline_neighbor_offsets(
    relative_receiver_x_radius: int,
    source_x_line_radius: int,
    source_y_half_shot_radius: int,
    relative_receiver_y_radius: int,
) -> tuple[MultilineNeighborOffset, ...]:
    """Return deterministic offsets on the staggered SEG C3 source lattice.

    Offset components are ordered as ``(relative_rx, source_x_line,
    source_y_half_shot, relative_ry)``. Adjacent source-x lines are staggered
    by one 40 m source-y half-shot, so a source offset is physical exactly when
    ``delta_source_x_line + delta_source_y_half_shot`` is even.
    """
    relative_rx_radius = _validated_radius(
        relative_receiver_x_radius,
        name="relative_receiver_x_radius",
    )
    source_x_radius = _validated_radius(
        source_x_line_radius,
        name="source_x_line_radius",
    )
    source_y_radius = _validated_radius(
        source_y_half_shot_radius,
        name="source_y_half_shot_radius",
    )
    relative_ry_radius = _validated_radius(
        relative_receiver_y_radius,
        name="relative_receiver_y_radius",
    )

    values = [
        (delta_relative_rx, delta_source_x, delta_source_y, delta_relative_ry)
        for delta_source_x in range(-source_x_radius, source_x_radius + 1)
        for delta_source_y in range(-source_y_radius, source_y_radius + 1)
        if (delta_source_x + delta_source_y) % 2 == 0
        for delta_relative_rx in range(-relative_rx_radius, relative_rx_radius + 1)
        for delta_relative_ry in range(-relative_ry_radius, relative_ry_radius + 1)
        if (delta_relative_rx, delta_source_x, delta_source_y, delta_relative_ry) != (0, 0, 0, 0)
    ]
    values.sort(
        key=lambda value: (
            value[0] ** 2 + 16 * value[1] ** 2 + value[2] ** 2 + value[3] ** 2,
            value[1],
            value[2],
            value[0],
            value[3],
        )
    )
    return tuple(values)


def neighbor_offsets(
    relative_receiver_x_radius: int,
    source_x_line_radius: int,
    source_y_half_shot_radius: int,
    relative_receiver_y_radius: int,
) -> tuple[MultilineNeighborOffset, ...]:
    """Alias matching the offset helper name used by the single-line lookup."""
    return multiline_neighbor_offsets(
        relative_receiver_x_radius,
        source_x_line_radius,
        source_y_half_shot_radius,
        relative_receiver_y_radius,
    )


class MultilineNeighborGeometryLookup:
    """Map trace-table positions to train-only neighbors on all source lines.

    Geometry from every input row defines the observed acquisition lattice,
    while only rows selected by ``available`` can populate neighbor channels or
    fit target-coordinate scaling. Duplicate available physical cells retain
    the table position whose ``array_row`` is lowest.
    """

    def __init__(
        self,
        trace_table: pd.DataFrame,
        available: np.ndarray,
        *,
        relative_receiver_x_radius: int,
        source_x_line_radius: int,
        source_y_half_shot_radius: int,
        relative_receiver_y_radius: int,
    ) -> None:
        array_rows, coordinates = _validated_trace_table(trace_table)
        available_mask = _validated_available_mask(available, len(trace_table))
        if not np.any(available_mask):
            raise ValueError("available must select at least one trace row")

        offsets = multiline_neighbor_offsets(
            relative_receiver_x_radius,
            source_x_line_radius,
            source_y_half_shot_radius,
            relative_receiver_y_radius,
        )
        offset_array = np.asarray(offsets, dtype=np.int64).reshape(-1, 4)

        source_x = coordinates[:, 0]
        source_y = coordinates[:, 1]
        relative_receiver_x = coordinates[:, 2] - source_x
        relative_receiver_y = coordinates[:, 3] - source_y
        if not (
            np.all(np.isfinite(relative_receiver_x)) and np.all(np.isfinite(relative_receiver_y))
        ):
            raise ValueError("relative receiver coordinates contain non-finite values")

        source_x_origin = float(np.min(source_x))
        source_y_origin = float(np.min(source_y))
        source_x_indices = _regular_lattice_indices(
            source_x,
            origin=source_x_origin,
            spacing=SOURCE_X_LINE_SPACING_M,
            name="source_x_m",
        )
        source_y_half_indices = _regular_lattice_indices(
            source_y,
            origin=source_y_origin,
            spacing=SOURCE_Y_HALF_SHOT_SPACING_M,
            name="source_y_m",
        )
        source_parity = (source_x_indices + source_y_half_indices) % 2
        if not np.all(source_parity == source_parity[0]):
            raise ValueError(
                "source_x_m and source_y_m must follow one staggered SEG C3 source parity"
            )

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

        observed_source_x_indices, line_slots = np.unique(
            source_x_indices,
            return_inverse=True,
        )
        line_source_y_halves: list[np.ndarray] = []
        line_lattices: list[np.ndarray] = []
        collision_count = 0
        collision_cell_count = 0
        lattice_dtype = np.int32 if len(trace_table) <= np.iinfo(np.int32).max else np.int64

        for line_slot in range(len(observed_source_x_indices)):
            line_positions = np.flatnonzero(line_slots == line_slot)
            line_halves = source_y_half_indices[line_positions]
            observed_halves = np.unique(line_halves)
            line_source_y_halves.append(observed_halves)
            source_y_slots = np.searchsorted(observed_halves, line_halves)
            line_lattice = np.full(
                (
                    len(observed_halves),
                    _RELATIVE_RECEIVER_X_COUNT,
                    _RELATIVE_RECEIVER_Y_COUNT,
                ),
                -1,
                dtype=lattice_dtype,
            )

            line_available = available_mask[line_positions]
            available_positions = line_positions[line_available]
            if len(available_positions) > 0:
                available_cells = (
                    source_y_slots[line_available] * _RELATIVE_RECEIVER_X_COUNT
                    + relative_rx_indices[available_positions]
                ) * _RELATIVE_RECEIVER_Y_COUNT + relative_ry_indices[available_positions]
                order = np.lexsort((array_rows[available_positions], available_cells))
                sorted_cells = available_cells[order]
                sorted_positions = available_positions[order]
                first_in_cell = np.empty(len(sorted_cells), dtype=bool)
                first_in_cell[0] = True
                first_in_cell[1:] = sorted_cells[1:] != sorted_cells[:-1]
                winning_cells = sorted_cells[first_in_cell]
                line_lattice.reshape(-1)[winning_cells] = sorted_positions[first_in_cell]

                collision_count += len(sorted_cells) - len(winning_cells)
                if len(sorted_cells) != len(winning_cells):
                    cell_counts = np.diff(
                        np.append(np.flatnonzero(first_in_cell), len(sorted_cells))
                    )
                    collision_cell_count += int(np.count_nonzero(cell_counts > 1))
            line_lattices.append(line_lattice)

        self._row_count = len(trace_table)
        self._offsets = offsets
        self._offset_array = offset_array
        self._source_x_origin = source_x_origin
        self._source_y_origin = source_y_origin
        self._source_x_indices = _compact_nonnegative(source_x_indices)
        self._source_y_half_indices = _compact_nonnegative(source_y_half_indices)
        self._relative_rx_indices = _compact_nonnegative(relative_rx_indices)
        self._relative_ry_indices = _compact_nonnegative(relative_ry_indices)
        self._observed_source_x_indices = observed_source_x_indices
        self._line_source_y_halves = tuple(line_source_y_halves)
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
        """Number of configured neighbor channels returned for each target."""
        return len(self._offsets)

    @property
    def offsets(self) -> tuple[MultilineNeighborOffset, ...]:
        """Ordered relative geometry represented by the neighbor columns."""
        return self._offsets

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
    def coordinate_min(self) -> tuple[float, float, float, float]:
        """Train-only physical minima in ``TARGET_COORDINATE_ORDER``."""
        return tuple(float(value) for value in self._coordinate_min)

    @property
    def coordinate_max(self) -> tuple[float, float, float, float]:
        """Train-only physical maxima in ``TARGET_COORDINATE_ORDER``."""
        return tuple(float(value) for value in self._coordinate_max)

    def neighbor_positions(self, target_positions: np.ndarray) -> np.ndarray:
        """Return available table positions for every target and offset.

        Missing source lines, missing shots, receiver boundaries, and rows not
        selected by ``available`` produce ``-1``. Values are signed ``int64``
        table positions, independent of the DataFrame index and ``array_row``.
        """
        positions = _validated_target_positions(target_positions, self._row_count)
        result = np.full((len(positions), len(self._offsets)), -1, dtype=np.int64)
        if len(positions) == 0 or len(self._offsets) == 0:
            return result

        target_source_x = self._source_x_indices[positions].astype(np.int64, copy=False)
        target_source_y = self._source_y_half_indices[positions].astype(np.int64, copy=False)
        target_rx = self._relative_rx_indices[positions].astype(np.int64, copy=False)
        target_ry = self._relative_ry_indices[positions].astype(np.int64, copy=False)

        wanted_source_x = target_source_x[:, None] + self._offset_array[:, 1]
        wanted_source_y = target_source_y[:, None] + self._offset_array[:, 2]
        wanted_rx = target_rx[:, None] + self._offset_array[:, 0]
        wanted_ry = target_ry[:, None] + self._offset_array[:, 3]

        line_slots = np.searchsorted(self._observed_source_x_indices, wanted_source_x)
        line_in_bounds = line_slots < len(self._observed_source_x_indices)
        safe_line_slots = np.minimum(line_slots, len(self._observed_source_x_indices) - 1)
        line_exists = line_in_bounds & (
            self._observed_source_x_indices[safe_line_slots] == wanted_source_x
        )
        receiver_in_bounds = (
            (wanted_rx >= 0)
            & (wanted_rx < _RELATIVE_RECEIVER_X_COUNT)
            & (wanted_ry >= 0)
            & (wanted_ry < _RELATIVE_RECEIVER_Y_COUNT)
        )
        potentially_valid = line_exists & receiver_in_bounds

        for line_slot in np.unique(safe_line_slots[potentially_valid]):
            line_mask = potentially_valid & (safe_line_slots == line_slot)
            target_grid, offset_grid = np.nonzero(line_mask)
            observed_halves = self._line_source_y_halves[int(line_slot)]
            desired_halves = wanted_source_y[target_grid, offset_grid]
            source_y_slots = np.searchsorted(observed_halves, desired_halves)
            source_y_in_bounds = source_y_slots < len(observed_halves)
            safe_source_y_slots = np.minimum(source_y_slots, len(observed_halves) - 1)
            source_y_exists = source_y_in_bounds & (
                observed_halves[safe_source_y_slots] == desired_halves
            )
            if not np.any(source_y_exists):
                continue

            valid_targets = target_grid[source_y_exists]
            valid_offsets = offset_grid[source_y_exists]
            lattice_values = self._line_lattices[int(line_slot)][
                safe_source_y_slots[source_y_exists],
                wanted_rx[valid_targets, valid_offsets],
                wanted_ry[valid_targets, valid_offsets],
            ]
            result[valid_targets, valid_offsets] = lattice_values
        return result

    def target_coordinates(self, target_positions: np.ndarray) -> np.ndarray:
        """Return coordinates min-max scaled with available rows only."""
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
        source_x = (
            self._source_x_origin
            + self._source_x_indices[positions].astype(np.float64, copy=False)
            * SOURCE_X_LINE_SPACING_M
        )
        source_y = (
            self._source_y_origin
            + self._source_y_half_indices[positions].astype(np.float64, copy=False)
            * SOURCE_Y_HALF_SHOT_SPACING_M
        )
        relative_rx = (
            RELATIVE_RECEIVER_X_MIN_M
            + self._relative_rx_indices[positions].astype(np.float64, copy=False)
            * RECEIVER_SPACING_M
        )
        relative_ry = (
            RELATIVE_RECEIVER_Y_MIN_M
            + self._relative_ry_indices[positions].astype(np.float64, copy=False)
            * RECEIVER_SPACING_M
        )
        return np.column_stack((source_x, source_y, relative_rx, relative_ry))


def _validated_radius(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    try:
        radius = index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}") from error
    if radius < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return radius


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


def _regular_lattice_indices(
    values: np.ndarray,
    *,
    origin: float,
    spacing: float,
    name: str,
) -> np.ndarray:
    ratios = (values - origin) / spacing
    nearest = np.rint(ratios)
    aligned = np.abs(values - (origin + nearest * spacing)) <= _LATTICE_ATOL_M
    int64_info = np.iinfo(np.int64)
    representable = (nearest >= 0) & (nearest <= int64_info.max)
    if not np.all(aligned & representable):
        bad_value = float(values[np.flatnonzero(~(aligned & representable))[0]])
        raise ValueError(
            f"{name} must align to the {spacing:g} m SEG C3 lattice, got {bad_value:g}"
        )
    return nearest.astype(np.int64)


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
