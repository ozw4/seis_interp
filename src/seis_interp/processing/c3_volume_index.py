"""Build deterministic dense SEG C3 NA shot-receiver volume indices."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_complex_dtype,
    is_integer_dtype,
    is_numeric_dtype,
)

from seis_interp.data.trace_table import validated_array_rows
from seis_interp.processing.c3_receiver_grid import receiver_grid_offsets
from seis_interp.processing.multiline_neighbor_geometry import (
    SOURCE_X_LINE_SPACING_M,
    SOURCE_Y_HALF_SHOT_SPACING_M,
)
from seis_interp.processing.neighbor_geometry import SOURCE_SHOT_SPACING_M

VOLUME_AXIS_ORDER = (
    "time",
    "source_line",
    "shot_in_line",
    "relative_receiver_x",
    "relative_receiver_y",
)

SPATIAL_AXIS_ORDER = VOLUME_AXIS_ORDER[1:]

VOLUME_INDEX_COLUMNS = (
    "array_row",
    "ffid",
    "source_line_index",
    "shot_in_line_index",
    "relative_receiver_x_index",
    "relative_receiver_y_index",
    "source_x_m",
    "source_y_m",
    "relative_receiver_x_m",
    "relative_receiver_y_m",
)

INDEX_CONTRACT = {
    "selection_ranges": "zero_based_half_open",
    "table_indices": "zero_based_local_to_selection",
    "source_line": "ascending_unique_source_x_m",
    "shot_in_line": "ascending_unique_source_y_m_within_source_line",
    "relative_receiver_x": "ascending_unique_receiver_x_m_minus_source_x_m",
    "relative_receiver_y": "ascending_unique_receiver_y_m_minus_source_y_m",
}

_REQUIRED_COLUMNS = (
    "array_row",
    "ffid",
    "source_x_m",
    "source_y_m",
    "receiver_x_m",
    "receiver_y_m",
)
_INDEX_COLUMNS = (
    "source_line_index",
    "shot_in_line_index",
    "relative_receiver_x_index",
    "relative_receiver_y_index",
)
_COORDINATE_COLUMNS = ("source_x_m", "source_y_m", "receiver_x_m", "receiver_y_m")
_SOURCE_COORDINATE_COLUMNS = ("source_x_m", "source_y_m")
_LATTICE_ATOL_M = 1.0e-6


def validated_index_range(value: object, *, name: str) -> tuple[int, int]:
    """Return one validated zero-based half-open index range."""
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, (Sequence, np.ndarray))
        or len(value) != 2
    ):
        raise ValueError(f"{name} must be a two-element integer sequence")
    start, stop = value
    if any(isinstance(item, bool) or not isinstance(item, Integral) for item in (start, stop)):
        raise ValueError(f"{name} must be a two-element integer sequence")
    result = (int(start), int(stop))
    if result[0] < 0 or result[1] < 0:
        raise ValueError(f"{name} values must be nonnegative")
    if result[0] >= result[1]:
        raise ValueError(f"{name} start must be less than stop")
    return result


def selected_spatial_shape(
    *,
    source_line_range: tuple[int, int],
    shot_in_line_range: tuple[int, int],
    relative_receiver_x_range: tuple[int, int],
    relative_receiver_y_range: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Return the spatial shape selected by four half-open ranges."""
    ranges = (
        validated_index_range(source_line_range, name="source_line_range"),
        validated_index_range(shot_in_line_range, name="shot_in_line_range"),
        validated_index_range(relative_receiver_x_range, name="relative_receiver_x_range"),
        validated_index_range(relative_receiver_y_range, name="relative_receiver_y_range"),
    )
    return tuple(stop - start for start, stop in ranges)  # type: ignore[return-value]


def c3_source_indices(trace_table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return source-line and line-local shot ranks aligned to table rows."""
    if not isinstance(trace_table, pd.DataFrame):
        raise TypeError(f"trace_table must be a pandas DataFrame, got {type(trace_table).__name__}")
    if trace_table.empty:
        raise ValueError("trace table is empty")
    missing = [column for column in _SOURCE_COORDINATE_COLUMNS if column not in trace_table.columns]
    if missing:
        raise ValueError(f"trace table is missing required columns: {missing}")

    coordinates: dict[str, np.ndarray] = {}
    for column in _SOURCE_COORDINATE_COLUMNS:
        values = trace_table[column]
        if (
            not is_numeric_dtype(values.dtype)
            or is_bool_dtype(values.dtype)
            or is_complex_dtype(values.dtype)
        ):
            raise ValueError(f"{column} must contain real numeric values")
        try:
            numeric = values.to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{column} must contain real numeric values") from error
        if not np.all(np.isfinite(numeric)):
            raise ValueError(f"{column} must contain finite values")
        coordinates[column] = numeric

    source_x = coordinates["source_x_m"]
    source_y = coordinates["source_y_m"]
    _, source_line_indices = np.unique(source_x, return_inverse=True)
    shot_in_line_indices = np.empty(len(trace_table), dtype=np.int64)
    for source_line_index in range(int(source_line_indices.max()) + 1):
        line_positions = source_line_indices == source_line_index
        _, line_shot_indices = np.unique(source_y[line_positions], return_inverse=True)
        shot_in_line_indices[line_positions] = line_shot_indices
    return (
        source_line_indices.astype(np.int64, copy=False),
        shot_in_line_indices,
    )


def build_c3_volume_index(
    trace_table: pd.DataFrame,
    candidate_array_rows: np.ndarray,
    *,
    source_line_range: tuple[int, int],
    shot_in_line_range: tuple[int, int],
    relative_receiver_x_range: tuple[int, int],
    relative_receiver_y_range: tuple[int, int],
) -> pd.DataFrame:
    """Map canonical candidate traces into one dense selected spatial crop."""
    table = _validated_trace_table(trace_table)
    candidates = _validated_candidate_rows(candidate_array_rows, table)
    ranges = (
        validated_index_range(source_line_range, name="source_line_range"),
        validated_index_range(shot_in_line_range, name="shot_in_line_range"),
        validated_index_range(relative_receiver_x_range, name="relative_receiver_x_range"),
        validated_index_range(relative_receiver_y_range, name="relative_receiver_y_range"),
    )

    source_line_indices, shot_ranks = c3_source_indices(table)
    source_x_values = np.sort(table["source_x_m"].unique())
    if ranges[0][1] > len(source_x_values):
        raise ValueError(
            f"source_line_range {ranges[0]} is outside the {len(source_x_values)} source lines"
        )

    selected_source_y_values: list[np.ndarray] = []
    for source_line_rank in range(*ranges[0]):
        positions = source_line_indices == source_line_rank
        source_y_values = np.sort(table.loc[positions, "source_y_m"].unique())
        if ranges[1][1] > len(source_y_values):
            raise ValueError(
                f"shot_in_line_range {ranges[1]} is outside source line "
                f"{source_line_rank}, which has {len(source_y_values)} shots"
            )
        selected_source_y_values.append(source_y_values[slice(*ranges[1])])
    _validate_selected_source_grid(
        source_x_values[slice(*ranges[0])],
        selected_source_y_values,
        first_source_line_index=ranges[0][0],
    )

    receiver_x_values, receiver_y_values = receiver_grid_offsets(table)
    if ranges[2][1] > len(receiver_x_values):
        raise ValueError(f"relative_receiver_x_range {ranges[2]} is outside the receiver grid")
    if ranges[3][1] > len(receiver_y_values):
        raise ValueError(f"relative_receiver_y_range {ranges[3]} is outside the receiver grid")

    source_x = table["source_x_m"].to_numpy(dtype=np.float64)
    source_y = table["source_y_m"].to_numpy(dtype=np.float64)
    relative_x = table["receiver_x_m"].to_numpy(dtype=np.float64) - source_x
    relative_y = table["receiver_y_m"].to_numpy(dtype=np.float64) - source_y
    global_indices = (
        source_line_indices,
        shot_ranks,
        np.searchsorted(receiver_x_values, relative_x),
        np.searchsorted(receiver_y_values, relative_y),
    )

    selected = np.isin(table["array_row"].to_numpy(dtype=np.int64), candidates)
    for values, (start, stop) in zip(global_indices, ranges, strict=True):
        selected &= (values >= start) & (values < stop)

    result = pd.DataFrame(
        {
            "array_row": table.loc[selected, "array_row"].to_numpy(dtype=np.int64),
            "ffid": table.loc[selected, "ffid"].to_numpy(dtype=np.int64),
            **{
                column: values[selected].astype(np.int64, copy=False) - index_range[0]
                for column, values, index_range in zip(
                    _INDEX_COLUMNS, global_indices, ranges, strict=True
                )
            },
            "source_x_m": source_x[selected],
            "source_y_m": source_y[selected],
            "relative_receiver_x_m": relative_x[selected],
            "relative_receiver_y_m": relative_y[selected],
        },
        columns=VOLUME_INDEX_COLUMNS,
    )

    _require_dense_crop(result, tuple(stop - start for start, stop in ranges))
    result = result.sort_values(list(_INDEX_COLUMNS), kind="stable").reset_index(drop=True)
    for column in VOLUME_INDEX_COLUMNS[:6]:
        result[column] = result[column].astype(np.int64)
    for column in VOLUME_INDEX_COLUMNS[6:]:
        result[column] = result[column].astype(np.float64)
    return result


def _validated_trace_table(trace_table: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(trace_table, pd.DataFrame):
        raise TypeError(f"trace_table must be a pandas DataFrame, got {type(trace_table).__name__}")
    missing = [column for column in _REQUIRED_COLUMNS if column not in trace_table.columns]
    if missing:
        raise ValueError(f"trace table is missing required columns: {missing}")
    array_rows = validated_array_rows(trace_table)
    if np.any(array_rows < 0):
        raise ValueError("array_row values must be nonnegative")

    ffids = trace_table["ffid"]
    if ffids.isna().any() or is_bool_dtype(ffids.dtype) or not is_integer_dtype(ffids.dtype):
        raise ValueError("ffid must have a non-missing integer dtype")
    int64_info = np.iinfo(np.int64)
    if int(ffids.min()) < int64_info.min or int(ffids.max()) > int64_info.max:
        raise ValueError("ffid values must fit in int64")

    for column in _COORDINATE_COLUMNS:
        values = trace_table[column]
        if (
            not is_numeric_dtype(values.dtype)
            or is_bool_dtype(values.dtype)
            or is_complex_dtype(values.dtype)
        ):
            raise ValueError(f"{column} must contain real numeric values")
        try:
            numeric = values.to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{column} must contain real numeric values") from error
        if not np.all(np.isfinite(numeric)):
            raise ValueError(f"{column} must contain finite values")

    source_per_ffid = trace_table.groupby("ffid", sort=False)[["source_x_m", "source_y_m"]].nunique(
        dropna=False
    )
    if bool((source_per_ffid > 1).any(axis=None)):
        raise ValueError("each FFID must correspond to exactly one source position")
    ffids_per_source = trace_table.groupby(["source_x_m", "source_y_m"], sort=False)[
        "ffid"
    ].nunique(dropna=False)
    if bool((ffids_per_source > 1).any()):
        raise ValueError("each source position must correspond to exactly one FFID")
    return trace_table.copy(deep=False)


def _validated_candidate_rows(values: np.ndarray, trace_table: pd.DataFrame) -> np.ndarray:
    candidates = np.asarray(values)
    if candidates.ndim != 1 or candidates.dtype.kind not in "iu" or candidates.dtype.kind == "b":
        raise ValueError("candidate_array_rows must be a one-dimensional integer array")
    if len(candidates) == 0:
        raise ValueError("candidate_array_rows must not be empty")
    if len(np.unique(candidates)) != len(candidates):
        raise ValueError("candidate_array_rows must not contain duplicates")
    int64_info = np.iinfo(np.int64)
    if int(candidates.min()) < 0 or int(candidates.max()) > int64_info.max:
        raise ValueError("candidate_array_rows values must be nonnegative and fit in int64")
    candidates = candidates.astype(np.int64, copy=False)
    known = trace_table["array_row"].to_numpy(dtype=np.int64)
    missing = candidates[~np.isin(candidates, known)]
    if len(missing):
        raise ValueError(
            "candidate_array_rows contains values absent from the trace table: "
            f"{missing[:5].tolist()}"
        )
    return candidates


def _validate_selected_source_grid(
    source_x_values: np.ndarray,
    source_y_values_by_line: list[np.ndarray],
    *,
    first_source_line_index: int,
) -> None:
    _require_regular_spacing(
        source_x_values,
        spacing=SOURCE_X_LINE_SPACING_M,
        name="selected source lines",
    )
    for offset, source_y_values in enumerate(source_y_values_by_line):
        source_line_index = first_source_line_index + offset
        _require_regular_spacing(
            source_y_values,
            spacing=SOURCE_SHOT_SPACING_M,
            name=f"selected shots in source line {source_line_index}",
        )

    for offset in range(len(source_y_values_by_line) - 1):
        first = source_y_values_by_line[offset]
        second = source_y_values_by_line[offset + 1]
        differences = np.abs(second - first)
        valid = np.isclose(
            differences,
            SOURCE_Y_HALF_SHOT_SPACING_M,
            rtol=0.0,
            atol=_LATTICE_ATOL_M,
        )
        if not np.all(valid):
            bad_difference = float(differences[np.flatnonzero(~valid)[0]])
            first_line = first_source_line_index + offset
            raise ValueError(
                "adjacent selected source lines must have a "
                f"{SOURCE_Y_HALF_SHOT_SPACING_M:g} m source-y stagger at matching "
                f"shot_in_line indices; source lines {first_line} and {first_line + 1} "
                f"have a {bad_difference:g} m difference"
            )


def _require_regular_spacing(values: np.ndarray, *, spacing: float, name: str) -> None:
    differences = np.diff(values)
    valid = np.isclose(differences, spacing, rtol=0.0, atol=_LATTICE_ATOL_M)
    if not np.all(valid):
        bad_difference = float(differences[np.flatnonzero(~valid)[0]])
        raise ValueError(
            f"{name} must be contiguous on the {spacing:g} m SEG C3 grid; "
            f"got a {bad_difference:g} m difference"
        )


def _require_dense_crop(index_table: pd.DataFrame, shape: tuple[int, int, int, int]) -> None:
    expected_count = int(np.prod(shape, dtype=np.int64))
    cells = index_table[list(_INDEX_COLUMNS)]
    duplicated = cells.duplicated(keep=False)
    if bool(duplicated.any()):
        examples = cells.loc[duplicated].head(5).to_records(index=False).tolist()
        raise ValueError(f"selected crop contains duplicate spatial cells: {examples}")

    flat = (
        np.ravel_multi_index(
            tuple(cells[column].to_numpy(dtype=np.int64) for column in _INDEX_COLUMNS),
            shape,
        )
        if len(cells)
        else np.empty(0, dtype=np.int64)
    )
    missing = np.setdiff1d(np.arange(expected_count, dtype=np.int64), flat)
    if len(index_table) != expected_count or len(missing):
        examples = [
            tuple(int(value) for value in np.unravel_index(cell, shape)) for cell in missing[:5]
        ]
        raise ValueError(
            f"selected crop is not dense; missing local spatial cells include {examples}"
        )
