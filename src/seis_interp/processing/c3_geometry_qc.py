"""Mask-independent source numbering, sampling, and grid summaries for C3."""

from __future__ import annotations

import numpy as np
import pandas as pd

from seis_interp.data.trace_table import validated_array_rows
from seis_interp.processing.c3_receiver_grid import receiver_grid_offsets
from seis_interp.processing.c3_volume_index import c3_source_indices, validated_index_range
from seis_interp.processing.trace_canonicalization import PHYSICAL_COORDINATE_COLUMNS


def summarize_time_axis(time_s: np.ndarray, time_range: tuple[int, int]) -> dict[str, object]:
    """Report sample indices and measured physical time without changing the origin."""
    start, stop = validated_index_range(time_range, name="time_range")
    times = np.asarray(time_s)
    if times.ndim != 1 or times.dtype.kind not in "fiu" or not np.all(np.isfinite(times)):
        raise ValueError("time_s must be a finite real vector")
    if stop > len(times):
        raise ValueError(f"time range requires {stop} samples, but time_s has {len(times)}")
    differences = np.diff(times.astype(np.float64))
    if len(differences) and (
        np.any(differences <= 0)
        or not np.allclose(differences, differences[0], rtol=1e-6, atol=1e-12)
    ):
        raise ValueError("time_s must be strictly increasing and uniformly sampled")
    return {
        "sample_start": start,
        "sample_stop": stop,
        "sample_last": stop - 1,
        "available_sample_count": len(times),
        "sample_count": stop - start,
        "time_start_s": float(times[start]),
        "time_last_s": float(times[stop - 1]),
        "sample_interval_s": float(differences[0]) if len(differences) else None,
    }


def c3_sail_line_mapping(
    trace_table: pd.DataFrame,
    requested_inclusive: tuple[int, int],
    *,
    original_number_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Map original sail numbers, when present, to full-survey source-line ranks."""
    if len(requested_inclusive) != 2 or any(type(v) is not int for v in requested_inclusive):
        raise ValueError("requested_inclusive must be two integers")
    lo, hi = requested_inclusive
    if lo < 0 or hi < lo:
        raise ValueError("requested_inclusive must be nonnegative and increasing")
    validated_array_rows(trace_table)
    line_indices, shot_indices = c3_source_indices(trace_table)
    column = original_number_column
    if column is None and "sail_line_number" in trace_table:
        column = "sail_line_number"
    columns = ["ffid", "source_x_m", "source_y_m"]
    if column is not None:
        if column not in trace_table or trace_table[column].dtype.kind not in "iu":
            raise ValueError("original sail-line number column must contain integers")
        columns.append(column)
    shots = (
        trace_table[columns]
        .assign(source_line_index=line_indices, shot_in_line_index=shot_indices)
        .drop_duplicates()
        .sort_values(["source_line_index", "shot_in_line_index"])
    )
    if shots.groupby("ffid")["source_line_index"].nunique().max() > 1:
        raise ValueError("FFID belongs to multiple source lines")
    if shots.groupby("ffid")["source_y_m"].nunique().max() > 1:
        raise ValueError("FFID belongs to multiple source positions")
    if shots.groupby(["source_line_index", "shot_in_line_index"])["ffid"].nunique().max() > 1:
        raise ValueError("source position belongs to multiple FFIDs")
    if column is not None:
        if (shots.groupby("source_line_index")[column].nunique() != 1).any() or (
            shots.groupby(column)["source_line_index"].nunique() != 1
        ).any():
            raise ValueError("original sail-line numbers must map one-to-one to global indices")
        number = shots[column]
        numbering = "original_sail_line_number"
    else:
        number = shots["source_line_index"]
        numbering = "global_source_line_index"
    selected = shots.loc[number.between(lo, hi)].copy()
    selected["requested_sail_line_number"] = number.loc[selected.index]
    actual = sorted(selected["requested_sail_line_number"].unique().tolist())
    if actual != list(range(lo, hi + 1)):
        raise ValueError(
            f"selected sail lines are missing: requested {lo}..{hi}, available {actual}"
        )
    indices = sorted(selected["source_line_index"].unique().tolist())
    if indices != list(range(indices[0], indices[-1] + 1)):
        raise ValueError("selected sail lines do not form a contiguous global-index range")
    if not np.allclose(np.diff(selected["source_x_m"].unique()), 160.0, rtol=0, atol=1e-6):
        raise ValueError("selected sail lines have a gap on the 160 m grid")
    mapping = {
        "requested_inclusive": [lo, hi],
        "numbering": numbering,
        "original_number_column": column,
        "index_range": [indices[0], indices[-1] + 1],
        "source_line_count": int(line_indices.max()) + 1,
        "lines": selected[["requested_sail_line_number", "source_line_index", "source_x_m"]]
        .drop_duplicates()
        .to_dict("records"),
    }
    return selected.reset_index(drop=True), mapping


def summarize_c3_geometry(
    trace_table: pd.DataFrame,
    time_s: np.ndarray,
    *,
    requested_inclusive: tuple[int, int],
    time_range: tuple[int, int],
    original_number_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Summarize geometry only; incomplete shots are reported, not globally excluded."""
    times = summarize_time_axis(time_s, time_range)
    shots, mapping = c3_sail_line_mapping(
        trace_table, requested_inclusive, original_number_column=original_number_column
    )
    x_values, y_values = receiver_grid_offsets(trace_table)
    selected = trace_table.loc[trace_table["ffid"].isin(shots["ffid"])]
    physical = selected.drop_duplicates(list(PHYSICAL_COORDINATE_COLUMNS))
    counts = physical.groupby("ffid").size()
    expected = len(x_values) * len(y_values)
    lines = []
    for line_index, line_shots in shots.groupby("source_line_index", sort=True):
        y = np.sort(line_shots["source_y_m"].unique())
        differences = np.diff(y)
        lines.append(
            {
                "source_line_index": int(line_index),
                "shot_count": len(y),
                "shot_index_range": [0, len(y)],
                "source_y_spacing_m": np.unique(differences).tolist(),
                "missing_intermediate_shots": int(
                    np.maximum(np.rint(differences / 80) - 1, 0).sum()
                ),
                "regular_shot_spacing": bool(np.allclose(differences, 80.0, rtol=0, atol=1e-6)),
                "source_y_start_m": float(y[0]),
            }
        )
    stagger = []
    for first, second in zip(lines, lines[1:], strict=False):
        y1 = shots.loc[
            shots["source_line_index"].eq(first["source_line_index"]), "source_y_m"
        ].to_numpy()
        y2 = shots.loc[
            shots["source_line_index"].eq(second["source_line_index"]), "source_y_m"
        ].to_numpy()
        differences = np.abs(y2[: min(len(y1), len(y2))] - y1[: min(len(y1), len(y2))])
        stagger.append(
            {
                "first_source_line_index": first["source_line_index"],
                "absolute_differences_m": np.unique(differences).tolist(),
            }
        )
    return shots, {
        "sail_lines": mapping,
        "time": times,
        "lines": lines,
        "stagger": stagger,
        "receiver_x_offsets_m": x_values.tolist(),
        "receiver_y_offsets_m": y_values.tolist(),
        "common_shot_index_range": [0, min(line["shot_count"] for line in lines)],
        "duplicate_physical_row_count": len(selected) - len(physical),
        "incomplete_ffid_count": int(counts.ne(expected).sum()),
        "missing_receiver_cell_count": int((expected - counts).clip(lower=0).sum()),
        "receiver_count_range": [int(counts.min()), int(counts.max())],
    }
