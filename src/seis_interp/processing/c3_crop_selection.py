"""Resolve deterministic C3 crop starts using existing canonical/grid contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import product

import numpy as np
import pandas as pd

from seis_interp.processing.c3_geometry_qc import summarize_time_axis
from seis_interp.processing.c3_receiver_grid import receiver_grid_offsets
from seis_interp.processing.c3_volume_index import (
    VOLUME_AXIS_ORDER,
    build_c3_volume_index,
    c3_source_indices,
    validated_index_range,
)
from seis_interp.processing.trace_canonicalization import canonicalize_eligible_physical_coordinates


def canonical_c3_rows(table: pd.DataFrame) -> np.ndarray:
    """Use the existing lowest-array-row policy without a mask or an amplitude filter."""
    canonical, _ = canonicalize_eligible_physical_coordinates(
        table.reset_index(drop=True).assign(split="train")
    )
    return np.sort(canonical["array_row"].to_numpy(dtype=np.int64))


def resolve_c3_crop(
    table: pd.DataFrame,
    time_s: np.ndarray,
    *,
    time_range: tuple[int, int],
    source_line_range: tuple[int, int],
    spatial_lengths: Sequence[int],
    explicit_ranges: Mapping[str, object] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Choose the closest valid contiguous window, keeping time and lines fixed."""
    times = summarize_time_axis(time_s, time_range)
    source_range = validated_index_range(source_line_range, name="source_line_range")
    if len(spatial_lengths) != 3 or any(type(v) is not int or v < 1 for v in spatial_lengths):
        raise ValueError("spatial_lengths must contain three positive integers")
    lines, shots = c3_source_indices(table)
    if source_range[1] > int(lines.max()) + 1:
        raise ValueError("source_line_range exceeds available source lines")
    x_values, y_values = receiver_grid_offsets(table)
    bounds = [(0, min(int(shots[lines == line].max()) + 1 for line in range(*source_range)))]
    for axis, coordinates in (("x", x_values), ("y", y_values)):
        offsets = (table[f"receiver_{axis}_m"] - table[f"source_{axis}_m"]).to_numpy()
        ranks = np.searchsorted(coordinates, offsets)
        available = [ranks[lines == line] for line in range(*source_range)]
        bounds.append(
            (
                max(int(values.min()) for values in available),
                min(int(values.max()) + 1 for values in available),
            )
        )
    axes = VOLUME_AXIS_ORDER[2:]
    initial, candidates = [], []
    for axis, (lo, hi), length in zip(axes, bounds, spatial_lengths, strict=True):
        if length > hi - lo:
            raise ValueError(
                f"{axis} needs {length} indices, common available range is [{lo},{hi})"
            )
        explicit = (explicit_ranges or {}).get(axis)
        if explicit is None:
            start = lo + (hi - lo - length) // 2
            candidates.append(range(lo, hi - length + 1))
        else:
            start, stop = validated_index_range(explicit, name=axis)
            if stop - start != length or start < lo or stop > hi:
                raise ValueError(f"explicit {axis} must select {length} indices inside [{lo},{hi})")
            candidates.append([start])
        initial.append(start)
    ordered = sorted(
        product(*candidates),
        key=lambda p: (sum(abs(a - b) for a, b in zip(p, initial, strict=True)), p),
    )
    canonical = canonical_c3_rows(table)
    first_failure = None
    for starts in ordered:
        selection = {"time": list(time_range), "source_line": list(source_range)}
        selection.update(
            {
                axis: [start, start + length]
                for axis, start, length in zip(axes, starts, spatial_lengths, strict=True)
            }
        )
        try:
            index = build_c3_volume_index(
                table,
                canonical,
                **{f"{axis}_range": tuple(selection[axis]) for axis in VOLUME_AXIS_ORDER[1:]},
            )
        except ValueError as error:
            if first_failure is None:
                first_failure = str(error)
            continue
        return index, {
            "selection": selection,
            "shape": [stop - start for start, stop in selection.values()],
            "time": times,
            "start_rule": "centered_contiguous",
            "initial_starts": dict(zip(axes, initial, strict=True)),
            "resolved_starts": dict(zip(axes, starts, strict=True)),
            "adjustment_reason": first_failure,
            "candidate_order": "manhattan_distance_then_lexicographic",
            "common_available_ranges": {
                axis: list(bounds) for axis, bounds in zip(axes, bounds, strict=True)
            },
            "trace_count": len(index),
        }
    raise ValueError(f"no valid crop under fixed time and source lines: {first_failure}")
