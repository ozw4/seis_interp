"""Interpolate nonnegative trace RMS from observed geometry in bounded chunks."""

from __future__ import annotations

from math import isfinite
from numbers import Integral, Real

import numpy as np


def interpolate_trace_rms(
    observed_coordinates: np.ndarray,
    observed_scales: np.ndarray,
    query_coordinates: np.ndarray,
    *,
    observed_array_rows: np.ndarray,
    neighbors: int,
    power: float,
    query_chunk_size: int = 128,
) -> np.ndarray:
    """Use Euclidean k-neighbor IDW on the caller's distance-scaled coordinates.

    Only observed scales are accepted. Equal distances are ordered by original
    array row, including ties at the k-neighbor boundary. An exact coordinate
    match returns the lowest-array-row observed scale without averaging.
    Zero RMS is valid. No full query-by-observation matrix is materialized.
    """
    observed = _coordinates(observed_coordinates, "observed_coordinates")
    queries = _coordinates(query_coordinates, "query_coordinates")
    if not len(observed) or queries.shape[1] != observed.shape[1]:
        raise ValueError("observed coordinates must be nonempty and match query coordinate width")
    scales = np.asarray(observed_scales)
    if (
        scales.shape != (len(observed),)
        or scales.dtype.kind not in "fiu"
        or not np.all(np.isfinite(scales))
        or np.any(scales < 0)
    ):
        raise ValueError(
            "observed_scales must be finite nonnegative values aligned with observations"
        )
    scales = scales.astype(np.float64)
    rows = np.asarray(observed_array_rows)
    if (
        rows.shape != (len(observed),)
        or rows.dtype.kind not in "iu"
        or len(np.unique(rows)) != len(rows)
        or np.any(rows < 0)
        or np.any(rows > np.iinfo(np.int64).max)
    ):
        raise ValueError(
            "observed_array_rows must be aligned unique nonnegative int64-compatible integers"
        )
    count = min(_positive_integer(neighbors, "neighbors"), len(observed))
    chunk_size = _positive_integer(query_chunk_size, "query_chunk_size")
    if isinstance(power, bool) or not isinstance(power, Real):
        raise ValueError("power must be positive and finite")
    try:
        power = float(power)
    except (OverflowError, ValueError) as error:
        raise ValueError("power must be positive and finite") from error
    if not isfinite(power) or power <= 0:
        raise ValueError("power must be positive and finite")
    result = np.empty(len(queries), dtype=np.float64)
    for start in range(0, len(queries), chunk_size):
        chunk = queries[start : start + chunk_size]
        distances = _euclidean_distances(chunk, observed)
        cutoffs = np.partition(distances, count - 1, axis=1)[:, count - 1]
        for offset, (distance, cutoff) in enumerate(zip(distances, cutoffs, strict=True)):
            candidates = np.flatnonzero(distance <= cutoff)
            ordered = np.lexsort((rows[candidates], distance[candidates]))[:count]
            selected = candidates[ordered]
            result[start + offset] = _weighted_scale(distance[selected], scales[selected], power)
    return result


def _coordinates(values, name):
    coordinates = np.asarray(values)
    if (
        coordinates.ndim != 2
        or not coordinates.shape[1]
        or coordinates.dtype.kind not in "fiu"
        or not np.all(np.isfinite(coordinates))
    ):
        raise ValueError(f"{name} must be a finite real matrix with coordinate columns")
    return coordinates.astype(np.float64, copy=False)


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _euclidean_distances(queries, observed):
    distances = np.zeros((len(queries), len(observed)), dtype=np.float64)
    nonzero = np.zeros(distances.shape, dtype=bool)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        for axis in range(observed.shape[1]):
            difference = queries[:, None, axis] - observed[None, :, axis]
            nonzero |= difference != 0
            distances += np.square(difference)
        # Summed squares preserve exact distance ties on the physical lattice;
        # a repeated hypot fold can round coordinate permutations differently.
        fallback = ~np.isfinite(distances) | ((distances == 0) & nonzero)
        np.sqrt(distances, out=distances)
        if np.any(fallback):
            query_rows, observed_rows = np.nonzero(fallback)
            stable = np.zeros(len(query_rows), dtype=np.float64)
            for axis in range(observed.shape[1]):
                difference = queries[query_rows, axis] - observed[observed_rows, axis]
                np.hypot(stable, difference, out=stable)
            distances[fallback] = stable
    if not np.all(np.isfinite(distances)):
        raise ValueError("coordinate distances must be finite")
    return distances


def _weighted_scale(distances, scales, power):
    if distances[0] == 0:
        return scales[0]
    maximum = float(scales.max())
    if maximum == 0:
        return 0.0
    # Nearest weight is exactly one; no reciprocal or positive exponent can
    # overflow. Scale the convex mean as well to handle large finite RMS.
    with np.errstate(under="ignore"):
        weights = np.power(distances[0] / distances, power)
    mean = np.sum(weights * (scales / maximum), dtype=np.float64) / weights.sum(dtype=np.float64)
    return maximum * min(float(mean), 1.0)
