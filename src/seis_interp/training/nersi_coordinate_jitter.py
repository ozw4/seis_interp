"""Sub-cell coordinate perturbations for observed-profile NeRSI training."""

from __future__ import annotations

from numbers import Real

import numpy as np


def jitter_profile_coordinates(
    coordinates: np.ndarray,
    spatial_shape: tuple[int, int, int, int],
    *,
    jitter_cells: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw uniformly within the sub-cell radius intersected with the analysis domain.

    Unchanged profile labels impose local smoothness, not an exact physical symmetry.
    Singleton axes stay fixed; no target amplitudes or masks enter this operation.
    """
    if (
        isinstance(jitter_cells, bool)
        or not isinstance(jitter_cells, Real)
        or not np.isfinite(jitter_cells)
        or not 0 < jitter_cells <= 0.5
    ):
        raise ValueError("jitter_cells must be finite and in (0, 0.5]")
    if (
        coordinates.ndim != 2
        or coordinates.shape[1] != 3
        or coordinates.dtype.kind != "f"
        or not np.all(np.isfinite(coordinates))
        or np.any(coordinates < 0)
        or np.any(coordinates > 1)
    ):
        raise ValueError("coordinates must be finite normalized (N, 3) floating values")
    if len(spatial_shape) != 4 or any(n < 1 for n in spatial_shape):
        raise ValueError("spatial_shape must contain four positive sizes")
    lengths = np.asarray(spatial_shape[:3])
    radius = np.where(lengths > 1, jitter_cells / np.maximum(lengths - 1, 1), 0.0)
    lower = np.maximum(-radius, -coordinates)
    upper = np.minimum(radius, 1.0 - coordinates)
    offsets = rng.uniform(lower, upper)
    return np.ascontiguousarray(coordinates + offsets, dtype=coordinates.dtype)
