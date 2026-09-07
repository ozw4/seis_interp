"""Reduce complex matrix rank with singular-value damping from Chen et al. (2016)."""

from __future__ import annotations

from numbers import Integral

import numpy as np


def damped_rank_reduce(
    matrix: np.ndarray,
    *,
    rank: int,
    damping_power: int,
) -> np.ndarray:
    """Return a contiguous complex128 damped TSVD of a finite complex 2D array.

    ``rank`` must satisfy ``1 <= rank < min(matrix.shape)`` and
    ``damping_power`` must be a positive integer. The first discarded singular
    value, ``singular_values[rank]``, is the damping reference. Each retained
    value is multiplied by ``1 - (reference / value) ** damping_power``;
    retained zeros stay zero. Exact NumPy SVD uses ``full_matrices=False``.
    The input matrix is not modified.
    """
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        raise ValueError("matrix must be a two-dimensional NumPy array")
    if not np.issubdtype(matrix.dtype, np.complexfloating):
        raise ValueError("matrix must have a complex dtype")
    values = np.asarray(matrix, dtype=np.complex128)
    if not np.all(np.isfinite(values)):
        raise ValueError("matrix must contain only finite values")
    kept_rank = _positive_integer(rank, "rank")
    if kept_rank >= min(values.shape):
        raise ValueError("rank must satisfy 1 <= rank < min(matrix.shape)")
    power = _positive_integer(damping_power, "damping_power")

    left, singular_values, right_adjoint = np.linalg.svd(values, full_matrices=False)
    retained = singular_values[:kept_rank]
    reference = singular_values[kept_rank]
    ratios = np.divide(reference, retained, out=np.zeros_like(retained), where=retained > 0.0)
    weights = np.clip(1.0 - ratios**power, 0.0, 1.0)
    damped = retained * weights
    return np.ascontiguousarray(
        (left[:, :kept_rank] * damped) @ right_adjoint[:kept_rank, :],
        dtype=np.complex128,
    )


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)
