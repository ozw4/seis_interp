"""Embed complex 4D grids in level-four Hankel matrices and average them back."""

from __future__ import annotations

from math import prod
from numbers import Integral

import numpy as np


def level_four_hankel_matrix_shape(
    spatial_shape: tuple[int, int, int, int],
) -> tuple[int, int]:
    """Return the row and column counts for a positive four-axis grid shape."""
    embedding_shape, start_shape = _hankel_shapes(spatial_shape)
    return prod(embedding_shape), prod(start_shape)


def hankelize_level_four(values: np.ndarray) -> np.ndarray:
    """Embed a complex64/128 4D grid as a contiguous complex128 matrix.

    Rows flatten window offsets in C order; columns flatten window starts in
    C order. Entry ``(row(offset), column(start))`` is ``values[start + offset]``.
    Each window axis has length ``values.shape[i] // 2 + 1``. The input is not
    modified.
    """
    if not isinstance(values, np.ndarray) or values.ndim != 4:
        raise ValueError("values must be a four-dimensional NumPy array")
    if values.dtype not in (np.dtype(np.complex64), np.dtype(np.complex128)):
        raise ValueError("values must have dtype complex64 or complex128")
    embedding_shape, start_shape = _hankel_shapes(values.shape)
    windows = np.lib.stride_tricks.sliding_window_view(values, embedding_shape)
    offset_then_start = windows.transpose(4, 5, 6, 7, 0, 1, 2, 3)
    return np.ascontiguousarray(
        offset_then_start.reshape(prod(embedding_shape), prod(start_shape)),
        dtype=np.complex128,
    )


def average_level_four_hankel(
    matrix: np.ndarray,
    spatial_shape: tuple[int, int, int, int],
) -> np.ndarray:
    """Average all matching anti-diagonal entries onto a complex128 4D grid.

    The complex 2D matrix uses C-order offset rows and C-order start columns,
    matching :func:`hankelize_level_four`. It need not have Hankel structure:
    every window contribution is included before dividing by overlap counts.
    The input matrix is not modified.
    """
    embedding_shape, start_shape = _hankel_shapes(spatial_shape)
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        raise ValueError("matrix must be a two-dimensional NumPy array")
    if not np.issubdtype(matrix.dtype, np.complexfloating):
        raise ValueError("matrix must have a complex dtype")
    expected_shape = (prod(embedding_shape), prod(start_shape))
    if matrix.shape != expected_shape:
        raise ValueError(f"matrix shape must be {expected_shape}, got {matrix.shape}")

    windows = np.asarray(matrix, dtype=np.complex128).reshape(*embedding_shape, *start_shape)
    total = np.zeros(spatial_shape, dtype=np.complex128)
    count = np.zeros(spatial_shape, dtype=np.int64)
    for start in np.ndindex(start_shape):
        region = tuple(
            slice(first, first + length)
            for first, length in zip(start, embedding_shape, strict=True)
        )
        total[region] += windows[(slice(None),) * 4 + start]
        count[region] += 1
    return total / count


def _hankel_shapes(
    spatial_shape: tuple[int, int, int, int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if (
        not isinstance(spatial_shape, tuple)
        or len(spatial_shape) != 4
        or any(
            isinstance(length, bool) or not isinstance(length, Integral) or length < 1
            for length in spatial_shape
        )
    ):
        raise ValueError("spatial_shape must be a tuple of four positive integers")
    embedding_shape = tuple(int(length) // 2 + 1 for length in spatial_shape)
    start_shape = tuple(
        int(length) - window + 1
        for length, window in zip(spatial_shape, embedding_shape, strict=True)
    )
    return embedding_shape, start_shape
