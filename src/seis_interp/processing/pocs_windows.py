"""Sequential overlapping-window assembly for five-dimensional POCS."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from numbers import Integral

import numpy as np

from seis_interp.processing.pocs import (
    _validated_block_inputs,
    _validated_pocs_parameters,
    interpolate_pocs_block,
)

_AXIS_COUNT = 5


@dataclass(frozen=True)
class WindowedPocsResult:
    """A reconstructed volume and coverage counts from window assembly."""

    values: np.ndarray
    block_count: int
    empty_block_count: int
    uncovered_sample_count: int


def interpolate_pocs_volume(
    observed: np.ndarray,
    observed_trace_mask: np.ndarray,
    *,
    n_iterations: int,
    threshold_start: float,
    threshold_end: float,
    window_shape: tuple[int, int, int, int, int] | None,
    overlap: tuple[int, int, int, int, int] | None,
) -> WindowedPocsResult:
    """Interpolate a 5D volume one overlapping block at a time.

    A window dimension larger than its corresponding volume dimension uses
    the available extent once. Its overlap therefore has no effect on that
    axis. Blocks are blended only after reconstruction with positive Hann
    interior weights; no analysis taper or padding is applied.
    """
    values, mask = _validated_block_inputs(observed, observed_trace_mask)
    iteration_count, start_ratio, end_ratio = _validated_pocs_parameters(
        n_iterations,
        threshold_start,
        threshold_end,
    )
    window, window_overlap = _validated_window_spec(window_shape, overlap)

    if window is None:
        return _interpolate_single_block(
            values,
            mask,
            n_iterations=iteration_count,
            threshold_start=start_ratio,
            threshold_end=end_ratio,
        )

    starts_by_axis = tuple(
        _axis_window_starts(length, window_length, axis_overlap)
        for length, window_length, axis_overlap in zip(
            values.shape,
            window,
            window_overlap,
            strict=True,
        )
    )
    if all(len(starts) == 1 for starts in starts_by_axis):
        return _interpolate_single_block(
            values,
            mask,
            n_iterations=iteration_count,
            threshold_start=start_ratio,
            threshold_end=end_ratio,
        )

    weighted_sum = np.zeros(values.shape, dtype=np.float64)
    weight_sum = np.zeros(values.shape, dtype=np.float64)
    block_count = 0
    empty_block_count = 0

    for starts in product(*starts_by_axis):
        block_count += 1
        block_slices = tuple(
            slice(start, min(start + window_length, length))
            for start, window_length, length in zip(
                starts,
                window,
                values.shape,
                strict=True,
            )
        )
        spatial_slices = block_slices[1:]
        block_mask = mask[spatial_slices]
        if not np.any(block_mask):
            empty_block_count += 1
            continue

        prediction = interpolate_pocs_block(
            values[block_slices],
            block_mask,
            n_iterations=iteration_count,
            threshold_start=start_ratio,
            threshold_end=end_ratio,
        )
        weights = _positive_hann_weights(prediction.shape)
        weighted_sum[block_slices] += weights * prediction
        weight_sum[block_slices] += weights

    covered = weight_sum > 0.0
    reconstructed = np.zeros(values.shape, dtype=np.float64)
    np.divide(weighted_sum, weight_sum, out=reconstructed, where=covered)
    uncovered_sample_count = int(np.count_nonzero(~covered))

    result_values = reconstructed.astype(values.dtype, copy=False)
    result_values[:, mask] = values[:, mask]
    return WindowedPocsResult(
        values=result_values,
        block_count=block_count,
        empty_block_count=empty_block_count,
        uncovered_sample_count=uncovered_sample_count,
    )


def _interpolate_single_block(
    observed: np.ndarray,
    observed_trace_mask: np.ndarray,
    *,
    n_iterations: int,
    threshold_start: float,
    threshold_end: float,
) -> WindowedPocsResult:
    is_empty = not np.any(observed_trace_mask)
    values = interpolate_pocs_block(
        observed,
        observed_trace_mask,
        n_iterations=n_iterations,
        threshold_start=threshold_start,
        threshold_end=threshold_end,
    )
    return WindowedPocsResult(
        values=values,
        block_count=1,
        empty_block_count=int(is_empty),
        uncovered_sample_count=observed.size if is_empty else 0,
    )


def _validated_window_spec(
    window_shape: tuple[int, int, int, int, int] | None,
    overlap: tuple[int, int, int, int, int] | None,
) -> tuple[tuple[int, ...] | None, tuple[int, ...] | None]:
    if (window_shape is None) != (overlap is None):
        raise ValueError("window_shape and overlap must either both be None or both be set")
    if window_shape is None:
        return None, None

    window = _validated_integer_sequence(window_shape, "window_shape", positive=True)
    window_overlap = _validated_integer_sequence(overlap, "overlap", positive=False)
    if any(
        axis_overlap >= window_length
        for window_length, axis_overlap in zip(window, window_overlap, strict=True)
    ):
        raise ValueError("overlap must be less than window_shape on every axis")
    return window, window_overlap


def _validated_integer_sequence(
    values: object,
    name: str,
    *,
    positive: bool,
) -> tuple[int, ...]:
    if (
        isinstance(values, (str, bytes))
        or not isinstance(values, (Sequence, np.ndarray))
        or len(values) != _AXIS_COUNT
    ):
        raise ValueError(f"{name} must contain exactly five integers")
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in values):
        raise ValueError(f"{name} must contain exactly five integers")

    converted = tuple(int(value) for value in values)
    if positive and any(value <= 0 for value in converted):
        raise ValueError("window_shape values must be positive")
    if not positive and any(value < 0 for value in converted):
        raise ValueError("overlap values must be nonnegative")
    return converted


def _axis_window_starts(length: int, window_length: int, overlap: int) -> tuple[int, ...]:
    if length <= window_length:
        return (0,)

    final_start = length - window_length
    starts = list(range(0, final_start + 1, window_length - overlap))
    if starts[-1] != final_start:
        starts.append(final_start)
    return tuple(starts)


def _positive_hann_weights(shape: tuple[int, ...]) -> np.ndarray:
    weights = np.ones(shape, dtype=np.float64)
    for axis, length in enumerate(shape):
        axis_weights = np.hanning(length + 2)[1:-1]
        broadcast_shape = [1] * len(shape)
        broadcast_shape[axis] = length
        weights *= axis_weights.reshape(broadcast_shape)
    return weights
