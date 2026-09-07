"""Sequential spatial-window assembly for five-dimensional DRR interpolation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from numbers import Integral

import numpy as np

from seis_interp.processing.drr import (
    interpolate_drr_block,
    select_drr_frequencies,
    validate_drr_block_inputs,
    validate_drr_parameters,
)


@dataclass(frozen=True)
class WindowedDrrResult:
    """A reconstructed volume and spatial-window coverage counts."""

    values: np.ndarray
    block_count: int
    empty_block_count: int
    uncovered_trace_count: int


def interpolate_drr_volume(
    observed: np.ndarray,
    observed_trace_mask: np.ndarray,
    time_s: np.ndarray,
    *,
    rank: int,
    damping_power: int,
    n_iterations: int,
    frequency_min_hz: float,
    frequency_max_hz: float | None,
    spatial_window_shape: tuple[int, int, int, int] | None,
    spatial_overlap: tuple[int, int, int, int] | None,
) -> WindowedDrrResult:
    """Interpolate spatial blocks, retaining the complete time axis in each.

    Blend nonempty blocks with positive interior-Hann synthesis weights and
    restore observations exactly. Empty blocks contribute no weight; traces
    covered only by empty blocks remain zero and are counted once each.
    """
    values, mask = validate_drr_block_inputs(observed, observed_trace_mask, time_s)
    window, overlap = _validated_window_spec(spatial_window_shape, spatial_overlap)
    if window is None:
        window = values.shape[1:]
        overlap = (0, 0, 0, 0)
    block_shape = tuple(
        min(length, requested) for length, requested in zip(values.shape[1:], window, strict=True)
    )
    retained_rank, power, iterations = validate_drr_parameters(
        rank, damping_power, n_iterations, block_shape
    )
    select_drr_frequencies(
        time_s, frequency_min_hz=frequency_min_hz, frequency_max_hz=frequency_max_hz
    )
    starts_by_axis = tuple(
        _axis_window_starts(length, window_length, axis_overlap)
        for length, window_length, axis_overlap in zip(
            values.shape[1:], window, overlap, strict=True
        )
    )

    if all(len(starts) == 1 for starts in starts_by_axis):
        if not np.any(mask):
            return WindowedDrrResult(np.zeros_like(values), 1, 1, mask.size)
        prediction = interpolate_drr_block(
            values,
            mask,
            time_s,
            rank=retained_rank,
            damping_power=power,
            n_iterations=iterations,
            frequency_min_hz=frequency_min_hz,
            frequency_max_hz=frequency_max_hz,
        )
        return WindowedDrrResult(prediction, 1, 0, 0)

    weighted_sum = np.zeros(values.shape, dtype=np.float64)
    weight_sum = np.zeros(mask.shape, dtype=np.float64)
    weights = _positive_hann_weights(block_shape)
    block_count = 0
    empty_block_count = 0

    for starts in product(*starts_by_axis):
        block_count += 1
        spatial_slices = tuple(
            slice(start, start + length) for start, length in zip(starts, block_shape, strict=True)
        )
        block_mask = mask[spatial_slices]
        if not np.any(block_mask):
            empty_block_count += 1
            continue
        block_slices = (slice(None), *spatial_slices)
        prediction = interpolate_drr_block(
            values[block_slices],
            block_mask,
            time_s,
            rank=retained_rank,
            damping_power=power,
            n_iterations=iterations,
            frequency_min_hz=frequency_min_hz,
            frequency_max_hz=frequency_max_hz,
        )
        weighted_sum[block_slices] += weights[None] * prediction
        weight_sum[spatial_slices] += weights

    covered = weight_sum > 0.0
    np.divide(weighted_sum, weight_sum[None], out=weighted_sum, where=covered[None])
    result_values = weighted_sum.astype(values.dtype, copy=False)
    result_values[:, mask] = values[:, mask]
    return WindowedDrrResult(
        values=result_values,
        block_count=block_count,
        empty_block_count=empty_block_count,
        uncovered_trace_count=int(np.count_nonzero(~covered)),
    )


def _validated_window_spec(
    window_shape: tuple[int, int, int, int] | None,
    overlap: tuple[int, int, int, int] | None,
) -> tuple[tuple[int, ...] | None, tuple[int, ...] | None]:
    if (window_shape is None) != (overlap is None):
        raise ValueError(
            "spatial_window_shape and spatial_overlap must either both be None or both be set"
        )
    if window_shape is None:
        return None, None
    window = _validated_integer_sequence(window_shape, "spatial_window_shape", positive=True)
    window_overlap = _validated_integer_sequence(overlap, "spatial_overlap", positive=False)
    if any(
        axis_overlap >= length for length, axis_overlap in zip(window, window_overlap, strict=True)
    ):
        raise ValueError("spatial_overlap must be less than spatial_window_shape on every axis")
    return window, window_overlap


def _validated_integer_sequence(
    value: object,
    name: str,
    *,
    positive: bool,
) -> tuple[int, ...]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, (Sequence, np.ndarray))
        or (isinstance(value, np.ndarray) and value.ndim != 1)
        or len(value) != 4
    ):
        raise ValueError(f"{name} must contain exactly four integers")
    if any(isinstance(item, bool) or not isinstance(item, Integral) for item in value):
        raise ValueError(f"{name} must contain exactly four integers")
    converted = tuple(int(item) for item in value)
    if positive and any(item <= 0 for item in converted):
        raise ValueError("spatial_window_shape values must be positive")
    if not positive and any(item < 0 for item in converted):
        raise ValueError("spatial_overlap values must be nonnegative")
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
        broadcast_shape = [1] * 4
        broadcast_shape[axis] = length
        weights *= axis_weights.reshape(broadcast_shape)
    return weights
