"""Geometry-only linear moveout alignment of a time-first observed volume.

Each trace is delayed by ``offset / velocity`` so that a linear event common to
the volume becomes flat on an extended time canvas. No waveform is inspected, no
sample is truncated, and the integer shifts invert exactly.
"""

from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np


def moveout_delay_samples(
    offset_m: np.ndarray, velocity_m_per_s: float, sample_interval_s: float
) -> np.ndarray:
    """Return the nonnegative integer delay of ``offset / velocity`` per trace."""
    offsets = np.asarray(offset_m, dtype=np.float64)
    if offsets.ndim != 1 or not np.isfinite(offsets).all() or np.any(offsets < 0):
        raise ValueError("offset_m must be a finite nonnegative vector")
    for name, value in (
        ("velocity_m_per_s", velocity_m_per_s),
        ("sample_interval_s", sample_interval_s),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be positive and finite")
    return np.rint(offsets / float(velocity_m_per_s) / float(sample_interval_s)).astype(np.int64)


def align_moveout(
    values: np.ndarray, delay_samples: np.ndarray, max_delay: int | None = None, *, progress=None
) -> np.ndarray:
    """Shift trace ``i`` later by ``max_delay - delay[i]`` onto a canvas of length T + max.

    ``max_delay`` defaults to the largest supplied delay. An explicit value lets
    disjoint trace subsets share one canvas; it must cover every supplied delay.
    """
    values, delays, maximum = _validated(values, delay_samples, max_delay)
    canvas = np.zeros((values.shape[0] + maximum, *values.shape[1:]), values.dtype)
    flat_values = values.reshape(values.shape[0], -1)
    flat_canvas = canvas.reshape(canvas.shape[0], -1)
    completed = 0
    for shift, cells in _shift_groups(delays, maximum):
        flat_canvas[shift : shift + values.shape[0], cells] = flat_values[:, cells]
        completed += len(cells)
        if progress is not None:
            progress(f"aligning {completed}/{len(delays)} cells")
    return canvas


def unalign_moveout(
    canvas: np.ndarray,
    delay_samples: np.ndarray,
    time_sample_count: int,
    max_delay: int | None = None,
    *,
    progress=None,
) -> np.ndarray:
    """Invert :func:`align_moveout`, returning the original ``time_sample_count`` samples."""
    canvas, delays, maximum = _validated(canvas, delay_samples, max_delay)
    if canvas.shape[0] != time_sample_count + maximum:
        raise ValueError("canvas length does not match the delays and time sample count")
    values = np.empty((time_sample_count, *canvas.shape[1:]), canvas.dtype)
    flat_canvas = canvas.reshape(canvas.shape[0], -1)
    flat_values = values.reshape(time_sample_count, -1)
    completed = 0
    for shift, cells in _shift_groups(delays, maximum):
        flat_values[:, cells] = flat_canvas[shift : shift + time_sample_count, cells]
        completed += len(cells)
        if progress is not None:
            progress(f"restoring alignment {completed}/{len(delays)} cells")
    return values


def _validated(
    values: np.ndarray, delay_samples: np.ndarray, max_delay: int | None
) -> tuple[np.ndarray, np.ndarray, int]:
    array = np.asarray(values)
    delays = np.asarray(delay_samples)
    if array.ndim < 2 or not array.size:
        raise ValueError("values must be a nonempty time-first array")
    if (
        delays.dtype.kind not in "iu"
        or delays.shape != array.shape[1:]
        or delays.size == 0
        or np.any(delays < 0)
    ):
        raise ValueError("delay_samples must be nonnegative integers matching the spatial shape")
    delays = delays.reshape(-1).astype(np.int64)
    if max_delay is None:
        return array, delays, int(delays.max())
    if (
        isinstance(max_delay, bool)
        or not isinstance(max_delay, Integral)
        or max_delay < int(delays.max())
    ):
        raise ValueError("max_delay must be an integer covering every delay")
    return array, delays, int(max_delay)


def _shift_groups(delays: np.ndarray, maximum: int):
    shifts = maximum - delays
    for shift in np.unique(shifts):
        yield int(shift), np.flatnonzero(shifts == shift)
