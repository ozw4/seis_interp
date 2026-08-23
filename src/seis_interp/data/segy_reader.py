"""Read amplitudes for selected SEG-Y traces and build the time axis."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import segyio


def read_trace_amplitudes(
    path: Path,
    trace_indices: Sequence[int],
) -> np.ndarray:
    """Read the amplitudes of the given file traces in the given order.

    Returns a C-contiguous ``float32`` array of shape
    ``(len(trace_indices), n_samples)``. Only the requested traces are read;
    the file is never loaded as a whole.
    """
    segy_path = Path(path)
    if not segy_path.is_file():
        raise FileNotFoundError(f"SEG-Y file not found: {segy_path}")

    indices = [int(index) for index in trace_indices]
    if not indices:
        raise ValueError("trace_indices must not be empty")
    if len(set(indices)) != len(indices):
        raise ValueError("trace_indices must not contain duplicates")
    if min(indices) < 0:
        raise ValueError(f"trace_indices must not be negative, got {min(indices)}")

    with segyio.open(
        str(segy_path),
        mode="r",
        strict=False,
        ignore_geometry=True,
    ) as handle:
        trace_count = int(handle.tracecount)
        if max(indices) >= trace_count:
            raise ValueError(
                f"trace index {max(indices)} is out of range for {segy_path.name} "
                f"with {trace_count} traces"
            )

        amplitudes = np.empty((len(indices), len(handle.samples)), dtype=np.float32)
        for row, trace_index in enumerate(indices):
            amplitudes[row] = handle.trace[trace_index]

    if not np.all(np.isfinite(amplitudes)):
        raise ValueError(f"amplitudes contain non-finite values: {segy_path.name}")
    return amplitudes


def build_time_axis(
    sample_count: int,
    sample_interval_s: float,
) -> np.ndarray:
    """Return a zero-based time axis in seconds.

    This POC ignores the SEG-Y recording delay, so the first sample is at
    ``0.0`` seconds.
    """
    if sample_count < 1:
        raise ValueError(f"sample_count must be at least 1, got {sample_count}")
    if not np.isfinite(sample_interval_s) or sample_interval_s <= 0.0:
        raise ValueError(f"sample_interval_s must be positive, got {sample_interval_s}")

    return np.arange(sample_count, dtype=np.float64) * float(sample_interval_s)
