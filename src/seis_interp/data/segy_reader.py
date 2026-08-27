"""Read amplitudes for selected SEG-Y traces and build the time axis."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from numbers import Integral
from pathlib import Path

import numpy as np
import segyio


def iter_trace_amplitude_chunks(
    path: Path,
    trace_indices: Sequence[int],
    *,
    chunk_size: int,
) -> Iterator[np.ndarray]:
    """Yield selected traces in bounded chunks while opening the SEG-Y once.

    Trace order follows ``trace_indices``. Every yielded array is C-contiguous
    ``float32`` and contains at most ``chunk_size`` rows.
    """
    segy_path = _validated_segy_path(path)
    indices = _validated_trace_indices(trace_indices)
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, Integral) or chunk_size < 1:
        raise ValueError(f"chunk_size must be a positive integer, got {chunk_size!r}")
    stored_chunk_size = int(chunk_size)

    with segyio.open(
        str(segy_path),
        mode="r",
        strict=False,
        ignore_geometry=True,
    ) as handle:
        _validate_trace_index_range(indices, int(handle.tracecount), segy_path.name)
        sample_count = len(handle.samples)
        for start in range(0, len(indices), stored_chunk_size):
            stop = min(start + stored_chunk_size, len(indices))
            yield _read_amplitudes_from_handle(
                handle,
                indices[start:stop],
                sample_count=sample_count,
                file_name=segy_path.name,
            )


def read_trace_amplitudes(
    path: Path,
    trace_indices: Sequence[int],
) -> np.ndarray:
    """Read the amplitudes of the given file traces in the given order.

    Returns a C-contiguous ``float32`` array of shape
    ``(len(trace_indices), n_samples)``. Only the requested traces are read;
    the file is never loaded as a whole.
    """
    segy_path = _validated_segy_path(path)
    indices = _validated_trace_indices(trace_indices)

    with segyio.open(
        str(segy_path),
        mode="r",
        strict=False,
        ignore_geometry=True,
    ) as handle:
        trace_count = int(handle.tracecount)
        _validate_trace_index_range(indices, trace_count, segy_path.name)
        return _read_amplitudes_from_handle(
            handle,
            indices,
            sample_count=len(handle.samples),
            file_name=segy_path.name,
        )


def _validated_segy_path(path: Path) -> Path:
    segy_path = Path(path)
    if not segy_path.is_file():
        raise FileNotFoundError(f"SEG-Y file not found: {segy_path}")
    return segy_path


def _validated_trace_indices(trace_indices: Sequence[int]) -> np.ndarray:
    try:
        raw_indices = np.asarray(trace_indices)
        if raw_indices.ndim != 1:
            raise ValueError("trace_indices must be one-dimensional")
        indices = raw_indices.astype(np.int64, copy=False)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("trace_indices must contain integer-convertible values") from error
    if indices.size == 0:
        raise ValueError("trace_indices must not be empty")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("trace_indices must not contain duplicates")
    minimum = int(np.min(indices))
    if minimum < 0:
        raise ValueError(f"trace_indices must not be negative, got {minimum}")
    return indices


def _validate_trace_index_range(
    indices: np.ndarray,
    trace_count: int,
    file_name: str,
) -> None:
    maximum = int(np.max(indices))
    if maximum >= trace_count:
        raise ValueError(
            f"trace index {maximum} is out of range for {file_name} with {trace_count} traces"
        )


def _read_amplitudes_from_handle(
    handle: segyio.SegyFile,
    indices: np.ndarray,
    *,
    sample_count: int,
    file_name: str,
) -> np.ndarray:
    amplitudes = np.empty((len(indices), sample_count), dtype=np.float32)
    for row, trace_index in enumerate(indices):
        amplitudes[row] = handle.trace[int(trace_index)]
    if not np.all(np.isfinite(amplitudes)):
        raise ValueError(f"amplitudes contain non-finite values: {file_name}")
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
