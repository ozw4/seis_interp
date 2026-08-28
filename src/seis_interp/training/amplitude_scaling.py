"""Apply training-target amplitude scaling without changing prepared data."""

from __future__ import annotations

import numpy as np

TRAIN_GLOBAL_RMS_SCALING = "train_global_rms"
PER_TRACE_RMS_SCALING = "per_trace_rms"
AMPLITUDE_SCALINGS = (TRAIN_GLOBAL_RMS_SCALING, PER_TRACE_RMS_SCALING)
TRAIN_GLOBAL_RMS_VALIDATION_DOMAIN = "train_global_rms"
ORACLE_PER_TRACE_RMS_VALIDATION_DOMAIN = "oracle_per_trace_unit_rms"

_ROW_CHUNK_SIZE = 4096


def validated_amplitude_scaling(
    value: object,
    *,
    name: str = "amplitude_scaling",
) -> str:
    """Return one supported training-target scaling name."""
    if not isinstance(value, str) or value not in AMPLITUDE_SCALINGS:
        raise ValueError(f"{name} must be one of {list(AMPLITUDE_SCALINGS)}, got {value!r}")
    return value


def validation_metric_domain_for_scaling(amplitude_scaling: object) -> str:
    """Return the metric domain implied by one target scaling."""
    scaling = validated_amplitude_scaling(amplitude_scaling)
    if scaling == TRAIN_GLOBAL_RMS_SCALING:
        return TRAIN_GLOBAL_RMS_VALIDATION_DOMAIN
    return ORACLE_PER_TRACE_RMS_VALIDATION_DOMAIN


def per_trace_rms_scaled_amplitudes(
    amplitudes: np.ndarray,
    *,
    array_rows: np.ndarray | None = None,
) -> np.ndarray:
    """Return traces divided by their own RMS using bounded working memory.

    The returned array preserves floating dtypes and promotes integer or float16
    inputs to at least float32. The input is never modified.
    """
    amplitude_array = _validated_amplitude_array(amplitudes)
    row_labels = _validated_array_rows(array_rows, trace_count=amplitude_array.shape[0])
    output_dtype = np.result_type(amplitude_array.dtype, np.float32)
    scaled = np.empty(amplitude_array.shape, dtype=output_dtype)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for start in range(0, amplitude_array.shape[0], _ROW_CHUNK_SIZE):
            stop = min(start + _ROW_CHUNK_SIZE, amplitude_array.shape[0])
            chunk = amplitude_array[start:stop]
            finite_rows = np.all(np.isfinite(chunk), axis=1)
            if not np.all(finite_rows):
                row = int(row_labels[start + int(np.flatnonzero(~finite_rows)[0])])
                raise ValueError(f"amplitudes contain non-finite values for array_row {row}")
            float_chunk = chunk.astype(np.float64, copy=False)
            rms = np.sqrt(
                np.mean(
                    np.square(float_chunk),
                    axis=1,
                    dtype=np.float64,
                )
            )
            invalid = ~np.isfinite(rms) | (rms <= 0.0)
            if np.any(invalid):
                row = int(row_labels[start + int(np.flatnonzero(invalid)[0])])
                raise ValueError(f"per-trace RMS must be positive and finite for array_row {row}")
            np.divide(
                chunk,
                rms[:, np.newaxis],
                out=scaled[start:stop],
                casting="unsafe",
            )
            if not np.all(np.isfinite(scaled[start:stop])):
                raise ValueError("per-trace RMS scaling produced non-finite amplitudes")
    return scaled


def per_trace_rms_scaled_rows(
    amplitudes: np.ndarray,
    array_rows: np.ndarray,
) -> np.ndarray:
    """Scale selected rows and return a row-aligned array with zeros elsewhere."""
    amplitude_array = _validated_amplitude_array(amplitudes)
    rows = _validated_selected_array_rows(array_rows, trace_count=amplitude_array.shape[0])
    output_dtype = np.result_type(amplitude_array.dtype, np.float32)
    scaled = np.zeros(amplitude_array.shape, dtype=output_dtype)
    for start in range(0, len(rows), _ROW_CHUNK_SIZE):
        chunk_rows = rows[start : start + _ROW_CHUNK_SIZE]
        scaled[chunk_rows] = per_trace_rms_scaled_amplitudes(
            amplitude_array[chunk_rows],
            array_rows=chunk_rows,
        )
    return scaled


def _validated_amplitude_array(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2 or array.size == 0:
        raise ValueError(
            f"amplitudes must be a non-empty two-dimensional array, got shape {array.shape}"
        )
    if array.dtype.kind not in "iuf" or array.dtype.kind == "b":
        raise ValueError("amplitudes must contain real numeric values")
    return array


def _validated_array_rows(values: np.ndarray | None, *, trace_count: int) -> np.ndarray:
    if values is None:
        return np.arange(trace_count, dtype=np.int64)
    rows = np.asarray(values)
    if rows.shape != (trace_count,) or rows.dtype.kind not in "iu" or rows.dtype.kind == "b":
        raise ValueError(
            "array_rows must be a one-dimensional integer array matching the amplitude rows"
        )
    if (
        len(np.unique(rows)) != len(rows)
        or np.any(rows < 0)
        or np.any(rows > np.iinfo(np.int64).max)
    ):
        raise ValueError("array_rows must contain unique non-negative int64-compatible values")
    return rows.astype(np.int64, copy=False)


def _validated_selected_array_rows(values: np.ndarray, *, trace_count: int) -> np.ndarray:
    rows = np.asarray(values)
    if rows.ndim != 1 or rows.size == 0:
        raise ValueError("array_rows must be a non-empty one-dimensional array")
    if rows.dtype.kind not in "iu" or rows.dtype.kind == "b":
        raise ValueError("array_rows must have an integer dtype")
    if np.any(rows < 0) or np.any(rows >= trace_count):
        raise ValueError(f"array_rows must be within [0, {trace_count})")
    if len(np.unique(rows)) != len(rows):
        raise ValueError("array_rows must contain unique values")
    return rows.astype(np.int64, copy=False)
