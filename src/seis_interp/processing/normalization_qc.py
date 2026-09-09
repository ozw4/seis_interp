"""Measure float32 signal retention after a fixed amplitude normalization."""

from __future__ import annotations

from numbers import Real

import numpy as np

from seis_interp.processing.c3_volume_index import validated_index_range


def summarize_amplitude_normalization(
    amplitudes: np.ndarray,
    array_rows: np.ndarray,
    *,
    time_range: tuple[int, int],
    amplitude_scale: float,
    chunk_rows: int = 4096,
) -> dict[str, object]:
    """Inspect only the caller's authorized rows/time without fitting or changing a scale.

    Division in float64 followed by a float32 cast matches the graph input and
    label readers. Exact zero traces are valid; losing every squared sample of
    a physically nonzero trace is a failed numerical check.
    """
    rows = np.asarray(array_rows)
    if (
        rows.ndim != 1
        or rows.dtype.kind not in "iu"
        or not rows.size
        or len(np.unique(rows)) != len(rows)
    ):
        raise ValueError("array_rows must be a nonempty unique integer vector")
    start, stop = validated_index_range(time_range, name="time_range")
    if (
        amplitudes.ndim != 2
        or amplitudes.dtype != np.float32
        or rows.min() < 0
        or rows.max() >= amplitudes.shape[0]
        or stop > amplitudes.shape[1]
    ):
        raise ValueError("float32 amplitudes must contain the selected rows and time samples")
    if (
        isinstance(amplitude_scale, bool)
        or not isinstance(amplitude_scale, Real)
        or not np.isfinite(amplitude_scale)
        or amplitude_scale <= 0
    ):
        raise ValueError("amplitude_scale must be positive and finite")
    if type(chunk_rows) is not int or chunk_rows < 1:
        raise ValueError("chunk_rows must be a positive integer")

    physical_rms = np.empty(len(rows), dtype=np.float64)
    physical_energy = 0.0
    nonfinite_samples = lost_samples = lost_energy_traces = nonfinite_energy_traces = 0
    maximum = normalized_maximum = 0.0
    for offset in range(0, len(rows), chunk_rows):
        values = np.asarray(amplitudes[rows[offset : offset + chunk_rows], start:stop])
        if not np.all(np.isfinite(values)):
            raise ValueError("selected physical amplitudes must be finite")
        values64 = values.astype(np.float64)
        trace_energy = np.sum(values64 * values64, axis=1, dtype=np.float64)
        physical_rms[offset : offset + len(values)] = np.sqrt(trace_energy / (stop - start))
        physical_energy += float(trace_energy.sum(dtype=np.float64))
        maximum = max(maximum, float(np.max(np.abs(values))))
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            normalized = (values64 / amplitude_scale).astype(np.float32)
            normalized_energy = np.sum(normalized * normalized, axis=1, dtype=np.float64)
        nonfinite_samples += int((~np.isfinite(normalized)).sum())
        lost_samples += int(((values != 0) & (normalized == 0)).sum())
        lost_energy_traces += int(((trace_energy > 0) & (normalized_energy == 0)).sum())
        nonfinite_energy_traces += int((~np.isfinite(normalized_energy)).sum())
        normalized_maximum = max(normalized_maximum, float(np.max(np.abs(normalized))))

    positive_rms = physical_rms[physical_rms > 0]
    with np.errstate(over="ignore", under="ignore"):
        normalized_rms = positive_rms / amplitude_scale
    failures = nonfinite_samples + lost_energy_traces + nonfinite_energy_traces
    return {
        "status": "passed" if failures == 0 else "failed",
        "scope": "caller_authorized_rows_and_selected_time_only",
        "time_samples": [start, stop],
        "trace_count": len(rows),
        "sample_count": len(rows) * (stop - start),
        "amplitude_scale": float(amplitude_scale),
        "physical_rms": float(np.sqrt(physical_energy / (len(rows) * (stop - start)))),
        "physical_max_abs_amplitude": maximum,
        "physical_zero_trace_count": int((physical_rms == 0).sum()),
        "physical_positive_trace_count": len(positive_rms),
        "physical_positive_trace_rms_quantiles": _rms_quantiles(positive_rms),
        "normalized_positive_trace_rms_quantiles": _rms_quantiles(normalized_rms),
        "normalized_max_abs_amplitude": normalized_maximum
        if np.isfinite(normalized_maximum)
        else None,
        "normalized_nonfinite_sample_count": nonfinite_samples,
        "nonzero_samples_cast_to_zero": lost_samples,
        "nonzero_traces_with_zero_float32_squared_energy": lost_energy_traces,
        "traces_with_nonfinite_float32_squared_energy": nonfinite_energy_traces,
        "quantile_probabilities": [0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0],
    }


def _rms_quantiles(values: np.ndarray) -> list[float] | None:
    if not len(values) or not np.all(np.isfinite(values)):
        return None
    return np.quantile(values, [0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0]).tolist()
