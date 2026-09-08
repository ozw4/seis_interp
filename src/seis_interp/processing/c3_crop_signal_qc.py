"""Stream fixed-crop signal statistics without interpreting zeros as missing."""

from __future__ import annotations

import numpy as np

from seis_interp.processing.c3_geometry_qc import summarize_time_axis


def summarize_c3_crop_signal(
    amplitudes: np.ndarray,
    array_rows: np.ndarray,
    time_s: np.ndarray,
    *,
    time_range: tuple[int, int],
    chunk_rows: int = 1024,
) -> dict[str, object]:
    """Read just the selected rows/samples, reporting nonfinite samples explicitly."""
    times = summarize_time_axis(time_s, time_range)
    rows = np.asarray(array_rows)
    if (
        rows.ndim != 1
        or rows.dtype.kind not in "iu"
        or not len(rows)
        or len(np.unique(rows)) != len(rows)
    ):
        raise ValueError("array_rows must be a nonempty unique integer vector")
    if (
        amplitudes.ndim != 2
        or amplitudes.shape[1] != len(time_s)
        or rows.min() < 0
        or rows.max() >= len(amplitudes)
    ):
        raise ValueError("amplitude rows/time do not match the crop")
    if type(chunk_rows) is not int or chunk_rows < 1:
        raise ValueError("chunk_rows must be a positive integer")
    start, stop = time_range
    energy_by_sample = np.zeros(stop - start, dtype=np.float64)
    finite_count = zero_count = 0
    maximum = 0.0
    for offset in range(0, len(rows), chunk_rows):
        values = np.asarray(
            amplitudes[rows[offset : offset + chunk_rows], start:stop], dtype=np.float64
        )
        finite = np.isfinite(values)
        finite_count += int(finite.sum())
        zero_count += int((values == 0).sum())
        values = np.where(finite, values, 0.0)
        maximum = max(maximum, float(np.max(np.abs(values))))
        with np.errstate(over="ignore"):
            energy_by_sample += np.sum(values**2, axis=0, dtype=np.float64)
    count = len(rows) * (stop - start)
    energy = float(energy_by_sample.sum())
    energy_finite = bool(np.isfinite(energy))
    edges = np.unique(np.linspace(start, stop, min(4, stop - start) + 1, dtype=int))
    bands = [
        {
            "time_samples": [int(a), int(b)],
            "energy": float(energy_by_sample[a - start : b - start].sum())
            if energy_finite
            else None,
        }
        for a, b in zip(edges, edges[1:], strict=False)
    ]
    return {
        "time": times,
        "trace_count": len(rows),
        "sample_count": count,
        "finite_count": finite_count,
        "nonfinite_count": count - finite_count,
        "zero_count": zero_count,
        "zero_fraction": zero_count / count,
        "energy": energy if energy_finite else None,
        "rms": float(np.sqrt(energy / count)) if energy_finite and finite_count == count else None,
        "max_abs_amplitude": maximum,
        "time_band_energy": bands,
        "energy_by_sample": energy_by_sample.tolist() if energy_finite else None,
        "ok": finite_count == count and energy_finite,
    }
