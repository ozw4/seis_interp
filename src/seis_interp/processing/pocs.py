"""Fourier POCS interpolation for regular five-dimensional seismic grids."""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np

_SPATIAL_AXES = (0, 1, 2, 3)
_COMPLEX_DTYPES = frozenset((np.dtype(np.complex64), np.dtype(np.complex128)))
_REAL_DTYPES = frozenset((np.dtype(np.float32), np.dtype(np.float64)))


def interpolate_pocs_frequency_slice(
    observed: np.ndarray,
    observed_trace_mask: np.ndarray,
    *,
    n_iterations: int,
    threshold_start: float,
    threshold_end: float,
) -> np.ndarray:
    """Interpolate one complex four-dimensional spatial frequency slice.

    Only values selected by ``observed_trace_mask`` contribute to the
    reconstruction. The returned array is always complex128, and observed
    values are reinserted exactly after every iteration.
    """
    values, mask = _validated_frequency_slice_inputs(observed, observed_trace_mask)
    iteration_count, start_ratio, end_ratio = _validated_pocs_parameters(
        n_iterations,
        threshold_start,
        threshold_end,
    )

    zero_filled = np.zeros(values.shape, dtype=np.complex128)
    zero_filled[mask] = values[mask]
    if not np.any(mask):
        return zero_filled
    if np.all(mask):
        return zero_filled.copy()

    initial_spectrum = np.fft.fftn(zero_filled, axes=_SPATIAL_AXES, norm="ortho")
    initial_maximum = float(np.max(np.abs(initial_spectrum)))
    if initial_maximum == 0.0:
        return zero_filled

    reconstructed = zero_filled.copy()
    ratios = np.geomspace(start_ratio, end_ratio, iteration_count)
    for ratio in ratios:
        spectrum = np.fft.fftn(reconstructed, axes=_SPATIAL_AXES, norm="ortho")
        spectrum[np.abs(spectrum) < ratio * initial_maximum] = 0.0
        reconstructed = np.fft.ifftn(spectrum, axes=_SPATIAL_AXES, norm="ortho")
        reconstructed[mask] = zero_filled[mask]

    return reconstructed


def interpolate_pocs_block(
    observed: np.ndarray,
    observed_trace_mask: np.ndarray,
    *,
    n_iterations: int,
    threshold_start: float,
    threshold_end: float,
) -> np.ndarray:
    """Interpolate one real 5D block using time rFFT and spatial POCS.

    The axis order is time, source line, shot in line, relative receiver x,
    and relative receiver y. The trace mask is shared by every time sample.
    """
    values, mask = _validated_block_inputs(observed, observed_trace_mask)
    iteration_count, start_ratio, end_ratio = _validated_pocs_parameters(
        n_iterations,
        threshold_start,
        threshold_end,
    )

    if not np.any(mask):
        return np.zeros(values.shape, dtype=values.dtype)
    if np.all(mask):
        return values.copy()
    if values.shape[0] == 0:
        raise ValueError("observed time axis must contain at least one sample")

    zero_filled = np.zeros(values.shape, dtype=np.float64)
    zero_filled[:, mask] = values[:, mask]
    frequency_values = np.fft.rfft(zero_filled, axis=0, norm="ortho")

    nyquist_index = len(frequency_values) - 1 if values.shape[0] % 2 == 0 else None
    for frequency_index in range(len(frequency_values)):
        reconstructed_slice = interpolate_pocs_frequency_slice(
            frequency_values[frequency_index],
            mask,
            n_iterations=iteration_count,
            threshold_start=start_ratio,
            threshold_end=end_ratio,
        )
        if frequency_index == 0 or frequency_index == nyquist_index:
            reconstructed_slice = reconstructed_slice.real
        frequency_values[frequency_index] = reconstructed_slice

    reconstructed = np.fft.irfft(
        frequency_values,
        n=values.shape[0],
        axis=0,
        norm="ortho",
    ).astype(values.dtype, copy=False)
    reconstructed[:, mask] = values[:, mask]
    return reconstructed


def _validated_frequency_slice_inputs(
    observed: np.ndarray,
    observed_trace_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(observed, np.ndarray) or observed.ndim != 4:
        raise ValueError("observed must be a four-dimensional NumPy array")
    if observed.dtype not in _COMPLEX_DTYPES:
        raise ValueError("observed must have dtype complex64 or complex128")
    mask = _validated_boolean_mask(
        observed_trace_mask,
        expected_shape=observed.shape,
    )
    if not np.all(np.isfinite(observed[mask])):
        raise ValueError("observed values selected by observed_trace_mask must be finite")
    return observed, mask


def _validated_block_inputs(
    observed: np.ndarray,
    observed_trace_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(observed, np.ndarray) or observed.ndim != 5:
        raise ValueError("observed must be a five-dimensional NumPy array")
    if observed.dtype not in _REAL_DTYPES:
        raise ValueError("observed must have dtype float32 or float64")
    mask = _validated_boolean_mask(
        observed_trace_mask,
        expected_shape=observed.shape[1:],
    )
    if not np.all(np.isfinite(observed[:, mask])):
        raise ValueError("observed values selected by observed_trace_mask must be finite")
    return observed, mask


def _validated_boolean_mask(
    observed_trace_mask: np.ndarray,
    *,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    if not isinstance(observed_trace_mask, np.ndarray):
        raise ValueError("observed_trace_mask must be a NumPy array")
    if observed_trace_mask.dtype != np.dtype(np.bool_):
        raise ValueError("observed_trace_mask must have boolean dtype")
    if observed_trace_mask.shape != expected_shape:
        raise ValueError(
            "observed_trace_mask shape must match the observed spatial shape; "
            f"got {observed_trace_mask.shape}, expected {expected_shape}"
        )
    return observed_trace_mask


def _validated_pocs_parameters(
    n_iterations: int,
    threshold_start: float,
    threshold_end: float,
) -> tuple[int, float, float]:
    if isinstance(n_iterations, bool) or not isinstance(n_iterations, Integral):
        raise ValueError("n_iterations must be an integer greater than or equal to 2")
    iteration_count = int(n_iterations)
    if iteration_count < 2:
        raise ValueError("n_iterations must be an integer greater than or equal to 2")

    start_ratio = _finite_real(threshold_start, "threshold_start")
    end_ratio = _finite_real(threshold_end, "threshold_end")
    if not 0.0 < end_ratio <= start_ratio <= 1.0:
        raise ValueError("thresholds must satisfy 0 < threshold_end <= threshold_start <= 1")
    return iteration_count, start_ratio, end_ratio


def _finite_real(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    converted = float(value)
    if not np.isfinite(converted):
        raise ValueError(f"{name} must be a finite real number")
    return converted
