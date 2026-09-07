"""Damped rank-reduction interpolation on regular seismic spatial grids."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from seis_interp.processing.damped_rank_reduction import damped_rank_reduce
from seis_interp.processing.level_four_hankel import (
    average_level_four_hankel,
    hankelize_level_four,
    level_four_hankel_matrix_shape,
)


def interpolate_drr_frequency_slice(
    observed: np.ndarray,
    observed_trace_mask: np.ndarray,
    *,
    rank: int,
    damping_power: int,
    n_iterations: int,
) -> np.ndarray:
    """Reconstruct a complex 4D slice, keeping observed coefficients exact.

    Only masked observations must be finite; all other input values are
    ignored. Return an independent complex128 array. The rank must be less
    than both dimensions of the slice's level-four Hankel matrix, even for
    all-observed or all-missing inputs.
    """
    values, mask = _validated_frequency_inputs(observed, observed_trace_mask)
    retained_rank, power, iterations = _validated_drr_parameters(
        rank, damping_power, n_iterations, values.shape
    )
    zero_filled = np.zeros(values.shape, dtype=np.complex128)
    zero_filled[mask] = values[mask]
    if np.all(mask) or not np.any(zero_filled):
        return zero_filled

    current = zero_filled.copy()
    for _ in range(iterations):
        reduced = damped_rank_reduce(
            hankelize_level_four(current), rank=retained_rank, damping_power=power
        )
        current = average_level_four_hankel(reduced, values.shape)
        current[mask] = zero_filled[mask]
    return current


@dataclass(frozen=True)
class DrrFrequencySelection:
    """Validated time sampling and selected padded-rFFT bins for a DRR run."""

    sample_interval_s: float
    fft_length: int
    frequencies_hz: np.ndarray
    bin_indices: np.ndarray


def select_drr_frequencies(
    time_s: np.ndarray,
    *,
    frequency_min_hz: float,
    frequency_max_hz: float | None,
) -> DrrFrequencySelection:
    """Select inclusive frequency limits on a next-power-of-two time rFFT.

    Time must contain at least two finite, uniformly spaced, increasing
    samples. The time origin is ignored. A null upper limit means Nyquist;
    explicit limits above Nyquist and ranges containing no bins are rejected.
    This same selection is used by reconstruction and recorded run metadata.
    """
    interval = _uniform_sample_interval(time_s)
    lower = _finite_frequency(frequency_min_hz, "frequency_min_hz")
    if lower < 0:
        raise ValueError("frequency_min_hz must be nonnegative")
    nyquist = 0.5 / interval
    upper = nyquist
    if frequency_max_hz is not None:
        upper = _finite_frequency(frequency_max_hz, "frequency_max_hz")
        if upper <= lower:
            raise ValueError("frequency_max_hz must be greater than frequency_min_hz")
        if upper > nyquist:
            raise ValueError(f"frequency_max_hz must not exceed Nyquist ({nyquist} Hz)")
    fft_length = 1 << (len(time_s) - 1).bit_length()
    frequencies = np.fft.rfftfreq(fft_length, d=interval)
    selected = frequencies >= lower
    if frequency_max_hz is not None:
        selected &= frequencies <= upper
    indices = np.flatnonzero(selected)
    if not len(indices):
        raise ValueError("requested frequency range contains no rFFT bins")
    return DrrFrequencySelection(interval, fft_length, frequencies, indices)


def interpolate_drr_block(
    observed: np.ndarray,
    observed_trace_mask: np.ndarray,
    time_s: np.ndarray,
    *,
    rank: int,
    damping_power: int,
    n_iterations: int,
    frequency_min_hz: float,
    frequency_max_hz: float | None,
) -> np.ndarray:
    """Reconstruct a real (time, X0, X1, X2, X3) block with padded time rFFT.

    Use orthonormal FFTs and only the selected bins, truncate the inverse to
    the input time length, and return the input float32/float64 dtype. The
    missing prediction has zero coefficients outside the selected range
    before truncation. Observed traces are reinserted exactly in time.
    """
    values, mask = _validated_block_inputs(observed, observed_trace_mask, time_s)
    retained_rank, power, iterations = _validated_drr_parameters(
        rank, damping_power, n_iterations, values.shape[1:]
    )
    selection = select_drr_frequencies(
        time_s, frequency_min_hz=frequency_min_hz, frequency_max_hz=frequency_max_hz
    )
    if np.all(mask):
        return values.copy()
    zero_filled = np.zeros(values.shape, dtype=np.float64)
    zero_filled[:, mask] = values[:, mask]
    if not np.any(zero_filled):
        return np.zeros(values.shape, dtype=values.dtype)

    transformed = np.fft.rfft(zero_filled, n=selection.fft_length, axis=0, norm="ortho")
    predicted = np.zeros_like(transformed)
    for index in selection.bin_indices:
        frequency_slice = interpolate_drr_frequency_slice(
            transformed[index],
            mask,
            rank=retained_rank,
            damping_power=power,
            n_iterations=iterations,
        )
        if index == 0 or index == selection.fft_length // 2:
            frequency_slice = frequency_slice.real
        predicted[index] = frequency_slice
    result = np.fft.irfft(predicted, n=selection.fft_length, axis=0, norm="ortho")
    result = result[: values.shape[0]].astype(values.dtype, copy=False)
    result[:, mask] = values[:, mask]
    return result


def _validated_block_inputs(
    observed: np.ndarray, observed_trace_mask: np.ndarray, time_s: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(observed, np.ndarray) or observed.ndim != 5:
        raise ValueError("observed must be a five-dimensional NumPy array")
    if observed.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("observed must have dtype float32 or float64")
    if any(length < 1 for length in observed.shape[1:]):
        raise ValueError("observed spatial axes must be nonempty")
    if not isinstance(time_s, np.ndarray) or time_s.shape != (observed.shape[0],):
        raise ValueError("time_s shape must match the observed time axis")
    mask = _validated_mask(observed_trace_mask, observed.shape[1:])
    if not np.all(np.isfinite(observed[:, mask])):
        raise ValueError("observed values selected by observed_trace_mask must be finite")
    return observed, mask


def _uniform_sample_interval(time_s: np.ndarray) -> float:
    if not isinstance(time_s, np.ndarray) or time_s.ndim != 1 or len(time_s) < 2:
        raise ValueError("time_s must be a one-dimensional array with at least two samples")
    if time_s.dtype.kind not in "fiu" or not np.all(np.isfinite(time_s)):
        raise ValueError("time_s must contain finite real values")
    times = time_s.astype(np.float64)
    differences = np.diff(times)
    if np.any(differences <= 0):
        raise ValueError("time_s must be strictly increasing")
    interval = float((times[-1] - times[0]) / (len(times) - 1))
    # Differences between stored times inherit rounding at their absolute scale.
    epsilon = np.finfo(time_s.dtype).eps if time_s.dtype.kind == "f" else np.finfo(float).eps
    tolerance = 4 * epsilon * max(float(np.max(np.abs(times))), interval)
    if not np.allclose(differences, interval, rtol=1e-6, atol=tolerance):
        raise ValueError("time_s must be uniformly sampled")
    return interval


def _finite_frequency(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


def _validated_frequency_inputs(
    observed: np.ndarray, observed_trace_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(observed, np.ndarray) or observed.ndim != 4:
        raise ValueError("observed must be a four-dimensional NumPy array")
    if observed.dtype not in (np.dtype(np.complex64), np.dtype(np.complex128)):
        raise ValueError("observed must have dtype complex64 or complex128")
    mask = _validated_mask(observed_trace_mask, observed.shape)
    if not np.all(np.isfinite(observed[mask])):
        raise ValueError("observed values selected by observed_trace_mask must be finite")
    return observed, mask


def _validated_mask(mask: np.ndarray, spatial_shape: tuple[int, ...]) -> np.ndarray:
    if not isinstance(mask, np.ndarray) or mask.dtype != np.dtype(np.bool_):
        raise ValueError("observed_trace_mask must be a boolean NumPy array")
    if mask.shape != spatial_shape:
        raise ValueError("observed_trace_mask shape must match the four-dimensional spatial shape")
    return mask


def _validated_drr_parameters(
    rank: int,
    damping_power: int,
    n_iterations: int,
    spatial_shape: tuple[int, int, int, int],
) -> tuple[int, int, int]:
    parameters = []
    for name, value in (
        ("rank", rank),
        ("damping_power", damping_power),
        ("n_iterations", n_iterations),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
        parameters.append(int(value))
    matrix_shape = level_four_hankel_matrix_shape(spatial_shape)
    if parameters[0] >= min(matrix_shape):
        raise ValueError(
            f"rank must be less than min(Hankel matrix shape {matrix_shape}) "
            f"for spatial shape {spatial_shape}"
        )
    return parameters[0], parameters[1], parameters[2]
