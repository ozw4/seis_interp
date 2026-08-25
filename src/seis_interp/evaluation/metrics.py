"""Focused scalar metrics for interpolation predictions."""

from __future__ import annotations

import numpy as np


def signal_to_noise_ratio_db(reference: np.ndarray, prediction: np.ndarray) -> float:
    """Return global signal-to-noise ratio in decibels over every supplied value."""
    reference_float, prediction_float = _validated_metric_arrays(reference, prediction)

    reference_energy = float(np.sum(np.square(reference_float), dtype=np.float64))
    if reference_energy == 0.0:
        raise ValueError("reference energy must be greater than zero")
    error_energy = float(np.sum(np.square(reference_float - prediction_float), dtype=np.float64))
    if error_energy == 0.0:
        return float("inf")
    return float(10.0 * np.log10(reference_energy / error_energy))


def trace_signal_to_noise_ratio_db(reference: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Return one signal-to-noise ratio per trace for arrays shaped (n_traces, n_samples)."""
    reference_float, prediction_float = _validated_metric_arrays(reference, prediction)
    if reference_float.ndim != 2:
        raise ValueError(
            f"reference and prediction must be two-dimensional, got {reference_float.shape}"
        )

    reference_energy = np.sum(np.square(reference_float), axis=1, dtype=np.float64)
    if not np.all(reference_energy > 0.0):
        raise ValueError("every reference trace energy must be greater than zero")
    error_energy = np.sum(np.square(reference_float - prediction_float), axis=1, dtype=np.float64)
    with np.errstate(divide="ignore"):
        return 10.0 * np.log10(reference_energy / error_energy)


def median_trace_signal_to_noise_ratio_db(reference: np.ndarray, prediction: np.ndarray) -> float:
    """Return the median of the per-trace signal-to-noise ratios in decibels."""
    return float(np.median(trace_signal_to_noise_ratio_db(reference, prediction)))


def _validated_metric_arrays(
    reference: np.ndarray,
    prediction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return matching, non-empty, finite float64 copies of both arrays."""
    reference_array = np.asarray(reference)
    prediction_array = np.asarray(prediction)
    if reference_array.shape != prediction_array.shape:
        raise ValueError(
            f"reference and prediction shapes must match, got "
            f"{reference_array.shape} and {prediction_array.shape}"
        )
    if reference_array.size == 0:
        raise ValueError("reference and prediction must not be empty")
    try:
        reference_float = reference_array.astype(np.float64, copy=False)
        prediction_float = prediction_array.astype(np.float64, copy=False)
    except (TypeError, ValueError) as error:
        raise ValueError("reference and prediction must contain real numeric values") from error
    if not np.all(np.isfinite(reference_float)) or not np.all(np.isfinite(prediction_float)):
        raise ValueError("reference and prediction must contain only finite values")
    return reference_float, prediction_float
