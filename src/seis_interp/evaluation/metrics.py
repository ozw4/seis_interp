"""Focused scalar metrics for interpolation predictions."""

from __future__ import annotations

import numpy as np


def signal_to_noise_ratio_db(reference: np.ndarray, prediction: np.ndarray) -> float:
    """Return global signal-to-noise ratio in decibels."""
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

    reference_energy = float(np.sum(np.square(reference_float), dtype=np.float64))
    if reference_energy == 0.0:
        raise ValueError("reference energy must be greater than zero")
    error_energy = float(np.sum(np.square(reference_float - prediction_float), dtype=np.float64))
    if error_energy == 0.0:
        return float("inf")
    return float(10.0 * np.log10(reference_energy / error_energy))
