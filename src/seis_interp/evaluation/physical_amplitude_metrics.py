"""Shared physical-amplitude energy and strict-JSON metric calculations."""

from __future__ import annotations

import math

import numpy as np


def physical_amplitude_energies(
    reference: np.ndarray, prediction: np.ndarray
) -> tuple[float, float]:
    """Return float64 target energy and SSE for physical amplitude arrays."""
    reference = np.asarray(reference, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        reference_energy = float(np.sum(np.square(reference), dtype=np.float64))
        difference = reference - prediction
        error_energy = float(np.sum(np.square(difference), dtype=np.float64))
    if not math.isfinite(reference_energy) or not math.isfinite(error_energy):
        raise ValueError("evaluation energies must be finite")
    return reference_energy, error_energy


def physical_amplitude_mean_trace_relative_mse(
    reference: np.ndarray,
    prediction: np.ndarray,
) -> float:
    """Return the mean trace-relative MSE using a divisor of one for zero traces."""
    reference = np.asarray(reference, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if reference.shape != prediction.shape or reference.ndim != 2:
        raise ValueError("reference and prediction must have matching [trace, time] shape")
    if min(reference.shape) < 1:
        raise ValueError("reference and prediction must contain at least one complete trace")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(prediction)):
        raise ValueError("reference and prediction must contain finite values")

    peak = np.max(np.abs(reference), axis=1, keepdims=True)
    safe_peak = np.where(peak > 0.0, peak, 1.0)
    with np.errstate(over="ignore", invalid="ignore"):
        rms = peak * np.sqrt(
            np.mean(
                np.square(reference / safe_peak),
                axis=1,
                dtype=np.float64,
                keepdims=True,
            )
        )
        divisor = np.where(rms > 0.0, rms, 1.0)
        residual = prediction - reference
        trace_relative_mse = np.mean(
            np.square(residual / divisor),
            axis=1,
            dtype=np.float64,
        )
        result = float(np.mean(trace_relative_mse, dtype=np.float64))
    _require_finite_metric(result, "mean trace-relative MSE")
    return result


def physical_amplitude_target_metrics(
    *,
    trace_count: int,
    sample_count: int,
    reference_energy: float,
    error_energy: float,
) -> dict[str, object]:
    """Summarize global energies with nullable dB for degenerate signals."""
    snr_db, snr_status = physical_amplitude_snr(reference_energy, error_energy)
    rmse = float(math.sqrt(error_energy / sample_count))
    relative_l2 = (
        None if reference_energy == 0.0 else float(math.sqrt(error_energy / reference_energy))
    )
    _require_finite_metric(rmse, "RMSE")
    if relative_l2 is not None:
        _require_finite_metric(relative_l2, "relative L2")
    return {
        "trace_count": trace_count,
        "sample_count": sample_count,
        "reference_energy": reference_energy,
        "error_energy": error_energy,
        "snr_db": snr_db,
        "snr_status": snr_status,
        "rmse": rmse,
        "relative_l2": relative_l2,
    }


def physical_amplitude_snr(
    reference_energy: float, error_energy: float
) -> tuple[float | None, str]:
    """Preserve zero-target and perfect-prediction meanings in strict JSON."""
    if reference_energy == 0.0:
        return None, "undefined_zero_reference"
    if error_energy == 0.0:
        return None, "perfect_reconstruction"
    snr_db = float(10.0 * (math.log10(reference_energy) - math.log10(error_energy)))
    _require_finite_metric(snr_db, "SNR")
    return snr_db, "finite"


def _require_finite_metric(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
