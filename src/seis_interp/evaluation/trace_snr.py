"""Per-trace physical SNR without clipping or silent nonfinite exclusions."""

import numpy as np


def trace_snr_db(reference: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if reference.ndim != 2 or reference.shape != prediction.shape or min(reference.shape) < 1:
        raise ValueError("reference and prediction must match nonempty [trace, time] shape")
    if not np.isfinite(reference).all() or not np.isfinite(prediction).all():
        raise ValueError("trace amplitudes must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        energy = np.sum(reference**2, axis=1)
        error = np.sum((prediction - reference) ** 2, axis=1)
    if not np.isfinite(energy).all() or not np.isfinite(error).all():
        raise ValueError("trace energies must be finite")
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10 * (np.log10(energy) - np.log10(error))


def summarize_trace_snr(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("trace SNR must be a nonempty vector")
    positive = int(np.isposinf(values).sum())
    negative = int(np.isneginf(values).sum())
    undefined = int(np.isnan(values).sum())
    if undefined or (positive and negative):
        status = "undefined"
    elif positive:
        status = "positive_infinity"
    elif negative:
        status = "negative_infinity"
    else:
        status = "finite"
    return {
        "mean_trace_snr_db": float(values.mean()) if status == "finite" else None,
        "mean_trace_snr_status": status,
        "trace_snr_trace_count": len(values),
        "trace_snr_positive_infinity_count": positive,
        "trace_snr_negative_infinity_count": negative,
        "trace_snr_undefined_count": undefined,
    }
