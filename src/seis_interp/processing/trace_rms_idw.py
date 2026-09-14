"""Observed-only trace RMS interpolation on a regular spatial grid."""

from itertools import product
from numbers import Integral, Real

import numpy as np


def interpolate_trace_rms_idw(
    values: np.ndarray,
    observed_mask: np.ndarray,
    *,
    radius: int,
    power: float,
    axis_scales: tuple[float, ...],
) -> np.ndarray:
    """Keep exact O RMS; interpolate other cells from O in a Chebyshev-radius stencil.

    Distances are Euclidean grid offsets multiplied by axis_scales. All observed
    neighbors in the stencil contribute with inverse-distance weights. No target
    amplitude is read. An unsupported target is an error, not an implicit fill.
    """
    if not isinstance(values, np.ndarray) or values.ndim != 5 or values.dtype.kind != "f":
        raise ValueError("values must be a floating [time, Sx, Sy, Rx, Ry] array")
    if (
        not isinstance(observed_mask, np.ndarray)
        or observed_mask.dtype != np.bool_
        or observed_mask.shape != values.shape[1:]
        or not values.size
        or not observed_mask.any()
    ):
        raise ValueError("observed_mask must be boolean, match the spatial shape, and select O")
    if isinstance(radius, bool) or not isinstance(radius, Integral) or radius < 1:
        raise ValueError("radius must be a positive integer")
    if (
        isinstance(power, bool)
        or not isinstance(power, Real)
        or not np.isfinite(power)
        or power <= 0
    ):
        raise ValueError("power must be finite and positive")
    scales = np.asarray(axis_scales, dtype=np.float64)
    if scales.shape != (4,) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("axis_scales must contain four finite positive values")
    observed = values[:, observed_mask].astype(np.float64)
    if not np.isfinite(observed).all():
        raise ValueError("observed amplitudes must be finite")
    peak = np.max(np.abs(observed), axis=0)
    divisor = np.where(peak > 0, peak, 1.0)
    rms = peak * np.sqrt(np.mean(np.square(observed / divisor), axis=0))
    field = np.zeros(observed_mask.shape, dtype=np.float64)
    field[observed_mask] = rms
    numerator = np.zeros_like(field)
    denominator = np.zeros_like(field)
    ranges = [range(-min(radius, n - 1), min(radius, n - 1) + 1) for n in field.shape]
    for offset in product(*ranges):
        if not any(offset):
            continue
        source = tuple(
            slice(max(0, d), min(n, n + d)) for n, d in zip(field.shape, offset, strict=True)
        )
        query = tuple(
            slice(max(0, -d), min(n, n - d)) for n, d in zip(field.shape, offset, strict=True)
        )
        weight = float(np.linalg.norm(np.array(offset) * scales) ** (-power))
        numerator[query] += weight * field[source]
        denominator[query] += weight * observed_mask[source]
    target = ~observed_mask
    if np.any(denominator[target] == 0):
        raise ValueError("IDW stencil has uncovered target traces; increase radius explicitly")
    field[target] = numerator[target] / denominator[target]
    if not np.isfinite(field).all():
        raise ValueError("IDW RMS must be finite")
    return field


def match_prediction_trace_rms(
    prediction: np.ndarray, rms: np.ndarray, observed_mask: np.ndarray
) -> np.ndarray:
    """Rescale nonzero target predictions to O-interpolated RMS, leaving O exact.

    A zero prediction remains zero: a scalar amplitude estimate cannot supply a
    waveform direction. No reference target amplitudes are accepted.
    """
    if (
        not isinstance(prediction, np.ndarray)
        or prediction.ndim != 5
        or prediction.dtype.kind != "f"
        or not prediction.size
        or not np.isfinite(prediction).all()
    ):
        raise ValueError("prediction must be a finite floating five-dimensional array")
    if (
        not isinstance(observed_mask, np.ndarray)
        or observed_mask.shape != prediction.shape[1:]
        or observed_mask.dtype != np.bool_
        or not isinstance(rms, np.ndarray)
        or rms.shape != observed_mask.shape
        or rms.dtype.kind != "f"
        or not np.isfinite(rms).all()
        or np.any(rms < 0)
    ):
        raise ValueError("RMS and boolean mask must match the prediction spatial shape")
    result = prediction.copy()
    selected = prediction[:, ~observed_mask].astype(np.float64)
    peak = np.max(np.abs(selected), axis=0)
    unit = selected / np.where(peak > 0, peak, 1)
    unit_rms = np.sqrt(np.mean(unit**2, axis=0))
    result[:, ~observed_mask] = unit / np.where(unit_rms > 0, unit_rms, 1) * rms[~observed_mask]
    if not np.isfinite(result).all():
        raise ValueError("RMS-corrected prediction must be finite")
    return result
