"""Reversible signed amplitude transforms with one observed-data percentile scale."""

import numpy as np


def observed_percentile_scale(observed_values, percentile=90.0):
    """Fit one scalar to observed samples only, without any per-trace normalization."""
    values = np.asarray(observed_values, dtype=np.float64)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError("expected finite nonempty observed [trace, time] values")
    if not 0 < percentile <= 100:
        raise ValueError("percentile must be in (0, 100]")
    scale = float(np.percentile(np.abs(values), percentile))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("observed percentile scale must be finite and positive")
    return scale


def transform_amplitudes(values, scale, mode, *, inverse=False):
    """Linear x/s or asinh(x/s), with a checked physical-amplitude inverse."""
    if mode not in ("linear", "asinh"):
        raise ValueError("mode must be linear or asinh")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("amplitudes must be finite")
    with np.errstate(over="raise", invalid="raise"):
        if inverse:
            result = scale * (np.sinh(values) if mode == "asinh" else values)
        else:
            result = values / scale
            if mode == "asinh":
                result = np.arcsinh(result)
        result = result.astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("transformed amplitudes must be finite")
    return result
