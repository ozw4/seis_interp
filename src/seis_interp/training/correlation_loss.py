"""Trace-balanced correlation loss for fixed-length trace batches."""

from __future__ import annotations

import math
from numbers import Real

import torch

DEFAULT_TRACE_CORRELATION_EPS = 1.0e-4
MSE_PLUS_TRACE_CORRELATION = "mse_plus_trace_correlation"


def trace_correlation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = DEFAULT_TRACE_CORRELATION_EPS,
) -> torch.Tensor:
    """Return mean 1-correlation over traces shaped (n_traces, n_samples)."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target shapes must match, got {tuple(prediction.shape)} "
            f"and {tuple(target.shape)}"
        )
    if prediction.ndim != 2:
        raise ValueError(
            f"prediction and target must be two-dimensional, got {tuple(prediction.shape)}"
        )
    if prediction.numel() == 0:
        raise ValueError("prediction and target must not be empty")
    if isinstance(eps, bool) or not isinstance(eps, Real):
        raise ValueError("eps must be a positive finite number")
    epsilon = float(eps)
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("eps must be a positive finite number")

    prediction_centered = prediction - prediction.mean(dim=1, keepdim=True)
    target_centered = target - target.mean(dim=1, keepdim=True)
    numerator = (prediction_centered * target_centered).sum(dim=1)
    prediction_norm = torch.sqrt(prediction_centered.square().sum(dim=1) + epsilon)
    target_norm = torch.sqrt(target_centered.square().sum(dim=1) + epsilon)
    correlation = numerator / (prediction_norm * target_norm)
    return (1.0 - correlation).mean()
