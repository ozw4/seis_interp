"""Trace-relative reconstruction loss shared by neural training methods."""

from __future__ import annotations

import torch


def masked_trace_relative_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    trace_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return relative MSE over selected complete traces with time last."""
    _validate_trace_pair(prediction, target)
    prediction_rows, target_rows = _selected_trace_rows(prediction, target, trace_mask)
    if not bool(torch.isfinite(prediction_rows).all()) or not bool(
        torch.isfinite(target_rows).all()
    ):
        raise ValueError("selected prediction and target traces must be finite")

    teacher = target_rows.detach().double()
    peak = teacher.abs().amax(dim=1, keepdim=True)
    safe_peak = torch.where(peak > 0, peak, torch.ones_like(peak))
    rms = peak * (teacher / safe_peak).square().mean(dim=1, keepdim=True).sqrt()
    divisor = torch.where(rms > 0, rms, torch.ones_like(rms))
    residual = prediction_rows.double() - target_rows.double()
    return (residual / divisor).square().mean()


def _validate_trace_pair(prediction: torch.Tensor, target: torch.Tensor) -> None:
    for name, value in (("prediction", prediction), ("target", target)):
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise TypeError(f"{name} must be a floating-point torch.Tensor")
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have matching shapes")
    if prediction.ndim < 1 or prediction.shape[-1] < 1:
        raise ValueError("prediction and target must have a positive time-last shape")
    if prediction.device != target.device:
        raise ValueError("prediction and target must share a device")


def _selected_trace_rows(
    prediction: torch.Tensor,
    target: torch.Tensor,
    trace_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    time_count = prediction.shape[-1]
    prediction_rows = prediction.reshape(-1, time_count)
    target_rows = target.reshape(-1, time_count)
    if trace_mask is not None:
        if not isinstance(trace_mask, torch.Tensor):
            raise TypeError("trace_mask must be a torch.Tensor or None")
        if trace_mask.dtype != torch.bool:
            raise TypeError(f"trace_mask must have dtype torch.bool, got {trace_mask.dtype}")
        if trace_mask.shape != prediction.shape[:-1]:
            raise ValueError(
                f"trace_mask must have shape {tuple(prediction.shape[:-1])}, "
                f"got {tuple(trace_mask.shape)}"
            )
        if trace_mask.device != prediction.device:
            raise ValueError("prediction, target, and trace_mask must share a device")
        selected = trace_mask.reshape(-1)
        prediction_rows = prediction_rows[selected]
        target_rows = target_rows[selected]
    if prediction_rows.shape[0] == 0:
        raise ValueError("trace_mask must select at least one trace")
    return prediction_rows, target_rows
