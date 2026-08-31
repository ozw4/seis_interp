"""Composite reconstruction losses for trace-graph gather interpolation."""

from __future__ import annotations

import math
from numbers import Integral, Real

import torch
from torch.nn import functional as F

DEFAULT_SPECTRUM_EPSILON = 1.0e-8
DEFAULT_SLOPE_EPSILON = 1.0e-4
DEFAULT_SLOPE_SMOOTHING_RECEIVER_SPAN = 3
DEFAULT_SLOPE_SMOOTHING_TIME_SPAN = 9
DEFAULT_ENVELOPE_WINDOW = 25


def masked_mean_square(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_availability: torch.Tensor,
) -> torch.Tensor:
    """Return the mean squared error over available target traces."""
    _validated_gather_pair(prediction, target, target_availability)
    squared_error = torch.square(prediction.float() - target.float())
    mask = target_availability.to(dtype=squared_error.dtype)[..., None]
    total = torch.sum(mask)
    if not bool((total > 0).item()):
        raise ValueError("target_availability must select at least one trace")
    return torch.sum(squared_error * mask) / (total * prediction.shape[-1])


def spectrum_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_availability: torch.Tensor,
    *,
    epsilon: float = DEFAULT_SPECTRUM_EPSILON,
) -> torch.Tensor:
    """Return log-magnitude plus magnitude-weighted phase spectrum mismatch."""
    _validated_gather_pair(prediction, target, target_availability)
    spectrum_epsilon = _positive_finite_float(epsilon, "epsilon")
    prediction_rows, target_rows = _selected_rows(
        prediction,
        target,
        target_availability,
    )
    prediction_spectrum = torch.fft.rfft(prediction_rows, dim=1)
    target_spectrum = torch.fft.rfft(target_rows, dim=1)
    prediction_magnitude = torch.abs(prediction_spectrum)
    target_magnitude = torch.abs(target_spectrum)
    log_magnitude_error = torch.mean(
        torch.square(torch.log1p(prediction_magnitude) - torch.log1p(target_magnitude))
    )
    cross = prediction_spectrum * torch.conj(target_spectrum)
    cosine_phase_difference = cross.real / (
        prediction_magnitude * target_magnitude + spectrum_epsilon
    )
    phase_weight = target_magnitude / (torch.mean(target_magnitude) + spectrum_epsilon)
    phase_error = torch.mean(phase_weight * (1.0 - cosine_phase_difference))
    return log_magnitude_error + phase_error


def slope_consistency_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_availability: torch.Tensor,
    *,
    epsilon: float = DEFAULT_SLOPE_EPSILON,
    receiver_span: int = DEFAULT_SLOPE_SMOOTHING_RECEIVER_SPAN,
    time_span: int = DEFAULT_SLOPE_SMOOTHING_TIME_SPAN,
) -> torch.Tensor:
    """Return local plane-wave destruction-residual mismatch along receiver y.

    The local slope field is estimated only from the target gather with a
    smoothed structure tensor and is detached, so the loss constrains the
    prediction's receiver-axis moveout without letting gradients reshape the
    slope estimate itself.
    """
    _validated_gather_pair(prediction, target, target_availability)
    slope_epsilon = _positive_finite_float(epsilon, "epsilon")
    receiver_window = _odd_positive_integer(receiver_span, "receiver_span")
    time_window = _odd_positive_integer(time_span, "time_span")
    prediction = prediction.float()
    target = target.float()

    target_dy, target_dt = _midpoint_derivatives(target)
    prediction_dy, prediction_dt = _midpoint_derivatives(prediction)
    smoothed_cross = _smoothed(target_dy * target_dt, receiver_window, time_window)
    smoothed_energy = _smoothed(torch.square(target_dt), receiver_window, time_window)
    slope = (-smoothed_cross / (smoothed_energy + slope_epsilon)).detach()

    prediction_residual = prediction_dy + slope * prediction_dt
    target_residual = target_dy + slope * target_dt
    pair_mask = (target_availability[:, :, 1:] & target_availability[:, :, :-1]).to(
        dtype=prediction.dtype
    )[..., None]
    total = torch.sum(pair_mask)
    if not bool((total > 0).item()):
        return torch.zeros((), dtype=prediction.dtype, device=prediction.device)
    squared_error = torch.square(prediction_residual - target_residual)
    return torch.sum(squared_error * pair_mask) / (total * squared_error.shape[-1])


def amplitude_envelope_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_availability: torch.Tensor,
    *,
    window: int = DEFAULT_ENVELOPE_WINDOW,
) -> torch.Tensor:
    """Return the mean squared windowed-RMS envelope mismatch."""
    _validated_gather_pair(prediction, target, target_availability)
    envelope_window = _odd_positive_integer(window, "window")
    prediction_rows, target_rows = _selected_rows(
        prediction,
        target,
        target_availability,
    )
    prediction_envelope = _windowed_rms(prediction_rows, envelope_window)
    target_envelope = _windowed_rms(target_rows, envelope_window)
    return torch.mean(torch.square(prediction_envelope - target_envelope))


def _selected_rows(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_availability: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    prediction_rows = prediction[target_availability].float()
    target_rows = target[target_availability].float()
    if prediction_rows.shape[0] == 0:
        raise ValueError("target_availability must select at least one trace")
    return prediction_rows, target_rows


def _midpoint_derivatives(gather: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return receiver-y and time derivatives co-located at cell midpoints."""
    time_difference = gather[..., 1:] - gather[..., :-1]
    receiver_difference = gather[:, :, 1:, :] - gather[:, :, :-1, :]
    receiver_at_midpoint = receiver_difference[..., 1:] + receiver_difference[..., :-1]
    time_at_midpoint = time_difference[:, :, 1:, :] + time_difference[:, :, :-1, :]
    return receiver_at_midpoint * 0.5, time_at_midpoint * 0.5


def _smoothed(values: torch.Tensor, receiver_window: int, time_window: int) -> torch.Tensor:
    batch_size, receiver_x, receiver_pairs, time_count = values.shape
    flat = values.reshape(batch_size * receiver_x, 1, receiver_pairs, time_count)
    smoothed = F.avg_pool2d(
        flat,
        kernel_size=(receiver_window, time_window),
        stride=1,
        padding=(receiver_window // 2, time_window // 2),
        count_include_pad=False,
    )
    return smoothed.reshape(values.shape)


def _windowed_rms(rows: torch.Tensor, window: int) -> torch.Tensor:
    smoothed_power = F.avg_pool1d(
        torch.square(rows)[:, None],
        kernel_size=window,
        stride=1,
        padding=window // 2,
        count_include_pad=False,
    )[:, 0]
    return torch.sqrt(smoothed_power.clamp_min(0.0) + 1.0e-12)


def _validated_gather_pair(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_availability: torch.Tensor,
) -> None:
    for name, tensor in (("prediction", prediction), ("target", target)):
        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            raise TypeError(f"{name} must be a floating-point torch.Tensor")
    if not isinstance(target_availability, torch.Tensor):
        raise TypeError("target_availability must be a torch.Tensor")
    if target_availability.dtype != torch.bool:
        raise TypeError(
            f"target_availability must have dtype torch.bool, got {target_availability.dtype}"
        )
    if prediction.ndim != 4:
        raise ValueError(
            "prediction must have shape (batch, receiver_x, receiver_y, time), "
            f"got {tuple(prediction.shape)}"
        )
    if target.shape != prediction.shape:
        raise ValueError(
            f"target shape {tuple(target.shape)} must match prediction {tuple(prediction.shape)}"
        )
    if target_availability.shape != prediction.shape[:3]:
        raise ValueError(
            "target_availability must have shape "
            f"{tuple(prediction.shape[:3])}, got {tuple(target_availability.shape)}"
        )
    if prediction.shape[-1] < 2 or prediction.shape[2] < 2:
        raise ValueError("prediction receiver-y and time dimensions must be at least 2")


def _positive_finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    return converted


def _odd_positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    converted = int(value)
    if converted % 2 == 0:
        raise ValueError(f"{name} must be odd, got {converted}")
    return converted
