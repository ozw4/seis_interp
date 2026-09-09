"""Observed-trace Gaussian RMS envelopes and an explicit auxiliary-loss schedule."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

TRACE_ENVELOPE_EPSILON = 1.0e-6


def validate_trace_envelope_loss_options(
    value: object,
    *,
    time_count: int | None = None,
    microbatch_size: int | None = None,
) -> dict[str, object] | None:
    """Validate the opt-in contract, optionally binding it to complete trace sizes."""
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {
        "weight",
        "sigma_samples",
        "decay_steps",
    }:
        raise ValueError("envelope_loss requires exactly weight, sigma_samples, decay_steps")
    weight = _positive_finite(value["weight"], "envelope_loss.weight")
    sigmas = _validated_sigmas(value["sigma_samples"])
    decay = value["decay_steps"]
    if isinstance(decay, bool) or not isinstance(decay, Integral) or decay < 2:
        raise ValueError("envelope_loss.decay_steps must be an integer >= 2")
    if time_count is not None:
        if isinstance(time_count, bool) or not isinstance(time_count, Integral) or time_count <= 0:
            raise ValueError("envelope_loss time_count must be a positive integer")
        if any(math.ceil(4.0 * sigma) >= time_count for sigma in sigmas):
            raise ValueError("envelope_loss Gaussian radius must be smaller than the trace length")
    if microbatch_size is not None:
        if (
            isinstance(microbatch_size, bool)
            or not isinstance(microbatch_size, Integral)
            or microbatch_size <= 0
        ):
            raise ValueError("envelope_loss microbatch_size must be a positive integer")
        if time_count is not None and microbatch_size < time_count:
            raise ValueError("envelope_loss microbatch_size must cover at least one complete trace")
    return {"weight": weight, "sigma_samples": sigmas, "decay_steps": int(decay)}


def trace_envelope_weight(step: int, *, weight: float, decay_steps: int) -> float:
    """Return the declared linear weight; its horizon is independent of run length."""
    if isinstance(step, bool) or not isinstance(step, Integral) or step <= 0:
        raise ValueError("envelope loss step must be a positive integer")
    initial = _positive_finite(weight, "envelope_loss.weight")
    if isinstance(decay_steps, bool) or not isinstance(decay_steps, Integral) or decay_steps < 2:
        raise ValueError("envelope_loss.decay_steps must be an integer >= 2")
    if step >= decay_steps:
        return 0.0
    return initial * (1.0 - (step - 1) / (decay_steps - 1))


def gaussian_envelope_kernels(
    sigma_samples: Sequence[float],
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    """Build normalized radius-ceil(4*sigma) kernels once, without random draws."""
    import torch

    if dtype not in (torch.float32, torch.float64):
        raise ValueError("envelope kernels require float32 or float64")
    kernels = []
    for sigma in _validated_sigmas(sigma_samples):
        radius = math.ceil(4.0 * sigma)
        positions = torch.arange(-radius, radius + 1, dtype=torch.float64)
        kernel = torch.exp(-0.5 * (positions / sigma).square())
        kernel = kernel / kernel.sum()
        kernels.append(kernel.to(device=device, dtype=dtype).reshape(1, 1, -1))
    return tuple(kernels)


def local_rms_envelope(waveforms: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Return per-trace envelopes in original sample order, with reflected edges."""
    import torch
    import torch.nn.functional as functional

    if waveforms.ndim != 2 or not waveforms.numel():
        raise ValueError("envelope waveforms must be a nonempty trace-by-time matrix")
    if waveforms.dtype not in (torch.float32, torch.float64):
        raise ValueError("envelope waveforms must have dtype float32 or float64")
    if kernel.ndim != 3 or kernel.shape[:2] != (1, 1) or kernel.shape[-1] % 2 != 1:
        raise ValueError("envelope kernel must have shape [1, 1, odd_length]")
    if kernel.dtype != waveforms.dtype or kernel.device != waveforms.device:
        raise ValueError("envelope kernel and waveforms must share dtype and device")
    radius = kernel.shape[-1] // 2
    if radius >= waveforms.shape[1]:
        raise ValueError("envelope Gaussian radius must be smaller than the trace length")
    squared = waveforms.square().unsqueeze(1)
    padded = functional.pad(squared, (radius, radius), mode="reflect")
    energy = functional.conv1d(padded, kernel)
    return torch.sqrt(energy.squeeze(1) + TRACE_ENVELOPE_EPSILON)


def trace_envelope_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    kernels: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Average envelope MSE equally over scales, traces, and time samples."""
    import torch
    import torch.nn.functional as functional

    if prediction.shape != target.shape:
        raise ValueError("envelope prediction and target shapes must match")
    if prediction.dtype != target.dtype or prediction.device != target.device:
        raise ValueError("envelope prediction and target must share dtype and device")
    if not kernels:
        raise ValueError("envelope loss requires at least one Gaussian kernel")
    losses = []
    for kernel in kernels:
        predicted = local_rms_envelope(prediction, kernel)
        with torch.no_grad():
            observed = local_rms_envelope(target, kernel)
        losses.append(functional.mse_loss(predicted, observed))
    return torch.stack(losses).mean()


def _validated_sigmas(values):
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not len(values):
        raise ValueError("envelope_loss.sigma_samples must be a nonempty sequence")
    sigmas = [_positive_finite(value, "envelope_loss.sigma_samples") for value in values]
    if any(not math.isfinite(4.0 * sigma) for sigma in sigmas):
        raise ValueError("envelope_loss.sigma_samples must have finite Gaussian radii")
    return sigmas


def _positive_finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite number")
    try:
        converted = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be a positive finite number") from error
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return converted
