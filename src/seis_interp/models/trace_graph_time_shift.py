"""Differentiable edge-value time shifts with explicit zero boundaries."""

from __future__ import annotations

import torch


def shift_trace_graph_values(values: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
    """Linearly sample ``values[E,C,F]`` at frame ``f + shifts[e]``.

    Positive shifts advance the sender waveform into earlier output frames.
    The same shift applies to every channel. Samples outside ``[0,F)`` are
    zero; there is no circular wrap. Shifts use latent-frame units, share the
    values' floating dtype/device, and may have gradients. At integer shifts
    autograd uses the right-hand piecewise-linear derivative.

    Fractional shifts attenuate high temporal frequencies: a half-frame shift
    averages adjacent samples and has amplitude response ``abs(cos(omega/2))``.
    This is linear interpolation, not an amplitude-preserving delay operator.
    """
    if not isinstance(values, torch.Tensor) or not isinstance(shifts, torch.Tensor):
        raise TypeError("values and shifts must be torch.Tensor objects")
    if values.ndim != 3 or min(values.shape[1:]) < 1:
        raise ValueError("values must have shape (edges, positive channels, positive frames)")
    if shifts.shape != (values.shape[0],):
        raise ValueError("shifts must have shape (edges,)")
    if not values.is_floating_point() or not shifts.is_floating_point():
        raise TypeError("values and shifts must have floating-point dtypes")
    if values.dtype != shifts.dtype or values.device != shifts.device:
        raise ValueError("values and shifts must share dtype and device")
    if not bool(torch.isfinite(shifts).all()):
        raise ValueError("shifts must be finite")
    frame_count = values.shape[2]
    position = torch.arange(frame_count, dtype=shifts.dtype, device=shifts.device)[None]
    position = position + shifts[:, None]
    lower = position.floor()
    fraction = position - lower
    lower_index = lower.to(torch.int64)
    upper_index = lower_index + 1
    lower_values = _zero_padded_gather(values, lower_index)
    upper_values = _zero_padded_gather(values, upper_index)
    return lower_values * (1 - fraction[:, None]) + upper_values * fraction[:, None]


def _zero_padded_gather(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    frame_count = values.shape[2]
    selected = torch.gather(values, 2, index.clamp(0, frame_count - 1)[:, None].expand_as(values))
    valid = (index >= 0) & (index < frame_count)
    return selected.masked_fill(~valid[:, None], 0)
