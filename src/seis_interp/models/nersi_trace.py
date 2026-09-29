"""Trace-wise NeRSI input/output adaptation for four continuous coordinates."""

import math
from numbers import Integral, Real

import numpy as np
import torch
from torch import nn

from seis_interp.models.nersi import FourierFeatureMapping, NersiBlock

# The residual slowness reads the polar coordinates through its own small fixed encoder.
RESIDUAL_SLOWNESS_FOURIER_COMPONENTS = 8
RESIDUAL_SLOWNESS_FREQUENCY_BASE = (1.05, 1.05, 1.1, 1.1)
RESIDUAL_SLOWNESS_WIDTH = 128
OUTPUT_WINDOW_KEYS = ("max_delay_samples", "window_samples", "delay_samples_per_radius")
# Raised-cosine band edges: the taper rises over [lo - 2, lo] Hz and falls over [hi, hi + 5] Hz.
BAND_LOW_RAMP_HZ = 2
BAND_HIGH_RAMP_HZ = 5


class NersiTrace(nn.Module):
    """Use NeRSI's encoder/2D decoder with a minimal 8-column output canvas.

    Four actual coordinates identify one trace. Three 2x shuffle stages require
    an 8-multiple canvas. Only column zero and the requested time samples are
    returned; no interpolation of coordinates, time, or waveform values occurs.
    This is an explicit trace-wise adaptation, not the C3 profile-wise baseline.

    Optional multi-slowness heads (``head_samples_per_radius``) give the decoder
    one output channel per head; channel k is read along its own linear moveout
    ``m_k * r`` and the reads are summed. Zero-initialized surface-consistent
    statics and a coordinate-conditioned residual slowness refine the read
    positions. A fixed zero-phase band taper and a geometry-only output window
    may follow, in that order. Every option defaults to off, which keeps the
    module structure, parameter creation order and output unchanged.
    """

    def __init__(
        self,
        *,
        time_sample_count,
        input_features=4,
        fourier_components=160,
        frequency_base=1.25,
        encoder_width=1536,
        latent_channels=192,
        decoder_channels=(192, 96, 48),
        upsample_scales=(2, 2, 2),
        kernel_size=5,
        activation="gelu",
        output_activation="linear",
        coordinate_transform=None,
        polar_offset=False,
        moveout_samples_per_radius=0.0,
        head_samples_per_radius=None,
        static_shift_samples=0.0,
        residual_slowness_samples_per_radius=0.0,
        spatial_shape=None,
        output_band_hz=None,
        sample_interval_s=None,
        output_window=None,
    ):
        super().__init__()
        if input_features != 4 or time_sample_count < 1:
            raise ValueError("trace NeRSI requires four coordinates and positive time count")
        if tuple(upsample_scales) != (2, 2, 2) or len(decoder_channels) != 3:
            raise ValueError("trace NeRSI requires three 2x decoder stages")
        if activation != "gelu" or output_activation != "linear":
            raise ValueError("trace NeRSI requires GELU and linear output")
        self.time_sample_count = int(time_sample_count)
        self.latent_channels = latent_channels
        transform = None
        if coordinate_transform is not None:
            transform = torch.as_tensor(coordinate_transform, dtype=torch.float32)
            if transform.shape != (4, 5) or not torch.isfinite(transform).all():
                raise ValueError("coordinate_transform must be a finite 4 by 5 affine matrix")
        self.register_buffer("coordinate_transform", transform)
        if not isinstance(polar_offset, bool) or (polar_offset and transform is None):
            raise ValueError("polar_offset requires an explicit coordinate_transform")
        self.polar_offset = polar_offset
        moveout = float(moveout_samples_per_radius)
        if not math.isfinite(moveout) or moveout < 0 or (moveout and not polar_offset):
            raise ValueError("nonnegative finite moveout requires polar_offset coordinates")
        self.moveout_samples_per_radius = moveout
        heads = _head_shifts(head_samples_per_radius, polar_offset, moveout)
        self.uses_heads = heads is not None
        self.head_count = len(heads) if self.uses_heads else 1
        self.static_shift_samples = _head_option(
            static_shift_samples, "static_shift_samples", heads
        )
        self.residual_slowness_samples_per_radius = _head_option(
            residual_slowness_samples_per_radius, "residual_slowness_samples_per_radius", heads
        )
        self.spatial_shape = _static_grid_shape(spatial_shape, self.static_shift_samples)
        if self.uses_heads:
            self.pad_left, self.pad_right = self._head_padding(heads)
            self.time_padding = self.pad_left + self.pad_right
        else:
            self.time_padding = math.ceil(moveout) + 1 if moveout else 0
            self.pad_left, self.pad_right = self.time_padding, 0
        self.decoder_time_count = self.time_sample_count + self.time_padding
        self.base_time = math.ceil(self.decoder_time_count / 8)
        self.fourier_mapping = FourierFeatureMapping(
            input_features=4, fourier_components=fourier_components, frequency_base=frequency_base
        )
        self.encoder = nn.Sequential(
            nn.Linear(self.fourier_mapping.output_features, encoder_width),
            nn.GELU(),
            nn.Linear(encoder_width, latent_channels * self.base_time),
            nn.GELU(),
        )
        incoming = (latent_channels, *decoder_channels[:-1])
        self.decoder = nn.Sequential(
            *(
                NersiBlock(a, b, kernel_size=kernel_size)
                for a, b in zip(incoming, decoder_channels, strict=True)
            )
        )
        self.output_convolution = nn.Conv2d(decoder_channels[-1], self.head_count, 1)
        if self.uses_heads:
            self.register_buffer(
                "head_samples_per_radius", torch.tensor(heads, dtype=torch.float32)
            )
        if self.static_shift_samples:
            self._build_statics()
        if self.residual_slowness_samples_per_radius:
            self._build_residual_slowness()
        band = band_taper(self.time_sample_count, output_band_hz, sample_interval_s)
        self.register_buffer(
            "output_band", None if band is None else torch.tensor(band, dtype=torch.float32)
        )
        self.output_window = _output_window(output_window, polar_offset)

    def _head_padding(self, heads):
        """Decoder samples before and after the canvas that every head read can reach."""
        margin = 2 * self.static_shift_samples + self.residual_slowness_samples_per_radius
        pad_left = math.ceil(max(0.0, -min(heads)) + margin) + 1
        pad_right = math.ceil(max(0.0, max(heads)) + margin) + 1
        return pad_left, pad_right

    def _build_statics(self):
        """Zero-initialized source and receiver shifts on the native line/point grids."""
        source_lines, source_points, receiver_lines, receiver_points = self.spatial_shape
        self.src_static = nn.Parameter(torch.zeros(1, 1, source_lines, source_points))
        self.rcv_static = nn.Parameter(torch.zeros(1, 1, receiver_lines, receiver_points))

    def _build_residual_slowness(self):
        """Per-head slowness correction from the polar coordinates; the last layer starts at 0."""
        self.residual_slowness_fourier = FourierFeatureMapping(
            input_features=4,
            fourier_components=RESIDUAL_SLOWNESS_FOURIER_COMPONENTS,
            frequency_base=list(RESIDUAL_SLOWNESS_FREQUENCY_BASE),
        )
        self.residual_slowness_network = nn.Sequential(
            nn.Linear(self.residual_slowness_fourier.output_features, RESIDUAL_SLOWNESS_WIDTH),
            nn.GELU(),
            nn.Linear(RESIDUAL_SLOWNESS_WIDTH, self.head_count),
        )
        nn.init.zeros_(self.residual_slowness_network[-1].weight)
        nn.init.zeros_(self.residual_slowness_network[-1].bias)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        local = coordinates
        if self.coordinate_transform is not None:
            coordinates = nn.functional.linear(
                coordinates, self.coordinate_transform[:, :4], self.coordinate_transform[:, 4]
            )
        if self.polar_offset:
            offset = coordinates[:, 2:]
            radius = torch.linalg.vector_norm(offset, dim=1)
            x = torch.where(radius > 0, offset[:, 0], torch.ones_like(radius))
            angle = torch.atan2(offset[:, 1], x) / math.pi
            coordinates = torch.cat((coordinates[:, :2], radius[:, None], angle[:, None]), dim=1)
        features = self.fourier_mapping(coordinates)
        latent = self.encoder(features).reshape(
            len(coordinates), self.latent_channels, self.base_time, 1
        )
        if self.uses_heads:
            decoded = self.output_convolution(self.decoder(latent))[
                :, :, : self.decoder_time_count, 0
            ]
            output = self._apply_heads(decoded, coordinates, radius, local)
        else:
            decoded = self.output_convolution(self.decoder(latent))[
                :, 0, : self.decoder_time_count, 0
            ]
            output = (
                self._apply_moveout(decoded, radius) if self.moveout_samples_per_radius else decoded
            )
        if self.output_band is not None:
            output = self._apply_band(output)
        if self.output_window is not None:
            output = self._apply_window(output, radius)
        return output

    def _apply_moveout(self, decoded: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        """Sample the padded decoder without truncating the requested physical time axis."""
        time = torch.arange(self.time_sample_count, dtype=decoded.dtype, device=decoded.device)
        index = (
            time[None, :] + self.time_padding - self.moveout_samples_per_radius * radius[:, None]
        )
        index = index.clamp(0, self.decoder_time_count - 1)
        lower = index.floor().long()
        upper = (lower + 1).clamp(max=self.decoder_time_count - 1)
        fraction = index - lower
        return decoded.gather(1, lower).lerp(decoded.gather(1, upper), fraction)

    def _apply_heads(self, decoded, polar, radius, local):
        """Read head k at ``t + pad_left + (m_k + dm_k) * r + shift`` and sum the heads."""
        time = torch.arange(self.time_sample_count, dtype=decoded.dtype, device=decoded.device)
        slope = self.head_samples_per_radius[None, :].expand(len(decoded), -1)
        if self.residual_slowness_samples_per_radius:
            correction = self.residual_slowness_network(self.residual_slowness_fourier(polar))
            slope = slope + self.residual_slowness_samples_per_radius * torch.tanh(correction)
        index = time[None, None, :] + self.pad_left + slope[:, :, None] * radius[:, None, None]
        if self.static_shift_samples:
            index = index + self._static_shift(local)[:, None, None]
        index = index.clamp(0, self.decoder_time_count - 1)
        lower = index.floor().long()
        upper = (lower + 1).clamp(max=self.decoder_time_count - 1)
        fraction = index - lower
        return decoded.gather(2, lower).lerp(decoded.gather(2, upper), fraction).sum(1)

    def _static_shift(self, local: torch.Tensor) -> torch.Tensor:
        """``S * (tanh(source static) + tanh(receiver static))`` at the input grid position."""
        source = torch.tanh(_sample_grid(self.src_static, local[:, :2]))
        receiver = torch.tanh(_sample_grid(self.rcv_static, local[:, 2:]))
        return self.static_shift_samples * (source + receiver)

    def _apply_band(self, output: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(output, dim=1) * self.output_band
        return torch.fft.irfft(spectrum, n=self.time_sample_count, dim=1)

    def _apply_window(self, output: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        """Zero samples outside ``[max_delay - rint(r * delay), +window)`` of each trace.

        The float32 radius follows the prototype, so where ``offset / (v dt)`` sits within
        rounding of a half sample the start can differ from the canvas's integer alignment
        delay by one sample (7 of 49,151 study_059 cells); one recorded sample is then zeroed.
        """
        window = self.output_window
        time = torch.arange(self.time_sample_count, dtype=output.dtype, device=output.device)
        start = window["max_delay_samples"] - torch.round(
            radius * window["delay_samples_per_radius"]
        )
        keep = (time[None, :] >= start[:, None]) & (
            time[None, :] < start[:, None] + window["window_samples"]
        )
        return output * keep


def band_taper(time_count, band_hz, sample_interval_s):
    """Zero-phase raised-cosine weights over ``rfftfreq(time_count, dt)``, or None when off.

    Zero below ``lo - 2`` Hz, a raised-cosine rise to 1 at ``lo``, 1 up to ``hi``,
    a raised-cosine fall to 0 at ``hi + 5`` Hz and zero above.
    """
    if band_hz is None:
        if sample_interval_s is not None:
            raise ValueError("sample_interval_s is used only with output_band_hz")
        return None
    interval = sample_interval_s
    if not _is_real(interval) or interval <= 0:
        raise ValueError("output_band_hz requires a finite positive sample_interval_s")
    if len(band_hz) != 2 or not all(_is_real(value) for value in band_hz):
        raise ValueError("output_band_hz must be two finite frequencies (lo, hi)")
    low, high = band_hz
    if not BAND_LOW_RAMP_HZ < low < high or high + BAND_HIGH_RAMP_HZ >= 0.5 / interval:
        raise ValueError("output_band_hz needs 2 < lo < hi and hi + 5 below the Nyquist frequency")
    frequency = np.fft.rfftfreq(time_count, interval)
    rise = np.clip((frequency - (low - BAND_LOW_RAMP_HZ)) / BAND_LOW_RAMP_HZ, 0, 1)
    fall = np.clip(((high + BAND_HIGH_RAMP_HZ) - frequency) / BAND_HIGH_RAMP_HZ, 0, 1)
    weight = np.where(frequency < low - BAND_LOW_RAMP_HZ, 0, rise * fall)
    return 0.5 - 0.5 * np.cos(np.pi * weight)


def _sample_grid(grid: torch.Tensor, local: torch.Tensor) -> torch.Tensor:
    """Bilinear value of a (1, 1, lines, points) grid at (line, point) in [0, 1]."""
    points = (local * 2 - 1).flip(1).reshape(1, len(local), 1, 2)
    return nn.functional.grid_sample(grid, points, align_corners=True).reshape(len(local))


def _is_real(value) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _head_shifts(values, polar_offset, moveout):
    if values is None:
        return None
    values = list(values)
    if not values or not all(_is_real(value) for value in values):
        raise ValueError("head_samples_per_radius must be a nonempty list of finite numbers")
    if not polar_offset:
        raise ValueError("head_samples_per_radius requires polar_offset coordinates")
    if moveout:
        raise ValueError("head_samples_per_radius and moveout_samples_per_radius are exclusive")
    return [float(value) for value in values]


def _head_option(value, name, heads):
    if not _is_real(value) or value < 0:
        raise ValueError(f"{name} must be a finite number >= 0")
    if value and heads is None:
        raise ValueError(f"{name} requires head_samples_per_radius")
    return float(value)


def _static_grid_shape(spatial_shape, static_shift_samples):
    if not static_shift_samples:
        if spatial_shape is not None:
            raise ValueError("spatial_shape is used only with static_shift_samples")
        return None
    if (
        spatial_shape is None
        or len(spatial_shape) != 4
        or not all(isinstance(n, Integral) and not isinstance(n, bool) for n in spatial_shape)
        or min(spatial_shape) < 1
    ):
        raise ValueError("static_shift_samples requires spatial_shape of four positive integers")
    return tuple(int(n) for n in spatial_shape)


def _output_window(window, polar_offset):
    if window is None:
        return None
    if not isinstance(window, dict) or set(window) != set(OUTPUT_WINDOW_KEYS):
        raise ValueError(f"output_window must have exactly {', '.join(OUTPUT_WINDOW_KEYS)}")
    if not polar_offset:
        raise ValueError("output_window requires polar_offset coordinates")
    maximum, samples, delay = (window[key] for key in OUTPUT_WINDOW_KEYS)
    integers = all(isinstance(n, Integral) and not isinstance(n, bool) for n in (maximum, samples))
    if not integers or maximum < 0 or samples < 1 or not _is_real(delay) or delay < 0:
        raise ValueError(
            "output_window needs integer max_delay_samples >= 0, window_samples >= 1 "
            "and finite delay_samples_per_radius >= 0"
        )
    return {
        "max_delay_samples": int(maximum),
        "window_samples": int(samples),
        "delay_samples_per_radius": float(delay),
    }
