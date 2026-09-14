"""Profile-wise neural implicit representation for C3 interpolation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from copy import deepcopy
from numbers import Integral, Real

import torch
from torch import nn


class FourierFeatureMapping(nn.Module):
    """Encode normalized coordinates with an exponential Fourier schedule.

    ``frequency_base=1.25`` is a repository reimplementation choice.  For each
    coordinate axis, output columns interleave cosine and sine at
    ``pi * frequency_base**i`` for ``i=1..fourier_components`` before moving to
    the next axis.
    """

    def __init__(
        self,
        input_features: int = 3,
        fourier_components: int = 40,
        frequency_base: float = 1.25,
        axis_frequency_limits: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        self.input_features = _positive_integer(input_features, "input_features")
        self.fourier_components = _positive_integer(fourier_components, "fourier_components")
        self.frequency_base = _frequency_base(frequency_base)
        frequencies = []
        try:
            for exponent in range(1, self.fourier_components + 1):
                frequencies.append(math.pi * self.frequency_base**exponent)
        except OverflowError as error:
            raise ValueError("Fourier frequencies must be finite float32 values") from error
        if any(
            not math.isfinite(value) or value > torch.finfo(torch.float32).max
            for value in frequencies
        ):
            raise ValueError("Fourier frequencies must be finite float32 values")
        self.register_buffer("frequencies", torch.tensor(frequencies, dtype=torch.float32))
        self.axis_frequency_limits = None
        if axis_frequency_limits is not None:
            if len(axis_frequency_limits) != self.input_features or any(
                isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) or v < 0
                for v in axis_frequency_limits
            ):
                raise ValueError(
                    "axis_frequency_limits must contain finite nonnegative angular frequencies"
                )
            self.axis_frequency_limits = tuple(float(v) for v in axis_frequency_limits)
            limits = torch.tensor(self.axis_frequency_limits, dtype=torch.float64)
            self.register_buffer(
                "frequency_mask",
                torch.tensor(frequencies, dtype=torch.float64)[None, :] <= limits[:, None],
            )

    @property
    def output_features(self) -> int:
        """Return the encoded feature width."""
        return self.input_features * 2 * self.fourier_components

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        if coordinates.ndim != 2 or coordinates.shape[1] != self.input_features:
            raise ValueError(
                f"coordinates must have shape (B, {self.input_features}), "
                f"got {tuple(coordinates.shape)}"
            )
        if not torch.is_floating_point(coordinates):
            raise ValueError("coordinates must contain floating-point values")
        frequencies = self.frequencies
        if self.axis_frequency_limits is not None:
            frequencies = frequencies * self.frequency_mask
        angles = coordinates.unsqueeze(-1) * frequencies
        encoded = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
        if self.axis_frequency_limits is not None:
            encoded = encoded * self.frequency_mask[None, :, :, None]
        return encoded.reshape(coordinates.shape[0], self.output_features)


class NersiBlock(nn.Module):
    """Apply same-padded convolution, 2x PixelShuffle, and GELU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        upsample_scale: int = 2,
    ) -> None:
        super().__init__()
        self.in_channels = _positive_integer(in_channels, "in_channels")
        self.out_channels = _positive_integer(out_channels, "out_channels")
        self.kernel_size = _odd_kernel_size(kernel_size)
        if (
            isinstance(upsample_scale, bool)
            or not isinstance(upsample_scale, Integral)
            or upsample_scale != 2
        ):
            raise ValueError("upsample_scale must be 2")
        self.upsample_scale = 2
        self.convolution = nn.Conv2d(
            self.in_channels,
            self.out_channels * self.upsample_scale**2,
            self.kernel_size,
            padding=self.kernel_size // 2,
        )
        self.pixel_shuffle = nn.PixelShuffle(self.upsample_scale)
        self.activation = nn.GELU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 4 or values.shape[1] != self.in_channels:
            raise ValueError(
                f"values must have shape (B, {self.in_channels}, H, W), got {tuple(values.shape)}"
            )
        return self.activation(self.pixel_shuffle(self.convolution(values)))


class Nersi(nn.Module):
    """Generate a complete ``(time, receiver-y)`` profile from three indices.

    Model weights alone do not identify the represented function: profile
    shape and Fourier frequencies determine both the decoder base grid and the
    coordinate mapping.  Callers saving weights must therefore also store all
    values returned by :meth:`constructor_config`.
    """

    def __init__(
        self,
        input_features: int = 3,
        fourier_components: int = 40,
        frequency_base: float = 1.25,
        encoder_width: int = 256,
        latent_channels: int = 64,
        decoder_channels: Sequence[int] = (64, 32, 16),
        profile_shape: Sequence[int] = (384, 32),
        upsample_scales: Sequence[int] = (2, 2, 2),
        kernel_size: int = 3,
        activation: str = "gelu",
        output_activation: str = "linear",
        coordinate_mapping: dict | None = None,
        axis_frequency_limits: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        if isinstance(input_features, bool) or not isinstance(input_features, Integral):
            raise ValueError("input_features must be 3")
        if input_features != 3:
            raise ValueError("input_features must be 3")
        self.input_features = 3
        self.fourier_components = _positive_integer(fourier_components, "fourier_components")
        self.frequency_base = _frequency_base(frequency_base)
        self.encoder_width = _positive_integer(encoder_width, "encoder_width")
        self.latent_channels = _positive_integer(latent_channels, "latent_channels")
        self.decoder_channels = _three_positive_integers(decoder_channels, name="decoder_channels")
        self.profile_shape = _profile_shape(profile_shape)
        self.upsample_scales = _upsample_scales(upsample_scales)
        self.kernel_size = _odd_kernel_size(kernel_size)
        if activation != "gelu":
            raise ValueError("activation must be 'gelu'")
        if output_activation != "linear":
            raise ValueError("output_activation must be 'linear'")
        self.activation = activation
        self.output_activation = output_activation
        self.coordinate_mapping = deepcopy(coordinate_mapping)
        if coordinate_mapping is not None:
            anchors = torch.tensor(coordinate_mapping["normalized_anchors"], dtype=torch.float32)
            shape = _three_positive_integers(
                coordinate_mapping["profile_grid_shape"], name="profile_grid_shape"
            )
            if (
                anchors.shape != (math.prod(shape), 3)
                or not torch.isfinite(anchors).all()
                or torch.any(anchors < 0)
                or torch.any(anchors > 1)
            ):
                raise ValueError(
                    "coordinate_mapping requires unit-interval anchors for every profile"
                )
            self.register_buffer("coordinate_anchors", anchors)
            self.register_buffer(
                "coordinate_index_scale", torch.tensor([n - 1 for n in shape], dtype=torch.float32)
            )
            self.coordinate_grid_shape = shape

        total_scale = math.prod(self.upsample_scales)
        time_count, receiver_y = self.profile_shape
        self.base_profile_shape = (time_count // total_scale, receiver_y // total_scale)
        latent_size = self.latent_channels * math.prod(self.base_profile_shape)

        self.fourier_mapping = FourierFeatureMapping(
            input_features=self.input_features,
            fourier_components=self.fourier_components,
            frequency_base=self.frequency_base,
            axis_frequency_limits=axis_frequency_limits,
        )
        self.encoder = nn.Sequential(
            nn.Linear(self.fourier_mapping.output_features, self.encoder_width),
            nn.GELU(),
            nn.Linear(self.encoder_width, latent_size),
            nn.GELU(),
        )
        incoming_channels = (self.latent_channels, *self.decoder_channels[:-1])
        self.decoder = nn.Sequential(
            *(
                NersiBlock(
                    incoming,
                    outgoing,
                    kernel_size=self.kernel_size,
                    upsample_scale=scale,
                )
                for incoming, outgoing, scale in zip(
                    incoming_channels,
                    self.decoder_channels,
                    self.upsample_scales,
                    strict=True,
                )
            )
        )
        self.output_convolution = nn.Conv2d(self.decoder_channels[-1], 1, kernel_size=1)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        if self.coordinate_mapping is not None:
            indices = torch.round(coordinates * self.coordinate_index_scale).long()
            flat = (
                indices[:, 0] * self.coordinate_grid_shape[1] + indices[:, 1]
            ) * self.coordinate_grid_shape[2] + indices[:, 2]
            coordinates = self.coordinate_anchors[flat]
        encoded = self.encoder(self.fourier_mapping(coordinates))
        latent = encoded.reshape(
            coordinates.shape[0],
            self.latent_channels,
            *self.base_profile_shape,
        )
        output = self.output_convolution(self.decoder(latent))
        expected = (coordinates.shape[0], 1, *self.profile_shape)
        if output.shape != expected:
            raise RuntimeError(f"NeRSI output shape must be {expected}, got {tuple(output.shape)}")
        return output

    def constructor_config(self) -> dict[str, object]:
        """Return every constructor value needed to restore the same function."""
        return {
            "input_features": self.input_features,
            "fourier_components": self.fourier_components,
            "frequency_base": self.frequency_base,
            "encoder_width": self.encoder_width,
            "latent_channels": self.latent_channels,
            "decoder_channels": self.decoder_channels,
            "profile_shape": self.profile_shape,
            "upsample_scales": self.upsample_scales,
            "kernel_size": self.kernel_size,
            "activation": self.activation,
            "output_activation": self.output_activation,
            **(
                {"coordinate_mapping": deepcopy(self.coordinate_mapping)}
                if self.coordinate_mapping is not None
                else {}
            ),
            **(
                {"axis_frequency_limits": self.fourier_mapping.axis_frequency_limits}
                if self.fourier_mapping.axis_frequency_limits is not None
                else {}
            ),
        }


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _frequency_base(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("frequency_base must be a finite number greater than 1")
    number = float(value)
    if not math.isfinite(number) or number <= 1.0:
        raise ValueError("frequency_base must be a finite number greater than 1")
    return number


def _odd_kernel_size(value: object) -> int:
    kernel_size = _positive_integer(value, "kernel_size")
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    return kernel_size


def _three_positive_integers(values: object, *, name: str) -> tuple[int, int, int]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or len(values) != 3:
        raise ValueError(f"{name} must contain exactly three positive integers")
    converted = tuple(_positive_integer(value, name) for value in values)
    return converted  # type: ignore[return-value]


def _profile_shape(values: object) -> tuple[int, int]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or len(values) != 2:
        raise ValueError("profile_shape must contain two positive integers divisible by 8")
    converted = tuple(_positive_integer(value, "profile_shape") for value in values)
    if any(value % 8 != 0 for value in converted):
        raise ValueError("profile_shape dimensions must be divisible by 8")
    return converted  # type: ignore[return-value]


def _upsample_scales(values: object) -> tuple[int, int, int]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or len(values) != 3:
        raise ValueError("upsample_scales must be (2, 2, 2)")
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) or value != 2 for value in values
    ):
        raise ValueError("upsample_scales must be (2, 2, 2)")
    return (2, 2, 2)
