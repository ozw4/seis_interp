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
        frequency_base: float | Sequence[float] = 1.25,
        axis_frequency_limits: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        self.input_features = _positive_integer(input_features, "input_features")
        self.fourier_components = _positive_integer(fourier_components, "fourier_components")
        self.frequency_base = _frequency_bases(frequency_base, self.input_features)
        bases = (
            (self.frequency_base,) * self.input_features
            if isinstance(self.frequency_base, float)
            else self.frequency_base
        )
        try:
            per_axis_frequencies = [
                [math.pi * base**exponent for exponent in range(1, self.fourier_components + 1)]
                for base in bases
            ]
        except OverflowError as error:
            raise ValueError("Fourier frequencies must be finite float32 values") from error
        if any(
            not math.isfinite(value) or value > torch.finfo(torch.float32).max
            for frequencies in per_axis_frequencies
            for value in frequencies
        ):
            raise ValueError("Fourier frequencies must be finite float32 values")
        frequency_tensor = torch.tensor(per_axis_frequencies, dtype=torch.float32)
        if isinstance(self.frequency_base, float):
            frequency_tensor = frequency_tensor[0]
        self.register_buffer("frequencies", frequency_tensor)
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
                torch.tensor(per_axis_frequencies, dtype=torch.float64) <= limits[:, None],
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
        convolution: str = "standard",
    ) -> None:
        super().__init__()
        self.in_channels = _positive_integer(in_channels, "in_channels")
        self.out_channels = _positive_integer(out_channels, "out_channels")
        self.kernel_size = _odd_kernel_size(kernel_size)
        if convolution not in ("standard", "depthwise_separable"):
            raise ValueError("convolution must be 'standard' or 'depthwise_separable'")
        self.convolution_kind = convolution
        if (
            isinstance(upsample_scale, bool)
            or not isinstance(upsample_scale, Integral)
            or upsample_scale != 2
        ):
            raise ValueError("upsample_scale must be 2")
        self.upsample_scale = 2
        if convolution == "standard":
            self.convolution = nn.Conv2d(
                self.in_channels,
                self.out_channels * self.upsample_scale**2,
                self.kernel_size,
                padding=self.kernel_size // 2,
            )
        else:
            self.depthwise_convolution = nn.Conv2d(
                self.in_channels,
                self.in_channels,
                self.kernel_size,
                padding=self.kernel_size // 2,
                groups=self.in_channels,
            )
            self.pointwise_convolution = nn.Conv2d(
                self.in_channels,
                self.out_channels * self.upsample_scale**2,
                1,
            )
        self.pixel_shuffle = nn.PixelShuffle(self.upsample_scale)
        self.activation = nn.GELU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 4 or values.shape[1] != self.in_channels:
            raise ValueError(
                f"values must have shape (B, {self.in_channels}, H, W), got {tuple(values.shape)}"
            )
        if self.convolution_kind == "standard":
            convolved = self.convolution(values)
        else:
            convolved = self.pointwise_convolution(self.depthwise_convolution(values))
        return self.activation(self.pixel_shuffle(convolved))


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
        frequency_base: float | Sequence[float] = 1.25,
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
        decoder_convolution: str = "standard",
        profile_embedding_channels: int | None = None,
        profile_grid_shape: Sequence[int] | None = None,
        latent_spatial_rank: int | None = None,
        temporal_basis_components: int | None = None,
    ) -> None:
        super().__init__()
        if isinstance(input_features, bool) or not isinstance(input_features, Integral):
            raise ValueError("input_features must be 3")
        if input_features != 3:
            raise ValueError("input_features must be 3")
        self.input_features = 3
        self.fourier_components = _positive_integer(fourier_components, "fourier_components")
        self.frequency_base = _frequency_bases(frequency_base, self.input_features)
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
        if decoder_convolution not in ("standard", "depthwise_separable"):
            raise ValueError("decoder_convolution must be 'standard' or 'depthwise_separable'")
        self.decoder_convolution = decoder_convolution
        self.latent_spatial_rank = (
            None
            if latent_spatial_rank is None
            else _positive_integer(latent_spatial_rank, "latent_spatial_rank")
        )
        self.temporal_basis_components = (
            None
            if temporal_basis_components is None
            else _positive_integer(temporal_basis_components, "temporal_basis_components")
        )
        self.coordinate_mapping = deepcopy(coordinate_mapping)
        self.profile_embedding_channels = None
        self.profile_grid_shape = None
        if (profile_embedding_channels is None) != (profile_grid_shape is None):
            raise ValueError(
                "profile_embedding_channels and profile_grid_shape must be provided together"
            )
        if profile_embedding_channels is not None:
            if coordinate_mapping is not None:
                raise ValueError("profile embeddings require index profile coordinates")
            self.profile_embedding_channels = _positive_integer(
                profile_embedding_channels, "profile_embedding_channels"
            )
            self.profile_grid_shape = _three_positive_integers(
                profile_grid_shape, name="profile_grid_shape"
            )
            self.register_buffer(
                "profile_index_scale",
                torch.tensor(
                    [length - 1 for length in self.profile_grid_shape], dtype=torch.float32
                ),
            )
            self.profile_embedding = nn.Embedding(
                math.prod(self.profile_grid_shape), self.profile_embedding_channels
            )
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
        decoder_time_count = self.temporal_basis_components or time_count
        if decoder_time_count > time_count or decoder_time_count % total_scale:
            raise ValueError(
                "temporal_basis_components must not exceed profile time and must be divisible "
                "by the total upsample scale"
            )
        self.decoder_profile_shape = (decoder_time_count, receiver_y)
        self.base_profile_shape = (
            decoder_time_count // total_scale,
            receiver_y // total_scale,
        )
        latent_size = (
            self.latent_channels * math.prod(self.base_profile_shape)
            if self.latent_spatial_rank is None
            else self.latent_channels * self.latent_spatial_rank * sum(self.base_profile_shape)
        )

        self.fourier_mapping = FourierFeatureMapping(
            input_features=self.input_features,
            fourier_components=self.fourier_components,
            frequency_base=self.frequency_base,
            axis_frequency_limits=axis_frequency_limits,
        )
        encoder_input_features = self.fourier_mapping.output_features + (
            self.profile_embedding_channels or 0
        )
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_features, self.encoder_width),
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
                    convolution=self.decoder_convolution,
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
        if self.temporal_basis_components is not None:
            samples = torch.arange(time_count, dtype=torch.float32) + 0.5
            orders = torch.arange(self.temporal_basis_components, dtype=torch.float32)[:, None]
            basis = torch.cos(math.pi * orders * samples / time_count)
            basis[0].mul_(math.sqrt(1 / time_count))
            basis[1:].mul_(math.sqrt(2 / time_count))
            self.temporal_basis = nn.Parameter(basis)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        profile_indices = None
        if self.profile_embedding_channels is not None:
            indices = torch.round(coordinates * self.profile_index_scale).long()
            profile_indices = (
                indices[:, 0] * self.profile_grid_shape[1] + indices[:, 1]
            ) * self.profile_grid_shape[2] + indices[:, 2]
        if self.coordinate_mapping is not None:
            indices = torch.round(coordinates * self.coordinate_index_scale).long()
            flat = (
                indices[:, 0] * self.coordinate_grid_shape[1] + indices[:, 1]
            ) * self.coordinate_grid_shape[2] + indices[:, 2]
            coordinates = self.coordinate_anchors[flat]
        features = self.fourier_mapping(coordinates)
        if profile_indices is not None:
            features = torch.cat((features, self.profile_embedding(profile_indices)), dim=1)
        encoded = self.encoder(features)
        if self.latent_spatial_rank is None:
            latent = encoded.reshape(
                coordinates.shape[0],
                self.latent_channels,
                *self.base_profile_shape,
            )
        else:
            base_time, base_receiver = self.base_profile_shape
            time_value_count = self.latent_channels * self.latent_spatial_rank * base_time
            time_factors = encoded[:, :time_value_count].reshape(
                coordinates.shape[0],
                self.latent_channels,
                self.latent_spatial_rank,
                base_time,
                1,
            )
            receiver_factors = encoded[:, time_value_count:].reshape(
                coordinates.shape[0],
                self.latent_channels,
                self.latent_spatial_rank,
                1,
                base_receiver,
            )
            latent = (time_factors * receiver_factors).sum(dim=2) / math.sqrt(
                self.latent_spatial_rank
            )
        output = self.output_convolution(self.decoder(latent))
        if self.temporal_basis_components is not None:
            output = torch.einsum("bckr,kt->bctr", output, self.temporal_basis)
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
            **(
                {"decoder_convolution": self.decoder_convolution}
                if self.decoder_convolution != "standard"
                else {}
            ),
            **(
                {
                    "profile_embedding_channels": self.profile_embedding_channels,
                    "profile_grid_shape": self.profile_grid_shape,
                }
                if self.profile_embedding_channels is not None
                else {}
            ),
            **(
                {"latent_spatial_rank": self.latent_spatial_rank}
                if self.latent_spatial_rank is not None
                else {}
            ),
            **(
                {"temporal_basis_components": self.temporal_basis_components}
                if self.temporal_basis_components is not None
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


def _frequency_bases(value: object, input_features: int) -> float | tuple[float, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != input_features:
            raise ValueError(f"frequency_base must contain {input_features} values")
        return tuple(_frequency_base(item) for item in value)
    return _frequency_base(value)


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
