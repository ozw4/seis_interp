"""Trace-wise NeRSI input/output adaptation for four continuous coordinates."""

import math

import torch
from torch import nn

from seis_interp.models.nersi import FourierFeatureMapping, NersiBlock


class NersiTrace(nn.Module):
    """Use NeRSI's encoder/2D decoder with a minimal 8-column output canvas.

    Four actual coordinates identify one trace. Three 2x shuffle stages require
    an 8-multiple canvas. Only column zero and the requested time samples are
    returned; no interpolation of coordinates, time, or waveform values occurs.
    This is an explicit trace-wise adaptation, not the C3 profile-wise baseline.
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
        self.time_padding = math.ceil(moveout) + 1 if moveout else 0
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
        self.output_convolution = nn.Conv2d(decoder_channels[-1], 1, 1)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
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
        decoded = self.output_convolution(self.decoder(latent))[:, 0, : self.decoder_time_count, 0]
        return self._apply_moveout(decoded, radius) if self.moveout_samples_per_radius else decoded

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
