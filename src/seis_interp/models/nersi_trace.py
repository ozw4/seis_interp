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
    ):
        super().__init__()
        if input_features != 4 or time_sample_count < 1:
            raise ValueError("trace NeRSI requires four coordinates and positive time count")
        if tuple(upsample_scales) != (2, 2, 2) or len(decoder_channels) != 3:
            raise ValueError("trace NeRSI requires three 2x decoder stages")
        if activation != "gelu" or output_activation != "linear":
            raise ValueError("trace NeRSI requires GELU and linear output")
        self.time_sample_count = int(time_sample_count)
        self.base_time = math.ceil(self.time_sample_count / 8)
        self.latent_channels = latent_channels
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
        features = self.fourier_mapping(coordinates)
        latent = self.encoder(features).reshape(
            len(coordinates), self.latent_channels, self.base_time, 1
        )
        return self.output_convolution(self.decoder(latent))[:, 0, : self.time_sample_count, 0]
