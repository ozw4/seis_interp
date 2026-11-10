"""Implicit trace field conditioned on the nearest observed traces."""

from numbers import Integral

import torch
from torch import nn

from seis_interp.models.nersi import FourierFeatureMapping
from seis_interp.models.trace_codec import TraceNodeDecoder, TraceNodeEncoder


class TraceContextField(nn.Module):
    """Decode one complete trace from a query coordinate and observed neighbours.

    The unconditional field of ``models/trace_coordinate_field.py`` holds the whole
    wavefield in its weights, which on a seventy percent missing survey memorizes
    the observed traces without interpolating, and its single linear output layer
    confines every prediction to a subspace of its own width. Here a query reads
    ``context`` observed traces instead: the shared trace codec turns each into a
    time-downsampled latent sequence, the Fourier features of the coordinate offset
    from the query enter every latent frame, the sequences are combined by weights
    the same offsets produce, and the query coordinate conditions the decode. The
    output stays a continuous function of the four coordinates, no output sample is
    discarded, and because the codec is convolutional the prediction is not confined
    to any fixed subspace and the parameter count does not depend on the time axis.

    ``input_features`` is four local coordinates, or five when the caller appends the
    log-offset coordinate; the module only requires that queries and context carry
    the same width.

    The caller supplies the neighbourhood, so which traces are visible is a
    property of the declared geometry and of the preparation's observed set, never
    of this module.
    """

    def __init__(
        self,
        *,
        time_sample_count,
        input_features=4,
        fourier_components=40,
        frequency_base=1.1,
        width=120,
        stem_kernel_size=9,
        time_downsample_factor=4,
        activation="gelu",
        output_activation="linear",
    ):
        super().__init__()
        if input_features not in (4, 5):
            raise ValueError("the trace context field requires four or five coordinates")
        if activation != "gelu" or output_activation != "linear":
            raise ValueError("the trace context field requires GELU and linear output")
        self.time_sample_count = _positive_integer(time_sample_count, "time_sample_count")
        self.time_downsample_factor = _positive_integer(
            time_downsample_factor, "time_downsample_factor"
        )
        if self.time_sample_count % self.time_downsample_factor:
            raise ValueError("the time axis must be divisible by time_downsample_factor")
        self.encoding = FourierFeatureMapping(
            input_features=input_features,
            fourier_components=fourier_components,
            frequency_base=frequency_base,
        )
        features = self.encoding.output_features
        self.encoder = TraceNodeEncoder(
            width,
            stem_kernel_size=stem_kernel_size,
            time_downsample_factor=self.time_downsample_factor,
        )
        self.width = self.encoder.width
        self.offset_mixer = nn.Sequential(
            nn.Conv1d(self.width + features, self.width, kernel_size=1), nn.GELU()
        )
        self.offset_weight = nn.Sequential(
            nn.Linear(features, self.width), nn.GELU(), nn.Linear(self.width, 1)
        )
        self.query_mixer = nn.Sequential(
            nn.Conv1d(self.width + features, self.width, kernel_size=1), nn.GELU()
        )
        self.decoder = TraceNodeDecoder(
            self.width, time_downsample_factor=self.time_downsample_factor
        )

    def forward(self, coordinates, context_values, context_coordinates):
        """``(batch, 4)`` and ``(batch, context, …)`` to ``(batch, time_sample_count)``."""
        if coordinates.ndim != 2 or context_values.ndim != 3 or context_coordinates.ndim != 3:
            raise ValueError(
                "expected (batch, features), (batch, context, time), (batch, context, features)"
            )
        if coordinates.shape[1] != self.encoding.input_features:
            raise ValueError("coordinates must carry the declared number of features")
        if context_coordinates.shape[2] != self.encoding.input_features:
            raise ValueError("context coordinates must carry the declared number of features")
        if context_values.shape[:2] != context_coordinates.shape[:2]:
            raise ValueError("context values and coordinates must describe the same neighbours")
        if context_values.shape[0] != coordinates.shape[0]:
            raise ValueError("every query needs its own context")
        if context_values.shape[2] != self.time_sample_count:
            raise ValueError("context traces must span the declared time axis")
        batch, context = context_values.shape[:2]
        offsets = self.encoding(
            (coordinates.unsqueeze(1) - context_coordinates).reshape(batch * context, -1)
        )
        latents = self.encoder(context_values.reshape(batch * context, 1, self.time_sample_count))
        frames = latents.shape[2]
        mixed = self.offset_mixer(
            torch.cat((latents, offsets[:, :, None].expand(-1, -1, frames)), dim=1)
        )
        weights = torch.softmax(self.offset_weight(offsets).reshape(batch, context, 1, 1), dim=1)
        pooled = (weights * mixed.reshape(batch, context, self.width, frames)).sum(dim=1)
        conditioned = self.query_mixer(
            torch.cat(
                (pooled, self.encoding(coordinates)[:, :, None].expand(-1, -1, frames)), dim=1
            )
        )
        return self.decoder(conditioned)


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)
