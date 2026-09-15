"""Observed-spectrum graph aggregation followed by real-waveform synthesis."""

import math

import torch
from torch import nn

from seis_interp.processing.trace_graph_geometry import EDGE_FEATURE_NAMES, RELATION_NAMES


class SpectralTraceGraphInputBlock(nn.Module):
    """Learn frequency-dependent complex edge responses without time resampling.

    Inputs have already passed the masked graph contract. Observed rows remain
    unchanged; only missing rows receive synthesized waveforms. Consequently this
    block needs direct observed neighbors, not an extra dependency-graph hop.
    """

    def __init__(self) -> None:
        super().__init__()
        self.rank = 8
        self.relation_embedding = nn.Embedding(len(RELATION_NAMES), 8)
        self.edge_response = nn.Sequential(
            nn.Linear(len(EDGE_FEATURE_NAMES) + 8, 32),
            nn.SiLU(),
            nn.Linear(32, 2 * self.rank + 1),
        )
        self.frequency_basis = nn.Sequential(
            nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, 2 * self.rank)
        )
        # Start at a distance-weighted observed-spectrum average.
        nn.init.zeros_(self.edge_response[-1].weight)
        nn.init.zeros_(self.edge_response[-1].bias)

    def forward(self, waveforms, observed_mask, edge_index, edge_type, edge_features):
        visible = waveforms.masked_fill(~observed_mask[:, None], 0)
        if edge_type.numel() == 0:
            return visible
        sender, destination = edge_index
        edge = self.edge_response(
            torch.cat((edge_features, self.relation_embedding(edge_type)), dim=1)
        ).float()
        with torch.autocast(device_type=waveforms.device.type, enabled=False):
            spectrum = torch.fft.rfft(visible.float(), dim=-1)
            frequency = torch.linspace(
                0, 1, spectrum.shape[-1], device=waveforms.device, dtype=torch.float32
            )[:, None]
            basis = self.frequency_basis(frequency).reshape(-1, self.rank, 2)
            coefficients = edge[:, 1:].reshape(-1, self.rank, 2)
            real = 1 + (
                coefficients[:, :, 0] @ basis[:, :, 0].T - coefficients[:, :, 1] @ basis[:, :, 1].T
            ) / math.sqrt(self.rank)
            imaginary = (
                coefficients[:, :, 0] @ basis[:, :, 1].T + coefficients[:, :, 1] @ basis[:, :, 0].T
            ) / math.sqrt(self.rank)
            logits = edge[:, 0] - edge_features[:, -1].float()
            maxima = logits.new_full((len(waveforms),), -torch.inf)
            maxima.scatter_reduce_(0, destination, logits, reduce="amax", include_self=True)
            weights = (logits - maxima[destination]).exp()
            totals = weights.new_zeros(len(waveforms)).index_add_(0, destination, weights)
            weights = weights / totals[destination]
            messages = spectrum[sender] * torch.complex(real, imaginary) * weights[:, None]
            aggregated = spectrum.new_zeros(spectrum.shape).index_add_(0, destination, messages)
            # Real time signals require real DC and, for even lengths, Nyquist bins.
            imaginary_mask = torch.ones_like(frequency[:, 0])
            imaginary_mask[0] = 0
            if waveforms.shape[-1] % 2 == 0:
                imaginary_mask[-1] = 0
            aggregated = torch.complex(aggregated.real, aggregated.imag * imaginary_mask)
            reconstructed = torch.fft.irfft(aggregated, n=waveforms.shape[-1], dim=-1)
        return torch.where(observed_mask[:, None], visible, reconstructed.to(waveforms.dtype))
