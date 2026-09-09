"""Observed-only waveform factors for masked trace graph prediction."""

from __future__ import annotations

import torch

TRACE_GRAPH_AMPLITUDE_MODES = ("train_global_rms", "observed_trace_rms")


def normalize_observed_trace_rms(
    waveforms: torch.Tensor, observed_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unit-RMS visible waveforms and their input-unit RMS factors.

    Inputs follow the validated graph's ``[N,T]`` waveform and ``[N]`` mask
    contracts. Hidden samples never enter either output. Only an exactly zero
    RMS uses divisor one; zero waveforms and their factors remain zero.
    Rescaling before squaring avoids overflow/underflow for finite waveforms,
    and the zero branch has finite derivatives.
    """
    visible = waveforms.masked_fill(~observed_mask[:, None], 0)
    peak = visible.abs().amax(dim=1)
    nonzero = peak > 0
    peak_divisor = torch.where(nonzero, peak, torch.ones_like(peak))
    mean_square = (visible / peak_divisor[:, None]).square().mean(dim=1)
    rms = peak * torch.where(nonzero, mean_square, torch.ones_like(mean_square)).sqrt()
    divisor = torch.where(rms > 0, rms, torch.ones_like(rms))
    return visible / divisor[:, None], rms


def interpolate_query_trace_rms(
    observed_rms: torch.Tensor,
    edge_index: torch.Tensor,
    query_indices: torch.Tensor,
    common_edge_distances: torch.Tensor,
) -> torch.Tensor:
    """Interpolate RMS over each query's unique direct observed senders.

    Tensors follow validated graph contracts, including observed-only senders
    and a common D0 for each physical sender/destination pair across relations.
    Weights are ``1 / max(D0, 1e-6)**2``. Support-node edges and other queries'
    senders do not contribute. Results follow query order; no context gives zero.
    """
    query_count = query_indices.numel()
    if not query_count or not edge_index.shape[1]:
        return observed_rms.new_zeros(query_count)
    node_count = observed_rms.numel()
    query_position = query_indices.new_full((node_count,), -1)
    query_position[query_indices] = torch.arange(query_count, device=query_indices.device)
    sender, destination = edge_index
    position = query_position[destination]
    direct = position >= 0
    pair, inverse = torch.unique(
        position[direct] * node_count + sender[direct], return_inverse=True
    )
    # A pair can occur in several relation lists; its common D0 counts once.
    distances = common_edge_distances.new_full((pair.numel(),), torch.inf)
    distances.scatter_reduce_(
        0, inverse, common_edge_distances[direct], reduce="amin", include_self=True
    )
    group = torch.div(pair, node_count, rounding_mode="floor")
    sender = pair.remainder(node_count)
    clamped = distances.clamp_min(1e-6)
    nearest = observed_rms.new_full((query_count,), torch.inf)
    nearest.scatter_reduce_(0, group, clamped, reduce="amin", include_self=True)
    # Multiplying each query's IDW weights by its minimum distance squared
    # cancels on normalization and keeps at least one weight exactly one.
    weights = (nearest[group] / clamped).square()
    totals = observed_rms.new_zeros(query_count).index_add_(0, group, weights)
    normalized_weights = weights / totals[group]
    return observed_rms.new_zeros(query_count).index_add_(
        0, group, normalized_weights * observed_rms[sender]
    )
