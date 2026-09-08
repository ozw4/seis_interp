"""Unique-pair graph controls sharing the trace model's temporal update."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def unique_trace_edge_indices(edge_index: torch.Tensor, node_count: int) -> torch.Tensor:
    """Return one edge per directed pair, in destination/sender order."""
    keys = edge_index[1] * node_count + edge_index[0]
    order = torch.argsort(keys, stable=True)
    keys = keys[order]
    first = torch.ones_like(keys, dtype=torch.bool)
    first[1:] = keys[1:] != keys[:-1]
    return order[first]


class TraceGraphComparisonBlock(nn.Module):
    """Row-normalized GCN or one untyped edge-conditioned incoming message."""

    def __init__(
        self,
        width: int,
        *,
        method_variant: str,
        temporal_kernel_size: int,
        temporal_dilation: int,
        attention_width: int,
    ) -> None:
        super().__init__()
        if method_variant not in ("plain_gcn_row_normalized", "untyped_edge_conditioned"):
            raise ValueError("unknown trace graph comparison method_variant")
        if isinstance(width, bool) or not isinstance(width, int) or width < 8 or width % 8:
            raise ValueError("width must be positive and divisible by 8")
        for value, name in (
            (temporal_kernel_size, "temporal_kernel_size"),
            (temporal_dilation, "temporal_dilation"),
            (attention_width, "attention_width"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if temporal_kernel_size % 2 == 0:
            raise ValueError("temporal_kernel_size must be odd")
        self.method_variant = method_variant
        self.width = width
        self.temporal_norm = nn.GroupNorm(min(8, width // 2), width)
        self.temporal = nn.Conv1d(
            width,
            width,
            temporal_kernel_size,
            padding=(temporal_kernel_size // 2) * temporal_dilation,
            dilation=temporal_dilation,
            groups=width,
        )
        self.temporal_update = nn.Sequential(
            nn.Conv1d(width, width, 1), nn.SiLU(), nn.Conv1d(width, width, 1)
        )
        self.value_projection = nn.Conv1d(width, width, 1)
        self.message_update = nn.Sequential(
            nn.Conv1d(width, width, 1), nn.SiLU(), nn.Conv1d(width, width, 1)
        )
        if method_variant == "untyped_edge_conditioned":
            self.attention = nn.Sequential(
                nn.Linear(2 * width + 15, attention_width), nn.SiLU(), nn.Linear(attention_width, 1)
            )
            self.gamma = nn.Sequential(
                nn.Linear(15, attention_width), nn.SiLU(), nn.Linear(attention_width, width)
            )

    def forward(
        self,
        latents: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_features: torch.Tensor,
        coverage: torch.Tensor | None = None,
        *,
        diagnostics: dict[str, torch.Tensor] | None = None,
        diagnostic_query_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Aggregate unique observed pairs; untyped features must already use D0.

        Relation IDs, relation multiplicities and coverage do not enter either
        control. The caller validates observed-only senders and geometry.
        """
        updated = latents + self.temporal_update(self.temporal(F.silu(self.temporal_norm(latents))))
        selected = unique_trace_edge_indices(edge_index, len(latents))
        sender, destination = edge_index[:, selected]
        count = torch.bincount(destination, minlength=len(latents))
        if self.method_variant == "plain_gcn_row_normalized":
            aggregate = updated.clone().index_add(0, destination, updated[sender])
            aggregate = aggregate / (count + 1)[:, None, None]
            aggregate = self.value_projection(aggregate)
        else:
            features = edge_features[selected]
            pooled = updated.mean(dim=-1)
            logits = self.attention(
                torch.cat((pooled[destination], pooled[sender], features), dim=1)
            )[:, 0]
            maxima = logits.new_full((len(latents),), -torch.inf)
            maxima.scatter_reduce_(0, destination, logits, reduce="amax", include_self=True)
            weights = torch.exp(logits - maxima[destination])
            totals = weights.new_zeros(len(latents)).index_add(0, destination, weights)
            weights = weights / totals[destination]
            gamma = 2 * torch.sigmoid(self.gamma(features))
            values = self.value_projection(updated)[sender]
            aggregate = torch.zeros_like(updated).index_add(
                0, destination, weights[:, None, None] * gamma[:, :, None] * values
            )
        if diagnostics is not None:
            if diagnostic_query_indices is None:
                raise ValueError("diagnostic_query_indices are required for diagnostics")
            valid = (count[diagnostic_query_indices] > 0).detach()
            context_count = valid.sum()
            diagnostics.update(
                gate_sum=context_count.to(latents.dtype)[None],
                available_query_count=context_count[None],
                context_query_count=context_count,
                no_context_query_count=(~valid).sum(),
            )
        return updated + self.message_update(aggregate)
