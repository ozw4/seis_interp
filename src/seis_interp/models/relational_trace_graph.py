"""Shared temporal messages on directed, typed trace dependency graphs."""

from __future__ import annotations

from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F

from seis_interp.data.masked_trace_inputs import (
    MaskedTraceGraphInputs,
    validate_masked_trace_graph_inputs,
)
from seis_interp.models.trace_codec import TraceNodeDecoder, TraceNodeEncoder
from seis_interp.models.trace_graph_amplitude import (
    TRACE_GRAPH_AMPLITUDE_MODES,
    interpolate_query_trace_rms,
    normalize_observed_trace_rms,
)
from seis_interp.models.trace_graph_comparison import TraceGraphComparisonBlock
from seis_interp.models.trace_graph_time_shift import shift_trace_graph_values
from seis_interp.processing.trace_graph_geometry import (
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    RELATION_NAMES,
)

RELATION_FUSION_MODES = ("mean", "learned_gate")
TRACE_GRAPH_METHOD_VARIANTS = ("relational", "plain_gcn_row_normalized", "untyped_edge_conditioned")
_RELATION_COUNT = len(RELATION_NAMES)


class RelationalTraceGraphMessageBlock(nn.Module):
    """One synchronous temporal and incoming-relation update of ``[N,C,F]``."""

    def __init__(
        self,
        width: int,
        *,
        temporal_kernel_size: int = 5,
        temporal_dilation: int = 1,
        attention_width: int = 32,
        relation_embedding_dim: int = 8,
        relation_fusion: str = "mean",
    ) -> None:
        super().__init__()
        self.width = _positive_integer(width, "width")
        if self.width % 8:
            raise ValueError("width must be divisible by 8")
        kernel = _positive_integer(temporal_kernel_size, "temporal_kernel_size")
        if kernel % 2 == 0:
            raise ValueError("temporal_kernel_size must be odd")
        dilation = _positive_integer(temporal_dilation, "temporal_dilation")
        hidden = _positive_integer(attention_width, "attention_width")
        embedding_dim = _positive_integer(relation_embedding_dim, "relation_embedding_dim")
        if relation_fusion not in RELATION_FUSION_MODES:
            raise ValueError(f"relation_fusion must be one of {RELATION_FUSION_MODES}")
        self.relation_fusion = relation_fusion

        # At least two channels per group keep one-frame, one-node inputs valid.
        self.temporal_norm = nn.GroupNorm(min(8, self.width // 2), self.width)
        self.temporal = nn.Conv1d(
            self.width,
            self.width,
            kernel,
            padding=(kernel // 2) * dilation,
            dilation=dilation,
            groups=self.width,
        )
        self.temporal_update = nn.Sequential(
            nn.Conv1d(self.width, self.width, 1),
            nn.SiLU(),
            nn.Conv1d(self.width, self.width, 1),
        )
        self.relation_embedding = nn.Embedding(_RELATION_COUNT, embedding_dim)
        edge_width = len(EDGE_FEATURE_NAMES) + embedding_dim
        self.attention = _mlp(2 * self.width + edge_width, hidden, 1)
        self.gamma = _mlp(edge_width, hidden, self.width)
        self.value_projection = nn.Conv1d(self.width, self.width, 1)
        self.message_update = nn.Sequential(
            nn.Conv1d(self.width, self.width, 1),
            nn.SiLU(),
            nn.Conv1d(self.width, self.width, 1),
        )
        if relation_fusion == "learned_gate":
            self.relation_gate = _mlp(2 * self.width + embedding_dim + 2, hidden, 1)
            nn.init.zeros_(self.relation_gate[-1].weight)
            nn.init.zeros_(self.relation_gate[-1].bias)

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
        edge_time_shifts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Update from sender/destination edges without modifying input states.

        ``coverage[N,4,2]`` contains selected degree/k and minimum distance,
        with zero for empty relations. Diagnostics require explicit query indices
        and hold detached gate sums and query counts, excluding support nodes.
        Indices must be validated, unique int64 local query indices in a 1-D
        tensor on the latent device; the interpolator validates its inputs.
        Gate weights describe the model, not causal importance.
        Optional ``edge_time_shifts[E]`` advance sender values in latent frames;
        attention logits retain their existing time-pooled computation.
        """
        _validate_graph_tensors(latents, self.width, edge_index, edge_type, edge_features)
        if diagnostics is not None and diagnostic_query_indices is None:
            raise ValueError("diagnostic_query_indices are required for diagnostics")
        updated = latents + self.temporal_update(self.temporal(F.silu(self.temporal_norm(latents))))
        messages, valid = self._relation_messages(
            updated, edge_index, edge_type, edge_features, edge_time_shifts=edge_time_shifts
        )
        weights = self._relation_weights(updated, messages, valid, coverage)
        aggregate = (weights[:, :, None, None] * messages).sum(dim=1)
        if diagnostics is not None:
            query_valid = valid.detach()[diagnostic_query_indices]
            has_context = query_valid.any(dim=1)
            diagnostics.update(
                gate_sum=weights.detach()[diagnostic_query_indices][has_context].sum(dim=0),
                available_query_count=query_valid.sum(dim=0),
                context_query_count=has_context.sum(),
                no_context_query_count=(~has_context).sum(),
            )
        return updated + self.message_update(aggregate)

    def _relation_messages(
        self,
        latents: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_features: torch.Tensor,
        *,
        edge_time_shifts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_count, channels, frames = latents.shape
        sender, destination = edge_index
        group = destination * _RELATION_COUNT + edge_type
        group_count = node_count * _RELATION_COUNT
        counts = torch.bincount(group, minlength=group_count)
        valid = counts.reshape(node_count, _RELATION_COUNT) > 0
        pooled = latents.mean(dim=-1)
        edge_context = torch.cat((edge_features, self.relation_embedding(edge_type)), dim=1)
        logits = self.attention(
            torch.cat((pooled[destination], pooled[sender], edge_context), dim=1)
        )[:, 0]
        attention = _group_softmax(logits, group, group_count)
        gamma = 2 * torch.sigmoid(self.gamma(edge_context))
        values = self.value_projection(latents)[sender]
        if edge_time_shifts is not None:
            values = shift_trace_graph_values(values, edge_time_shifts)
        weighted = attention[:, None, None] * gamma[:, :, None] * values
        messages = weighted.new_zeros(group_count, channels, frames)
        messages.index_add_(0, group, weighted)
        return messages.reshape(node_count, _RELATION_COUNT, channels, frames), valid

    def _relation_weights(
        self,
        latents: torch.Tensor,
        messages: torch.Tensor,
        valid: torch.Tensor,
        coverage: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.relation_fusion == "mean":
            return valid.to(latents.dtype) / valid.sum(dim=1, keepdim=True).clamp_min(1)
        if coverage is None or coverage.shape != (*valid.shape, 2):
            raise ValueError("learned_gate requires coverage with shape (nodes, 4, 2)")
        pooled = latents.mean(dim=-1)[:, None].expand(-1, _RELATION_COUNT, -1)
        relation = self.relation_embedding.weight[None].expand(latents.shape[0], -1, -1)
        logits = self.relation_gate(
            torch.cat((pooled, messages.mean(dim=-1), relation, coverage), dim=-1)
        )[:, :, 0]
        # Only valid entries enter softmax, including when an entire node is empty.
        node, relation_id = valid.nonzero(as_tuple=True)
        normalized = _group_softmax(logits[node, relation_id], node, latents.shape[0])
        weights = normalized.new_zeros(logits.shape)
        weights[node, relation_id] = normalized
        return weights


class RelationalTraceGraphInterpolator(nn.Module):
    """Direct normalized query-waveform prediction with shared temporal codecs."""

    def __init__(
        self,
        *,
        width: int = 64,
        message_passing_rounds: int = 2,
        time_downsample_factor: int = 2,
        stem_kernel_size: int = 7,
        temporal_kernel_size: int = 5,
        temporal_dilations: tuple[int, ...] = (1, 2),
        attention_width: int = 32,
        relation_embedding_dim: int = 8,
        relation_fusion: str = "mean",
        method_variant: str = "relational",
        explicit_azimuth_features: bool = True,
        amplitude_mode: str = "train_global_rms",
        max_edge_time_shift_samples: int = 0,
    ) -> None:
        super().__init__()
        if method_variant not in TRACE_GRAPH_METHOD_VARIANTS:
            raise ValueError(f"method_variant must be one of {TRACE_GRAPH_METHOD_VARIANTS}")
        if not isinstance(explicit_azimuth_features, bool):
            raise ValueError("explicit_azimuth_features must be a boolean")
        if method_variant != "relational" and relation_fusion != "mean":
            raise ValueError("comparison models require relation_fusion=mean")
        if amplitude_mode not in TRACE_GRAPH_AMPLITUDE_MODES:
            raise ValueError(f"amplitude_mode must be one of {TRACE_GRAPH_AMPLITUDE_MODES}")
        if (
            isinstance(max_edge_time_shift_samples, bool)
            or not isinstance(max_edge_time_shift_samples, Integral)
            or max_edge_time_shift_samples < 0
        ):
            raise ValueError("max_edge_time_shift_samples must be a nonnegative integer")
        if max_edge_time_shift_samples and method_variant != "relational":
            raise ValueError("edge time shifts require method_variant=relational")
        self.method_variant = method_variant
        self.explicit_azimuth_features = explicit_azimuth_features
        self.amplitude_mode = amplitude_mode
        self.max_edge_time_shift_samples = int(max_edge_time_shift_samples)
        rounds = _positive_integer(message_passing_rounds, "message_passing_rounds")
        if len(temporal_dilations) != rounds:
            raise ValueError("temporal_dilations must match message_passing_rounds")
        self.encoder = TraceNodeEncoder(
            width,
            stem_kernel_size=stem_kernel_size,
            time_downsample_factor=time_downsample_factor,
        )
        self.node_embedding = _mlp(len(NODE_FEATURE_NAMES), width, width)
        self.rounds = nn.ModuleList(
            (
                RelationalTraceGraphMessageBlock(
                    width,
                    temporal_kernel_size=temporal_kernel_size,
                    temporal_dilation=dilation,
                    attention_width=attention_width,
                    relation_embedding_dim=relation_embedding_dim,
                    relation_fusion=relation_fusion,
                )
                if method_variant == "relational"
                else TraceGraphComparisonBlock(
                    width,
                    method_variant=method_variant,
                    temporal_kernel_size=temporal_kernel_size,
                    temporal_dilation=dilation,
                    attention_width=attention_width,
                )
            )
            for dilation in temporal_dilations
        )
        self.decoder = TraceNodeDecoder(width, time_downsample_factor=time_downsample_factor)
        if self.max_edge_time_shift_samples:
            # One bias-free geometry vector for all relations and all rounds.
            # zeros consumes no RNG and does not change existing initialization.
            self.edge_time_shift_weights = nn.Parameter(torch.zeros(4))
        self._config = {
            "width": int(width),
            "message_passing_rounds": rounds,
            "time_downsample_factor": int(time_downsample_factor),
            "stem_kernel_size": int(stem_kernel_size),
            "temporal_kernel_size": int(temporal_kernel_size),
            "temporal_dilations": [int(value) for value in temporal_dilations],
            "attention_width": int(attention_width),
            "relation_embedding_dim": int(relation_embedding_dim),
            "relation_fusion": relation_fusion,
        }
        if method_variant != "relational":
            self._config["method_variant"] = method_variant
        if not explicit_azimuth_features:
            self._config["explicit_azimuth_features"] = False
        if amplitude_mode != "train_global_rms":
            self._config["amplitude_mode"] = amplitude_mode
        if self.max_edge_time_shift_samples:
            self._config["max_edge_time_shift_samples"] = self.max_edge_time_shift_samples

    def constructor_config(self) -> dict[str, object]:
        """Return independent, JSON-compatible constructor values."""
        return {**self._config, "temporal_dilations": list(self._config["temporal_dilations"])}

    @property
    def message_passing_rounds(self) -> int:
        """Required dependency rounds for graph construction."""
        return len(self.rounds)

    def forward(
        self,
        inputs: MaskedTraceGraphInputs,
        *,
        diagnostics: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``([Q,T], [Q])`` in query order; no-context predictions are zero."""
        validate_masked_trace_graph_inputs(inputs)
        if (
            inputs.relation_names == ("untyped",)
            and self.method_variant != "untyped_edge_conditioned"
        ):
            raise ValueError("single_4d untyped input requires untyped_edge_conditioned model")
        if (
            self.method_variant == "untyped_edge_conditioned"
            and inputs.common_edge_distances is None
        ):
            raise ValueError("untyped_edge_conditioned requires common_edge_distances for D0")
        if self.amplitude_mode == "observed_trace_rms" and inputs.common_edge_distances is None:
            raise ValueError("observed_trace_rms requires common_edge_distances for D0")
        if inputs.dependency_rounds < self.message_passing_rounds:
            raise ValueError(
                f"dependency_rounds ({inputs.dependency_rounds}) must be at least "
                f"message_passing_rounds ({self.message_passing_rounds})"
            )
        waveforms = inputs.waveforms
        if waveforms.ndim != 2 or min(waveforms.shape) < 1:
            raise ValueError("waveforms must have shape (positive nodes, positive time)")
        time_count = waveforms.shape[1]
        factor = self.encoder.time_downsample_factor
        # The existing codec's eight-group norm also needs >=2 time samples
        # for width=8, factor=1, T=1. Its implementation remains unchanged.
        padded_time = ((max(time_count, 2) + factor - 1) // factor) * factor
        visible = waveforms.masked_fill(~inputs.observed_mask[:, None], 0)
        if self.amplitude_mode == "observed_trace_rms":
            visible, observed_rms = normalize_observed_trace_rms(visible, inputs.observed_mask)
            query_rms = interpolate_query_trace_rms(
                observed_rms,
                inputs.edge_index,
                inputs.query_indices,
                inputs.common_edge_distances,
            )
        padded = F.pad(visible[:, None], (0, padded_time - time_count))
        node_features, edge_features = self.input_features(inputs)
        latents = self.encoder(padded) + self.node_embedding(node_features)[:, :, None]
        shift_arguments = {}
        if self.max_edge_time_shift_samples:
            # Features 0:4 are sender-minus-destination source/receiver deltas.
            shift_arguments["edge_time_shifts"] = (self.max_edge_time_shift_samples / factor) * (
                edge_features[:, :4] @ self.edge_time_shift_weights
            ).tanh()
        round_summaries = []
        for block in self.rounds:
            summary = {} if diagnostics is not None else None
            latents = block(
                latents,
                inputs.edge_index,
                inputs.edge_type,
                edge_features,
                inputs.coverage,
                diagnostics=summary,
                diagnostic_query_indices=inputs.query_indices,
                **shift_arguments,
            )
            if summary is not None:
                round_summaries.append(summary)
        context = torch.zeros_like(inputs.observed_mask)
        context[inputs.edge_index[1]] = True
        has_context = context[inputs.query_indices]
        predictions = (
            self.decoder(latents[inputs.query_indices])[:, :time_count]
            if inputs.query_indices.numel()
            else waveforms.new_empty((0, time_count))
        )
        if self.amplitude_mode == "observed_trace_rms":
            predictions = predictions * query_rms[:, None]
        predictions = predictions.masked_fill(~has_context[:, None], 0)
        if diagnostics is not None:
            diagnostics.update(
                {
                    name: torch.stack([summary[name] for summary in round_summaries])
                    for name in round_summaries[0]
                }
            )
        return predictions, has_context

    def input_features(self, inputs: MaskedTraceGraphInputs) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the actual node/edge features for this fixed model ablation.

        Untyped messages replace relation-specific distance with the common
        midpoint/offset D0. Explicit azimuth removal retains all vector and
        length columns and never changes input tensors or graph connectivity.
        """
        nodes, edges = inputs.node_features, inputs.edge_features
        if self.method_variant == "untyped_edge_conditioned":
            if inputs.common_edge_distances is None:
                raise ValueError("untyped_edge_conditioned requires common_edge_distances for D0")
            edges = edges.clone()
            edges[:, -1] = inputs.common_edge_distances
        if not self.explicit_azimuth_features:
            nodes = nodes.clone()
            edges = edges.clone()
            nodes[:, 5:8] = 0
            edges[:, 11:14] = 0
        return nodes, edges


def _mlp(input_width: int, hidden_width: int, output_width: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_width, hidden_width),
        nn.SiLU(),
        nn.Linear(hidden_width, output_width),
    )


def _group_softmax(logits: torch.Tensor, group: torch.Tensor, group_count: int) -> torch.Tensor:
    # AMP linear logits may be low precision; softmax reductions stay in FP32.
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()
    maxima = logits.new_full((group_count,), -torch.inf)
    maxima.scatter_reduce_(0, group, logits, reduce="amax", include_self=True)
    exponentials = torch.exp(logits - maxima[group])
    totals = logits.new_zeros(group_count).index_add_(0, group, exponentials)
    return exponentials / totals[group]


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _validate_graph_tensors(
    latents: torch.Tensor,
    width: int,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    edge_features: torch.Tensor,
) -> None:
    if latents.ndim != 3 or latents.shape[1] != width or min(latents.shape) < 1:
        raise ValueError(f"latents must have shape (positive nodes, {width}, positive frames)")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, edges)")
    edge_count = edge_index.shape[1]
    if edge_type.shape != (edge_count,):
        raise ValueError("edge_type must have shape (edges,)")
    if edge_features.shape != (edge_count, len(EDGE_FEATURE_NAMES)):
        raise ValueError("edge_features must have shape (edges, 15)")
    if edge_index.dtype != torch.int64 or edge_type.dtype != torch.int64:
        raise TypeError("edge_index and edge_type must have dtype torch.int64")
    if torch.any((edge_index < 0) | (edge_index >= latents.shape[0])):
        raise ValueError("edge_index contains an out-of-range node")
    if torch.any((edge_type < 0) | (edge_type >= _RELATION_COUNT)):
        raise ValueError("edge_type must contain relation IDs from 0 to 3")
