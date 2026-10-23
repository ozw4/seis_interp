"""Shared temporal messages on directed, typed trace dependency graphs."""

from __future__ import annotations

from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from seis_interp.data.masked_trace_inputs import (
    MaskedTraceGraphInputs,
    validate_masked_trace_graph_inputs,
)
from seis_interp.models.nersi import FourierFeatureMapping
from seis_interp.models.spectral_trace_graph import SpectralTraceGraphInputBlock
from seis_interp.models.trace_codec import TraceNodeDecoder, TraceNodeEncoder
from seis_interp.models.trace_graph_amplitude import (
    TRACE_GRAPH_AMPLITUDE_MODES,
    interpolate_query_trace_rms,
    normalize_observed_trace_rms,
)
from seis_interp.models.trace_graph_comparison import TraceGraphComparisonBlock
from seis_interp.models.trace_graph_edge_aggregation import aggregate_edge_messages
from seis_interp.models.trace_graph_time_shift import shift_trace_graph_values
from seis_interp.processing.trace_graph_geometry import (
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    RELATION_NAMES,
)

RELATION_FUSION_MODES = ("mean", "learned_gate")
TRACE_GRAPH_METHOD_VARIANTS = ("relational", "plain_gcn_row_normalized", "untyped_edge_conditioned")
# Exact memory/speed execution options; none adds parameters or changes the math.
TRACE_GRAPH_MEMORY_OPTIONS = (
    "receptive_field_pruning",
    "fused_edge_aggregation",
    "activation_checkpointing",
)
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
        attention_pooling: str = "mean",
        relation_gate_pooling: str = "mean",
        fused_edge_aggregation: bool = False,
        activation_checkpointing: bool = False,
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
        _validate_pooling(attention_pooling, relation_gate_pooling, relation_fusion)
        self.attention_pooling = attention_pooling
        self.relation_gate_pooling = relation_gate_pooling
        for name, value in (
            ("fused_edge_aggregation", fused_edge_aggregation),
            ("activation_checkpointing", activation_checkpointing),
        ):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be boolean")
        # Execution options only: they add no parameters and consume no RNG.
        self.fused_edge_aggregation = fused_edge_aggregation
        self.activation_checkpointing = activation_checkpointing

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
        output_nodes: torch.Tensor | None = None,
        host_checks: bool = True,
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

        Optional ``output_nodes[D]`` (receptive-field pruning) are unique rows of
        ``latents`` that contain every edge destination. Only their updates are
        computed and returned as ``[D,C,F]`` in that order; ``coverage[D,4,2]``
        and the diagnostic query indices then refer to those output rows.

        ``host_checks=False`` is for already validated graphs: it skips the value
        checks that wait for the device and replaces ``bincount``/``nonzero`` by
        bit-identical, synchronization-free group operations.
        """
        _validate_graph_tensors(
            latents, self.width, edge_index, edge_type, edge_features, values=host_checks
        )
        if diagnostics is not None and diagnostic_query_indices is None:
            raise ValueError("diagnostic_query_indices are required for diagnostics")
        if self.fused_edge_aggregation and edge_time_shifts is not None:
            raise ValueError("fused_edge_aggregation does not support edge time shifts")
        updated = (
            checkpoint(self._temporal_residual, latents, use_reentrant=False)
            if self.activation_checkpointing and torch.is_grad_enabled()
            else self._temporal_residual(latents)
        )
        messages, valid = self._relation_messages(
            updated,
            edge_index,
            edge_type,
            edge_features,
            edge_time_shifts=edge_time_shifts,
            output_nodes=output_nodes,
            host_checks=host_checks,
        )
        if output_nodes is not None:
            updated = updated[output_nodes]
        weights = self._relation_weights(updated, messages, valid, coverage, host_checks)
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

    def _temporal_residual(self, latents: torch.Tensor) -> torch.Tensor:
        return latents + self.temporal_update(self.temporal(F.silu(self.temporal_norm(latents))))

    def _relation_messages(
        self,
        latents: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_features: torch.Tensor,
        *,
        edge_time_shifts: torch.Tensor | None = None,
        output_nodes: torch.Tensor | None = None,
        host_checks: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_count, channels, frames = latents.shape
        sender, destination = edge_index
        receiver_count = node_count
        receiver = destination
        if output_nodes is not None:
            receiver_count = output_nodes.numel()
            receiver = _output_rows(output_nodes, destination, node_count, check=host_checks)
        group = receiver * _RELATION_COUNT + edge_type
        group_count = receiver_count * _RELATION_COUNT
        if host_checks:
            valid = torch.bincount(group, minlength=group_count) > 0
        else:
            # Same Boolean presence without the device-to-host size query of bincount.
            valid = torch.zeros(group_count, dtype=torch.bool, device=group.device)
            valid = valid.index_fill_(0, group, True)
        valid = valid.reshape(receiver_count, _RELATION_COUNT)
        pooled = _temporal_pool(latents, self.attention_pooling)
        edge_context = torch.cat((edge_features, self.relation_embedding(edge_type)), dim=1)
        logits = self.attention(
            torch.cat((pooled[destination], pooled[sender], edge_context), dim=1)
        )[:, 0]
        attention = _group_softmax(logits, group, group_count)
        gamma = 2 * torch.sigmoid(self.gamma(edge_context))
        if self.fused_edge_aggregation:
            coefficients = attention[:, None] * gamma
            messages = aggregate_edge_messages(
                self.value_projection(latents), coefficients, sender, group, group_count
            )
            return messages.reshape(receiver_count, _RELATION_COUNT, channels, frames), valid
        values = self.value_projection(latents)[sender]
        if edge_time_shifts is not None:
            values = shift_trace_graph_values(values, edge_time_shifts)
        weighted = attention[:, None, None] * gamma[:, :, None] * values
        messages = weighted.new_zeros(group_count, channels, frames)
        messages.index_add_(0, group, weighted)
        return messages.reshape(receiver_count, _RELATION_COUNT, channels, frames), valid

    def _relation_weights(
        self,
        latents: torch.Tensor,
        messages: torch.Tensor,
        valid: torch.Tensor,
        coverage: torch.Tensor | None,
        host_checks: bool = True,
    ) -> torch.Tensor:
        if self.relation_fusion == "mean":
            return valid.to(latents.dtype) / valid.sum(dim=1, keepdim=True).clamp_min(1)
        if coverage is None or coverage.shape != (*valid.shape, 2):
            raise ValueError("learned_gate requires coverage with shape (nodes, 4, 2)")
        pooled = _temporal_pool(latents, self.relation_gate_pooling)[:, None].expand(
            -1, _RELATION_COUNT, -1
        )
        relation = self.relation_embedding.weight[None].expand(latents.shape[0], -1, -1)
        logits = self.relation_gate(
            torch.cat(
                (pooled, _temporal_pool(messages, self.relation_gate_pooling), relation, coverage),
                dim=-1,
            )
        )[:, :, 0]
        if not host_checks:
            return _dense_relation_softmax(logits, valid)
        # Only valid entries enter softmax, including when an entire node is empty.
        node, relation_id = valid.nonzero(as_tuple=True)
        normalized = _group_softmax(logits[node, relation_id], node, latents.shape[0])
        weights = normalized.new_zeros(logits.shape)
        weights[node, relation_id] = normalized
        return weights


class RelationalTraceGraphInterpolator(nn.Module):
    """Direct normalized query-waveform prediction with shared temporal codecs.

    The relational-only execution options in ``TRACE_GRAPH_MEMORY_OPTIONS``
    (default off) add no parameters and are exact in real arithmetic:
    ``receptive_field_pruning`` updates only the states a later round or the
    decoder reads, ``fused_edge_aggregation`` recomputes edge messages in
    chunks instead of saving ``[E,C,F]`` tensors, and ``activation_checkpointing``
    recomputes the encoder and each round's temporal residual in backward.
    """

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
        attention_pooling: str = "mean",
        relation_gate_pooling: str = "mean",
        method_variant: str = "relational",
        explicit_azimuth_features: bool = True,
        amplitude_mode: str = "train_global_rms",
        max_edge_time_shift_samples: int = 0,
        node_fourier_components: int = 0,
        spectral_input_block: bool = False,
        receptive_field_pruning: bool = False,
        fused_edge_aggregation: bool = False,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if method_variant not in TRACE_GRAPH_METHOD_VARIANTS:
            raise ValueError(f"method_variant must be one of {TRACE_GRAPH_METHOD_VARIANTS}")
        _validate_pooling(attention_pooling, relation_gate_pooling, relation_fusion)
        if method_variant != "relational" and (
            attention_pooling != "mean" or relation_gate_pooling != "mean"
        ):
            raise ValueError("RMS pooling requires method_variant=relational")
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
        memory_options = dict(
            zip(
                TRACE_GRAPH_MEMORY_OPTIONS,
                (receptive_field_pruning, fused_edge_aggregation, activation_checkpointing),
                strict=True,
            )
        )
        _validate_memory_options(memory_options, method_variant, max_edge_time_shift_samples)
        self.receptive_field_pruning = receptive_field_pruning
        self.activation_checkpointing = activation_checkpointing
        self.method_variant = method_variant
        self.explicit_azimuth_features = explicit_azimuth_features
        self.amplitude_mode = amplitude_mode
        self.max_edge_time_shift_samples = int(max_edge_time_shift_samples)
        if not isinstance(spectral_input_block, bool):
            raise ValueError("spectral_input_block must be boolean")
        if spectral_input_block and (
            amplitude_mode != "train_global_rms"
            or max_edge_time_shift_samples
            or method_variant != "relational"
        ):
            raise ValueError(
                "spectral_input_block requires relational global RMS and no edge time shift"
            )
        if (
            isinstance(node_fourier_components, bool)
            or not isinstance(node_fourier_components, Integral)
            or node_fourier_components < 0
        ):
            raise ValueError("node_fourier_components must be a nonnegative integer")
        rounds = _positive_integer(message_passing_rounds, "message_passing_rounds")
        if len(temporal_dilations) != rounds:
            raise ValueError("temporal_dilations must match message_passing_rounds")
        self.encoder = TraceNodeEncoder(
            width,
            stem_kernel_size=stem_kernel_size,
            time_downsample_factor=time_downsample_factor,
        )
        self.node_fourier_mapping = (
            FourierFeatureMapping(input_features=4, fourier_components=node_fourier_components)
            if node_fourier_components
            else None
        )
        node_width = len(NODE_FEATURE_NAMES) + (
            self.node_fourier_mapping.output_features if self.node_fourier_mapping else 0
        )
        self.node_embedding = _mlp(node_width, width, width)
        self.rounds = nn.ModuleList(
            (
                RelationalTraceGraphMessageBlock(
                    width,
                    temporal_kernel_size=temporal_kernel_size,
                    temporal_dilation=dilation,
                    attention_width=attention_width,
                    relation_embedding_dim=relation_embedding_dim,
                    relation_fusion=relation_fusion,
                    attention_pooling=attention_pooling,
                    relation_gate_pooling=relation_gate_pooling,
                    fused_edge_aggregation=fused_edge_aggregation,
                    activation_checkpointing=activation_checkpointing,
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
        self.spectral_input = SpectralTraceGraphInputBlock() if spectral_input_block else None
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
        if attention_pooling != "mean":
            self._config["attention_pooling"] = attention_pooling
        if relation_gate_pooling != "mean":
            self._config["relation_gate_pooling"] = relation_gate_pooling
        if method_variant != "relational":
            self._config["method_variant"] = method_variant
        if not explicit_azimuth_features:
            self._config["explicit_azimuth_features"] = False
        if amplitude_mode != "train_global_rms":
            self._config["amplitude_mode"] = amplitude_mode
        if self.max_edge_time_shift_samples:
            self._config["max_edge_time_shift_samples"] = self.max_edge_time_shift_samples
        if node_fourier_components:
            self._config["node_fourier_components"] = int(node_fourier_components)
        if spectral_input_block:
            self._config["spectral_input_block"] = True
        self._config.update({name: True for name, value in memory_options.items() if value})

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
        host_checks: bool = True,
        receptive_field: tuple[list[torch.Tensor], list[torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``([Q,T], [Q])`` in query order; no-context predictions are zero.

        ``host_checks=False`` is for inputs already checked by
        ``validate_masked_trace_graph_inputs`` (for example on the CPU before the
        device transfer): it skips that re-validation and every relational block
        value check that waits for the device. Results are bit-identical.
        With ``receptive_field_pruning``, ``receptive_field`` may supply
        ``trace_graph_receptive_field`` of these inputs computed elsewhere (for
        example on the CPU), so the device never waits for its index sizes.
        """
        if host_checks:
            validate_masked_trace_graph_inputs(inputs)
        elif not isinstance(inputs, MaskedTraceGraphInputs):
            raise TypeError("inputs must be a MaskedTraceGraphInputs object")
        if receptive_field is not None and not self.receptive_field_pruning:
            raise ValueError("receptive_field requires receptive_field_pruning")
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
        node_features, edge_features = self.input_features(inputs)
        spectral_seed = None
        if self.spectral_input is not None:
            spectral_seed = self.spectral_input(
                visible, inputs.observed_mask, inputs.edge_index, inputs.edge_type, edge_features
            )
        padded = F.pad(
            (visible if spectral_seed is None else spectral_seed)[:, None],
            (0, padded_time - time_count),
        )
        if self.node_fourier_mapping is not None:
            # Only fixed midpoint/offset coordinates are periodic; visibility is not.
            node_features = torch.cat(
                (node_features, self.node_fourier_mapping(node_features[:, :4])), dim=1
            )
        pruned = self.receptive_field_pruning and inputs.query_indices.numel() > 0
        if pruned:
            nodes, edges = (
                _receptive_field(
                    inputs.edge_index, inputs.query_indices, len(self.rounds), waveforms.shape[0]
                )
                if receptive_field is None
                else _supplied_receptive_field(receptive_field, len(self.rounds), waveforms.device)
            )
            padded, node_features = padded[nodes[0]], node_features[nodes[0]]
        encoded = (
            checkpoint(self.encoder, padded, use_reentrant=False)
            if self.activation_checkpointing and torch.is_grad_enabled()
            else self.encoder(padded)
        )
        latents = encoded + self.node_embedding(node_features)[:, :, None]
        block_options = {}
        if self.max_edge_time_shift_samples:
            # Features 0:4 are sender-minus-destination source/receiver deltas.
            block_options["edge_time_shifts"] = (self.max_edge_time_shift_samples / factor) * (
                edge_features[:, :4] @ self.edge_time_shift_weights
            ).tanh()
        if self.method_variant == "relational":
            block_options["host_checks"] = host_checks
        round_summaries = []
        for index, block in enumerate(self.rounds):
            summary = {} if diagnostics is not None else None
            if pruned:
                latents = _pruned_update(
                    block,
                    latents,
                    inputs,
                    edge_features,
                    block_options,
                    nodes[index],
                    nodes[index + 1],
                    edges[index],
                    diagnostics=summary,
                )
            else:
                latents = block(
                    latents,
                    inputs.edge_index,
                    inputs.edge_type,
                    edge_features,
                    inputs.coverage,
                    diagnostics=summary,
                    diagnostic_query_indices=inputs.query_indices,
                    **block_options,
                )
            if summary is not None:
                round_summaries.append(summary)
        context = torch.zeros_like(inputs.observed_mask)
        if host_checks:
            context[inputs.edge_index[1]] = True
        else:
            # The same Boolean scatter without copying a host scalar to the device.
            context.index_fill_(0, inputs.edge_index[1], True)
        has_context = context[inputs.query_indices]
        # Pruned final states hold exactly the queries in ascending node order.
        query_rows = (
            torch.searchsorted(nodes[-1], inputs.query_indices) if pruned else inputs.query_indices
        )
        predictions = (
            self.decoder(latents[query_rows])[:, :time_count]
            if inputs.query_indices.numel()
            else waveforms.new_empty((0, time_count))
        )
        if self.amplitude_mode == "observed_trace_rms":
            predictions = predictions * query_rms[:, None]
        if spectral_seed is not None:
            predictions = predictions + spectral_seed[inputs.query_indices]
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


def trace_graph_receptive_field(
    edge_index: torch.Tensor, query_indices: torch.Tensor, rounds: int, node_count: int
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Public form of the pruning receptive field, on whatever device the indices are."""
    return _receptive_field(edge_index, query_indices, rounds, node_count)


def _supplied_receptive_field(
    receptive_field: tuple[list[torch.Tensor], list[torch.Tensor]],
    rounds: int,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    nodes, edges = receptive_field
    if (
        len(nodes) != rounds + 1
        or len(edges) != rounds
        or any(
            value.ndim != 1 or value.dtype != torch.int64 or value.device != device
            for value in (*nodes, *edges)
        )
    ):
        raise ValueError("receptive_field must hold rounds + 1 node and rounds edge index vectors")
    return list(nodes), list(edges)


def _receptive_field(
    edge_index: torch.Tensor, query_indices: torch.Tensor, rounds: int, node_count: int
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Return the states each synchronous round must provide for the queries.

    ``nodes[r]`` are the ascending nodes whose state after ``r`` rounds a later
    round or the decoder reads (``nodes[rounds]`` are the queries). ``edges[r]``
    are all edges into ``nodes[r + 1]`` in input order; their senders and
    destinations lie in ``nodes[r]``, so round ``r + 1`` restricted to them is exact.
    """
    sender, destination = edge_index
    needed = torch.zeros(node_count, dtype=torch.bool, device=edge_index.device)
    needed[query_indices] = True
    nodes = [needed.nonzero()[:, 0]]
    edges = []
    for _ in range(rounds):
        selected = needed[destination]
        edges.append(selected.nonzero()[:, 0])
        needed[sender[selected]] = True
        nodes.append(needed.nonzero()[:, 0])
    return nodes[::-1], edges[::-1]


def _pruned_update(
    block: RelationalTraceGraphMessageBlock,
    latents: torch.Tensor,
    inputs: MaskedTraceGraphInputs,
    edge_features: torch.Tensor,
    block_options: dict[str, object],
    input_nodes: torch.Tensor,
    output_nodes: torch.Tensor,
    edges: torch.Tensor,
    *,
    diagnostics: dict[str, torch.Tensor] | None,
) -> torch.Tensor:
    """Apply ``block`` to the states of ``input_nodes`` and return ``output_nodes`` only."""
    return block(
        latents,
        torch.searchsorted(input_nodes, inputs.edge_index[:, edges]),
        inputs.edge_type[edges],
        edge_features[edges],
        inputs.coverage[output_nodes],
        diagnostics=diagnostics,
        diagnostic_query_indices=torch.searchsorted(output_nodes, inputs.query_indices),
        output_nodes=torch.searchsorted(input_nodes, output_nodes),
        **{
            name: value[edges] if isinstance(value, torch.Tensor) else value
            for name, value in block_options.items()
        },
    )


def _output_rows(
    output_nodes: torch.Tensor, destination: torch.Tensor, node_count: int, *, check: bool = True
) -> torch.Tensor:
    """Map each edge destination to its row in ``output_nodes``.

    ``check=False`` skips the value checks that wait for the device.
    """
    if (
        output_nodes.ndim != 1
        or output_nodes.dtype != torch.int64
        or output_nodes.device != destination.device
    ):
        raise ValueError("output_nodes must be a 1-D int64 tensor on the latent device")
    if check and bool(((output_nodes < 0) | (output_nodes >= node_count)).any()):
        raise ValueError("output_nodes contains an out-of-range node")
    rows = torch.full((node_count,), -1, dtype=torch.int64, device=destination.device)
    rows[output_nodes] = torch.arange(output_nodes.numel(), device=destination.device)
    if check and bool((rows >= 0).sum() != output_nodes.numel()):
        raise ValueError("output_nodes must be unique")
    receiver = rows[destination]
    if check and bool((receiver < 0).any()):
        raise ValueError("every edge destination must be one of output_nodes")
    return receiver


def _validate_memory_options(
    options: dict[str, object], method_variant: str, max_edge_time_shift_samples: int
) -> None:
    for name, value in options.items():
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be boolean")
    if any(options.values()) and method_variant != "relational":
        raise ValueError("memory options require method_variant=relational")
    if options["fused_edge_aggregation"] and max_edge_time_shift_samples:
        raise ValueError("fused_edge_aggregation requires max_edge_time_shift_samples=0")


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


def _dense_relation_softmax(logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Softmax of ``logits[N,4]`` over each node's valid relations; empty entries are zero.

    Same values as the compact ``nonzero`` form without its host synchronization:
    an empty relation enters its node's group as ``-inf``, whose exponential adds
    exactly zero, in the same relation order.
    """
    node_count = logits.shape[0]
    group = torch.arange(node_count, device=logits.device).repeat_interleave(_RELATION_COUNT)
    masked = logits.masked_fill(~valid, -torch.inf).reshape(-1)
    normalized = _group_softmax(masked, group, node_count).reshape(node_count, _RELATION_COUNT)
    return torch.where(
        valid, normalized, torch.zeros((), dtype=normalized.dtype, device=logits.device)
    )


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
    *,
    values: bool = True,
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
    if not values:
        return
    if torch.any((edge_index < 0) | (edge_index >= latents.shape[0])):
        raise ValueError("edge_index contains an out-of-range node")
    if torch.any((edge_type < 0) | (edge_type >= _RELATION_COUNT)):
        raise ValueError("edge_type must contain relation IDs from 0 to 3")


def _validate_pooling(attention: str, gate: str, fusion: str) -> None:
    for name, value in (("attention_pooling", attention), ("relation_gate_pooling", gate)):
        if value not in ("mean", "rms"):
            raise ValueError(f"{name} must be mean or rms")
    if gate != "mean" and fusion != "learned_gate":
        raise ValueError("relation_gate_pooling=rms requires relation_fusion=learned_gate")


def _temporal_pool(values: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "mean":
        return values.mean(dim=-1)
    # vector_norm gives finite zero gradients for all-zero features.
    return torch.linalg.vector_norm(values, dim=-1) / values.shape[-1] ** 0.5
