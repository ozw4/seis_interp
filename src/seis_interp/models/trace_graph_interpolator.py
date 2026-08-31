"""Trace-node graph interpolation of whole shot gathers on the SEG C3 grid."""

from __future__ import annotations

import math
from collections.abc import Iterable
from numbers import Integral, Real

import torch
from torch import nn
from torch.nn import functional as F

from seis_interp.models.shot_gather_inpainter import (
    RECEIVER_X_COUNT,
    RECEIVER_Y_COUNT,
    inverse_distance_reference,
)

TRACE_LATTICE_GRAPH_MODE = "trace_lattice"
SOURCE_RECEIVER_BIPARTITE_GRAPH_MODE = "source_receiver_bipartite"
GRAPH_MODES = (
    TRACE_LATTICE_GRAPH_MODE,
    SOURCE_RECEIVER_BIPARTITE_GRAPH_MODE,
)
POOLED_ATTENTION_TIME_RESOLUTION = "pooled"
PER_FRAME_ATTENTION_TIME_RESOLUTION = "per_frame"
PER_FRAME_SHIFTED_ATTENTION_TIME_RESOLUTION = "per_frame_shifted"
ATTENTION_TIME_RESOLUTIONS = (
    POOLED_ATTENTION_TIME_RESOLUTION,
    PER_FRAME_ATTENTION_TIME_RESOLUTION,
    PER_FRAME_SHIFTED_ATTENTION_TIME_RESOLUTION,
)
_ATTENTION_FRAME_SHIFTS = (-1, 0, 1)
DEFAULT_WIDTH = 64
DEFAULT_MESSAGE_PASSING_ROUNDS = 4
DEFAULT_TIME_DOWNSAMPLE_FACTOR = 5
DEFAULT_STEM_KERNEL_SIZE = 7
DEFAULT_TEMPORAL_KERNEL_SIZE = 5
DEFAULT_TEMPORAL_DILATIONS = (1, 2, 4, 8)
DEFAULT_SPATIAL_KERNEL_SIZE = 3
DEFAULT_ATTENTION_WIDTH = 32
DEFAULT_DISTANCE_EPSILON = 1.0e-6
NODE_STATIC_FEATURE_NAMES: tuple[str, ...] = (
    "is_target_shot",
    "cell_availability",
    "source_direction_x",
    "source_direction_y",
    "normalized_source_distance",
    "receiver_coordinate_x",
    "receiver_coordinate_y",
    "target_coordinate_x",
    "target_coordinate_y",
)
"""Ordered static descriptors embedded into every trace node."""

SHOT_DESCRIPTOR_NAMES: tuple[str, ...] = (
    "is_target_shot",
    "source_direction_x",
    "source_direction_y",
    "normalized_source_distance",
)
"""Ordered per-shot descriptors conditioning source-axis message passing."""

_GROUP_COUNT = 8
_SOURCE_COORDINATE_COUNT = 2
_TARGET_COORDINATE_COUNT = 2


class TraceNodeEncoder(nn.Module):
    """Encode one trace waveform into a time-downsampled latent sequence."""

    def __init__(
        self,
        width: int,
        *,
        stem_kernel_size: int,
        time_downsample_factor: int,
    ) -> None:
        super().__init__()
        self.width = _validated_width(width)
        self.stem_kernel_size = _odd_positive_integer(stem_kernel_size, "stem_kernel_size")
        self.time_downsample_factor = _positive_integer(
            time_downsample_factor,
            "time_downsample_factor",
        )
        self.stem = nn.Conv1d(
            1,
            self.width,
            kernel_size=self.stem_kernel_size,
            padding=self.stem_kernel_size // 2,
        )
        self.norm = nn.GroupNorm(_GROUP_COUNT, self.width)
        factor = self.time_downsample_factor
        self.downsample = nn.Conv1d(
            self.width,
            self.width,
            kernel_size=2 * factor - 1,
            stride=factor,
            padding=factor - 1,
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Return latent frames with shape ``[nodes, width, time / factor]``."""
        waveforms = _require_floating_tensor(waveforms, "waveforms")
        if waveforms.ndim != 3 or waveforms.shape[1] != 1:
            raise ValueError(
                f"waveforms must have shape (nodes, 1, time), got {tuple(waveforms.shape)}"
            )
        if waveforms.shape[2] % self.time_downsample_factor != 0:
            raise ValueError(
                "waveforms time dimension must be divisible by "
                f"time_downsample_factor {self.time_downsample_factor}, "
                f"got {waveforms.shape[2]}"
            )
        return self.downsample(F.silu(self.norm(self.stem(waveforms))))


class TraceNodeDecoder(nn.Module):
    """Decode target-node latent sequences back to full-rate residual traces."""

    def __init__(self, width: int, *, time_downsample_factor: int) -> None:
        super().__init__()
        self.width = _validated_width(width)
        self.time_downsample_factor = _positive_integer(
            time_downsample_factor,
            "time_downsample_factor",
        )
        factor = self.time_downsample_factor
        self.upsample = nn.ConvTranspose1d(
            self.width,
            self.width,
            kernel_size=2 * factor - 1,
            stride=factor,
            padding=factor - 1,
            output_padding=factor - 1,
        )
        self.head = nn.Sequential(
            nn.GroupNorm(_GROUP_COUNT, self.width),
            nn.SiLU(),
            nn.Conv1d(self.width, self.width, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(self.width, 1, kernel_size=1),
        )
        final_projection = self.head[-1]
        if not isinstance(final_projection, nn.Conv1d):
            raise AssertionError("trace node decoder head must end with Conv1d")
        nn.init.zeros_(final_projection.weight)
        nn.init.zeros_(final_projection.bias)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """Return residual traces with shape ``[nodes, time]``."""
        latents = _require_floating_tensor(latents, "latents")
        if latents.ndim != 3 or latents.shape[1] != self.width:
            raise ValueError(
                f"latents must have shape (nodes, {self.width}, frames), got {tuple(latents.shape)}"
            )
        return self.head(self.upsample(latents))[:, 0]


class TraceGraphMessagePassingRound(nn.Module):
    """One round of factorized message passing over trace nodes.

    ``trace_lattice`` mode applies, in order, a per-node temporal update, a
    receiver-lattice spatial update inside each shot, and masked attention
    between shots at the same relative-receiver cell. The bipartite mode
    replaces the source-axis attention with explicit receiver-node and
    source-node aggregation whose messages are broadcast back to every
    incident observed-trace edge.
    """

    def __init__(
        self,
        width: int,
        *,
        graph_mode: str,
        temporal_kernel_size: int,
        temporal_dilation: int,
        spatial_kernel_size: int,
        attention_width: int,
        attention_time_resolution: str = POOLED_ATTENTION_TIME_RESOLUTION,
    ) -> None:
        super().__init__()
        self.width = _validated_width(width)
        self.graph_mode = _validated_graph_mode(graph_mode)
        self.attention_time_resolution = _validated_attention_time_resolution(
            attention_time_resolution,
            graph_mode=self.graph_mode,
        )
        self.temporal_kernel_size = _odd_positive_integer(
            temporal_kernel_size,
            "temporal_kernel_size",
        )
        self.temporal_dilation = _positive_integer(temporal_dilation, "temporal_dilation")
        self.spatial_kernel_size = _odd_positive_integer(
            spatial_kernel_size,
            "spatial_kernel_size",
        )
        self.attention_width = _positive_integer(attention_width, "attention_width")

        self.temporal_norm = nn.GroupNorm(_GROUP_COUNT, self.width)
        self.temporal = nn.Conv1d(
            self.width,
            self.width,
            kernel_size=self.temporal_kernel_size,
            padding=(self.temporal_kernel_size // 2) * self.temporal_dilation,
            dilation=self.temporal_dilation,
            groups=self.width,
        )
        self.temporal_expand = nn.Conv1d(self.width, self.width * 2, kernel_size=1)
        self.temporal_contract = nn.Conv1d(self.width, self.width, kernel_size=1)

        self.spatial_norm = nn.GroupNorm(_GROUP_COUNT, self.width)
        spatial_padding = self.spatial_kernel_size // 2
        self.spatial = nn.Conv3d(
            self.width,
            self.width,
            kernel_size=(self.spatial_kernel_size, self.spatial_kernel_size, 1),
            padding=(spatial_padding, spatial_padding, 0),
            groups=self.width,
        )
        self.spatial_expand = nn.Conv3d(self.width, self.width * 2, kernel_size=1)
        self.spatial_contract = nn.Conv3d(self.width, self.width, kernel_size=1)

        self.source_norm = nn.GroupNorm(_GROUP_COUNT, self.width)
        descriptor_count = len(SHOT_DESCRIPTOR_NAMES)
        if self.graph_mode == TRACE_LATTICE_GRAPH_MODE:
            self.query_projection = nn.Linear(self.width, self.attention_width)
            self.key_projection = nn.Linear(self.width, self.attention_width)
            self.query_descriptor = nn.Linear(descriptor_count, self.attention_width)
            self.key_descriptor = nn.Linear(descriptor_count, self.attention_width)
            self.value_projection = nn.Linear(self.width, self.width)
            if self.attention_time_resolution == PER_FRAME_SHIFTED_ATTENTION_TIME_RESOLUTION:
                self.frame_shift_bias = nn.Parameter(torch.zeros(len(_ATTENTION_FRAME_SHIFTS)))
            self.message_expand = nn.Conv1d(self.width, self.width * 2, kernel_size=1)
            self.message_contract = nn.Conv1d(self.width, self.width, kernel_size=1)
        else:
            self.receiver_score = nn.Linear(self.width, 1)
            self.receiver_score_descriptor = nn.Linear(descriptor_count, 1)
            self.receiver_value = nn.Linear(self.width, self.width)
            self.receiver_spatial = nn.Conv3d(
                self.width,
                self.width,
                kernel_size=(self.spatial_kernel_size, self.spatial_kernel_size, 1),
                padding=(spatial_padding, spatial_padding, 0),
                groups=self.width,
            )
            self.source_value = nn.Linear(self.width, self.width)
            self.source_query_projection = nn.Linear(self.width, self.attention_width)
            self.source_key_projection = nn.Linear(self.width, self.attention_width)
            self.source_query_descriptor = nn.Linear(descriptor_count, self.attention_width)
            self.source_key_descriptor = nn.Linear(descriptor_count, self.attention_width)
            self.edge_from_receiver = nn.Linear(self.width, self.width)
            self.edge_from_source = nn.Linear(self.width, self.width)
            self.message_expand = nn.Conv1d(self.width, self.width * 2, kernel_size=1)
            self.message_contract = nn.Conv1d(self.width, self.width, kernel_size=1)

    def forward(
        self,
        latents: torch.Tensor,
        *,
        presence: torch.Tensor,
        shot_descriptors: torch.Tensor,
    ) -> torch.Tensor:
        """Transform ``[B, shots, width, 8, 68, frames]`` preserving its shape."""
        batch_size, shot_count, width, receiver_x, receiver_y, frame_count = latents.shape
        if width != self.width:
            raise ValueError(f"latents channel dimension must be {self.width}, got {width}")
        latents = self._temporal_update(latents)
        latents = self._spatial_update(latents)
        if self.graph_mode == TRACE_LATTICE_GRAPH_MODE:
            latents = self._source_axis_attention(latents, presence, shot_descriptors)
        else:
            latents = self._bipartite_update(latents, presence, shot_descriptors)
        if latents.shape != (
            batch_size,
            shot_count,
            width,
            receiver_x,
            receiver_y,
            frame_count,
        ):
            raise AssertionError("message passing changed the latent shape")
        return latents

    def _temporal_update(self, latents: torch.Tensor) -> torch.Tensor:
        batch_size, shot_count, width, receiver_x, receiver_y, frame_count = latents.shape
        flat = latents.permute(0, 1, 3, 4, 2, 5).reshape(-1, width, frame_count)
        transformed = self.temporal(F.silu(self.temporal_norm(flat)))
        value, gate = self.temporal_expand(transformed).chunk(2, dim=1)
        flat = flat + self.temporal_contract(F.silu(value) * torch.sigmoid(gate))
        return flat.reshape(
            batch_size,
            shot_count,
            receiver_x,
            receiver_y,
            width,
            frame_count,
        ).permute(0, 1, 4, 2, 3, 5)

    def _spatial_update(self, latents: torch.Tensor) -> torch.Tensor:
        batch_size, shot_count, width, receiver_x, receiver_y, frame_count = latents.shape
        flat = latents.reshape(-1, width, receiver_x, receiver_y, frame_count)
        transformed = self.spatial(F.silu(self.spatial_norm(flat)))
        value, gate = self.spatial_expand(transformed).chunk(2, dim=1)
        flat = flat + self.spatial_contract(F.silu(value) * torch.sigmoid(gate))
        return flat.reshape(latents.shape)

    def _normalized_pooled(
        self,
        latents: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, shot_count, width, receiver_x, receiver_y, frame_count = latents.shape
        flat = latents.permute(0, 1, 3, 4, 2, 5).reshape(-1, width, frame_count)
        normalized = self.source_norm(flat).reshape(
            batch_size,
            shot_count,
            receiver_x,
            receiver_y,
            width,
            frame_count,
        )
        pooled = normalized.mean(dim=5)
        return normalized, pooled

    def _source_axis_attention(
        self,
        latents: torch.Tensor,
        presence: torch.Tensor,
        shot_descriptors: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, shot_count, width, receiver_x, receiver_y, frame_count = latents.shape
        normalized, pooled = self._normalized_pooled(latents)
        values = self.value_projection(normalized.permute(0, 1, 2, 3, 5, 4))
        if self.attention_time_resolution == POOLED_ATTENTION_TIME_RESOLUTION:
            queries = (
                self.query_projection(pooled)
                + self.query_descriptor(shot_descriptors)[:, :, None, None]
            )
            keys = (
                self.key_projection(pooled)
                + self.key_descriptor(shot_descriptors)[:, :, None, None]
            )
            logits = torch.einsum("bixya,bjxya->bxyij", queries.float(), keys.float())
            logits = logits / math.sqrt(self.attention_width)
            sender_mask = presence.permute(0, 2, 3, 1)[:, :, :, None, :]
            logits = logits.masked_fill(~sender_mask, torch.finfo(logits.dtype).min)
            weights = torch.softmax(logits, dim=4).to(dtype=latents.dtype)
            aggregated = torch.einsum("bxyij,bjxytc->bixytc", weights, values)
        else:
            frames = normalized.permute(0, 1, 2, 3, 5, 4)
            queries = (
                self.query_projection(frames)
                + self.query_descriptor(shot_descriptors)[:, :, None, None, None]
            )
            keys = (
                self.key_projection(frames)
                + self.key_descriptor(shot_descriptors)[:, :, None, None, None]
            )
            aggregated = self._per_frame_aggregate(queries, keys, values, presence)
        message = aggregated.permute(0, 1, 2, 3, 5, 4).reshape(-1, width, frame_count)
        value, gate = self.message_expand(message).chunk(2, dim=1)
        update = self.message_contract(F.silu(value) * torch.sigmoid(gate))
        update = update.reshape(
            batch_size,
            shot_count,
            receiver_x,
            receiver_y,
            width,
            frame_count,
        ).permute(0, 1, 4, 2, 3, 5)
        return latents + update

    def _per_frame_aggregate(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        presence: torch.Tensor,
    ) -> torch.Tensor:
        """Attend per latent frame, optionally over shifted sender frames."""
        frame_count = queries.shape[4]
        scale = math.sqrt(self.attention_width)
        sender_mask = presence.permute(0, 2, 3, 1)[:, :, :, None, None, :]
        if self.attention_time_resolution == PER_FRAME_ATTENTION_TIME_RESOLUTION:
            logits = torch.einsum("bixyta,bjxyta->bxytij", queries.float(), keys.float()) / scale
            logits = logits.masked_fill(~sender_mask, torch.finfo(logits.dtype).min)
            weights = torch.softmax(logits, dim=5).to(dtype=values.dtype)
            return torch.einsum("bxytij,bjxytc->bixytc", weights, values)

        shift_logits = []
        shift_values = []
        for shift_index, shift in enumerate(_ATTENTION_FRAME_SHIFTS):
            shifted_keys = torch.roll(keys, shifts=-shift, dims=4)
            shifted_values = torch.roll(values, shifts=-shift, dims=4)
            logits = (
                torch.einsum("bixyta,bjxyta->bxytij", queries.float(), shifted_keys.float()) / scale
                + self.frame_shift_bias[shift_index].float()
            )
            frame_index = torch.arange(frame_count, device=queries.device)
            frame_valid = (frame_index + shift >= 0) & (frame_index + shift < frame_count)
            valid = sender_mask & frame_valid[None, None, None, :, None, None]
            logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
            shift_logits.append(logits)
            shift_values.append(shifted_values)
        stacked_logits = torch.stack(shift_logits, dim=6)
        flat_logits = stacked_logits.flatten(start_dim=5)
        weights = torch.softmax(flat_logits, dim=5).to(dtype=values.dtype)
        weights = weights.reshape(stacked_logits.shape)
        aggregated = None
        for shift_index in range(len(_ATTENTION_FRAME_SHIFTS)):
            contribution = torch.einsum(
                "bxytij,bjxytc->bixytc",
                weights[..., shift_index],
                shift_values[shift_index],
            )
            aggregated = contribution if aggregated is None else aggregated + contribution
        return aggregated

    def _bipartite_update(
        self,
        latents: torch.Tensor,
        presence: torch.Tensor,
        shot_descriptors: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, shot_count, width, receiver_x, receiver_y, frame_count = latents.shape
        normalized, pooled = self._normalized_pooled(latents)

        receiver_logits = (
            self.receiver_score(pooled)[..., 0]
            + self.receiver_score_descriptor(shot_descriptors)[:, :, None, None, 0]
        ).float()
        receiver_logits = receiver_logits.masked_fill(
            ~presence,
            torch.finfo(receiver_logits.dtype).min,
        )
        receiver_weights = torch.softmax(receiver_logits, dim=1).to(dtype=latents.dtype)
        edge_values = self.receiver_value(normalized.permute(0, 1, 2, 3, 5, 4))
        receiver_nodes = torch.einsum("bjxy,bjxytc->bxytc", receiver_weights, edge_values)
        receiver_nodes = receiver_nodes.permute(0, 4, 1, 2, 3)
        receiver_nodes = receiver_nodes + self.receiver_spatial(receiver_nodes)

        presence_fraction = presence.to(dtype=latents.dtype).flatten(start_dim=2).mean(dim=2)
        cell_weights = presence.to(dtype=latents.dtype)
        cell_totals = cell_weights.flatten(start_dim=2).sum(dim=2).clamp_min(1.0)
        source_values = self.source_value(normalized.permute(0, 1, 2, 3, 5, 4))
        source_nodes = (
            torch.einsum("bjxy,bjxytc->bjtc", cell_weights, source_values)
            / cell_totals[:, :, None, None]
        )
        source_pooled = source_nodes.mean(dim=2)
        source_queries = self.source_query_projection(source_pooled) + self.source_query_descriptor(
            shot_descriptors
        )
        source_keys = self.source_key_projection(source_pooled) + self.source_key_descriptor(
            shot_descriptors
        )
        source_logits = source_queries.float() @ source_keys.float().transpose(1, 2)
        source_logits = source_logits / math.sqrt(self.attention_width)
        shot_present = presence_fraction > 0.0
        source_logits = source_logits.masked_fill(
            ~shot_present[:, None, :],
            torch.finfo(source_logits.dtype).min,
        )
        source_weights = torch.softmax(source_logits, dim=2).to(dtype=latents.dtype)
        source_nodes = source_nodes + torch.einsum("bij,bjtc->bitc", source_weights, source_nodes)

        receiver_message = self.edge_from_receiver(receiver_nodes.permute(0, 2, 3, 4, 1)).permute(
            0, 4, 1, 2, 3
        )
        source_message = self.edge_from_source(source_nodes).permute(0, 1, 3, 2)
        message = (
            normalized.permute(0, 1, 4, 2, 3, 5)
            + receiver_message[:, None]
            + source_message[:, :, :, None, None, :]
        )
        message = message.permute(0, 1, 3, 4, 2, 5).reshape(-1, width, frame_count)
        value, gate = self.message_expand(message).chunk(2, dim=1)
        update = self.message_contract(F.silu(value) * torch.sigmoid(gate))
        update = update.reshape(
            batch_size,
            shot_count,
            receiver_x,
            receiver_y,
            width,
            frame_count,
        ).permute(0, 1, 4, 2, 3, 5)
        return latents + update


class TraceGraphInterpolator(nn.Module):
    """Reconstruct a hidden ``8 x 68`` shot gather with trace-node message passing.

    Every trace is one graph node; time is encoded as an in-node latent
    sequence and never appears as a graph coordinate. The node set is the
    target gather plus the ``K`` nearest train source gathers. Prediction is
    the deterministic inverse-source-distance reference plus a
    zero-initialized decoded residual, so the model starts exactly at the
    reference.
    """

    def __init__(
        self,
        *,
        width: int = DEFAULT_WIDTH,
        graph_mode: str = TRACE_LATTICE_GRAPH_MODE,
        message_passing_rounds: int = DEFAULT_MESSAGE_PASSING_ROUNDS,
        time_downsample_factor: int = DEFAULT_TIME_DOWNSAMPLE_FACTOR,
        stem_kernel_size: int = DEFAULT_STEM_KERNEL_SIZE,
        temporal_kernel_size: int = DEFAULT_TEMPORAL_KERNEL_SIZE,
        temporal_dilations: Iterable[int] = DEFAULT_TEMPORAL_DILATIONS,
        spatial_kernel_size: int = DEFAULT_SPATIAL_KERNEL_SIZE,
        attention_width: int = DEFAULT_ATTENTION_WIDTH,
        attention_time_resolution: str = POOLED_ATTENTION_TIME_RESOLUTION,
        distance_epsilon: float = DEFAULT_DISTANCE_EPSILON,
        use_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.width = _validated_width(width)
        self.graph_mode = _validated_graph_mode(graph_mode)
        if not isinstance(use_gradient_checkpointing, bool):
            raise ValueError("use_gradient_checkpointing must be a boolean")
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.attention_time_resolution = _validated_attention_time_resolution(
            attention_time_resolution,
            graph_mode=self.graph_mode,
        )
        self.message_passing_rounds = _positive_integer(
            message_passing_rounds,
            "message_passing_rounds",
        )
        self.time_downsample_factor = _positive_integer(
            time_downsample_factor,
            "time_downsample_factor",
        )
        self.stem_kernel_size = _odd_positive_integer(stem_kernel_size, "stem_kernel_size")
        self.temporal_kernel_size = _odd_positive_integer(
            temporal_kernel_size,
            "temporal_kernel_size",
        )
        self.temporal_dilations = _validated_temporal_dilations(temporal_dilations)
        if len(self.temporal_dilations) != self.message_passing_rounds:
            raise ValueError(
                "temporal_dilations length must equal message_passing_rounds "
                f"{self.message_passing_rounds}, got {len(self.temporal_dilations)}"
            )
        self.spatial_kernel_size = _odd_positive_integer(
            spatial_kernel_size,
            "spatial_kernel_size",
        )
        self.attention_width = _positive_integer(attention_width, "attention_width")
        self.distance_epsilon = _positive_finite_float(distance_epsilon, "distance_epsilon")
        self.node_static_feature_names = NODE_STATIC_FEATURE_NAMES

        self.encoder = TraceNodeEncoder(
            self.width,
            stem_kernel_size=self.stem_kernel_size,
            time_downsample_factor=self.time_downsample_factor,
        )
        self.static_embedding = nn.Linear(len(NODE_STATIC_FEATURE_NAMES), self.width)
        self.rounds = nn.ModuleList(
            TraceGraphMessagePassingRound(
                self.width,
                graph_mode=self.graph_mode,
                temporal_kernel_size=self.temporal_kernel_size,
                temporal_dilation=temporal_dilation,
                spatial_kernel_size=self.spatial_kernel_size,
                attention_width=self.attention_width,
                attention_time_resolution=self.attention_time_resolution,
            )
            for temporal_dilation in self.temporal_dilations
        )
        self.decoder = TraceNodeDecoder(
            self.width,
            time_downsample_factor=self.time_downsample_factor,
        )

    def forward(
        self,
        neighbors: torch.Tensor,
        availability: torch.Tensor,
        source_deltas: torch.Tensor,
        target_coordinates: torch.Tensor,
    ) -> torch.Tensor:
        """Return target amplitudes with shape ``[B, 8, 68, T]``."""
        batch_size, source_count, time_count = _validated_inputs(
            neighbors,
            availability,
            source_deltas,
            target_coordinates,
        )
        _validate_model_compatibility(self.static_embedding, neighbors)
        if time_count % self.time_downsample_factor != 0:
            raise ValueError(
                "neighbors time dimension must be divisible by time_downsample_factor "
                f"{self.time_downsample_factor}, got {time_count}"
            )
        reference = inverse_distance_reference(
            neighbors,
            availability,
            source_deltas,
            distance_epsilon=self.distance_epsilon,
        )
        availability_feature = availability.to(dtype=neighbors.dtype)
        waveforms = torch.cat(
            (reference[:, None], neighbors * availability_feature[..., None]),
            dim=1,
        )
        shot_count = source_count + 1
        presence = torch.cat(
            (
                torch.ones(
                    (batch_size, 1, RECEIVER_X_COUNT, RECEIVER_Y_COUNT),
                    dtype=torch.bool,
                    device=availability.device,
                ),
                availability,
            ),
            dim=1,
        )
        shot_descriptors = _shot_descriptors(
            source_deltas,
            distance_epsilon=self.distance_epsilon,
        )
        static_features = _node_static_features(
            shot_descriptors,
            availability_feature,
            target_coordinates,
            reference_coverage=availability.any(dim=1),
        )

        latents = self.encoder(waveforms.reshape(-1, 1, time_count))
        frame_count = latents.shape[2]
        latents = (
            latents
            + self.static_embedding(static_features.reshape(-1, len(NODE_STATIC_FEATURE_NAMES)))[
                :, :, None
            ]
        )
        latents = latents.reshape(
            batch_size,
            shot_count,
            RECEIVER_X_COUNT,
            RECEIVER_Y_COUNT,
            self.width,
            frame_count,
        ).permute(0, 1, 4, 2, 3, 5)
        run_checkpointed = (
            self.use_gradient_checkpointing and self.training and torch.is_grad_enabled()
        )
        for message_round in self.rounds:
            if run_checkpointed:
                latents = torch.utils.checkpoint.checkpoint(
                    _run_message_round,
                    message_round,
                    latents,
                    presence,
                    shot_descriptors,
                    use_reentrant=False,
                )
            else:
                latents = message_round(
                    latents,
                    presence=presence,
                    shot_descriptors=shot_descriptors,
                )
        target_latents = (
            latents[:, 0]
            .permute(0, 2, 3, 1, 4)
            .reshape(
                -1,
                self.width,
                frame_count,
            )
        )
        residual = self.decoder(target_latents).reshape(
            batch_size,
            RECEIVER_X_COUNT,
            RECEIVER_Y_COUNT,
            time_count,
        )
        return reference + residual


def _run_message_round(
    message_round: TraceGraphMessagePassingRound,
    latents: torch.Tensor,
    presence: torch.Tensor,
    shot_descriptors: torch.Tensor,
) -> torch.Tensor:
    """Invoke one round positionally so activation checkpointing can rerun it."""
    return message_round(latents, presence=presence, shot_descriptors=shot_descriptors)


def _shot_descriptors(
    source_deltas: torch.Tensor,
    *,
    distance_epsilon: float,
) -> torch.Tensor:
    """Return ``[B, sources + 1, 4]`` descriptors with the target shot first."""
    calculation_dtype = (
        torch.float32
        if source_deltas.dtype in (torch.float16, torch.bfloat16)
        else source_deltas.dtype
    )
    deltas = source_deltas.to(dtype=calculation_dtype)
    distances = torch.linalg.vector_norm(deltas, dim=2)
    direction_denominator = torch.where(
        distances > 0.0,
        distances,
        torch.ones_like(distances),
    )
    directions = deltas / direction_denominator[:, :, None]
    scale = distances.amax(dim=1, keepdim=True).clamp_min(distance_epsilon)
    normalized_distances = distances / scale
    neighbor_descriptors = torch.cat(
        (
            torch.zeros_like(distances)[:, :, None],
            directions,
            normalized_distances[:, :, None],
        ),
        dim=2,
    )
    target_descriptor = torch.zeros_like(neighbor_descriptors[:, :1])
    target_descriptor[:, :, 0] = 1.0
    return torch.cat((target_descriptor, neighbor_descriptors), dim=1).to(dtype=source_deltas.dtype)


def _node_static_features(
    shot_descriptors: torch.Tensor,
    availability_feature: torch.Tensor,
    target_coordinates: torch.Tensor,
    *,
    reference_coverage: torch.Tensor,
) -> torch.Tensor:
    """Return ``[B, shots, 8, 68, 9]`` static node descriptors."""
    batch_size, shot_count, _descriptor_count = shot_descriptors.shape
    cell_availability = torch.cat(
        (
            reference_coverage.to(dtype=availability_feature.dtype)[:, None],
            availability_feature,
        ),
        dim=1,
    )
    descriptor_grid = shot_descriptors[:, :, None, None, :].expand(
        -1,
        -1,
        RECEIVER_X_COUNT,
        RECEIVER_Y_COUNT,
        -1,
    )
    receiver_coordinates = _normalized_receiver_coordinates(
        dtype=availability_feature.dtype,
        device=availability_feature.device,
    )
    receiver_grid = receiver_coordinates.permute(1, 2, 0)[None, None].expand(
        batch_size,
        shot_count,
        -1,
        -1,
        -1,
    )
    target_grid = target_coordinates[:, None, None, None, :].expand(
        -1,
        shot_count,
        RECEIVER_X_COUNT,
        RECEIVER_Y_COUNT,
        -1,
    )
    features = torch.cat(
        (
            descriptor_grid[..., :1],
            cell_availability[..., None],
            descriptor_grid[..., 1:],
            receiver_grid,
            target_grid,
        ),
        dim=4,
    )
    if features.shape[-1] != len(NODE_STATIC_FEATURE_NAMES):
        raise AssertionError("node static feature construction changed its declared schema")
    return features


def _normalized_receiver_coordinates(
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return deterministic receiver-axis coordinates with shape ``[2, 8, 68]``."""
    receiver_x = torch.linspace(-1.0, 1.0, RECEIVER_X_COUNT, dtype=dtype, device=device)
    receiver_y = torch.linspace(-1.0, 1.0, RECEIVER_Y_COUNT, dtype=dtype, device=device)
    return torch.stack(
        (
            receiver_x[:, None].expand(-1, RECEIVER_Y_COUNT),
            receiver_y[None, :].expand(RECEIVER_X_COUNT, -1),
        )
    )


def _validated_inputs(
    neighbors: torch.Tensor,
    availability: torch.Tensor,
    source_deltas: torch.Tensor,
    target_coordinates: torch.Tensor,
) -> tuple[int, int, int]:
    neighbors = _require_floating_tensor(neighbors, "neighbors")
    availability = _require_tensor(availability, "availability")
    source_deltas = _require_floating_tensor(source_deltas, "source_deltas")
    target_coordinates = _require_floating_tensor(target_coordinates, "target_coordinates")
    if neighbors.ndim != 5:
        raise ValueError(
            f"neighbors must have shape (batch, sources, 8, 68, time), got {tuple(neighbors.shape)}"
        )
    batch_size, source_count, receiver_x, receiver_y, time_count = neighbors.shape
    if (receiver_x, receiver_y) != (RECEIVER_X_COUNT, RECEIVER_Y_COUNT):
        raise ValueError(
            "neighbors receiver dimensions must be "
            f"({RECEIVER_X_COUNT}, {RECEIVER_Y_COUNT}), got ({receiver_x}, {receiver_y})"
        )
    if batch_size == 0 or source_count == 0 or time_count == 0:
        raise ValueError("neighbors batch, source, and time dimensions must be non-empty")
    if availability.shape != (batch_size, source_count, RECEIVER_X_COUNT, RECEIVER_Y_COUNT):
        raise ValueError(
            "availability must have shape "
            f"({batch_size}, {source_count}, {RECEIVER_X_COUNT}, {RECEIVER_Y_COUNT}), "
            f"got {tuple(availability.shape)}"
        )
    if availability.dtype != torch.bool:
        raise TypeError(f"availability must have dtype torch.bool, got {availability.dtype}")
    if source_deltas.shape != (batch_size, source_count, _SOURCE_COORDINATE_COUNT):
        raise ValueError(
            "source_deltas must have shape "
            f"({batch_size}, {source_count}, {_SOURCE_COORDINATE_COUNT}), "
            f"got {tuple(source_deltas.shape)}"
        )
    if target_coordinates.shape != (batch_size, _TARGET_COORDINATE_COUNT):
        raise ValueError(
            "target_coordinates must have shape "
            f"({batch_size}, {_TARGET_COORDINATE_COUNT}), got {tuple(target_coordinates.shape)}"
        )
    for name, tensor in (
        ("source_deltas", source_deltas),
        ("target_coordinates", target_coordinates),
    ):
        if tensor.dtype != neighbors.dtype:
            raise TypeError(
                f"{name} must share the neighbors dtype, got {tensor.dtype} and {neighbors.dtype}"
            )
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"{name} must contain only finite values")
    if not (
        neighbors.device == availability.device == source_deltas.device == target_coordinates.device
    ):
        raise ValueError("all trace graph inputs must share a device")
    return batch_size, source_count, time_count


def _validate_model_compatibility(
    static_embedding: nn.Linear,
    neighbors: torch.Tensor,
) -> None:
    if neighbors.dtype != static_embedding.weight.dtype:
        raise TypeError(
            "neighbors dtype must match the model dtype, "
            f"got {neighbors.dtype} and {static_embedding.weight.dtype}"
        )
    if neighbors.device != static_embedding.weight.device:
        raise ValueError("neighbors and model parameters must share a device")


def _require_tensor(value: object, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    return value


def _require_floating_tensor(value: object, name: str) -> torch.Tensor:
    tensor = _require_tensor(value, name)
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype, got {tensor.dtype}")
    return tensor


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _validated_width(value: object) -> int:
    width = _positive_integer(value, "width")
    if width % _GROUP_COUNT != 0:
        raise ValueError(f"width must be divisible by {_GROUP_COUNT}, got {width}")
    return width


def _validated_graph_mode(value: object) -> str:
    if not isinstance(value, str) or value not in GRAPH_MODES:
        raise ValueError(f"graph_mode must be one of {GRAPH_MODES}, got {value!r}")
    return value


def _validated_attention_time_resolution(value: object, *, graph_mode: str) -> str:
    if not isinstance(value, str) or value not in ATTENTION_TIME_RESOLUTIONS:
        raise ValueError(
            f"attention_time_resolution must be one of {ATTENTION_TIME_RESOLUTIONS}, got {value!r}"
        )
    if (
        graph_mode == SOURCE_RECEIVER_BIPARTITE_GRAPH_MODE
        and value != POOLED_ATTENTION_TIME_RESOLUTION
    ):
        raise ValueError("attention_time_resolution must be 'pooled' for the bipartite graph mode")
    return value


def _odd_positive_integer(value: object, name: str) -> int:
    converted = _positive_integer(value, name)
    if converted % 2 == 0:
        raise ValueError(f"{name} must be odd, got {converted}")
    return converted


def _validated_temporal_dilations(values: Iterable[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("temporal_dilations must be a non-empty iterable")
    try:
        raw_values = tuple(values)
    except TypeError as error:
        raise ValueError("temporal_dilations must be a non-empty iterable") from error
    if not raw_values:
        raise ValueError("temporal_dilations must not be empty")
    return tuple(
        _positive_integer(value, f"temporal_dilations[{index}]")
        for index, value in enumerate(raw_values)
    )


def _positive_finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    return converted
