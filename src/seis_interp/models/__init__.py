"""Neural network models for coordinate-based interpolation."""

from seis_interp.models.neighbor_trace_inpainter import (
    TEMPORAL_DILATIONS,
    NeighborTraceInpainter,
    TemporalResidualBlock,
)
from seis_interp.models.shared_offset_attention_inpainter import (
    SharedOffsetAttentionInpainter,
)
from seis_interp.models.shot_gather_inpainter import (
    LEARNED_FILM_RECEIVER_POSITION_CONDITIONING,
    MOMENTS_SOURCE_FEATURE_MODE,
    NO_RECEIVER_POSITION_CONDITIONING,
    ORDERED_RAW_SOURCE_FEATURE_MODE,
    RECEIVER_POSITION_CONDITIONING_MODES,
    SHOT_GATHER_INPUT_FEATURE_NAMES,
    FactorizedGatherResidualBlock,
    ReceiverPositionFiLM,
    ShotGatherInpainter,
    inverse_distance_reference,
    ordered_raw_input_feature_names,
)
from seis_interp.models.siren import SineLayer, Siren

__all__ = [
    "TEMPORAL_DILATIONS",
    "FactorizedGatherResidualBlock",
    "LEARNED_FILM_RECEIVER_POSITION_CONDITIONING",
    "MOMENTS_SOURCE_FEATURE_MODE",
    "NeighborTraceInpainter",
    "NO_RECEIVER_POSITION_CONDITIONING",
    "ORDERED_RAW_SOURCE_FEATURE_MODE",
    "RECEIVER_POSITION_CONDITIONING_MODES",
    "ReceiverPositionFiLM",
    "SHOT_GATHER_INPUT_FEATURE_NAMES",
    "SharedOffsetAttentionInpainter",
    "ShotGatherInpainter",
    "SineLayer",
    "Siren",
    "TemporalResidualBlock",
    "inverse_distance_reference",
    "ordered_raw_input_feature_names",
]
