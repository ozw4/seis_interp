"""Physical geometry and ordered features for grid-free trace graphs.

Geometry remains float64; only the final normalized features become float32.
Origins and scales are supplied by the caller and are never fitted here.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np

from seis_interp.processing.geometry import compute_trace_geometry

RELATION_NAMES = ("source", "receiver", "cmp", "offset_azimuth")
RELATION_IDS = {name: index for index, name in enumerate(RELATION_NAMES)}

NODE_FEATURE_NAMES = (
    "midpoint_x_scaled",
    "midpoint_y_scaled",
    "offset_x_scaled",
    "offset_y_scaled",
    "offset_length_scaled",
    "azimuth_sin",
    "azimuth_cos",
    "azimuth_valid",
    "observed",
)

EDGE_FEATURE_NAMES = (
    "delta_source_x_scaled",
    "delta_source_y_scaled",
    "delta_receiver_x_scaled",
    "delta_receiver_y_scaled",
    "delta_midpoint_x_scaled",
    "delta_midpoint_y_scaled",
    "delta_offset_x_scaled",
    "delta_offset_y_scaled",
    "destination_offset_length_scaled",
    "sender_offset_length_scaled",
    "delta_offset_length_scaled",
    "azimuth_delta_cos",
    "azimuth_delta_sin",
    "azimuth_pair_valid",
    "relation_distance",
)


@dataclass(frozen=True)
class TraceGraphGeometry:
    """Per-trace physical arrays in input order, without IDs or amplitudes.

    Coordinate vectors have shape ``[N, 2]`` and scalars have shape ``[N]``.
    Coordinates and azimuth components are float64; validity is boolean.
    Build this object with :func:`compute_trace_graph_geometry`.
    """

    source_xy_m: np.ndarray
    receiver_xy_m: np.ndarray
    midpoint_xy_m: np.ndarray
    offset_xy_m: np.ndarray
    offset_m: np.ndarray
    azimuth_sin: np.ndarray
    azimuth_cos: np.ndarray
    azimuth_valid: np.ndarray


def compute_trace_graph_geometry(
    source_xy_m: np.ndarray,
    receiver_xy_m: np.ndarray,
    *,
    azimuth_min_offset_m: float,
) -> TraceGraphGeometry:
    """Derive CMP, full offset and the ``atan2(ox, oy)`` unit direction.

    Source and receiver are absolute horizontal positions in metres, each
    ``[N, 2]``. At or below the nonnegative offset threshold, both azimuth
    components are zero and validity is false; the offset vector is retained.
    Inputs are copied so subsequent caller changes do not alter the geometry.
    """
    threshold = _finite_length(azimuth_min_offset_m, "azimuth_min_offset_m", allow_zero=True)
    source = _xy_coordinates(source_xy_m, "source_xy_m")
    receiver = _xy_coordinates(receiver_xy_m, "receiver_xy_m")
    midpoint_x, midpoint_y, offset_m, _ = compute_trace_geometry(
        source[:, 0], source[:, 1], receiver[:, 0], receiver[:, 1]
    )
    offset_xy = source - receiver
    valid = offset_m > threshold
    direction = np.zeros_like(offset_xy)
    np.divide(offset_xy, offset_m[:, None], out=direction, where=valid[:, None])
    return TraceGraphGeometry(
        source_xy_m=source,
        receiver_xy_m=receiver,
        midpoint_xy_m=np.column_stack((midpoint_x, midpoint_y)),
        offset_xy_m=offset_xy,
        offset_m=offset_m,
        azimuth_sin=direction[:, 0],
        azimuth_cos=direction[:, 1],
        azimuth_valid=valid,
    )


def build_trace_graph_node_features(
    geometry: TraceGraphGeometry,
    *,
    midpoint_origin_m: np.ndarray,
    position_scale_m: float,
    offset_scale_m: float,
    observed_mask: np.ndarray,
) -> np.ndarray:
    """Return float32 ``[N, 9]`` in :data:`NODE_FEATURE_NAMES` order.

    ``midpoint_origin_m`` is a fixed two-component physical origin. The two
    positive scales are feature scales, independent of relation search scales.
    ``observed_mask`` must be a boolean vector in the same trace order.
    """
    position_scale = _finite_length(position_scale_m, "position_scale_m")
    offset_scale = _finite_length(offset_scale_m, "offset_scale_m")
    origin = np.asarray(midpoint_origin_m, dtype=np.float64)
    if origin.shape != (2,) or not np.all(np.isfinite(origin)):
        raise ValueError("midpoint_origin_m must be a finite vector with shape [2]")
    observed = np.asarray(observed_mask)
    if observed.shape != geometry.offset_m.shape or observed.dtype != np.bool_:
        raise ValueError("observed_mask must be a boolean vector with shape [N]")
    return np.column_stack(
        (
            (geometry.midpoint_xy_m - origin) / position_scale,
            geometry.offset_xy_m / offset_scale,
            geometry.offset_m / offset_scale,
            geometry.azimuth_sin,
            geometry.azimuth_cos,
            geometry.azimuth_valid,
            observed,
        )
    ).astype(np.float32)


def build_trace_graph_edge_features(
    geometry: TraceGraphGeometry,
    *,
    sender_indices: np.ndarray,
    destination_indices: np.ndarray,
    relation_distances: np.ndarray,
    position_scale_m: float,
    offset_scale_m: float,
) -> np.ndarray:
    """Return float32 ``[E, 15]`` in :data:`EDGE_FEATURE_NAMES` order.

    All differences are sender minus destination, including the azimuth angle.
    Offset lengths are ordered destination, sender, then their difference.
    ``relation_distances`` contains the already computed, dimensionless D (not
    D squared). Relation IDs are separate from these continuous features.
    Empty indices and distances produce ``[0, 15]``.
    """
    position_scale = _finite_length(position_scale_m, "position_scale_m")
    offset_scale = _finite_length(offset_scale_m, "offset_scale_m")
    node_count = len(geometry.offset_m)
    sender = _node_indices(sender_indices, "sender_indices", node_count)
    destination = _node_indices(destination_indices, "destination_indices", node_count)
    distance = np.asarray(relation_distances, dtype=np.float64)
    if sender.shape != destination.shape or distance.shape != sender.shape:
        raise ValueError(
            "sender_indices, destination_indices and relation_distances must share [E]"
        )
    if not np.all(np.isfinite(distance)) or np.any(distance < 0):
        raise ValueError("relation_distances must be finite and nonnegative")

    valid = geometry.azimuth_valid[sender] & geometry.azimuth_valid[destination]
    sin_sender, cos_sender = geometry.azimuth_sin[sender], geometry.azimuth_cos[sender]
    sin_destination = geometry.azimuth_sin[destination]
    cos_destination = geometry.azimuth_cos[destination]
    delta_cos = np.where(valid, sin_sender * sin_destination + cos_sender * cos_destination, 0.0)
    delta_sin = np.where(valid, sin_sender * cos_destination - cos_sender * sin_destination, 0.0)
    return np.column_stack(
        (
            (geometry.source_xy_m[sender] - geometry.source_xy_m[destination]) / position_scale,
            (geometry.receiver_xy_m[sender] - geometry.receiver_xy_m[destination]) / position_scale,
            (geometry.midpoint_xy_m[sender] - geometry.midpoint_xy_m[destination]) / position_scale,
            (geometry.offset_xy_m[sender] - geometry.offset_xy_m[destination]) / offset_scale,
            geometry.offset_m[destination] / offset_scale,
            geometry.offset_m[sender] / offset_scale,
            (geometry.offset_m[sender] - geometry.offset_m[destination]) / offset_scale,
            delta_cos,
            delta_sin,
            valid,
            distance,
        )
    ).astype(np.float32)


def _xy_coordinates(values: np.ndarray, name: str) -> np.ndarray:
    coordinates = np.array(values, dtype=np.float64, copy=True)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(f"{name} must have shape [N, 2]")
    return coordinates


def _finite_length(value: float, name: str, *, allow_zero: bool = False) -> float:
    requirement = "nonnegative" if allow_zero else "positive"
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or (value < 0 if allow_zero else value <= 0)
    ):
        raise ValueError(f"{name} must be finite and {requirement}")
    return float(value)


def _node_indices(values: np.ndarray, name: str, node_count: int) -> np.ndarray:
    indices = np.asarray(values)
    if indices.ndim != 1 or (indices.size and indices.dtype.kind not in "iu"):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if np.any(indices < 0) or np.any(indices >= node_count):
        raise ValueError(f"{name} must be in [0, N)")
    return indices.astype(np.int64, copy=False)
