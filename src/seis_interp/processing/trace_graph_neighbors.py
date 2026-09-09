"""Exact, visibility-first incoming neighbors for the four trace relations."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields
from numbers import Integral, Real

import numpy as np

from seis_interp.processing.trace_graph_geometry import RELATION_NAMES, TraceGraphGeometry


@dataclass(frozen=True)
class TraceGraphNeighbors:
    """Typed incoming edges, ordered by destination, relation, distance, ID.

    Destination order follows the supplied destination array. IDs and relation
    types are int64, distances are float64 and all arrays have shape ``[E]``.
    """

    sender_ids: np.ndarray
    destination_ids: np.ndarray
    edge_type: np.ndarray
    distances: np.ndarray


def select_trace_graph_neighbors(
    destination_geometry: TraceGraphGeometry,
    destination_trace_ids: np.ndarray,
    candidate_geometry: TraceGraphGeometry,
    candidate_trace_ids: np.ndarray,
    observed_mask: np.ndarray,
    *,
    relation_scales_m: np.ndarray,
    allowed_mask: np.ndarray | None = None,
    neighbors_per_relation: int = 8,
    radius: float = 1.0,
    candidate_chunk_size: int = 4096,
    topology: str = "multi_relation",
    excluded_relation: str | None = None,
    common_distance_scales_m: tuple[float, float] | None = None,
    single_4d_neighbors: int = 32,
) -> TraceGraphNeighbors:
    """Select directed radius-limited top-k from the original observed domain.

    ``relation_scales_m`` is positive float64 ``[4, 2]`` in the fixed relation
    order source, receiver, cmp, offset_azimuth. The first two rows scale source
    and receiver differences; the last two scale midpoint and full-offset
    differences. Study-specific physical scales have no defaults here.

    IDs must be unique within each geometry; physical canonicalization belongs
    to the data boundary. Only observed, allowed, non-self candidates compete
    for top-k. Different relations retain their own copy of a selected pair.
    Search uses one destination and a candidate chunk at a time: temporary
    distance storage is O(chunk_size + k), never a survey N-by-N matrix.
    """
    destination_ids = _trace_ids(destination_trace_ids, destination_geometry, "destination")
    candidate_ids = _trace_ids(candidate_trace_ids, candidate_geometry, "candidate")
    observed = _boolean_mask(observed_mask, len(candidate_ids), "observed_mask")
    allowed = (
        np.ones(len(candidate_ids), dtype=bool)
        if allowed_mask is None
        else _boolean_mask(allowed_mask, len(candidate_ids), "allowed_mask")
    )
    scales = np.asarray(relation_scales_m, dtype=np.float64)
    if scales.shape != (4, 2) or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("relation_scales_m must be finite, positive and have shape [4, 2]")
    k = _positive_integer(neighbors_per_relation, "neighbors_per_relation")
    chunk_size = _positive_integer(candidate_chunk_size, "candidate_chunk_size")
    if topology not in ("multi_relation", "single_4d"):
        raise ValueError("topology must be multi_relation or single_4d")
    if excluded_relation is not None and excluded_relation not in RELATION_NAMES:
        raise ValueError("excluded_relation must name one of the four seismic relations")
    if topology == "single_4d":
        common = np.asarray(common_distance_scales_m, dtype=np.float64)
        if common.shape != (2,) or not np.all(np.isfinite(common)) or np.any(common <= 0):
            raise ValueError("single_4d requires two positive common_distance_scales_m")
        if excluded_relation is not None:
            raise ValueError("single_4d cannot exclude a seismic relation")
        k = _positive_integer(single_4d_neighbors, "single_4d_neighbors")
        relations = (0,)
    else:
        relations = tuple(
            index for index, name in enumerate(RELATION_NAMES) if name != excluded_relation
        )
    if (
        isinstance(radius, (bool, np.bool_))
        or not isinstance(radius, Real)
        or not np.isfinite(radius)
        or radius <= 0
    ):
        raise ValueError("radius must be finite and positive")
    eligible_rows = np.flatnonzero(observed & allowed)
    sender_blocks, destination_blocks, type_blocks, distance_blocks = [], [], [], []
    for destination_row, destination_id in enumerate(destination_ids):
        selected_ids = [np.empty(0, dtype=np.int64) for _ in range(4)]
        selected_distances = [np.empty(0, dtype=np.float64) for _ in range(4)]
        for start in range(0, len(eligible_rows), chunk_size):
            rows = eligible_rows[start : start + chunk_size]
            rows = rows[candidate_ids[rows] != destination_id]
            distances = _relation_distances(
                destination_geometry, destination_row, candidate_geometry, rows, scales
            )
            if topology == "single_4d":
                delta_midpoint = (
                    candidate_geometry.midpoint_xy_m[rows]
                    - destination_geometry.midpoint_xy_m[destination_row]
                )
                delta_offset = (
                    candidate_geometry.offset_xy_m[rows]
                    - destination_geometry.offset_xy_m[destination_row]
                )
                distances[:, 0] = np.sqrt(
                    np.sum(delta_midpoint**2, axis=1) / common[0] ** 2
                    + np.sum(delta_offset**2, axis=1) / common[1] ** 2
                )
            for relation in relations:
                inside = distances[:, relation] <= radius
                ids = np.concatenate((selected_ids[relation], candidate_ids[rows[inside]]))
                values = np.concatenate((selected_distances[relation], distances[inside, relation]))
                order = np.lexsort((ids, values))[:k]
                selected_ids[relation] = ids[order]
                selected_distances[relation] = values[order]
        for relation in relations:
            count = len(selected_ids[relation])
            sender_blocks.append(selected_ids[relation])
            destination_blocks.append(np.full(count, destination_id, dtype=np.int64))
            type_blocks.append(np.full(count, relation, dtype=np.int64))
            distance_blocks.append(selected_distances[relation])
    return TraceGraphNeighbors(
        sender_ids=_concatenate(sender_blocks, np.int64),
        destination_ids=_concatenate(destination_blocks, np.int64),
        edge_type=_concatenate(type_blocks, np.int64),
        distances=_concatenate(distance_blocks, np.float64),
    )


class FixedTraceGraphNeighborIndex:
    """Exact radius search over one owned geometry and fixed visibility mask.

    Sorted coordinate axes only remove candidates outside conservative boxes.
    The original float64 distance formula and distance/ID ordering decide every
    edge. Index storage is linear in candidates; no pairwise matrix is built.
    Construct a new index when an episode's visibility or allowed mask changes.
    """

    def __init__(
        self,
        candidate_geometry: TraceGraphGeometry,
        candidate_trace_ids: np.ndarray,
        observed_mask: np.ndarray,
        **search_settings,
    ) -> None:
        empty = TraceGraphGeometry(
            **{
                field.name: getattr(candidate_geometry, field.name)[:0]
                for field in fields(TraceGraphGeometry)
            }
        )
        # Reuse the brute boundary's validation, without searching any query.
        select_trace_graph_neighbors(
            empty,
            np.empty(0, dtype=np.int64),
            candidate_geometry,
            candidate_trace_ids,
            observed_mask,
            **search_settings,
        )
        self.geometry = TraceGraphGeometry(
            **{
                field.name: _owned_readonly(getattr(candidate_geometry, field.name))
                for field in fields(TraceGraphGeometry)
            }
        )
        self.trace_ids = _owned_readonly(np.asarray(candidate_trace_ids, dtype=np.int64))
        eligible = np.asarray(observed_mask).copy()
        if search_settings.get("allowed_mask") is not None:
            eligible &= np.asarray(search_settings["allowed_mask"])
        self.eligible_mask = _owned_readonly(eligible)
        self._settings = {
            "neighbors_per_relation": 8,
            "radius": 1.0,
            "candidate_chunk_size": 4096,
            "topology": "multi_relation",
            "excluded_relation": None,
            "common_distance_scales_m": None,
            "single_4d_neighbors": 32,
            **deepcopy(search_settings),
        }
        self._settings.pop("allowed_mask", None)
        self._scales = np.asarray(self._settings["relation_scales_m"], dtype=np.float64)
        self._eligible_rows = np.flatnonzero(self.eligible_mask)
        self._axes = {}
        for name in ("source_xy_m", "receiver_xy_m", "midpoint_xy_m", "offset_xy_m"):
            for axis in range(2):
                coordinates = getattr(self.geometry, name)[self._eligible_rows, axis]
                order = np.argsort(coordinates, kind="stable")
                self._axes[name, axis] = (coordinates[order], self._eligible_rows[order])

    @property
    def search_settings(self) -> dict:
        """Return settings independently of the owned visibility and arrays."""
        return deepcopy(self._settings)

    def select(
        self, destination_geometry: TraceGraphGeometry, destination_trace_ids: np.ndarray
    ) -> TraceGraphNeighbors:
        """Select exact incoming edges in destination, relation, distance/ID order."""
        ids = _trace_ids(destination_trace_ids, destination_geometry, "destination")
        single = self._settings["topology"] == "single_4d"
        relations = (
            (0,)
            if single
            else tuple(
                index
                for index, name in enumerate(RELATION_NAMES)
                if name != self._settings["excluded_relation"]
            )
        )
        k = self._settings["single_4d_neighbors" if single else "neighbors_per_relation"]
        chunk_size = self._settings["candidate_chunk_size"]
        blocks = ([], [], [], [])
        for row, trace_id in enumerate(ids):
            for relation in relations:
                candidates = self._box_rows(destination_geometry, row, relation)
                candidates = candidates[self.trace_ids[candidates] != trace_id]
                selected_ids = np.empty(0, dtype=np.int64)
                selected_distances = np.empty(0, dtype=np.float64)
                for start in range(0, len(candidates), chunk_size):
                    rows = candidates[start : start + chunk_size]
                    distances = self._distances(destination_geometry, row, rows, relation)
                    inside = distances <= self._settings["radius"]
                    combined_ids = np.concatenate((selected_ids, self.trace_ids[rows[inside]]))
                    combined_distances = np.concatenate((selected_distances, distances[inside]))
                    order = np.lexsort((combined_ids, combined_distances))[:k]
                    selected_ids = combined_ids[order]
                    selected_distances = combined_distances[order]
                count = len(selected_ids)
                blocks[0].append(selected_ids)
                blocks[1].append(np.full(count, trace_id, dtype=np.int64))
                blocks[2].append(np.full(count, relation, dtype=np.int64))
                blocks[3].append(selected_distances)
        return TraceGraphNeighbors(
            *(
                _concatenate(block, dtype)
                for block, dtype in zip(
                    blocks, (np.int64, np.int64, np.int64, np.float64), strict=True
                )
            )
        )

    def _box_rows(self, destination, row, relation):
        single = self._settings["topology"] == "single_4d"
        names = (
            ("source_xy_m", "receiver_xy_m")
            if relation < 2 and not single
            else ("midpoint_xy_m", "offset_xy_m")
        )
        scales = self._settings["common_distance_scales_m"] if single else self._scales[relation]
        bounds = []
        for name, scale in zip(names, scales, strict=True):
            width = _conservative_box_width(self._settings["radius"], scale)
            if width is None:
                continue
            for axis in range(2):
                center = getattr(destination, name)[row, axis]
                with np.errstate(over="ignore", invalid="ignore"):
                    lower = np.nextafter(center - width, -np.inf)
                    upper = np.nextafter(center + width, np.inf)
                if np.isnan(lower) or np.isnan(upper):
                    continue
                values, indices = self._axes[name, axis]
                start = np.searchsorted(values, lower, side="left")
                stop = np.searchsorted(values, upper, side="right")
                bounds.append((stop - start, indices, start, stop, name, axis, lower, upper))
        if not bounds:
            return self._eligible_rows
        _, indices, start, stop, *_ = min(bounds, key=lambda bound: bound[0])
        rows = indices[start:stop]
        # The shortest sorted range avoids allocating a full-domain Boolean mask.
        for _, _, _, _, name, axis, lower, upper in bounds:
            values = getattr(self.geometry, name)[rows, axis]
            rows = rows[(values >= lower) & (values <= upper)]
        return rows

    def _distances(self, destination, row, rows, relation):
        if self._settings["topology"] != "single_4d":
            return _relation_distances(destination, row, self.geometry, rows, self._scales)[
                :, relation
            ]
        common = np.asarray(self._settings["common_distance_scales_m"], dtype=np.float64)
        delta_midpoint = self.geometry.midpoint_xy_m[rows] - destination.midpoint_xy_m[row]
        delta_offset = self.geometry.offset_xy_m[rows] - destination.offset_xy_m[row]
        return np.sqrt(
            np.sum(delta_midpoint**2, axis=1) / common[0] ** 2
            + np.sum(delta_offset**2, axis=1) / common[1] ** 2
        )


def _conservative_box_width(radius, scale):
    """Widen radius bounds for rounded products/sums; avoid unsafe extreme scales."""
    limits = np.finfo(np.float64)
    if radius < np.sqrt(limits.tiny):
        # A squared distance can underflow during division by a large scale.
        # Its zero result may pass a radius far below the geometric distance.
        return None
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        squared_scale = np.float64(scale) ** 2
        width = np.float64(radius) * scale
        if not np.isfinite(squared_scale) or squared_scale < limits.tiny:
            return None
        # The absolute allowance also covers subnormal squared differences.
        width = width * (1.0 + 16.0 * limits.eps) + np.sqrt(limits.tiny)
    return None if not np.isfinite(width) else np.nextafter(width, np.inf)


def _owned_readonly(values):
    result = np.array(values, copy=True)
    result.setflags(write=False)
    return result


def _relation_distances(
    destination: TraceGraphGeometry,
    destination_row: int,
    candidates: TraceGraphGeometry,
    rows: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    squared_lengths = []
    for name in ("source_xy_m", "receiver_xy_m", "midpoint_xy_m", "offset_xy_m"):
        delta = getattr(candidates, name)[rows] - getattr(destination, name)[destination_row]
        squared_lengths.append(np.sum(delta * delta, axis=1, dtype=np.float64))
    distances = np.empty((len(rows), 4), dtype=np.float64)
    for relation in range(4):
        first = 0 if relation < 2 else 2
        distances[:, relation] = np.sqrt(
            squared_lengths[first] / scales[relation, 0] ** 2
            + squared_lengths[first + 1] / scales[relation, 1] ** 2
        )
    return distances


def _trace_ids(values: np.ndarray, geometry: TraceGraphGeometry, name: str) -> np.ndarray:
    ids = np.asarray(values)
    if ids.shape != geometry.offset_m.shape or (ids.size and ids.dtype.kind not in "iu"):
        raise ValueError(f"{name}_trace_ids must be an integer vector matching geometry")
    if ids.size and ids.dtype.kind == "u" and np.max(ids) > np.iinfo(np.int64).max:
        raise ValueError(f"{name}_trace_ids must fit int64")
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"{name}_trace_ids must be unique after canonicalization")
    return ids.astype(np.int64, copy=False)


def _boolean_mask(values: np.ndarray, count: int, name: str) -> np.ndarray:
    mask = np.asarray(values)
    if mask.shape != (count,) or mask.dtype != np.bool_:
        raise ValueError(f"{name} must be a boolean vector matching candidates")
    return mask


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _concatenate(blocks: list[np.ndarray], dtype: type) -> np.ndarray:
    return np.concatenate(blocks) if blocks else np.empty(0, dtype=dtype)
