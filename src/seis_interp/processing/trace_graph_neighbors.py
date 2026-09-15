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


class TraceGraphSpatialIndex:
    """Owned geometry-only sorted axes shared across episode visibility views.

    Storage is linear in candidates. Eligibility is applied before exact
    distance/radius selection and is never retained by this object.
    Multiple destinations share buffers of at most candidate_chunk_size
    box-filtered pairs. Bounds scale with destinations, without a dense
    destination-by-candidate matrix; only top-k crosses a buffer boundary.
    """

    def __init__(
        self,
        candidate_geometry: TraceGraphGeometry,
        candidate_trace_ids: np.ndarray,
        **search_settings,
    ) -> None:
        if "allowed_mask" in search_settings:
            raise ValueError("allowed_mask belongs to the episode visibility view")
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
            np.ones(len(candidate_geometry.offset_m), dtype=bool),
            **search_settings,
        )
        self.geometry = TraceGraphGeometry(
            **{
                field.name: _owned_readonly(getattr(candidate_geometry, field.name))
                for field in fields(TraceGraphGeometry)
            }
        )
        self.trace_ids = _owned_readonly(np.asarray(candidate_trace_ids, dtype=np.int64))
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
        self._scales = np.asarray(self._settings["relation_scales_m"], dtype=np.float64)
        self._box_widths = self._relation_box_widths()
        self._rows = _owned_readonly(np.arange(len(self.trace_ids), dtype=np.int64))
        self._axes = {}
        self._columns = {}
        for name in ("source_xy_m", "receiver_xy_m", "midpoint_xy_m", "offset_xy_m"):
            for axis in range(2):
                coordinates = getattr(self.geometry, name)[:, axis]
                order = np.argsort(coordinates, kind="stable")
                self._columns[name, axis] = _owned_readonly(coordinates)
                self._axes[name, axis] = (
                    _owned_readonly(coordinates[order]),
                    _owned_readonly(self._rows[order]),
                )

    @property
    def search_settings(self) -> dict:
        """Return settings independently of the owned visibility and arrays."""
        return deepcopy(self._settings)

    def _relation_box_widths(self) -> dict[int, tuple]:
        """Widen each relation's bounds once; radius and scales never change."""
        radius = self._settings["radius"]
        if self._settings["topology"] == "single_4d":
            scale_rows = {0: self._settings["common_distance_scales_m"]}
        else:
            scale_rows = {relation: self._scales[relation] for relation in range(4)}
        return {
            relation: tuple(_conservative_box_width(radius, scale) for scale in scales)
            for relation, scales in scale_rows.items()
        }

    def _select(
        self,
        destination_geometry: TraceGraphGeometry,
        destination_trace_ids: np.ndarray,
        eligible_mask: np.ndarray,
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
        radius = self._settings["radius"]
        blocks = ([], [], [], [])
        # For very small batches, staging/reduction costs more than scalar top-k.
        if len(ids) > 3:
            return self._select_batched(destination_geometry, ids, eligible_mask, relations, k)
        boxes = (
            {relation: self._box_bounds(destination_geometry, relation) for relation in relations}
            if len(ids) > 1
            else {}
        )
        for row, trace_id in enumerate(ids):
            for relation in relations:
                candidates = (
                    self._batch_box_rows(*boxes[relation], row)
                    if boxes
                    else self._box_rows(destination_geometry, row, relation)
                )
                candidates = candidates[eligible_mask[candidates]]
                candidates = candidates[self.trace_ids[candidates] != trace_id]
                selected_ids = np.empty(0, dtype=np.int64)
                selected_distances = np.empty(0, dtype=np.float64)
                for start in range(0, len(candidates), chunk_size):
                    rows = candidates[start : start + chunk_size]
                    distances = self._distances(destination_geometry, row, rows, relation)
                    inside = distances <= radius
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

    def _select_batched(self, destination, ids, eligible, relations, k):
        blocks = [
            self._select_relation(destination, ids, eligible, relation, k) for relation in relations
        ]
        rows, senders, distances = (
            np.concatenate([block[column] for block in blocks]) for column in range(3)
        )
        types = np.concatenate(
            [
                np.full(len(block[0]), relation, dtype=np.int64)
                for relation, block in zip(relations, blocks, strict=True)
            ]
        )
        # Each relation is already ordered by destination, distance and sender ID.
        order = np.lexsort((types, rows))
        return TraceGraphNeighbors(senders[order], ids[rows[order]], types[order], distances[order])

    def _select_relation(self, destination, ids, eligible, relation, k):
        """Retain only top-k across a chunk boundary; emit completed destinations."""
        pending = (np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float64))
        blocks = ([], [], [])
        for last_row, rows, candidates in self._candidate_pair_chunks(
            destination, ids, eligible, relation
        ):
            distances = (
                self._distances(destination, rows, candidates, relation)
                if len(rows)
                else np.empty(0, dtype=np.float64)
            )
            inside = distances <= self._settings["radius"]
            values = (rows[inside], self.trace_ids[candidates[inside]], distances[inside])
            if len(pending[0]):
                values = tuple(
                    np.concatenate((old, new)) for old, new in zip(pending, values, strict=True)
                )
            selected = _relation_topk(*values, k)
            finished = np.searchsorted(selected[0], last_row, side="left")
            if finished:
                for block, array in zip(blocks, selected, strict=True):
                    block.append(array[:finished].copy())
            pending = tuple(array[finished:].copy() for array in selected)
        for block, array in zip(blocks, pending, strict=True):
            block.append(array)
        return tuple(np.concatenate(block) for block in blocks)

    def _candidate_pair_chunks(self, destination, ids, eligible, relation):
        """Batch only box-filtered pairs into reusable, chunk-size-bounded buffers."""
        bounds, shortest = self._box_bounds(destination, relation)
        capacity = min(self._settings["candidate_chunk_size"], len(self.trace_ids))
        pair_rows = np.empty(capacity, dtype=np.int64)
        pair_candidates = np.empty(capacity, dtype=np.int64)
        used = 0
        for row, trace_id in enumerate(ids):
            candidates = self._batch_box_rows(bounds, shortest, row)
            candidates = candidates[eligible[candidates] & (self.trace_ids[candidates] != trace_id)]
            offset = 0
            while offset < len(candidates):
                count = min(capacity - used, len(candidates) - offset)
                pair_rows[used : used + count] = row
                pair_candidates[used : used + count] = candidates[offset : offset + count]
                used += count
                offset += count
                if used == capacity:
                    # The consumer finishes each borrowed buffer before resuming us.
                    yield row, pair_rows, pair_candidates
                    used = 0
        if used:
            yield pair_rows[used - 1], pair_rows[:used], pair_candidates[:used]

    def _box_bounds(self, destination, relation):
        """Find sorted-axis ranges for all destinations with O(destinations) storage."""
        single = self._settings["topology"] == "single_4d"
        names = _relation_axis_names(relation, single)
        bounds = []
        lengths = []
        with np.errstate(over="ignore", invalid="ignore"):
            for name, width in zip(names, self._box_widths[relation], strict=True):
                if width is None:
                    continue
                for axis in range(2):
                    center = getattr(destination, name)[:, axis]
                    lower = np.nextafter(center - width, -np.inf)
                    upper = np.nextafter(center + width, np.inf)
                    valid = ~(np.isnan(lower) | np.isnan(upper))
                    values, indices = self._axes[name, axis]
                    start = np.searchsorted(values, lower, side="left")
                    stop = np.searchsorted(values, upper, side="right")
                    bounds.append(
                        (indices, self._columns[name, axis], start, stop, lower, upper, valid)
                    )
                    # Invalid bounds must lose even to a full-domain valid range.
                    lengths.append(np.where(valid, stop - start, len(self.trace_ids) + 1))
        shortest = np.argmin(lengths, axis=0) if bounds else None
        return bounds, shortest

    def _batch_box_rows(self, bounds, shortest, row):
        if not bounds:
            return self._rows
        selected = shortest[row]
        indices, _, start, stop, _, _, valid = bounds[selected]
        if not valid[row]:
            return self._rows
        rows = indices[start[row] : stop[row]]
        # The shortest sorted range avoids allocating a full-domain Boolean mask.
        # That range already satisfies its own bound, so only the others filter.
        for position, (_, column, start, stop, lower, upper, valid) in enumerate(bounds):
            if (
                position == selected
                or not valid[row]
                or (start[row] == 0 and stop[row] == len(self.trace_ids))
            ):
                continue
            values = column[rows]
            rows = rows[(values >= lower[row]) & (values <= upper[row])]
        return rows

    def _box_rows(self, destination, row, relation):
        """Avoid batching overhead when there is only one destination."""
        single = self._settings["topology"] == "single_4d"
        names = _relation_axis_names(relation, single)
        bounds = []
        with np.errstate(over="ignore", invalid="ignore"):
            for name, width in zip(names, self._box_widths[relation], strict=True):
                if width is None:
                    continue
                for axis in range(2):
                    center = getattr(destination, name)[row, axis]
                    lower = np.nextafter(center - width, -np.inf)
                    upper = np.nextafter(center + width, np.inf)
                    if np.isnan(lower) or np.isnan(upper):
                        continue
                    values, indices = self._axes[name, axis]
                    start = np.searchsorted(values, lower, side="left")
                    stop = np.searchsorted(values, upper, side="right")
                    bounds.append((stop - start, indices, start, stop, name, axis, lower, upper))
        if not bounds:
            return self._rows
        shortest = min(range(len(bounds)), key=lambda index: bounds[index][0])
        _, indices, start, stop, *_ = bounds[shortest]
        rows = indices[start:stop]
        # The shortest sorted range avoids allocating a full-domain Boolean mask.
        # That range already satisfies its own bound, so only the others filter.
        for position, (_, _, _, _, name, axis, lower, upper) in enumerate(bounds):
            if position == shortest:
                continue
            values = self._columns[name, axis][rows]
            rows = rows[(values >= lower) & (values <= upper)]
        return rows

    def _distances(self, destination, row, rows, relation):
        if self._settings["topology"] != "single_4d":
            # Only this relation's own coordinate pair contributes; the other two
            # squared lengths of the four-relation form are discarded anyway.
            scales = self._scales[relation]
            squared = []
            for name in _relation_axis_names(relation, False):
                coordinates = getattr(destination, name)
                first = self._columns[name, 0][rows] - coordinates[row, 0]
                second = self._columns[name, 1][rows] - coordinates[row, 1]
                # Summing a length-two axis is exactly this ordered pair of terms.
                squared.append(first * first + second * second)
            return np.sqrt(squared[0] / scales[0] ** 2 + squared[1] / scales[1] ** 2)
        common = np.asarray(self._settings["common_distance_scales_m"], dtype=np.float64)
        delta_midpoint = self.geometry.midpoint_xy_m[rows] - destination.midpoint_xy_m[row]
        delta_offset = self.geometry.offset_xy_m[rows] - destination.offset_xy_m[row]
        return np.sqrt(
            np.sum(delta_midpoint**2, axis=1) / common[0] ** 2
            + np.sum(delta_offset**2, axis=1) / common[1] ** 2
        )


class FixedTraceGraphNeighborIndex:
    """An episode's owned visibility over a reusable geometry-only index."""

    def __init__(
        self,
        candidate_geometry: TraceGraphGeometry,
        candidate_trace_ids: np.ndarray,
        observed_mask: np.ndarray,
        **search_settings,
    ) -> None:
        allowed = search_settings.pop("allowed_mask", None)
        spatial_index = TraceGraphSpatialIndex(
            candidate_geometry, candidate_trace_ids, **search_settings
        )
        self._bind(spatial_index, observed_mask, allowed)

    @classmethod
    def from_spatial_index(
        cls,
        spatial_index: TraceGraphSpatialIndex,
        observed_mask: np.ndarray,
        *,
        allowed_mask: np.ndarray | None = None,
    ) -> FixedTraceGraphNeighborIndex:
        """Bind masks in the shared index's candidate row order, without rebuilding."""
        if not isinstance(spatial_index, TraceGraphSpatialIndex):
            raise ValueError("spatial_index must be a TraceGraphSpatialIndex")
        view = cls.__new__(cls)
        view._bind(spatial_index, observed_mask, allowed_mask)
        return view

    def _bind(self, spatial_index, observed_mask, allowed_mask):
        count = len(spatial_index.trace_ids)
        observed = _boolean_mask(observed_mask, count, "observed_mask")
        eligible = observed.copy()
        if allowed_mask is not None:
            eligible &= _boolean_mask(allowed_mask, count, "allowed_mask")
        self.spatial_index = spatial_index
        self.eligible_mask = _owned_readonly(eligible)

    @property
    def geometry(self) -> TraceGraphGeometry:
        return self.spatial_index.geometry

    @property
    def trace_ids(self) -> np.ndarray:
        return self.spatial_index.trace_ids

    @property
    def search_settings(self) -> dict:
        return self.spatial_index.search_settings

    def select(
        self, destination_geometry: TraceGraphGeometry, destination_trace_ids: np.ndarray
    ) -> TraceGraphNeighbors:
        """Select exact incoming edges in destination, relation, distance/ID order."""
        return self.spatial_index._select(
            destination_geometry, destination_trace_ids, self.eligible_mask
        )


def _relation_topk(rows, senders, distances, k):
    """Select exact top-k from pairs already grouped in destination-row order."""
    if not len(rows):
        return rows, senders, distances
    if k <= 8:
        return _small_relation_topk(rows, senders, distances, k)
    order = np.lexsort((senders, distances, rows))
    ordered_rows = rows[order]
    positions = np.arange(len(order))
    first = np.r_[True, ordered_rows[1:] != ordered_rows[:-1]]
    group_starts = np.maximum.accumulate(np.where(first, positions, 0))
    order = order[positions - group_starts < k]
    return rows[order], senders[order], distances[order]


def _small_relation_topk(rows, senders, distances, k):
    """Reduce small fanouts without sorting every candidate in a distance chunk."""
    starts = np.r_[0, np.flatnonzero(rows[1:] != rows[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(rows)])
    groups = np.repeat(np.arange(len(starts)), counts)
    remaining = distances.copy()
    winners = []
    for _ in range(min(k, int(counts.max()))):
        minimum = np.minimum.reduceat(remaining, starts)
        nearest = (remaining == minimum[groups]) & np.isfinite(remaining)
        tied_ids = np.where(nearest, senders, np.iinfo(np.int64).max)
        smallest_id = np.minimum.reduceat(tied_ids, starts)
        chosen = np.flatnonzero(nearest & (senders == smallest_id[groups]))
        winners.append(chosen)
        remaining[chosen] = np.inf
    order = np.concatenate(winners)
    # Successive minima already order each destination by distance, then ID.
    order = order[np.argsort(rows[order], kind="stable")]
    return rows[order], senders[order], distances[order]


def _relation_axis_names(relation: int, single: bool) -> tuple[str, str]:
    """Name the two coordinate pairs whose scaled lengths form this distance."""
    if single or relation >= 2:
        return ("midpoint_xy_m", "offset_xy_m")
    return ("source_xy_m", "receiver_xy_m")


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
