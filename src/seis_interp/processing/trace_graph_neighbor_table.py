"""Ranked nearest candidates of every indexed trace, reused for exact episode searches."""

from __future__ import annotations

from dataclasses import fields
from numbers import Integral

import numpy as np

from seis_interp.processing.trace_graph_geometry import TraceGraphGeometry
from seis_interp.processing.trace_graph_neighbors import TraceGraphSpatialIndex

_RELATION_COUNT = 4
# Settings that change which neighbors are selected; the chunk size only bounds memory.
_RESULT_SETTINGS = (
    "relation_scales_m",
    "neighbors_per_relation",
    "radius",
    "topology",
    "excluded_relation",
    "common_distance_scales_m",
    "single_4d_neighbors",
)


class TraceGraphNeighborTable:
    """Each indexed trace's ``k + margin`` nearest candidates per relation.

    The table is built once with the exact index search, with every trace of
    ``spatial_index`` eligible. Ranking (distance, then ID), the radius rule and
    self-exclusion do not depend on visibility, so for any visibility that is a
    subset of the table's traces the exact top-k of a destination is the first
    ``k`` visible entries of its ranked list. A list with fewer than ``k`` visible
    entries that may continue beyond ``k + margin`` is searched exactly instead
    (see ``TraceGraphNeighborTableView.select``).
    """

    def __init__(
        self,
        spatial_index: TraceGraphSpatialIndex,
        *,
        margin: int = 16,
        destination_chunk_size: int = 2048,
    ) -> None:
        if not isinstance(spatial_index, TraceGraphSpatialIndex):
            raise ValueError("spatial_index must be a TraceGraphSpatialIndex")
        for name, value in (("margin", margin), ("destination_chunk_size", destination_chunk_size)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        settings = spatial_index.search_settings
        single = settings["topology"] == "single_4d"
        key = "single_4d_neighbors" if single else "neighbors_per_relation"
        self.neighbors = int(settings[key])
        width = self.neighbors + int(margin)
        wide = TraceGraphSpatialIndex(
            spatial_index.geometry, spatial_index.trace_ids, **{**settings, key: width}
        )
        self.geometry = spatial_index.geometry
        self.trace_ids = spatial_index.trace_ids
        self.search_settings = {name: settings[name] for name in _RESULT_SETTINGS}
        self._lookup = TraceIdLookup(self.trace_ids)
        count = len(self.trace_ids)
        self._senders = np.full((count, _RELATION_COUNT, width), -1, dtype=np.int64)
        self._distances = np.full((count, _RELATION_COUNT, width), np.inf, dtype=np.float64)
        everyone = np.ones(count, dtype=bool)
        for start in range(0, count, int(destination_chunk_size)):
            rows = np.arange(start, min(count, start + int(destination_chunk_size)))
            edges = wide._select(
                select_geometry_rows(self.geometry, rows), self.trace_ids[rows], everyone
            )
            destinations = self._lookup.rows(edges.destination_ids)
            group = destinations * _RELATION_COUNT + edges.edge_type
            positions = np.arange(len(group))
            first = np.r_[True, group[1:] != group[:-1]] if len(group) else np.empty(0, bool)
            slot = positions - np.maximum.accumulate(np.where(first, positions, 0))
            self._senders[destinations, edges.edge_type, slot] = self._lookup.rows(edges.sender_ids)
            self._distances[destinations, edges.edge_type, slot] = edges.distances
        # A list shorter than its width already holds every in-radius candidate.
        self._complete = self._senders[:, :, -1] < 0

    def view(
        self, spatial_index: TraceGraphSpatialIndex, eligible_mask: np.ndarray
    ) -> TraceGraphNeighborTableView:
        """Bind an index and its visibility; refuse anything that could change results."""
        return TraceGraphNeighborTableView(self, spatial_index, eligible_mask)


class TraceGraphNeighborTableView:
    """Exact incoming edges for rows of one index under one fixed visibility."""

    def __init__(
        self,
        table: TraceGraphNeighborTable,
        spatial_index: TraceGraphSpatialIndex,
        eligible_mask: np.ndarray,
    ) -> None:
        settings = spatial_index.search_settings
        if any(
            not np.array_equal(np.asarray(settings[name]), np.asarray(value))
            if name in ("relation_scales_m", "common_distance_scales_m") and value is not None
            else settings[name] != value
            for name, value in table.search_settings.items()
        ):
            raise ValueError("neighbor table search settings differ from the index")
        eligible = np.asarray(eligible_mask)
        if eligible.dtype != np.bool_ or eligible.shape != spatial_index.trace_ids.shape:
            raise ValueError("eligible_mask must be a boolean vector over the index")
        table_rows = table._lookup.rows(spatial_index.trace_ids)
        covered = table_rows >= 0
        if np.any(eligible & ~covered):
            raise ValueError("neighbor table must contain every eligible candidate")
        if not same_geometry_rows(
            select_geometry_rows(spatial_index.geometry, np.flatnonzero(covered)),
            table.geometry,
            table_rows[covered],
        ):
            raise ValueError("neighbor table geometry differs from the index")
        self._table = table
        self._table_rows = table_rows
        self._eligible = np.zeros(len(table.trace_ids), dtype=bool)
        self._eligible[table_rows[eligible]] = True

    def select(
        self, destination_rows: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return table edges and the destinations that need the exact search.

        Edges are ``(sender_ids, destination positions, edge_type, distances)`` in
        destination, relation, distance and ID order, where a position indexes
        ``destination_rows``. The last array lists the positions without edges here
        that must be searched exactly: rows absent from the table, and lists whose
        visible entries may be truncated by the table width.
        """
        table = self._table
        rows = self._table_rows[destination_rows]
        covered = rows >= 0
        rows = np.where(covered, rows, 0)
        candidates = table._senders[rows]
        visible = candidates >= 0
        visible[visible] = self._eligible[candidates[visible]]
        visible &= covered[:, None, None]
        rank = np.cumsum(visible, axis=-1)
        taken = visible & (rank <= table.neighbors)
        truncated = (rank[:, :, -1] < table.neighbors) & ~table._complete[rows]
        exact = np.flatnonzero(~covered | truncated.any(axis=1))
        taken[exact] = False
        position, relation, slot = np.nonzero(taken)
        senders = candidates[position, relation, slot]
        distances = table._distances[rows[position], relation, slot]
        return table.trace_ids[senders], position, relation.astype(np.int64), distances, exact


def select_geometry_rows(geometry: TraceGraphGeometry, rows: np.ndarray) -> TraceGraphGeometry:
    """Rows of every geometry field, in the given order."""
    return TraceGraphGeometry(
        **{field.name: getattr(geometry, field.name)[rows] for field in fields(TraceGraphGeometry)}
    )


def same_geometry_rows(
    geometry: TraceGraphGeometry, reference: TraceGraphGeometry, rows: np.ndarray
) -> bool:
    """Whether ``geometry`` equals ``reference`` at ``rows`` field by field (``np.array_equal``)."""
    return all(
        np.array_equal(getattr(geometry, field.name), getattr(reference, field.name)[rows])
        for field in fields(TraceGraphGeometry)
    )


class TraceIdLookup:
    """Vectorized trace ID to row lookup; absent IDs map to -1."""

    def __init__(self, trace_ids: np.ndarray) -> None:
        self._order = np.argsort(trace_ids, kind="stable")
        self._sorted = trace_ids[self._order]

    def rows(self, trace_ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(trace_ids, dtype=np.int64)
        if not len(self._sorted):
            return np.full(ids.shape, -1, dtype=np.int64)
        positions = np.minimum(np.searchsorted(self._sorted, ids), len(self._sorted) - 1)
        found = self._sorted[positions] == ids
        return np.where(found, self._order[positions], -1).astype(np.int64)
