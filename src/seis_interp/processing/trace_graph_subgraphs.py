"""Finite observed dependency closures for synchronous trace graph rounds."""

from __future__ import annotations

from dataclasses import dataclass, fields
from numbers import Integral

import numpy as np

from seis_interp.processing.trace_graph_geometry import RELATION_NAMES, TraceGraphGeometry
from seis_interp.processing.trace_graph_neighbors import (
    FixedTraceGraphNeighborIndex,
    TraceGraphNeighbors,
    select_trace_graph_neighbors,
)

_CACHED_EDGE_DTYPE = np.dtype(
    [("sender_ids", np.int64), ("edge_type", np.int64), ("distances", np.float64)]
)


@dataclass(frozen=True)
class TraceGraphPlan:
    """CPU geometry and topology; no amplitudes or array-row correspondence.

    Nodes are in ascending stable-ID order, while ``query_indices`` preserves
    input query order. ``edge_index`` is ``[sender, destination]``. Depth-L
    leaves have no incoming edges unless another query reaches them sooner.
    Degree and minimum distance describe the full selected incoming edges for
    expanded nodes; both are zero for an empty relation or an unexpanded leaf.
    ``dependency_rounds`` records the requested update depth, even when the
    observed domain is exhausted before reaching that depth.
    """

    trace_ids: np.ndarray
    geometry: TraceGraphGeometry
    observed_mask: np.ndarray
    query_indices: np.ndarray
    edge_index: np.ndarray
    edge_type: np.ndarray
    edge_distances: np.ndarray
    degree: np.ndarray
    min_distance: np.ndarray
    depth: np.ndarray
    diagnostics: dict[str, int]
    neighbors_per_relation: int
    dependency_rounds: int
    relation_names: tuple[str, ...] = RELATION_NAMES
    common_edge_distances: np.ndarray | None = None

    @property
    def coverage(self) -> np.ndarray:
        """Return float32 ``[N, 4, 2]`` containing degree/k and minimum D."""
        return np.stack(
            (self.degree / self.neighbors_per_relation, self.min_distance), axis=-1
        ).astype(np.float32)


def build_trace_graph_subgraph(
    query_geometry: TraceGraphGeometry,
    query_trace_ids: np.ndarray,
    candidate_geometry: TraceGraphGeometry,
    candidate_trace_ids: np.ndarray,
    observed_mask: np.ndarray,
    *,
    rounds: int,
    relation_scales_m: np.ndarray,
    allowed_mask: np.ndarray | None = None,
    neighbors_per_relation: int = 8,
    radius: float = 1.0,
    candidate_chunk_size: int = 4096,
    topology: str = "multi_relation",
    excluded_relation: str | None = None,
    common_distance_scales_m: tuple[float, float] | None = None,
    single_4d_neighbors: int = 32,
) -> TraceGraphPlan:
    """Collect exactly the incoming dependencies for ``rounds`` updates.

    Every search uses the same original candidate geometry and visibility;
    no search is restricted to the accumulated closure. A query may have an
    ID absent from candidates or share an invisible candidate's ID and exact
    geometry. Queries can never be eligible observed senders. Breadth-first
    traversal establishes minimum depth before expanding any destination.
    """
    if isinstance(rounds, (bool, np.bool_)) or not isinstance(rounds, Integral) or rounds < 1:
        raise ValueError("rounds must be a positive integer")
    search_settings = {
        "relation_scales_m": relation_scales_m,
        "allowed_mask": allowed_mask,
        "neighbors_per_relation": neighbors_per_relation,
        "radius": radius,
        "candidate_chunk_size": candidate_chunk_size,
        "topology": topology,
        "excluded_relation": excluded_relation,
        "common_distance_scales_m": common_distance_scales_m,
        "single_4d_neighbors": single_4d_neighbors,
    }
    initial_edges = select_trace_graph_neighbors(
        query_geometry,
        query_trace_ids,
        candidate_geometry,
        candidate_trace_ids,
        observed_mask,
        **search_settings,
    )
    query_ids = np.asarray(query_trace_ids, dtype=np.int64)
    candidate_ids = np.asarray(candidate_trace_ids, dtype=np.int64)
    eligible = np.asarray(observed_mask).copy()
    if allowed_mask is not None:
        eligible &= np.asarray(allowed_mask)
    candidate_rows = {int(trace_id): row for row, trace_id in enumerate(candidate_ids)}

    def search(geometry, ids):
        return select_trace_graph_neighbors(
            geometry,
            ids,
            candidate_geometry,
            candidate_ids,
            observed_mask,
            **search_settings,
        )

    return _build_dependency_plan(
        query_geometry,
        query_ids,
        candidate_geometry,
        eligible,
        rounds=rounds,
        initial_edges=initial_edges,
        search=search,
        candidate_rows=candidate_rows,
        search_settings=search_settings,
    )


class FixedTraceGraphSubgraphBuilder:
    """Reuse an exact index and ID lookup for one fixed observed domain.

    Geometry, IDs, visibility, allowed mask and settings are copied at creation.
    Build a new instance for each training episode. Plans contain no amplitudes,
    and every hop still searches the complete original visible candidate set.
    Incoming edges of eligible observed senders are computed lazily and retained
    for this builder's lifetime. Initial query searches are never cached.
    """

    def __init__(
        self,
        candidate_geometry: TraceGraphGeometry,
        candidate_trace_ids: np.ndarray,
        observed_mask: np.ndarray,
        **search_settings,
    ) -> None:
        self._index = FixedTraceGraphNeighborIndex(
            candidate_geometry, candidate_trace_ids, observed_mask, **search_settings
        )
        self._candidate_rows = {
            int(trace_id): row for row, trace_id in enumerate(self._index.trace_ids)
        }
        settings = self._index.search_settings
        single = settings["topology"] == "single_4d"
        relation_count = (
            1 if single else len(RELATION_NAMES) - int(settings["excluded_relation"] is not None)
        )
        k = settings["single_4d_neighbors"] if single else settings["neighbors_per_relation"]
        self._eligible_sender_count = int(np.count_nonzero(self._index.eligible_mask))
        self._edge_capacity = self._eligible_sender_count * relation_count * int(k)
        self._observed_neighbors: dict[int, np.ndarray] = {}
        self._cached_edge_count = 0

    def observed_neighbor_cache_info(self) -> dict[str, int]:
        """Describe the bounded cache; array bytes exclude Python container overhead."""
        return {
            "cached_sender_count": len(self._observed_neighbors),
            "cached_edge_count": self._cached_edge_count,
            "cached_array_bytes": self._cached_edge_count * _CACHED_EDGE_DTYPE.itemsize,
            "eligible_sender_count": self._eligible_sender_count,
            "edge_capacity": self._edge_capacity,
        }

    def build(
        self,
        query_geometry: TraceGraphGeometry,
        query_trace_ids: np.ndarray,
        *,
        rounds: int,
    ) -> TraceGraphPlan:
        """Build the same ordered finite closure as the brute-force public API."""
        if isinstance(rounds, (bool, np.bool_)) or not isinstance(rounds, Integral) or rounds < 1:
            raise ValueError("rounds must be a positive integer")
        initial_edges = self._index.select(query_geometry, query_trace_ids)
        return _build_dependency_plan(
            query_geometry,
            np.asarray(query_trace_ids, dtype=np.int64),
            self._index.geometry,
            self._index.eligible_mask,
            rounds=rounds,
            initial_edges=initial_edges,
            search=self._cached_observed_neighbors,
            candidate_rows=self._candidate_rows,
            search_settings=self._index.search_settings,
        )

    def _cached_observed_neighbors(self, geometry, trace_ids):
        missing = [
            row
            for row, trace_id in enumerate(trace_ids)
            if int(trace_id) not in self._observed_neighbors
        ]
        if missing:
            missing_ids = trace_ids[missing]
            for trace_id in missing_ids:
                candidate_row = self._candidate_rows.get(int(trace_id))
                if candidate_row is None or not self._index.eligible_mask[candidate_row]:
                    raise ValueError("cached incoming neighbors require eligible observed IDs")
            fresh = self._index.select(_select_geometry(geometry, missing), missing_ids)
            self._remember_observed_neighbors(missing_ids, fresh)
        records = [self._observed_neighbors[int(trace_id)] for trace_id in trace_ids]
        # Concatenation returns independent arrays even when only one record is used.
        return TraceGraphNeighbors(
            sender_ids=np.concatenate([record["sender_ids"] for record in records]),
            destination_ids=np.repeat(trace_ids, [len(record) for record in records]),
            edge_type=np.concatenate([record["edge_type"] for record in records]),
            distances=np.concatenate([record["distances"] for record in records]),
        )

    def _remember_observed_neighbors(self, trace_ids, edges):
        unique_ids, counts = np.unique(edges.destination_ids, return_counts=True)
        counts_by_id = dict(zip(unique_ids.tolist(), counts.tolist(), strict=True))
        start = 0
        for trace_id in trace_ids:
            count = counts_by_id.get(int(trace_id), 0)
            record = np.empty(count, dtype=_CACHED_EDGE_DTYPE)
            for name in _CACHED_EDGE_DTYPE.names:
                record[name] = getattr(edges, name)[start : start + count]
            record.setflags(write=False)
            self._observed_neighbors[int(trace_id)] = record
            self._cached_edge_count += count
            start += count


def _build_dependency_plan(
    query_geometry,
    query_ids,
    candidate_geometry,
    eligible,
    *,
    rounds,
    initial_edges,
    search,
    candidate_rows,
    search_settings,
):
    query_rows = {int(trace_id): row for row, trace_id in enumerate(query_ids)}
    for trace_id, query_row in query_rows.items():
        if trace_id in candidate_rows:
            candidate_row = candidate_rows[trace_id]
            if eligible[candidate_row]:
                raise ValueError("query IDs must not be eligible observed senders")
            for field in fields(TraceGraphGeometry):
                if not np.array_equal(
                    getattr(query_geometry, field.name)[query_row],
                    getattr(candidate_geometry, field.name)[candidate_row],
                ):
                    raise ValueError("shared query/candidate ID must have identical geometry")

    depth_by_id = {int(trace_id): 0 for trace_id in query_ids}
    frontier = query_ids
    edge_blocks = []
    for depth in range(rounds):
        if not len(frontier):
            break
        if depth == 0:
            edges = initial_edges
        else:
            edges = search(
                _select_geometry(candidate_geometry, [candidate_rows[int(i)] for i in frontier]),
                frontier,
            )
        edge_blocks.append(edges)
        next_frontier = []
        for trace_id in np.unique(edges.sender_ids):
            if int(trace_id) not in depth_by_id:
                depth_by_id[int(trace_id)] = depth + 1
                next_frontier.append(trace_id)
        frontier = np.asarray(next_frontier, dtype=np.int64)

    node_ids = np.array(sorted(depth_by_id), dtype=np.int64)
    query_indices = np.searchsorted(node_ids, query_ids)
    observed = np.ones(len(node_ids), dtype=bool)
    observed[query_indices] = False
    geometry = _assemble_geometry(
        node_ids, query_geometry, query_rows, candidate_geometry, candidate_rows
    )
    edge_index, edge_type, edge_distances = _local_edges(edge_blocks, node_ids)
    sender_rows = [candidate_rows.get(int(trace_id)) for trace_id in node_ids[edge_index[0]]]
    if not np.all(observed[edge_index[0]]) or any(
        row is None or not eligible[row] for row in sender_rows
    ):
        raise ValueError("all graph senders must belong to the original observed domain")
    degree = np.zeros((len(node_ids), 4), dtype=np.int64)
    minimum = np.full((len(node_ids), 4), np.inf, dtype=np.float64)
    np.add.at(degree, (edge_index[1], edge_type), 1)
    np.minimum.at(minimum, (edge_index[1], edge_type), edge_distances)
    minimum[degree == 0] = 0.0
    depths = np.array([depth_by_id[int(i)] for i in node_ids], dtype=np.int64)
    diagnostics = {
        "query_count": len(query_ids),
        "support_node_count": int(np.count_nonzero(observed)),
        "typed_edge_count": len(edge_type),
        "unique_pair_count": np.unique(edge_index, axis=1).shape[1],
        "max_depth": int(depths.max(initial=0)),
    }
    common_distances = None
    common_distance_scales_m = search_settings["common_distance_scales_m"]
    if common_distance_scales_m is not None:
        scales = np.asarray(common_distance_scales_m, dtype=np.float64)
        if scales.shape != (2,) or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
            raise ValueError("common_distance_scales_m must have two positive finite scales")
        sender, destination = edge_index
        midpoint_delta = geometry.midpoint_xy_m[sender] - geometry.midpoint_xy_m[destination]
        offset_delta = geometry.offset_xy_m[sender] - geometry.offset_xy_m[destination]
        common_distances = np.sqrt(
            np.sum(midpoint_delta**2, axis=1) / scales[0] ** 2
            + np.sum(offset_delta**2, axis=1) / scales[1] ** 2
        )
    return TraceGraphPlan(
        trace_ids=node_ids,
        geometry=geometry,
        observed_mask=observed,
        query_indices=query_indices,
        edge_index=edge_index,
        edge_type=edge_type,
        edge_distances=edge_distances,
        degree=degree,
        min_distance=minimum,
        depth=depths,
        diagnostics=diagnostics,
        neighbors_per_relation=int(
            search_settings["single_4d_neighbors"]
            if search_settings["topology"] == "single_4d"
            else search_settings["neighbors_per_relation"]
        ),
        dependency_rounds=int(rounds),
        relation_names=("untyped",)
        if search_settings["topology"] == "single_4d"
        else RELATION_NAMES,
        common_edge_distances=common_distances,
    )


def _select_geometry(geometry: TraceGraphGeometry, rows: list[int]) -> TraceGraphGeometry:
    return TraceGraphGeometry(
        **{field.name: getattr(geometry, field.name)[rows] for field in fields(TraceGraphGeometry)}
    )


def _assemble_geometry(
    node_ids: np.ndarray,
    queries: TraceGraphGeometry,
    query_rows: dict[int, int],
    candidates: TraceGraphGeometry,
    candidate_rows: dict[int, int],
) -> TraceGraphGeometry:
    arrays = {}
    for field in fields(TraceGraphGeometry):
        query_array = getattr(queries, field.name)
        candidate_array = getattr(candidates, field.name)
        array = np.empty((len(node_ids), *query_array.shape[1:]), dtype=query_array.dtype)
        for row, trace_id in enumerate(node_ids):
            if int(trace_id) in query_rows:
                array[row] = query_array[query_rows[int(trace_id)]]
            else:
                array[row] = candidate_array[candidate_rows[int(trace_id)]]
        arrays[field.name] = array
    return TraceGraphGeometry(**arrays)


def _local_edges(
    blocks: list[TraceGraphNeighbors], node_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not blocks:
        return np.empty((2, 0), dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0)
    sender = np.concatenate([block.sender_ids for block in blocks])
    destination = np.concatenate([block.destination_ids for block in blocks])
    edge_type = np.concatenate([block.edge_type for block in blocks])
    distances = np.concatenate([block.distances for block in blocks])
    order = np.lexsort((sender, distances, edge_type, destination))
    edge_index = np.stack(
        (np.searchsorted(node_ids, sender[order]), np.searchsorted(node_ids, destination[order]))
    )
    return edge_index, edge_type[order], distances[order]
