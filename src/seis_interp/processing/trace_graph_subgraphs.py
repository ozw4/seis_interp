"""Finite observed dependency closures for synchronous trace graph rounds."""

from __future__ import annotations

from dataclasses import dataclass, fields
from numbers import Integral

import numpy as np

from seis_interp.processing.trace_graph_geometry import TraceGraphGeometry
from seis_interp.processing.trace_graph_neighbors import (
    TraceGraphNeighbors,
    select_trace_graph_neighbors,
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
    query_rows = {int(trace_id): row for row, trace_id in enumerate(query_ids)}
    candidate_rows = {int(trace_id): row for row, trace_id in enumerate(candidate_ids)}
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
            edges = select_trace_graph_neighbors(
                _select_geometry(candidate_geometry, [candidate_rows[int(i)] for i in frontier]),
                frontier,
                candidate_geometry,
                candidate_ids,
                observed_mask,
                **search_settings,
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
    if not np.all(observed[edge_index[0]]) or not np.all(
        np.isin(node_ids[edge_index[0]], candidate_ids[eligible])
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
        neighbors_per_relation=int(neighbors_per_relation),
        dependency_rounds=int(rounds),
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
