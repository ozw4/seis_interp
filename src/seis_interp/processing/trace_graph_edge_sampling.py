"""Stable, uniform fixed fanout from visibility-filtered candidate neighbors."""

from numbers import Integral

import numpy as np

from seis_interp.processing.trace_graph_geometry import RELATION_NAMES
from seis_interp.processing.trace_graph_neighbors import TraceGraphNeighbors


def sample_trace_graph_relation_fanout(
    neighbors: TraceGraphNeighbors,
    *,
    fanout_per_relation: int,
    rng: np.random.Generator,
) -> TraceGraphNeighbors:
    """Sample each contiguous destination/relation group without replacement.

    Candidates follow the neighbor selector's destination/relation/distance/ID
    order. Output is an independent subsequence in that order. Groups already
    within fanout consume no randomness; all random state belongs to the caller.
    """
    if (
        isinstance(fanout_per_relation, (bool, np.bool_))
        or not isinstance(fanout_per_relation, Integral)
        or fanout_per_relation < 1
    ):
        raise ValueError("fanout_per_relation must be a positive integer")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy.random.Generator")
    arrays = (
        neighbors.sender_ids,
        neighbors.destination_ids,
        neighbors.edge_type,
        neighbors.distances,
    )
    for array, dtype in zip(arrays, (np.int64, np.int64, np.int64, np.float64), strict=True):
        if not isinstance(array, np.ndarray) or array.ndim != 1 or array.dtype != dtype:
            raise ValueError(
                "neighbor arrays must be vectors with int64 IDs/types and float64 distances"
            )
    count = len(arrays[0])
    if any(len(array) != count for array in arrays):
        raise ValueError("neighbor arrays must have matching lengths")
    if np.any((neighbors.edge_type < 0) | (neighbors.edge_type >= len(RELATION_NAMES))):
        raise ValueError("edge_type must identify an existing relation")
    boundaries = (
        np.flatnonzero(
            (neighbors.destination_ids[1:] != neighbors.destination_ids[:-1])
            | (neighbors.edge_type[1:] != neighbors.edge_type[:-1])
        )
        + 1
    )
    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [count]))
    keep = np.ones(count, dtype=bool)
    for start, stop in zip(starts, stops, strict=True):
        if stop - start > fanout_per_relation:
            keep[start:stop] = False
            selected = rng.choice(stop - start, size=fanout_per_relation, replace=False)
            keep[start + selected] = True
    return TraceGraphNeighbors(*(array[keep] for array in arrays))
