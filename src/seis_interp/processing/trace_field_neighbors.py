"""Nearest observed traces in whatever ranking space the caller supplies."""

from numbers import Integral

import numpy as np


def nearest_observed_positions(
    coordinates, observed_rows, query_rows, count, *, groups=None, chunk_size=512
):
    """Positions into ``observed_rows`` of each query's ``count`` nearest observed traces.

    Distance is Euclidean in the supplied coordinates. The caller chooses that
    space, and a trace field passes the coordinates it reads itself, so a
    neighbourhood is a property of the declared geometry alone and never of a
    waveform. A query that is itself observed excludes itself, so a fit never
    reconstructs a trace from its own amplitudes. ``groups`` widens that exclusion
    to every observed candidate sharing the query's group, which a trace field uses
    to drop a whole source gather so that a fit sees the neighbourhood a query with
    a missing gather will see. Ties break on the observed position, which follows
    the preparation's cell order.

    One query chunk competes against all observed traces at a time, so temporary
    storage is ``chunk_size`` by the observed count rather than a survey square.
    """
    coordinates = np.asarray(coordinates, dtype=np.float64)
    observed_rows = np.asarray(observed_rows, dtype=np.int64)
    query_rows = np.asarray(query_rows, dtype=np.int64)
    if coordinates.ndim != 2 or not coordinates.shape[1] or not np.isfinite(coordinates).all():
        raise ValueError("coordinates must be a finite (trace, feature) table")
    for rows, name in ((observed_rows, "observed_rows"), (query_rows, "query_rows")):
        if rows.ndim != 1 or not len(rows) or np.any(rows < 0) or np.any(rows >= len(coordinates)):
            raise ValueError(f"{name} must index the coordinate table")
    if len(np.unique(observed_rows)) != len(observed_rows):
        raise ValueError("observed_rows must be distinct")
    count = _positive_integer(count, "count")
    chunk_size = _positive_integer(chunk_size, "chunk_size")
    if count > len(observed_rows) - 1:
        raise ValueError("count must leave one observed trace after excluding the query itself")
    if groups is not None:
        groups = np.asarray(groups)
        if groups.shape != (len(coordinates),):
            raise ValueError("groups must label every coordinate row")

    observed_coordinates = coordinates[observed_rows]
    positions = np.empty((len(query_rows), count), dtype=np.int64)
    order = np.arange(len(observed_rows), dtype=np.int64)
    for start in range(0, len(query_rows), chunk_size):
        rows = query_rows[start : start + chunk_size]
        deltas = coordinates[rows][:, None, :] - observed_coordinates[None, :, :]
        distances = np.einsum("qoc,qoc->qo", deltas, deltas)
        excluded = (
            observed_rows[None, :] == rows[:, None]
            if groups is None
            else groups[observed_rows][None, :] == groups[rows][:, None]
        )
        distances[excluded] = np.inf
        if np.any((~excluded).sum(axis=1) < count):
            raise ValueError("a query has fewer admissible observed candidates than count")
        nearest = np.argpartition(distances, count - 1, axis=1)[:, :count]
        chosen = np.take_along_axis(distances, nearest, axis=1)
        # Stable lexicographic order: distance first, then the observed position.
        ranking = np.lexsort((order[nearest], chosen), axis=1)
        positions[start : start + len(rows)] = np.take_along_axis(nearest, ranking, axis=1)
    return positions


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)
