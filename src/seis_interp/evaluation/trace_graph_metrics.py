"""Evaluate completed trace predictions with target-only physical amplitudes."""

from __future__ import annotations

import math

import numpy as np

from seis_interp.data.trace_graph_domain import TraceGraphDomain
from seis_interp.data.trace_graph_prediction_store import trace_graph_query_positions
from seis_interp.evaluation.physical_amplitude_metrics import physical_amplitude_target_metrics


def evaluate_trace_graph_prediction(
    prediction: np.ndarray,
    domain: TraceGraphDomain,
    *,
    query_trace_ids: np.ndarray,
    has_observed_context: np.ndarray,
    amplitudes: np.ndarray | None = None,
    row_chunk_size: int = 1024,
) -> dict[str, object]:
    """Read labels after prediction and stream global target energies in float64.

    Query order may differ from storage order. Only mapped evaluation targets
    are read; observed rows, decoder padding, and excluded rows never enter the
    primary metric. Context-free queries remain in the overall result.
    """
    positions = trace_graph_query_positions(domain, query_trace_ids)
    if not len(positions):
        raise ValueError("evaluation requires at least one query")
    if domain.array_rows is None or np.any(domain.array_rows[positions] < 0):
        raise ValueError("evaluation queries must have mapped target array rows")
    context = np.asarray(has_observed_context)
    if context.dtype != np.bool_ or context.shape != (len(positions),):
        raise ValueError("has_observed_context must be a boolean vector with one flag per query")
    values = np.asarray(prediction)
    if values.shape != (len(positions), len(domain.time_s)) or values.dtype.kind not in "fiu":
        raise ValueError("prediction must have physical numeric shape [Q, T]")
    if not np.all(np.isfinite(values)):
        raise ValueError("prediction must contain finite physical amplitudes")
    if (
        isinstance(row_chunk_size, bool)
        or not isinstance(row_chunk_size, int)
        or row_chunk_size < 1
    ):
        raise ValueError("row_chunk_size must be a positive integer")
    if amplitudes is None:
        if domain.amplitudes_path is None:
            raise ValueError("target amplitudes or an amplitudes_path are required for evaluation")
        amplitudes = np.load(domain.amplitudes_path, mmap_mode="r", allow_pickle=False)
    rows = domain.array_rows[positions]
    start, stop = domain.time_samples
    if amplitudes.ndim != 2 or amplitudes.dtype.kind not in "fiu":
        raise ValueError("target amplitudes must have numeric shape [rows, time]")
    if stop > amplitudes.shape[1] or np.any(rows >= amplitudes.shape[0]):
        raise ValueError("target row and time selection must be inside amplitudes")

    # Accumulate trace contributions in query order, independent of read chunks.
    totals = {name: [0.0, 0.0] for name in ("all", "with_context", "without_context")}
    for offset in range(0, len(rows), row_chunk_size):
        end = min(offset + row_chunk_size, len(rows))
        reference = np.asarray(amplitudes[rows[offset:end], start:stop], dtype=np.float64)
        if reference.shape != (end - offset, len(domain.time_s)):
            raise ValueError("target row reader returned an inconsistent shape")
        if not np.all(np.isfinite(reference)):
            raise ValueError("target reference amplitudes must be finite")
        predicted = np.asarray(values[offset:end], dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            energies = np.sum(np.square(reference), axis=1, dtype=np.float64)
            errors = np.sum(np.square(reference - predicted), axis=1, dtype=np.float64)
        for index, (energy, error) in enumerate(zip(energies, errors, strict=True)):
            group = "with_context" if context[offset + index] else "without_context"
            for name in ("all", group):
                totals[name][0] += float(energy)
                totals[name][1] += float(error)
    if not all(math.isfinite(value) for pair in totals.values() for value in pair):
        raise ValueError("evaluation energies must be finite")
    no_context_count = int((~context).sum())
    return {
        "evaluation_domain": "evaluation_target",
        "amplitude_domain": "physical",
        "evaluation_target": _metrics(totals["all"], len(rows), len(domain.time_s)),
        "with_observed_context": _metrics(
            totals["with_context"], int(context.sum()), len(domain.time_s)
        ),
        "without_observed_context": _metrics(
            totals["without_context"], no_context_count, len(domain.time_s)
        ),
        "zero_context_query_count": no_context_count,
        "zero_context_query_fraction": no_context_count / len(rows),
    }


def _metrics(totals: list[float], trace_count: int, time_count: int) -> dict[str, object]:
    if trace_count == 0:
        return {
            "trace_count": 0,
            "sample_count": 0,
            "reference_energy": 0.0,
            "error_energy": 0.0,
            "snr_db": None,
            "snr_status": "no_queries",
            "rmse": None,
            "relative_l2": None,
        }
    return physical_amplitude_target_metrics(
        trace_count=trace_count,
        sample_count=trace_count * time_count,
        reference_energy=totals[0],
        error_energy=totals[1],
    )
