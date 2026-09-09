"""Bounded exact-graph and inference measurements without starting training."""

from __future__ import annotations

from numbers import Integral, Real
from time import perf_counter

import numpy as np
import torch

from seis_interp.data.trace_graph_domain import TraceGraphDomain
from seis_interp.data.trace_graph_prediction_store import trace_graph_query_positions
from seis_interp.evaluation.trace_graph_diagnostic_metrics import (
    TraceGraphDiagnosticBands,
    evaluate_trace_graph_baselines,
    evaluate_trace_graph_diagnostic_bands,
)
from seis_interp.evaluation.trace_graph_metrics import evaluate_trace_graph_prediction
from seis_interp.models.relational_trace_graph import RelationalTraceGraphInterpolator
from seis_interp.processing.trace_graph_diagnostics import (
    summarize_trace_graph_queries,
    trace_graph_resource_measurements,
)
from seis_interp.processing.trace_graph_geometry import compute_trace_graph_geometry
from seis_interp.processing.trace_graph_preprocessing import TraceGraphPreprocessing
from seis_interp.processing.trace_graph_settings import TraceGraphSettings
from seis_interp.processing.trace_graph_subgraphs import (
    FixedTraceGraphSubgraphBuilder,
    build_trace_graph_subgraph,
)
from seis_interp.training.relational_trace_graph_prediction import predict_relational_trace_graph


def run_trace_graph_preflight(
    model: RelationalTraceGraphInterpolator,
    domain: TraceGraphDomain,
    preprocessing: TraceGraphPreprocessing,
    *,
    graph_settings: TraceGraphSettings,
    query_trace_ids: np.ndarray,
    query_limit: int = 32,
    query_batch_size: int = 8,
    amplitudes: np.ndarray | None = None,
    observed_waveforms: np.ndarray | None = None,
    device: torch.device | str = "cpu",
    evaluate_baselines: bool = False,
    bands: TraceGraphDiagnosticBands | None = None,
    max_graph_seconds: float | None = None,
    max_process_rss_bytes: int | None = None,
    max_cuda_allocated_bytes: int | None = None,
) -> dict[str, object]:
    """Measure explicit queries and report budget blockers without approximation.

    The query selection is mandatory and never silently truncated. Geometry
    and model timing do not need labels. Optional physical scoring happens
    only after observed-only prediction; it requires mapped benchmark targets.
    Memory counters are measured process/device lifetime peaks, not estimates
    of a full survey or isolated operation allocations.
    """
    if isinstance(query_limit, bool) or not isinstance(query_limit, Integral) or query_limit < 1:
        raise ValueError("query_limit must be a positive integer")
    ids = np.asarray(query_trace_ids)
    if ids.ndim != 1 or not 0 < len(ids) <= query_limit:
        raise ValueError(
            "preflight requires an explicit nonempty query selection within query_limit"
        )
    rows = trace_graph_query_positions(domain, ids)
    limits = {
        "graph_build_seconds": max_graph_seconds,
        "process_max_rss_bytes": max_process_rss_bytes,
        "cuda_max_memory_allocated_bytes": max_cuda_allocated_bytes,
    }
    for name, limit in limits.items():
        if limit is not None and (
            isinstance(limit, bool)
            or not isinstance(limit, Real)
            or not np.isfinite(limit)
            or limit <= 0
        ):
            raise ValueError(f"{name} preflight limit must be finite and positive")
    if not isinstance(evaluate_baselines, bool):
        raise ValueError("evaluate_baselines must be boolean")
    if not np.array_equal(domain.time_s, preprocessing.time_s):
        raise ValueError("preflight time_s must match the fixed preprocessing time grid")
    graph_settings.validate_model_config(model.constructor_config())
    result = {
        "status": "success",
        "blockers": [],
        "query_trace_ids": ids.astype(np.int64).tolist(),
        "query_limit": int(query_limit),
        "geometry": None,
        "geometry_graph_seconds": None,
        "prediction_diagnostics": None,
        "resources": trace_graph_resource_measurements(device),
        "limits": limits,
        "scope": "specified_queries_exact_search_no_training",
    }
    started = perf_counter()
    try:
        candidates = compute_trace_graph_geometry(
            domain.source_xy_m,
            domain.receiver_xy_m,
            azimuth_min_offset_m=preprocessing.azimuth_min_offset_m,
        )
        queries = compute_trace_graph_geometry(
            domain.source_xy_m[rows],
            domain.receiver_xy_m[rows],
            azimuth_min_offset_m=preprocessing.azimuth_min_offset_m,
        )
        if graph_settings.neighbor_search == "exact_index":
            builder = FixedTraceGraphSubgraphBuilder(
                candidates,
                domain.trace_ids,
                domain.observed_mask,
                **graph_settings.subgraph_kwargs(),
            )
            plan = builder.build(queries, ids, rounds=model.message_passing_rounds)
        else:
            plan = build_trace_graph_subgraph(
                queries,
                ids,
                candidates,
                domain.trace_ids,
                domain.observed_mask,
                rounds=model.message_passing_rounds,
                **graph_settings.subgraph_kwargs(),
            )
    except (MemoryError, torch.cuda.OutOfMemoryError) as error:
        return _memory_blocked(result, error, "geometry", device)
    geometry_graph_seconds = perf_counter() - started
    result.update(
        geometry=summarize_trace_graph_queries(plan),
        geometry_graph_seconds=geometry_graph_seconds,
        resources=trace_graph_resource_measurements(device),
    )
    blockers = _budget_blockers(
        {**result["resources"], "graph_build_seconds": geometry_graph_seconds}, limits
    )
    if blockers:
        result.update(status="blocked", blockers=blockers, blocked_stage="geometry")
        return result
    try:
        prediction = predict_relational_trace_graph(
            model,
            domain,
            preprocessing,
            graph_settings=graph_settings,
            query_trace_ids=ids,
            query_batch_size=query_batch_size,
            amplitudes=amplitudes,
            observed_waveforms=observed_waveforms,
            device=device,
            measure_resources=True,
        )
    except (MemoryError, torch.cuda.OutOfMemoryError) as error:
        return _memory_blocked(result, error, "prediction", device)
    measurements = {
        **prediction.diagnostics["resources"],
        "graph_build_seconds": geometry_graph_seconds
        + prediction.diagnostics["timings"]["graph_build_seconds"],
    }
    blockers = _budget_blockers(measurements, limits)
    result.update(
        prediction_diagnostics=prediction.diagnostics, resources=prediction.diagnostics["resources"]
    )
    if blockers:
        result.update(status="blocked", blockers=blockers, blocked_stage="prediction")
        return result
    if evaluate_baselines:
        result["model_metrics"] = evaluate_trace_graph_prediction(
            prediction.prediction,
            domain,
            query_trace_ids=ids,
            has_observed_context=prediction.has_observed_context,
            amplitudes=amplitudes,
        )
        result["baselines"] = evaluate_trace_graph_baselines(
            domain,
            preprocessing,
            graph_settings=graph_settings,
            query_trace_ids=ids,
            query_batch_size=query_batch_size,
            amplitudes=amplitudes,
            observed_waveforms=observed_waveforms,
        )
        if bands is not None:
            result["error_bands"] = evaluate_trace_graph_diagnostic_bands(
                prediction.prediction,
                domain,
                query_trace_ids=ids,
                bands=bands,
                azimuth_min_offset_m=preprocessing.azimuth_min_offset_m,
                amplitudes=amplitudes,
            )
    return result


def _budget_blockers(measurements: dict[str, object], limits: dict[str, object]) -> list[dict]:
    blockers = []
    for name, limit in limits.items():
        if limit is None:
            continue
        measured = measurements[name]
        if measured is None or measured > limit:
            blockers.append(
                {
                    "measurement": name,
                    "measured": measured,
                    "limit": limit,
                    "reason": "unmeasured" if measured is None else "measured_limit_exceeded",
                }
            )
    return blockers


def _memory_blocked(result: dict, error: Exception, stage: str, device: object) -> dict:
    result.update(
        status="blocked",
        blocked_stage=stage,
        blockers=[
            {"reason": type(error).__name__, "measurement": "allocation_failed", "measured": None}
        ],
        resources=trace_graph_resource_measurements(device),
    )
    return result
