"""Frozen physical trace predictions from one fixed observed domain."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral
from time import perf_counter

import numpy as np
import torch

from seis_interp.data.masked_trace_source import (
    MaskedTraceSource,
    assemble_masked_trace_graph_inputs,
)
from seis_interp.data.trace_graph_domain import TraceGraphDomain
from seis_interp.models.relational_trace_graph import RelationalTraceGraphInterpolator
from seis_interp.processing.trace_graph_diagnostics import trace_graph_resource_measurements
from seis_interp.processing.trace_graph_geometry import RELATION_NAMES, compute_trace_graph_geometry
from seis_interp.processing.trace_graph_neighbor_table import TraceGraphNeighborTable
from seis_interp.processing.trace_graph_preprocessing import TraceGraphPreprocessing
from seis_interp.processing.trace_graph_settings import TraceGraphSettings
from seis_interp.processing.trace_graph_subgraphs import (
    FixedTraceGraphSubgraphBuilder,
    build_trace_graph_subgraph,
)
from seis_interp.training.phase_progress import PhaseProgress


@dataclass(frozen=True)
class TraceGraphPrediction:
    """Physical waveforms and geometry in the caller's query order."""

    query_trace_ids: np.ndarray
    source_xy_m: np.ndarray
    receiver_xy_m: np.ndarray
    prediction: np.ndarray
    has_observed_context: np.ndarray
    diagnostics: dict[str, object]


def predict_relational_trace_graph(
    model: RelationalTraceGraphInterpolator,
    domain: TraceGraphDomain,
    preprocessing: TraceGraphPreprocessing,
    *,
    graph_settings: TraceGraphSettings,
    query_trace_ids: np.ndarray | None = None,
    query_source_xy_m: np.ndarray | None = None,
    query_receiver_xy_m: np.ndarray | None = None,
    query_batch_size: int = 64,
    amplitudes: np.ndarray | None = None,
    observed_waveforms: np.ndarray | None = None,
    device: torch.device | str = "cpu",
    measure_resources: bool = True,
    neighbor_table: TraceGraphNeighborTable | None = None,
    reporter=None,
    phase="primary",
) -> TraceGraphPrediction:
    """Predict selected or arbitrary queries without reading their waveforms.

    By default, predict all domain queries. Explicit source/receiver arrays and
    unique IDs allow off-grid queries with no array-row mapping. To use a domain
    with no file rows at all, supply ``observed_waveforms[observed_count,T]`` in
    domain observed order; otherwise only required support file rows are read.
    Time samples must exactly match the fixed training preprocessing.

    The observed domain and search settings stay fixed across batches. Graph
    counts describe processed batch instances; gate sums and query counts can
    be accumulated across arbitrary query splits without counting support.
    An optional ``neighbor_table`` (for example the training table over the
    observed traces) serves support searches; predictions are unchanged.
    """
    return _predict(
        [model],
        domain,
        preprocessing,
        graph_settings=graph_settings,
        query_trace_ids=query_trace_ids,
        query_source_xy_m=query_source_xy_m,
        query_receiver_xy_m=query_receiver_xy_m,
        query_batch_size=query_batch_size,
        amplitudes=amplitudes,
        observed_waveforms=observed_waveforms,
        device=device,
        measure_resources=measure_resources,
        neighbor_table=neighbor_table,
        reporter=reporter,
        phases=[phase],
    )[0]


def predict_relational_trace_graph_states(
    model: RelationalTraceGraphInterpolator,
    states: Sequence[Mapping[str, torch.Tensor]],
    domain: TraceGraphDomain,
    preprocessing: TraceGraphPreprocessing,
    **options,
) -> list[TraceGraphPrediction]:
    """One prediction per state dict, from graphs and inputs built once per batch.

    Each result equals ``predict_relational_trace_graph`` of ``model`` with that
    state loaded; ``model`` itself keeps its weights. ``options`` are the keyword
    arguments of ``predict_relational_trace_graph``. Timings other than the
    forward pass describe the shared preparation.
    """
    if not isinstance(model, RelationalTraceGraphInterpolator):
        raise TypeError("model must be a RelationalTraceGraphInterpolator")
    models = []
    for state in states:
        copy = deepcopy(model)
        copy.load_state_dict(state)
        models.append(copy)
    return _predict(models, domain, preprocessing, **options)


def _predict(
    models: list[RelationalTraceGraphInterpolator],
    domain: TraceGraphDomain,
    preprocessing: TraceGraphPreprocessing,
    *,
    graph_settings: TraceGraphSettings,
    query_trace_ids: np.ndarray | None = None,
    query_source_xy_m: np.ndarray | None = None,
    query_receiver_xy_m: np.ndarray | None = None,
    query_batch_size: int = 64,
    amplitudes: np.ndarray | None = None,
    observed_waveforms: np.ndarray | None = None,
    device: torch.device | str = "cpu",
    measure_resources: bool = True,
    neighbor_table: TraceGraphNeighborTable | None = None,
    reporter=None,
    phases=None,
) -> list[TraceGraphPrediction]:
    if not models or any(
        not isinstance(model, RelationalTraceGraphInterpolator) for model in models
    ):
        raise TypeError("model must be a RelationalTraceGraphInterpolator")
    progress = PhaseProgress(reporter)
    phases = phases if phases is not None else ["primary"] * len(models)
    model = models[0]
    if not isinstance(graph_settings, TraceGraphSettings):
        raise TypeError("graph_settings must be TraceGraphSettings")
    graph_settings.validate_model_config(model.constructor_config())
    if (
        isinstance(query_batch_size, bool)
        or not isinstance(query_batch_size, Integral)
        or query_batch_size < 1
    ):
        raise ValueError("query_batch_size must be a positive integer")
    if not np.array_equal(domain.time_s, preprocessing.time_s):
        raise ValueError("observed time_s must match the fixed preprocessing time grid")
    query_ids, source_xy, receiver_xy = _query_geometry(
        domain, query_trace_ids, query_source_xy_m, query_receiver_xy_m
    )
    observed_ids = domain.trace_ids[domain.observed_mask]
    source = None
    if observed_waveforms is None:
        source = MaskedTraceSource(domain, preprocessing, amplitudes, device=device)
    else:
        if amplitudes is not None:
            raise ValueError("supply either amplitudes or observed_waveforms")
        if observed_waveforms.shape != (len(observed_ids), len(domain.time_s)):
            raise ValueError("observed_waveforms must have shape [observed traces, T]")
        if observed_waveforms.dtype != np.float32:
            raise ValueError("observed_waveforms must have dtype float32")

    if not isinstance(measure_resources, bool):
        raise ValueError("measure_resources must be boolean")
    requested_device = torch.device(device)
    timings = {
        name: 0.0
        for name in (
            "graph_build_seconds",
            "observed_read_seconds",
            "input_assembly_seconds",
            "forward_seconds",
        )
    }
    geometry_started = perf_counter() if measure_resources else None
    candidate_geometry = compute_trace_graph_geometry(
        domain.source_xy_m,
        domain.receiver_xy_m,
        azimuth_min_offset_m=preprocessing.azimuth_min_offset_m,
    )
    if neighbor_table is not None and graph_settings.neighbor_search != "exact_index":
        raise ValueError("neighbor_table requires neighbor_search=exact_index")
    builder = (
        FixedTraceGraphSubgraphBuilder(
            candidate_geometry,
            domain.trace_ids,
            domain.observed_mask,
            neighbor_table=neighbor_table,
            **graph_settings.subgraph_kwargs(),
        )
        if graph_settings.neighbor_search == "exact_index"
        else None
    )
    if measure_resources:
        timings["graph_build_seconds"] += perf_counter() - geometry_started
    rounds = model.message_passing_rounds
    relation_count = 4 if model.method_variant == "relational" else 1
    outputs = [
        _PredictionOutput.empty(len(query_ids), len(domain.time_s), rounds, relation_count)
        for _ in models
    ]
    forward_seconds = [0.0 for _ in models]
    diagnostics = {
        "query_count": len(query_ids),
        "query_batch_count": 0,
        "processed_support_node_count": 0,
        "processed_typed_edge_count": 0,
        "processed_unique_pair_count": 0,
        "processed_message_edge_count": 0,
        "max_support_node_count": 0,
        "gate_relation_names": list(RELATION_NAMES) if relation_count == 4 else ["untyped"],
    }
    was_training = [member.training for member in models]
    for member in models:
        member.to(device)
        member.eval()
    try:
        with torch.inference_mode():
            for start in range(0, len(query_ids), query_batch_size):
                stop = min(start + query_batch_size, len(query_ids))
                started = perf_counter() if measure_resources else None
                geometry = compute_trace_graph_geometry(
                    source_xy[start:stop],
                    receiver_xy[start:stop],
                    azimuth_min_offset_m=preprocessing.azimuth_min_offset_m,
                )
                if builder is None:
                    plan = build_trace_graph_subgraph(
                        geometry,
                        query_ids[start:stop],
                        candidate_geometry,
                        domain.trace_ids,
                        domain.observed_mask,
                        rounds=rounds,
                        **graph_settings.subgraph_kwargs(),
                    )
                else:
                    plan = builder.build(geometry, query_ids[start:stop], rounds=rounds)
                progress(
                    f"building prediction graph {start}/{len(query_ids)} cells "
                    f"({', '.join(phases)})"
                )
                if measure_resources:
                    timings["graph_build_seconds"] += perf_counter() - started
                if source is not None:
                    support_ids = plan.trace_ids[plan.observed_mask]
                    started = perf_counter() if measure_resources else None
                    support_waveforms = source.read_observed_rows(support_ids)
                    if measure_resources:
                        timings["observed_read_seconds"] += perf_counter() - started
                else:
                    support_ids, support_waveforms = observed_ids, observed_waveforms
                started = perf_counter() if measure_resources else None
                inputs = assemble_masked_trace_graph_inputs(
                    plan,
                    observed_trace_ids=support_ids,
                    observed_waveforms=support_waveforms,
                    time_s=domain.time_s,
                    preprocessing=preprocessing,
                    device=device,
                )
                if measure_resources:
                    _synchronize(requested_device)
                    timings["input_assembly_seconds"] += perf_counter() - started
                for index, member in enumerate(models):
                    batch_diagnostics = {}
                    started = perf_counter() if measure_resources else None
                    normalized, flags = member(inputs, diagnostics=batch_diagnostics)
                    if measure_resources:
                        _synchronize(requested_device)
                        forward_seconds[index] += perf_counter() - started
                    values = (
                        normalized.cpu().numpy().astype(np.float64) * preprocessing.amplitude_scale
                    )
                    if not np.all(np.isfinite(values)):
                        raise ValueError("model predictions must be finite")
                    outputs[index].prediction[start:stop] = values
                    outputs[index].context[start:stop] = flags.cpu().numpy()
                    progress(f"predicting {stop}/{len(query_ids)} cells ({phases[index]})")
                    for name, total in outputs[index].totals.items():
                        total += batch_diagnostics[name].cpu().numpy()
                diagnostics["query_batch_count"] += 1
                for name in ("support_node_count", "typed_edge_count", "unique_pair_count"):
                    diagnostics[f"processed_{name}"] += plan.diagnostics[name]
                diagnostics["processed_message_edge_count"] += plan.diagnostics[
                    "typed_edge_count" if relation_count == 4 else "unique_pair_count"
                ]
                diagnostics["max_support_node_count"] = max(
                    diagnostics["max_support_node_count"], plan.diagnostics["support_node_count"]
                )
    finally:
        for member, mode in zip(models, was_training, strict=True):
            member.train(mode)
    if measure_resources and source is None:
        timings["observed_read_seconds"] = None
    resources = trace_graph_resource_measurements(requested_device) if measure_resources else None
    results = []
    for output, seconds in zip(outputs, forward_seconds, strict=True):
        record = {
            **diagnostics,
            **{name: values.tolist() for name, values in output.totals.items()},
        }
        if measure_resources:
            record["timings"] = {**timings, "forward_seconds": seconds}
            record["resources"] = resources
        record["gate_interpretation"] = "model_weights_not_causal_importance"
        results.append(
            TraceGraphPrediction(
                query_trace_ids=query_ids,
                source_xy_m=source_xy,
                receiver_xy_m=receiver_xy,
                prediction=output.prediction,
                has_observed_context=output.context,
                diagnostics=record,
            )
        )
    return results


@dataclass(frozen=True)
class _PredictionOutput:
    """Accumulators of one weight set across query batches."""

    prediction: np.ndarray
    context: np.ndarray
    totals: dict[str, np.ndarray]

    @classmethod
    def empty(
        cls, query_count: int, time_count: int, rounds: int, relation_count: int
    ) -> _PredictionOutput:
        return cls(
            np.empty((query_count, time_count), dtype=np.float32),
            np.empty(query_count, dtype=bool),
            {
                "gate_sum": np.zeros((rounds, relation_count), dtype=np.float64),
                "available_query_count": np.zeros((rounds, relation_count), dtype=np.int64),
                "context_query_count": np.zeros(rounds, dtype=np.int64),
                "no_context_query_count": np.zeros(rounds, dtype=np.int64),
            },
        )


def _query_geometry(
    domain: TraceGraphDomain,
    trace_ids: np.ndarray | None,
    source_xy_m: np.ndarray | None,
    receiver_xy_m: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    explicit = source_xy_m is not None or receiver_xy_m is not None
    if explicit and (source_xy_m is None or receiver_xy_m is None or trace_ids is None):
        raise ValueError("arbitrary queries require IDs and both source/receiver coordinates")
    ids = domain.trace_ids[domain.query_mask] if trace_ids is None else np.asarray(trace_ids)
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or len(np.unique(ids)) != len(ids):
        raise ValueError("query_trace_ids must be a unique integer vector")
    ids = ids.astype(np.int64, copy=True)
    positions = {int(trace_id): row for row, trace_id in enumerate(domain.trace_ids)}
    if explicit:
        geometry = compute_trace_graph_geometry(source_xy_m, receiver_xy_m, azimuth_min_offset_m=0)
        source = geometry.source_xy_m
        receiver = geometry.receiver_xy_m
        if len(source) != len(ids):
            raise ValueError("query coordinates must contain one pair per query_trace_id")
    else:
        if any(int(trace_id) not in positions for trace_id in ids):
            raise ValueError(
                "queries outside the domain require explicit source/receiver coordinates"
            )
        rows = np.array([positions[int(trace_id)] for trace_id in ids], dtype=np.int64)
        source = domain.source_xy_m[rows].copy()
        receiver = domain.receiver_xy_m[rows].copy()
    for index, trace_id in enumerate(ids):
        row = positions.get(int(trace_id))
        if row is not None:
            if not domain.query_mask[row]:
                raise ValueError("domain query IDs must belong to the authorized query set")
            if not np.array_equal(source[index], domain.source_xy_m[row]) or not np.array_equal(
                receiver[index], domain.receiver_xy_m[row]
            ):
                raise ValueError("query coordinates do not match the domain trace ID")
    pairs = np.column_stack((source, receiver))
    if len(np.unique(pairs, axis=0)) != len(ids):
        raise ValueError("queries must have unique physical source/receiver pairs")
    observed_pairs = {
        tuple(pair)
        for pair in np.column_stack(
            (domain.source_xy_m[domain.observed_mask], domain.receiver_xy_m[domain.observed_mask])
        )
    }
    if any(tuple(pair) in observed_pairs for pair in pairs):
        raise ValueError("query duplicates an observed physical source/receiver pair")
    return ids, source, receiver


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
