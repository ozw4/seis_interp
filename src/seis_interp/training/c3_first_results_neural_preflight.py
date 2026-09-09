"""Disposable neural measurements and explicitly approximate pilot run budgets."""

from __future__ import annotations

import math
from time import perf_counter

import numpy as np
import torch

from seis_interp.data.masked_trace_source import MaskedTraceSource
from seis_interp.data.trace_graph_domain import TraceGraphDomain
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
from seis_interp.training.trace_graph_episodes import (
    TraceGraphEpisodeGenerator,
    TraceGraphEpisodeLabelReader,
    read_trace_graph_training_labels,
)


def measure_c3_first_results_graph_training_batch(
    domain: TraceGraphDomain,
    preprocessing: TraceGraphPreprocessing,
    *,
    model_config: dict,
    graph_settings: TraceGraphSettings,
    training_options: dict,
    device: torch.device,
    amplitudes: np.ndarray | None = None,
) -> dict[str, object]:
    """Measure one fresh episode's first batch using native graph/label primitives.

    The full episode visibility is chosen before batching. The temporary AdamW
    update, model, and episode RNG are discarded; global Torch RNG is restored.
    """
    if domain.pool != "all_train_traces":
        raise ValueError("pilot GNN preflight requires all_train_traces")
    options = training_options
    episode = TraceGraphEpisodeGenerator(
        domain,
        random_seed=options["random_seed"],
        kind_probabilities=options["episode_kind_probabilities"],
        missing_fractions=options["missing_fractions"],
    ).next_episode()
    query_ids = next(episode.query_batches(options["query_batch_size"]))
    positions = {int(trace_id): position for position, trace_id in enumerate(domain.trace_ids)}
    rows = np.array([positions[int(trace_id)] for trace_id in query_ids], dtype=np.int64)
    devices = list(range(torch.cuda.device_count())) if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(options["random_seed"])
        model = RelationalTraceGraphInterpolator(**model_config).float().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=options["learning_rate"],
            weight_decay=options["weight_decay"],
        )
        source = MaskedTraceSource(domain, preprocessing, amplitudes, device=device)
        synchronize_preflight_device(device)
        started = perf_counter()
        geometry = compute_trace_graph_geometry(
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
                geometry,
                domain.trace_ids,
                episode.visible_mask,
                **graph_settings.subgraph_kwargs(),
            )
            plan = builder.build(queries, query_ids, rounds=model.message_passing_rounds)
        else:
            plan = build_trace_graph_subgraph(
                queries,
                query_ids,
                geometry,
                domain.trace_ids,
                episode.visible_mask,
                rounds=model.message_passing_rounds,
                **graph_settings.subgraph_kwargs(),
            )
        graph_seconds = perf_counter() - started
        started = perf_counter()
        inputs = source.inputs(plan)
        if graph_settings.neighbor_search == "exact_index":
            labels = TraceGraphEpisodeLabelReader(
                domain, episode, preprocessing, amplitudes, device=device
            ).read(query_ids)
        else:
            labels = read_trace_graph_training_labels(
                domain, episode, query_ids, preprocessing, amplitudes, device=device
            )
        synchronize_preflight_device(device)
        read_seconds = perf_counter() - started
        started = perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction, context = model(inputs)
        if prediction.shape != labels.shape:
            raise ValueError("preflight predictions must match hidden training labels")
        loss = (prediction - labels).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError("preflight training loss is nonfinite")
        synchronize_preflight_device(device)
        forward_seconds = perf_counter() - started
        started = perf_counter()
        loss.backward()
        clip = options["gradient_clip_norm"]
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
        optimizer.step()
        synchronize_preflight_device(device)
        backward_seconds = perf_counter() - started
        report = {
            "scope": "first_batch_of_one_disposable_full_train_pool_episode",
            "training_started": False,
            "smoke_optimizer_steps": 1,
            "training_seed": options["random_seed"],
            "episode_seed": options["random_seed"],
            "pool": domain.pool,
            "pool_trace_count": int(len(domain.trace_ids)),
            "visible_trace_count": int(episode.visible_mask.sum()),
            "hidden_trace_count": int(len(episode.hidden_trace_ids)),
            "mask_kind": episode.kind,
            "missing_fraction": episode.missing_fraction,
            "query_trace_ids": query_ids.tolist(),
            "query_count": int(len(query_ids)),
            "sample_count": int(labels.numel()),
            "time_samples": list(domain.time_samples),
            "time_s": domain.time_s.tolist(),
            "geometry": summarize_trace_graph_queries(plan),
            "no_context_query_count": int((~context).sum().cpu()),
            "normalized_training_loss": float(loss.detach().cpu()),
            "timings": {
                "graph_build_seconds": graph_seconds,
                "observed_and_label_read_seconds": read_seconds,
                "forward_seconds": forward_seconds,
                "backward_and_optimizer_seconds": backward_seconds,
            },
            "resources": trace_graph_resource_measurements(device),
            "state_reused_by_pilot": False,
        }
    return report


def estimate_c3_first_results_graph_budget(
    training_measurement: dict,
    validation_measurement: dict,
    *,
    max_steps: int,
    validation_interval: int,
    validation_query_batch_size: int,
    diagnostic_baselines: bool = False,
) -> dict[str, object]:
    """Extrapolate measured scopes, including native best and final prediction passes."""
    if validation_measurement["status"] != "success":
        raise ValueError("GNN estimate requires a successful validation measurement")
    query_count = validation_measurement["total_case_query_count"]
    measured_count = len(validation_measurement["sampled_query_trace_ids"])
    if measured_count < min(query_count, validation_query_batch_size):
        raise ValueError("GNN estimate requires at least one full validation batch")
    measured_batches = math.ceil(measured_count / validation_query_batch_size)
    full_batches = math.ceil(query_count / validation_query_batch_size)
    timings = validation_measurement["prediction_diagnostics"]["timings"]
    projected_validation = {
        name: value * full_batches / measured_batches
        for name, value in timings.items()
        if isinstance(value, (int, float))
    }
    validation_seconds = sum(projected_validation.values())
    train_seconds = sum(training_measurement["timings"].values()) * max_steps
    validation_passes = math.ceil(max_steps / validation_interval)
    # Native training emits its best prediction after training. IDW baselines
    # run only when optional diagnostics are configured; the pilot omits them.
    baseline_allowance = int(diagnostic_baselines)
    return {
        "kind": "linear_extrapolation_not_measured_full_run",
        "validation_query_count": query_count,
        "validation_batch_count": full_batches,
        "training_steps": max_steps,
        "trainer_validation_passes": validation_passes,
        "native_best_prediction_passes": 1,
        "native_baseline_pass_equivalent_allowance": baseline_allowance,
        "final_frozen_prediction_passes": 1,
        "estimated_training_steps_seconds": train_seconds,
        "estimated_validation_prediction_seconds": validation_seconds,
        "estimated_validation_prediction_timings": projected_validation,
        "estimated_train_action_seconds": train_seconds
        + (validation_passes + 1 + baseline_allowance) * validation_seconds,
        "estimated_final_predict_action_seconds": validation_seconds,
        "estimated_combined_seconds": train_seconds
        + (validation_passes + 2 + baseline_allowance) * validation_seconds,
        "unmeasured_costs": [
            "input verification and preprocessing",
            "checkpoint and prediction artifact writes",
            "target scoring",
            "geometry variation beyond the sampled leading query IDs",
        ],
    }


def synchronize_preflight_device(device: torch.device) -> None:
    """Finish queued device work before reporting elapsed stage time."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
