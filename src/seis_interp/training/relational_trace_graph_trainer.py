"""Masked query MSE optimization and fixed physical validation selection."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral, Real
from time import perf_counter

import numpy as np
import torch

from seis_interp.data.masked_trace_source import MaskedTraceSource
from seis_interp.data.trace_graph_domain import TraceGraphDomain
from seis_interp.evaluation.trace_graph_diagnostic_metrics import (
    TraceGraphDiagnosticBands,
    evaluate_trace_graph_diagnostic_bands,
    summarize_trace_graph_prediction_bands,
)
from seis_interp.evaluation.trace_graph_metrics import evaluate_trace_graph_prediction
from seis_interp.models.relational_trace_graph import RelationalTraceGraphInterpolator
from seis_interp.processing.trace_graph_geometry import compute_trace_graph_geometry
from seis_interp.processing.trace_graph_preprocessing import TraceGraphPreprocessing
from seis_interp.processing.trace_graph_settings import TraceGraphSettings
from seis_interp.processing.trace_graph_subgraphs import (
    FixedTraceGraphSubgraphBuilder,
    build_trace_graph_subgraph,
)
from seis_interp.training.relational_trace_graph_prediction import predict_relational_trace_graph
from seis_interp.training.trace_graph_episodes import (
    TraceGraphEpisodeGenerator,
    TraceGraphEpisodeLabelReader,
    read_trace_graph_training_labels,
)
from seis_interp.training.trace_graph_losses import (
    TRACE_GRAPH_TRAINING_LOSSES,
    trace_graph_training_errors_and_loss,
)


@dataclass(frozen=True)
class RelationalTraceGraphTrainingResult:
    """Independent CPU snapshots, weighted losses, and episode completion records."""

    steps_completed: int
    episodes_completed: int
    best_step: int
    best_validation_metrics: dict[str, object]
    final_validation_metrics: dict[str, object]
    best_state_dict: dict[str, torch.Tensor]
    final_state_dict: dict[str, torch.Tensor]
    training_history: tuple[dict[str, object], ...]
    validation_history: tuple[dict[str, object], ...]
    episode_history: tuple[dict[str, object], ...]
    query_count: int
    no_context_query_count: int
    sample_count: int
    normalized_error_energy: float
    final_episode_interrupted: bool

    @property
    def train_loss(self) -> float:
        return self.normalized_error_energy / self.sample_count

    @property
    def no_context_rate(self) -> float:
        return self.no_context_query_count / self.query_count


def train_relational_trace_graph(
    model: RelationalTraceGraphInterpolator,
    training_domain: TraceGraphDomain,
    validation_domain: TraceGraphDomain,
    preprocessing: TraceGraphPreprocessing,
    *,
    graph_settings: TraceGraphSettings,
    episode_kind_probabilities: Mapping[str, float],
    missing_fractions: Sequence[float],
    random_seed: int,
    max_steps: int,
    query_batch_size: int,
    validation_interval: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float | None = None,
    validation_query_batch_size: int | None = None,
    device: torch.device | str = "cpu",
    training_amplitudes: np.ndarray | None = None,
    validation_amplitudes: np.ndarray | None = None,
    reporter: Callable[[str], None] | None = None,
    on_best_update: Callable[[dict[str, torch.Tensor], int, dict[str, object]], None] | None = None,
    diagnostic_bands: TraceGraphDiagnosticBands | None = None,
    loss: str = "masked_mse",
    cudnn_benchmark: bool = True,
) -> RelationalTraceGraphTrainingResult:
    """Train an already initialized model, selecting by target-only physical SSE.

    The caller seeds model initialization before constructing ``model``. This
    function uses ``random_seed`` only for a separate local episode RNG. The
    fixed preprocessing is never fitted here; validation labels enter scoring
    only after observed-only predictions have been produced.

    ``on_best_update(state_dict, step, metrics)`` can save each improved best
    snapshot immediately. It receives the independent CPU snapshot and its
    selection metrics before validation reporting; treat them as read-only.
    Callback exceptions propagate to the caller.
    """
    if not isinstance(model, RelationalTraceGraphInterpolator):
        raise TypeError("model must be a RelationalTraceGraphInterpolator")
    graph_settings.validate_model_config(model.constructor_config())
    if not isinstance(loss, str) or loss not in TRACE_GRAPH_TRAINING_LOSSES:
        raise ValueError(f"loss must be one of {TRACE_GRAPH_TRAINING_LOSSES}")
    if not isinstance(cudnn_benchmark, bool):
        raise ValueError("cudnn_benchmark must be a boolean")
    steps = _positive_integer(max_steps, "max_steps")
    batch_size = _positive_integer(query_batch_size, "query_batch_size")
    interval = _positive_integer(validation_interval, "validation_interval")
    validation_batch_size = (
        batch_size
        if validation_query_batch_size is None
        else _positive_integer(validation_query_batch_size, "validation_query_batch_size")
    )
    rate = _finite_number(learning_rate, "learning_rate", positive=True)
    decay = _finite_number(weight_decay, "weight_decay", positive=False)
    clip = (
        None
        if gradient_clip_norm is None
        else _finite_number(gradient_clip_norm, "gradient_clip_norm", positive=True)
    )
    if (
        validation_domain.inputs_lock.get("partition") != "validation"
        or validation_domain.pool is not None
    ):
        raise ValueError("fixed validation requires a validation partition case, never train/test")
    if not np.any(validation_domain.query_mask):
        raise ValueError("fixed validation must have evaluation target queries")
    if not np.array_equal(validation_domain.time_s, preprocessing.time_s):
        raise ValueError("validation time_s must match the fixed preprocessing time grid")
    episodes = TraceGraphEpisodeGenerator(
        training_domain,
        random_seed=random_seed,
        kind_probabilities=episode_kind_probabilities,
        missing_fractions=missing_fractions,
    )
    device_value = torch.device(device)
    if not cudnn_benchmark:
        torch.backends.cudnn.benchmark = False
    source = MaskedTraceSource(
        training_domain, preprocessing, training_amplitudes, device=device_value
    )
    geometry = compute_trace_graph_geometry(
        training_domain.source_xy_m,
        training_domain.receiver_xy_m,
        azimuth_min_offset_m=preprocessing.azimuth_min_offset_m,
    )
    positions = {int(trace_id): index for index, trace_id in enumerate(training_domain.trace_ids)}
    model.to(device_value)
    optimizer = torch.optim.AdamW(model.parameters(), lr=rate, weight_decay=decay)
    training_history: list[dict[str, object]] = []
    validation_history: list[dict[str, object]] = []
    episode_history: list[dict[str, object]] = []
    best_metrics = None
    best_state = None
    best_step = 0
    total_error = 0.0
    total_objective = 0.0
    total_samples = total_queries = no_context_queries = step = 0

    while step < steps:
        episode = episodes.next_episode()
        builder = (
            FixedTraceGraphSubgraphBuilder(
                geometry,
                training_domain.trace_ids,
                episode.visible_mask,
                **graph_settings.subgraph_kwargs(),
            )
            if graph_settings.neighbor_search == "exact_index"
            else None
        )
        label_reader = (
            TraceGraphEpisodeLabelReader(
                training_domain,
                episode,
                preprocessing,
                training_amplitudes,
                device=device_value,
            )
            if builder is not None
            else None
        )
        episode_queries = 0
        for query_ids in episode.query_batches(batch_size):
            graph_started = perf_counter()
            indices = np.array([positions[int(trace_id)] for trace_id in query_ids], dtype=np.int64)
            query_geometry = compute_trace_graph_geometry(
                training_domain.source_xy_m[indices],
                training_domain.receiver_xy_m[indices],
                azimuth_min_offset_m=preprocessing.azimuth_min_offset_m,
            )
            if builder is None:
                plan = build_trace_graph_subgraph(
                    query_geometry,
                    query_ids,
                    geometry,
                    training_domain.trace_ids,
                    episode.visible_mask,
                    rounds=model.message_passing_rounds,
                    **graph_settings.subgraph_kwargs(),
                )
            else:
                plan = builder.build(query_geometry, query_ids, rounds=model.message_passing_rounds)
            graph_seconds = perf_counter() - graph_started
            read_started = perf_counter()
            inputs = source.inputs(plan)
            if diagnostic_bands is not None and device_value.type == "cuda":
                torch.cuda.synchronize(device_value)
            read_seconds = perf_counter() - read_started
            labels = (
                read_trace_graph_training_labels(
                    training_domain,
                    episode,
                    query_ids,
                    preprocessing,
                    training_amplitudes,
                    device=device_value,
                )
                if label_reader is None
                else label_reader.read(query_ids)
            )
            model.train()
            optimizer.zero_grad(set_to_none=True)
            gate_diagnostics = None if diagnostic_bands is None else {}
            if diagnostic_bands is not None and device_value.type == "cuda":
                torch.cuda.synchronize(device_value)
            forward_started = perf_counter()
            prediction, context = model(inputs, diagnostics=gate_diagnostics)
            if diagnostic_bands is not None and device_value.type == "cuda":
                torch.cuda.synchronize(device_value)
            forward_seconds = perf_counter() - forward_started
            if prediction.shape != labels.shape:
                raise ValueError("training prediction must match unpadded query labels [Q, T]")
            errors, objective = trace_graph_training_errors_and_loss(prediction, labels, loss=loss)
            if not torch.isfinite(objective):
                raise RuntimeError(f"non-finite training loss at step {step + 1}")
            objective.backward()
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            batch_error = float(errors.detach().double().sum().cpu())
            batch_no_context = int((~context).sum().cpu())
            total_error += batch_error
            total_samples += labels.numel()
            total_queries += len(query_ids)
            no_context_queries += batch_no_context
            episode_queries += len(query_ids)
            training_history.append(
                {
                    "step": step,
                    "episode_id": episode.episode_id,
                    "query_count": len(query_ids),
                    "sample_count": labels.numel(),
                    "normalized_error_energy": batch_error,
                    "batch_loss": batch_error / labels.numel(),
                    "train_loss": total_error / total_samples,
                    "no_context_query_count": batch_no_context,
                }
            )
            if loss != "masked_mse":
                batch_objective = float(objective.detach().cpu())
                total_objective += batch_objective * labels.numel()
                training_history[-1].update(
                    batch_objective_loss=batch_objective,
                    objective_loss=total_objective / total_samples,
                )
            if diagnostic_bands is not None:
                scale = preprocessing.amplitude_scale
                training_history[-1]["diagnostics"] = {
                    "bands": summarize_trace_graph_prediction_bands(
                        prediction.detach().cpu().numpy().astype(np.float64) * scale,
                        labels.detach().cpu().numpy().astype(np.float64) * scale,
                        time_s=training_domain.time_s,
                        source_xy_m=training_domain.source_xy_m[indices],
                        receiver_xy_m=training_domain.receiver_xy_m[indices],
                        bands=diagnostic_bands,
                        azimuth_min_offset_m=preprocessing.azimuth_min_offset_m,
                    ),
                    "gate": {
                        name: value.cpu().tolist() for name, value in gate_diagnostics.items()
                    },
                    "gate_interpretation": "query aggregation weights, not causal importance",
                    "graph": dict(plan.diagnostics),
                    "timing_seconds": {
                        "graph_construction": graph_seconds,
                        "observed_input_loading": read_seconds,
                        "forward": forward_seconds,
                    },
                }
            if reporter is not None:
                reporter(
                    f"relational-trace-graph step {step}/{steps} episode {episode.episode_id}: "
                    f"loss={float(objective.detach().cpu()):.8g} queries={len(query_ids)} "
                    f"no_context={batch_no_context}"
                )
            if step % interval == 0 or step == steps:
                metrics = _evaluate_validation(
                    model,
                    validation_domain,
                    preprocessing,
                    graph_settings,
                    query_batch_size=validation_batch_size,
                    amplitudes=validation_amplitudes,
                    device=device_value,
                    diagnostic_bands=diagnostic_bands,
                )
                error = float(metrics["evaluation_target"]["error_energy"])
                if not np.isfinite(error):
                    raise RuntimeError("validation error energy must be finite")
                validation_history.append({"step": step, "metrics": deepcopy(metrics)})
                if (
                    best_metrics is None
                    or error < best_metrics["evaluation_target"]["error_energy"]
                ):
                    best_metrics = deepcopy(metrics)
                    best_state = _cpu_state(model)
                    best_step = step
                    if on_best_update is not None:
                        on_best_update(best_state, best_step, best_metrics)
                if reporter is not None:
                    reporter(
                        f"relational-trace-graph validation step {step}: physical_sse={error:.8g}"
                    )
            if step == steps:
                break
        episode_history.append(
            {
                "episode_id": episode.episode_id,
                "kind": episode.kind,
                "missing_fraction": episode.missing_fraction,
                "hidden_count": len(episode.hidden_trace_ids),
                "visible_count": int(episode.visible_mask.sum()),
                "query_count": episode_queries,
                "completed": episode_queries == len(episode.hidden_trace_ids),
            }
        )

    assert best_metrics is not None and best_state is not None
    return RelationalTraceGraphTrainingResult(
        steps_completed=step,
        episodes_completed=sum(bool(row["completed"]) for row in episode_history),
        best_step=best_step,
        best_validation_metrics=best_metrics,
        final_validation_metrics=deepcopy(validation_history[-1]["metrics"]),
        best_state_dict=best_state,
        final_state_dict=_cpu_state(model),
        training_history=tuple(training_history),
        validation_history=tuple(validation_history),
        episode_history=tuple(episode_history),
        query_count=total_queries,
        no_context_query_count=no_context_queries,
        sample_count=total_samples,
        normalized_error_energy=total_error,
        final_episode_interrupted=not episode_history[-1]["completed"],
    )


def _evaluate_validation(
    model: RelationalTraceGraphInterpolator,
    domain: TraceGraphDomain,
    preprocessing: TraceGraphPreprocessing,
    graph_settings: TraceGraphSettings,
    *,
    query_batch_size: int,
    amplitudes: np.ndarray | None,
    device: torch.device,
    diagnostic_bands: TraceGraphDiagnosticBands | None = None,
) -> dict[str, object]:
    prediction = predict_relational_trace_graph(
        model,
        domain,
        preprocessing,
        graph_settings=graph_settings,
        query_batch_size=query_batch_size,
        amplitudes=amplitudes,
        device=device,
    )
    metrics = evaluate_trace_graph_prediction(
        prediction.prediction,
        domain,
        query_trace_ids=prediction.query_trace_ids,
        has_observed_context=prediction.has_observed_context,
        amplitudes=amplitudes,
    )
    if diagnostic_bands is not None:
        metrics["diagnostic_bands"] = evaluate_trace_graph_diagnostic_bands(
            prediction.prediction,
            domain,
            query_trace_ids=prediction.query_trace_ids,
            bands=diagnostic_bands,
            azimuth_min_offset_m=preprocessing.azimuth_min_offset_m,
            amplitudes=amplitudes,
        )
    return metrics


def _cpu_state(model: RelationalTraceGraphInterpolator) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _finite_number(value: float, name: str, *, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return float(value)
