"""Fixed-step pseudo-mask training on the observed set of a single volume."""

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch

from seis_interp import config_values
from seis_interp.data.c3_poc_trace_graph import C3PocTraceGraphTrainingData
from seis_interp.models.relational_trace_graph import RelationalTraceGraphInterpolator
from seis_interp.processing.trace_graph_geometry import compute_trace_graph_geometry
from seis_interp.processing.trace_graph_neighbors import TraceGraphSpatialIndex
from seis_interp.processing.trace_graph_settings import TraceGraphSettings
from seis_interp.processing.trace_graph_subgraphs import (
    FixedTraceGraphSubgraphBuilder,
    build_trace_graph_subgraph,
)
from seis_interp.training.c3_poc_trace_graph_episodes import (
    PocTraceGraphEpisodeGenerator,
    PocTraceGraphEpisodeLabelReader,
    build_poc_trace_graph_context_source,
)
from seis_interp.training.mixed_precision import mixed_precision_dtype
from seis_interp.training.trace_graph_sampling import validate_trace_graph_edge_sampling
from seis_interp.training.trace_relative_loss import masked_trace_loss


@dataclass(frozen=True)
class TraceGraphPocTrainingResult:
    """Only fixed-step optimization history; no validation or selected state."""

    steps_completed: int
    episodes_started: int
    episodes_completed: int
    query_count: int
    no_context_query_count: int
    history: list[dict[str, object]]


def train_relational_trace_graph_poc(
    model: RelationalTraceGraphInterpolator,
    training: C3PocTraceGraphTrainingData,
    *,
    graph_settings: TraceGraphSettings,
    random_seed: int,
    inner_mask_fraction: float,
    max_steps: int,
    query_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float | None,
    report_interval: int,
    loss: str = "masked_trace_relative_mse",
    mixed_precision: str = "off",
    edge_sampling: dict[str, int] | None = None,
    device: torch.device | str = "cpu",
    reporter: Callable[[str], None] | None = None,
) -> TraceGraphPocTrainingResult:
    """Apply the objective to every hidden query, including no-context rows.

    History timings are relative wall-clock measurements without extra CUDA
    synchronization. Scalar reads finish the optimization interval; preparation
    may enqueue device work that is accounted for by that later interval.
    """
    amp_dtype = mixed_precision_dtype(mixed_precision, device)
    sampling = (
        None
        if edge_sampling is None
        else validate_trace_graph_edge_sampling(edge_sampling, graph_settings)
    )
    build_options = (
        {}
        if sampling is None
        else {
            "fanout_per_relation": sampling["fanout_per_relation"],
            "rng": np.random.default_rng(sampling["seed"]),
        }
    )
    steps = config_values.positive_integer(max_steps, "max_steps")
    batch_size = config_values.positive_integer(query_batch_size, "query_batch_size")
    interval = config_values.positive_integer(report_interval, "report_interval")
    rate = config_values.positive_float(learning_rate, "learning_rate")
    decay = config_values.nonnegative_float(weight_decay, "weight_decay")
    clip = (
        None
        if gradient_clip_norm is None
        else config_values.positive_float(gradient_clip_norm, "gradient_clip_norm")
    )
    graph_settings.validate_model_config(model.constructor_config())
    episodes = PocTraceGraphEpisodeGenerator(
        training, random_seed=random_seed, missing_fraction=inner_mask_fraction
    )
    domain = training.domain
    threshold = training.preprocessing.azimuth_min_offset_m
    geometry = compute_trace_graph_geometry(
        domain.source_xy_m, domain.receiver_xy_m, azimuth_min_offset_m=threshold
    )
    spatial_index = (
        TraceGraphSpatialIndex(geometry, domain.trace_ids, **graph_settings.subgraph_kwargs())
        if graph_settings.neighbor_search == "exact_index"
        else None
    )
    positions = {int(trace_id): i for i, trace_id in enumerate(domain.trace_ids)}
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=rate, weight_decay=decay)
    scaler = torch.cuda.amp.GradScaler() if mixed_precision == "fp16" else None
    history = []
    total_queries = no_context = completed = 0
    while len(history) < steps:
        episode = episodes.next_episode()
        source = build_poc_trace_graph_context_source(training, episode, device=device)
        labels = PocTraceGraphEpisodeLabelReader(training, episode, device=device)
        builder = (
            FixedTraceGraphSubgraphBuilder.from_spatial_index(spatial_index, episode.visible_mask)
            if spatial_index is not None
            else None
        )
        episode_queries = 0
        for query_ids in episode.query_batches(batch_size):
            started = perf_counter()
            indices = [positions[int(trace_id)] for trace_id in query_ids]
            queries = compute_trace_graph_geometry(
                domain.source_xy_m[indices],
                domain.receiver_xy_m[indices],
                azimuth_min_offset_m=threshold,
            )
            plan = (
                builder.build(
                    queries, query_ids, rounds=model.message_passing_rounds, **build_options
                )
                if builder
                else build_trace_graph_subgraph(
                    queries,
                    query_ids,
                    geometry,
                    domain.trace_ids,
                    episode.visible_mask,
                    rounds=model.message_passing_rounds,
                    **graph_settings.subgraph_kwargs(),
                )
            )
            batch = source.inputs(plan)
            target = labels.read(query_ids)
            prepared = perf_counter()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            if amp_dtype is None:
                prediction, context = model(batch)
            else:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    prediction, context = model(batch)
            if prediction.shape != target.shape or prediction.ndim != 2:
                raise ValueError("prediction must match hidden labels [query, time]")
            objective = (
                masked_trace_loss(prediction, target, loss_name=loss)
                if amp_dtype is None
                else masked_trace_loss(
                    prediction.float(),
                    target.float(),
                    loss_name=loss,
                    accumulation_dtype=torch.float32,
                )
            )
            if not torch.isfinite(objective):
                raise RuntimeError("non-finite PoC training loss")
            if scaler is None:
                objective.backward()
            else:
                scaler.scale(objective).backward()
                scaler.unscale_(optimizer)
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
            if scaler is None:
                optimizer.step()
            else:
                scaler.step(optimizer)
                scaler.update()
            count = int((~context).sum().item())
            loss_value = float(objective.detach().item())
            optimized = perf_counter()
            total_queries += len(query_ids)
            no_context += count
            episode_queries += len(query_ids)
            history.append(
                {
                    "step": len(history) + 1,
                    "episode_id": episode.episode_id,
                    "loss": loss_value,
                    "query_count": len(query_ids),
                    "no_context_query_count": count,
                    "seconds": perf_counter() - started,
                    "subgraph_node_count": len(plan.trace_ids),
                    "subgraph_support_node_count": plan.diagnostics["support_node_count"],
                    "subgraph_edge_count": plan.diagnostics["typed_edge_count"],
                    "subgraph_max_depth": plan.diagnostics["max_depth"],
                    "batch_preparation_seconds": prepared - started,
                    "optimization_seconds": optimized - prepared,
                }
            )
            if reporter and (len(history) % interval == 0 or len(history) == steps):
                reporter(f"GNN step {len(history)}/{steps}: loss={history[-1]['loss']:.8g}")
            if len(history) == steps:
                break
        completed += int(episode_queries == len(episode.hidden_trace_ids))
    return TraceGraphPocTrainingResult(
        len(history), episode.episode_id, completed, total_queries, no_context, history
    )
