"""Fixed-step pseudo-mask training on the observed set of a single volume."""

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

import torch

from seis_interp import config_values
from seis_interp.data.c3_poc_trace_graph import C3PocTraceGraphTrainingData
from seis_interp.models.relational_trace_graph import RelationalTraceGraphInterpolator
from seis_interp.processing.trace_graph_geometry import compute_trace_graph_geometry
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
    device: torch.device | str = "cpu",
    reporter: Callable[[str], None] | None = None,
) -> TraceGraphPocTrainingResult:
    """Apply the shared objective to every hidden query, including no-context rows."""
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
    positions = {int(trace_id): i for i, trace_id in enumerate(domain.trace_ids)}
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=rate, weight_decay=decay)
    history = []
    total_queries = no_context = completed = 0
    while len(history) < steps:
        episode = episodes.next_episode()
        source = build_poc_trace_graph_context_source(training, episode, device=device)
        labels = PocTraceGraphEpisodeLabelReader(training, episode, device=device)
        builder = (
            FixedTraceGraphSubgraphBuilder(
                geometry, domain.trace_ids, episode.visible_mask, **graph_settings.subgraph_kwargs()
            )
            if graph_settings.neighbor_search == "exact_index"
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
                builder.build(queries, query_ids, rounds=model.message_passing_rounds)
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
            model.train()
            optimizer.zero_grad(set_to_none=True)
            prediction, context = model(batch)
            if prediction.shape != target.shape or prediction.ndim != 2:
                raise ValueError("prediction must match hidden labels [query, time]")
            objective = masked_trace_loss(prediction, target, loss_name=loss)
            if not torch.isfinite(objective):
                raise RuntimeError("non-finite PoC training loss")
            objective.backward()
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
            optimizer.step()
            count = int((~context).sum().item())
            total_queries += len(query_ids)
            no_context += count
            episode_queries += len(query_ids)
            history.append(
                {
                    "step": len(history) + 1,
                    "episode_id": episode.episode_id,
                    "loss": float(objective.detach().item()),
                    "query_count": len(query_ids),
                    "no_context_query_count": count,
                    "seconds": perf_counter() - started,
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
