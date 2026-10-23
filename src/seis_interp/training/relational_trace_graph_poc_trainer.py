"""Fixed-step pseudo-mask training on the observed set of a single volume."""

import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from time import perf_counter

import numpy as np
import torch

from seis_interp import config_values
from seis_interp.data.c3_poc_trace_graph import C3PocTraceGraphTrainingData
from seis_interp.data.masked_trace_inputs import MaskedTraceGraphInputs
from seis_interp.models.relational_trace_graph import (
    RelationalTraceGraphInterpolator,
    trace_graph_receptive_field,
)
from seis_interp.processing.trace_graph_geometry import compute_trace_graph_geometry
from seis_interp.processing.trace_graph_neighbor_table import TraceGraphNeighborTable
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
from seis_interp.training.exact_resume import object_state, restore_objects
from seis_interp.training.mixed_precision import mixed_precision_dtype
from seis_interp.training.trace_graph_ema import TraceGraphEMA, update_trace_graph_averages
from seis_interp.training.trace_graph_execution import (
    TraceGraphExecution,
    torch_cpu_threads,
    validate_trace_graph_execution,
)
from seis_interp.training.trace_graph_optimization import (
    trace_graph_step_learning_rate,
    validate_trace_graph_schedule,
)
from seis_interp.training.trace_graph_sampling import validate_trace_graph_edge_sampling
from seis_interp.training.trace_relative_loss import NONFINITE_TRACES_MESSAGE, masked_trace_loss

NONFINITE_LOSS_MESSAGE = "non-finite PoC training loss"
# torch.nn.utils.clip_grad_norm_(error_if_nonfinite=True) text for the default L2 norm.
NONFINITE_GRADIENT_MESSAGE = (
    "The total norm of order 2.0 for gradients from `parameters` is non-finite, so it "
    "cannot be clipped. To disable this error and scale the gradients by the non-finite "
    "norm anyway, set `error_if_nonfinite=False`"
)
# Batches a prefetch worker may prepare ahead of the optimizer.
PREFETCH_BATCHES = 4


@dataclass(frozen=True)
class TraceGraphPocTrainingResult:
    """Only fixed-step optimization history; no validation or selected state."""

    steps_completed: int
    episodes_started: int
    episodes_completed: int
    query_count: int
    no_context_query_count: int
    history: list[dict[str, object]]


@dataclass(frozen=True)
class _PreparedBatch:
    """One update's graph inputs and labels with the record of how they were made."""

    batch: MaskedTraceGraphInputs
    target: torch.Tensor
    episode_id: int
    episode_query_count: int
    query_count: int
    node_count: int
    support_node_count: int
    edge_count: int
    max_depth: int
    started: float
    queried: float
    built: float
    assembled: float
    prepared: float
    sampler_state: dict
    receptive_field: tuple[list[torch.Tensor], list[torch.Tensor]] | None = None

    def pin_memory(self) -> "_PreparedBatch":
        """Page-locked copy for asynchronous transfers (used by the prefetch loader)."""
        return replace(
            self,
            batch=self.batch.pin_memory(),
            target=self.target.pin_memory(),
            receptive_field=_map_receptive_field(self.receptive_field, torch.Tensor.pin_memory),
        )


@dataclass(frozen=True)
class _PendingStep:
    """Device scalars of one update whose host reads are deferred."""

    row: int
    loss: torch.Tensor
    no_context_count: torch.Tensor
    finite_traces: torch.Tensor
    gradient_norm: torch.Tensor | None


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
    cudnn_benchmark: bool = True,
    learning_rate_schedule: dict | None = None,
    ema: TraceGraphEMA | None = None,
    ema_shadows: Sequence[TraceGraphEMA] = (),
    device: torch.device | str = "cpu",
    reporter: Callable[[str], None] | None = None,
    execution: Mapping[str, object] | TraceGraphExecution | None = None,
    neighbor_table: TraceGraphNeighborTable | None = None,
    diagnostics=None,
    recovery=None,
) -> TraceGraphPocTrainingResult:
    """Apply the objective to every hidden query, including no-context rows.

    History timings are relative wall-clock measurements without extra CUDA
    synchronization. Scalar reads finish the optimization interval; preparation
    may enqueue device work that is accounted for by that later interval.
    Query geometry, graph build, input assembly and label read partition the
    preparation interval in that order and sum to it. ``ema_shadows`` are extra
    averages advanced exactly when ``ema`` would be; they never affect fitting.
    With clipping, each history row records the pre-clip gradient norm.

    ``execution`` holds the opt-in switches of ``trace_graph_execution`` (all off
    by default). With ``prefetch``, preparation timings are measured in the worker
    and overlap earlier updates; ``seconds`` then spans waiting for the batch and
    the optimization. With ``deferred_checks``, scalar reads (and the timings they
    finish) happen at report intervals. ``neighbor_table`` optionally supplies the
    table that ``execution.neighbor_table`` would otherwise build here.
    """
    if not isinstance(cudnn_benchmark, bool):
        raise ValueError("cudnn_benchmark must be boolean")
    options = validate_trace_graph_execution(
        execution, neighbor_search=graph_settings.neighbor_search, device=device
    )
    if neighbor_table is not None and not options.neighbor_table:
        raise ValueError("neighbor_table requires execution.neighbor_table")
    if not cudnn_benchmark:
        torch.backends.cudnn.benchmark = False
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
    schedule = (
        None
        if learning_rate_schedule is None
        else validate_trace_graph_schedule(learning_rate_schedule, rate, steps)
    )
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
    geometry = _training_geometry(training)
    spatial_index = (
        TraceGraphSpatialIndex(geometry, domain.trace_ids, **graph_settings.subgraph_kwargs())
        if graph_settings.neighbor_search == "exact_index"
        else None
    )
    table = None
    if options.neighbor_table:
        table = TraceGraphNeighborTable(spatial_index) if neighbor_table is None else neighbor_table
    # Without device waits the batch is prepared on the CPU and copied asynchronously.
    transfer = options.prefetch or options.deferred_checks
    preparation = {
        "training": training,
        "episodes": episodes,
        "geometry": geometry,
        "spatial_index": spatial_index,
        "neighbor_table": table,
        "graph_settings": graph_settings,
        "positions": {int(trace_id): i for i, trace_id in enumerate(domain.trace_ids)},
        "batch_size": batch_size,
        "rounds": model.message_passing_rounds,
        "build_options": build_options,
        "device": "cpu" if transfer else device,
        # Sync-free pruning: index sizes are known before the batch reaches the device.
        "receptive_field": options.deferred_checks and model.receptive_field_pruning,
    }
    averages = ([] if ema is None else [ema]) + list(ema_shadows)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=rate, weight_decay=decay)
    scaler = torch.cuda.amp.GradScaler() if mixed_precision == "fp16" else None
    deferred = options.deferred_checks
    forward_options = {"host_checks": False} if deferred else {}
    loss_options = {"check_finite": False} if deferred else {}
    history = []
    pending: list[_PendingStep] = []
    total_queries = no_context = completed = episode_queries = 0
    episode_id = None
    objects = {"diagnostics": diagnostics} | {f"ema_{i}": value for i, value in enumerate(averages)}
    state = {} if recovery is None else recovery.load(model, optimizer)
    if state:
        restore_objects(objects, state["objects"])
        history = state["history"]
        total_queries, no_context, completed = state["counts"]
        episode_id, episode_queries = state["episode_progress"]
        preparation["resume_sampler"] = state["sampler"]
        if scaler is not None:
            scaler.load_state_dict(state["scaler"])
    with torch_cpu_threads(options.cpu_threads):
        stream = _batch_stream(preparation, prefetch=options.prefetch, device=device)
        try:
            while len(history) < steps:
                waited = perf_counter()
                item = next(stream)
                if transfer:
                    batch = item.batch.to(device, non_blocking=True)
                    target = item.target.to(device, non_blocking=True)
                else:
                    batch, target = item.batch, item.target
                step_options = dict(forward_options)
                if item.receptive_field is not None:
                    step_options["receptive_field"] = _map_receptive_field(
                        item.receptive_field, lambda value: value.to(device, non_blocking=True)
                    )
                # Worker clocks cover preparation only; the update interval starts at the wait.
                started, prepared = (
                    (waited, perf_counter()) if options.prefetch else (item.started, item.prepared)
                )
                if item.episode_id != episode_id:
                    episode_id, episode_queries = item.episode_id, 0
                model.train()
                if schedule is not None:
                    for group in optimizer.param_groups:
                        group["lr"] = trace_graph_step_learning_rate(
                            len(history) + 1, steps, rate, schedule
                        )
                optimizer.zero_grad(set_to_none=True)
                if amp_dtype is None:
                    prediction, context = model(batch, **step_options)
                else:
                    with torch.autocast(device_type=torch.device(device).type, dtype=amp_dtype):
                        prediction, context = model(batch, **step_options)
                if prediction.shape != target.shape or prediction.ndim != 2:
                    raise ValueError("prediction must match hidden labels [query, time]")
                objective = (
                    masked_trace_loss(prediction, target, loss_name=loss, **loss_options)
                    if amp_dtype is None
                    else masked_trace_loss(
                        prediction.float(),
                        target.float(),
                        loss_name=loss,
                        accumulation_dtype=torch.float32,
                        **loss_options,
                    )
                )
                if deferred:
                    finite_traces = torch.isfinite(prediction).all() & torch.isfinite(target).all()
                elif not torch.isfinite(objective):
                    raise RuntimeError(NONFINITE_LOSS_MESSAGE)
                if diagnostics is not None:
                    diagnostics.observe(len(history) + 1, prediction, target, objective)
                if scaler is None:
                    objective.backward()
                else:
                    scaler.scale(objective).backward()
                    scaler.unscale_(optimizer)
                if diagnostics is not None:
                    diagnostics.gradients(model)
                gradient_norm = (
                    None
                    if clip is None
                    else torch.nn.utils.clip_grad_norm_(
                        model.parameters(), clip, error_if_nonfinite=not deferred
                    )
                )
                if scaler is None:
                    optimizer.step()
                    update_trace_graph_averages(averages, model, foreach=options.foreach_ema)
                else:
                    previous_scale = scaler.get_scale() if averages else None
                    scaler.step(optimizer)
                    scaler.update()
                    if averages and scaler.get_scale() >= previous_scale:
                        update_trace_graph_averages(averages, model, foreach=options.foreach_ema)
                if deferred:
                    pending.append(
                        _PendingStep(
                            len(history),
                            objective.detach(),
                            (~context).sum(),
                            finite_traces,
                            gradient_norm,
                        )
                    )
                    count = loss_value = None
                else:
                    count = int((~context).sum().item())
                    loss_value = float(objective.detach().item())
                    no_context += count
                optimized = perf_counter()
                total_queries += item.query_count
                episode_queries += item.query_count
                history.append(
                    {
                        "step": len(history) + 1,
                        "episode_id": item.episode_id,
                        "loss": loss_value,
                        "query_count": item.query_count,
                        "no_context_query_count": count,
                        "seconds": perf_counter() - started,
                        "subgraph_node_count": item.node_count,
                        "subgraph_support_node_count": item.support_node_count,
                        "subgraph_edge_count": item.edge_count,
                        "subgraph_max_depth": item.max_depth,
                        "batch_preparation_seconds": item.prepared - item.started,
                        "query_geometry_seconds": item.queried - item.started,
                        "graph_build_seconds": item.built - item.queried,
                        "input_assembly_seconds": item.assembled - item.built,
                        "label_read_seconds": item.prepared - item.assembled,
                        "optimization_seconds": optimized - prepared,
                    }
                )
                if schedule is not None:
                    history[-1]["learning_rate"] = optimizer.param_groups[0]["lr"]
                if gradient_norm is not None and not deferred:
                    history[-1]["gradient_norm"] = float(gradient_norm)
                due = len(history) % interval == 0 or len(history) == steps
                if deferred and due:
                    no_context += _resolve_deferred(pending, history)
                if episode_queries == item.episode_query_count:
                    completed += 1
                if recovery is not None and (recovery.due(len(history)) or len(history) == steps):
                    if deferred:
                        no_context += _resolve_deferred(pending, history)
                    recovery.save(
                        len(history),
                        model,
                        optimizer,
                        {
                            "objects": object_state(objects),
                            "history": history,
                            "counts": (total_queries, no_context, completed),
                            "episode_progress": (episode_id, episode_queries),
                            "sampler": item.sampler_state,
                            "scaler": None if scaler is None else scaler.state_dict(),
                        },
                        force=True,
                        phase="before_prediction" if len(history) == steps else "training",
                    )
                if reporter and due:
                    reporter(f"GNN step {len(history)}/{steps}: loss={history[-1]['loss']:.8g}")
                if len(history) == steps:
                    break
        finally:
            del stream
    return TraceGraphPocTrainingResult(
        len(history), episode_id, completed, total_queries, no_context, history
    )


def build_training_neighbor_table(
    training: C3PocTraceGraphTrainingData, graph_settings: TraceGraphSettings
) -> TraceGraphNeighborTable:
    """The ``execution.neighbor_table`` table over the observed training traces.

    Built from the same index the trainer searches; prediction over a domain whose
    observed traces are these traces can reuse it for support searches.
    """
    if graph_settings.neighbor_search != "exact_index":
        raise ValueError("execution.neighbor_table requires neighbor_search=exact_index")
    return TraceGraphNeighborTable(
        TraceGraphSpatialIndex(
            _training_geometry(training),
            training.domain.trace_ids,
            **graph_settings.subgraph_kwargs(),
        )
    )


def _training_geometry(training: C3PocTraceGraphTrainingData):
    domain = training.domain
    return compute_trace_graph_geometry(
        domain.source_xy_m,
        domain.receiver_xy_m,
        azimuth_min_offset_m=training.preprocessing.azimuth_min_offset_m,
    )


def _batch_stream(preparation: dict, *, prefetch: bool, device) -> Iterator[_PreparedBatch]:
    """Prepared batches in training order, made here or by one worker process."""
    if not prefetch:
        return _prepared_batches(**preparation)
    loader = torch.utils.data.DataLoader(
        _PreparedBatchStream(preparation),
        batch_size=None,
        num_workers=1,
        prefetch_factor=PREFETCH_BATCHES,
        pin_memory=torch.device(device).type == "cuda",
        # A private generator keeps the global torch random stream untouched.
        generator=torch.Generator(),
        multiprocessing_context="fork",
    )
    return iter(loader)


class _PreparedBatchStream(torch.utils.data.IterableDataset):
    """The model-independent batch sequence, for one forked loader worker."""

    def __init__(self, preparation: dict) -> None:
        self._preparation = preparation

    def __iter__(self) -> Iterator[_PreparedBatch]:
        return _prepared_batches(**self._preparation)


def _prepared_batches(
    *,
    training,
    episodes,
    geometry,
    spatial_index,
    neighbor_table,
    graph_settings,
    positions,
    batch_size,
    rounds,
    build_options,
    device,
    receptive_field,
    resume_sampler=None,
) -> Iterator[_PreparedBatch]:
    """Episodes are drawn only when the previous episode's batches are exhausted."""
    domain = training.domain
    threshold = training.preprocessing.azimuth_min_offset_m
    episode = None
    next_query = 0
    if resume_sampler is not None:
        episodes.load_state_dict(resume_sampler["episodes"])
        episode, next_query = resume_sampler["episode"], resume_sampler["next_query"]
        if "rng" in build_options:
            build_options["rng"].bit_generator.state = resume_sampler["edge_rng"]
    while True:
        if episode is None:
            episode = episodes.next_episode()
        source = build_poc_trace_graph_context_source(training, episode, device=device)
        labels = PocTraceGraphEpisodeLabelReader(training, episode, device=device)
        builder = None
        if spatial_index is not None and neighbor_table is None:
            builder = FixedTraceGraphSubgraphBuilder.from_spatial_index(
                spatial_index, episode.visible_mask
            )
        elif spatial_index is not None:
            builder = FixedTraceGraphSubgraphBuilder.from_spatial_index(
                spatial_index, episode.visible_mask, neighbor_table=neighbor_table
            )
        for query_index, query_ids in enumerate(episode.query_batches(batch_size)):
            if query_index < next_query:
                continue
            started = perf_counter()
            indices = [positions[int(trace_id)] for trace_id in query_ids]
            queries = compute_trace_graph_geometry(
                domain.source_xy_m[indices],
                domain.receiver_xy_m[indices],
                azimuth_min_offset_m=threshold,
            )
            queried = perf_counter()
            plan = (
                builder.build(queries, query_ids, rounds=rounds, **build_options)
                if builder
                else build_trace_graph_subgraph(
                    queries,
                    query_ids,
                    geometry,
                    domain.trace_ids,
                    episode.visible_mask,
                    rounds=rounds,
                    **graph_settings.subgraph_kwargs(),
                )
            )
            built = perf_counter()
            batch = source.inputs(plan)
            assembled = perf_counter()
            target = labels.read(query_ids)
            prepared = perf_counter()
            field = (
                trace_graph_receptive_field(
                    batch.edge_index, batch.query_indices, rounds, batch.waveforms.shape[0]
                )
                if receptive_field and batch.query_indices.numel()
                else None
            )
            yield _PreparedBatch(
                batch=batch,
                target=target,
                episode_id=episode.episode_id,
                episode_query_count=len(episode.hidden_trace_ids),
                query_count=len(query_ids),
                node_count=len(plan.trace_ids),
                support_node_count=plan.diagnostics["support_node_count"],
                edge_count=plan.diagnostics["typed_edge_count"],
                max_depth=plan.diagnostics["max_depth"],
                started=started,
                queried=queried,
                built=built,
                assembled=assembled,
                prepared=prepared,
                receptive_field=field,
                sampler_state=deepcopy(
                    {
                        "episodes": episodes.state_dict(),
                        "episode": episode,
                        "next_query": query_index + 1,
                        "edge_rng": build_options["rng"].bit_generator.state
                        if "rng" in build_options
                        else None,
                    }
                ),
            )

        episode, next_query = None, 0


def _map_receptive_field(receptive_field, function):
    if receptive_field is None:
        return None
    nodes, edges = receptive_field
    return [function(value) for value in nodes], [function(value) for value in edges]


def _resolve_deferred(pending: list[_PendingStep], history: list[dict]) -> int:
    """Read deferred scalars once, fail like the per-update checks, fill history rows.

    Returns the no-context query count of the resolved updates.
    """
    if not pending:
        return 0
    losses = torch.stack([step.loss for step in pending]).cpu()
    counts = torch.stack([step.no_context_count for step in pending]).cpu()
    finite = torch.stack([step.finite_traces for step in pending]).cpu()
    norms = (
        None
        if pending[0].gradient_norm is None
        else torch.stack([step.gradient_norm for step in pending]).cpu()
    )
    total = 0
    for index, step in enumerate(pending):
        if not bool(finite[index]):
            raise ValueError(NONFINITE_TRACES_MESSAGE)
        loss_value = float(losses[index])
        if not math.isfinite(loss_value):
            raise RuntimeError(NONFINITE_LOSS_MESSAGE)
        row = history[step.row]
        if norms is not None:
            norm = float(norms[index])
            if not math.isfinite(norm):
                raise RuntimeError(NONFINITE_GRADIENT_MESSAGE)
            row["gradient_norm"] = norm
        row["loss"] = loss_value
        row["no_context_query_count"] = int(counts[index])
        total += row["no_context_query_count"]
    pending.clear()
    return total
