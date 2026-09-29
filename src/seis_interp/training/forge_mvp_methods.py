"""Six fixed-method adapters consuming only compact observed FORGE inputs."""

import hashlib
from copy import deepcopy
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
import torch

from seis_interp.data.c3_poc_trace_graph import C3PocTraceGraphTrainingData
from seis_interp.data.forge_aligned_inputs import align_model_inputs, unalign_predictions
from seis_interp.data.forge_mvp_artifacts import ForgeModelInputs
from seis_interp.data.trace_graph_domain import build_trace_graph_domain
from seis_interp.models.ccnet5d import CCNet5D
from seis_interp.models.nersi_trace import NersiTrace
from seis_interp.models.relational_trace_graph import RelationalTraceGraphInterpolator
from seis_interp.processing.drr_batched import interpolate_drr_volume_batched
from seis_interp.processing.drr_windows import interpolate_drr_volume
from seis_interp.processing.pocs_batched import interpolate_pocs_volume_batched
from seis_interp.processing.pocs_windows import interpolate_pocs_volume
from seis_interp.processing.trace_graph_preprocessing import build_poc_trace_graph_preprocessing
from seis_interp.processing.trace_graph_settings import TraceGraphSettings
from seis_interp.training.ccnet5d_observed_patches import CCNet5DObservedPatchSource
from seis_interp.training.ccnet5d_observed_training import train_ccnet5d_observed_steps
from seis_interp.training.ccnet5d_prediction import predict_ccnet5d_volume
from seis_interp.training.relational_trace_graph_poc_trainer import train_relational_trace_graph_poc
from seis_interp.training.relational_trace_graph_prediction import predict_relational_trace_graph
from seis_interp.training.trace_graph_ema import TraceGraphEMA, validate_trace_graph_ema_decay
from seis_interp.training.trace_relative_loss import masked_trace_relative_mse

LOSS_WINDOW_UPDATES = 1000


@dataclass
class ForgeMethodResult:
    """``shadow_predictions`` maps each EMA shadow decay string to its prediction.

    ``ema_states`` maps a decay string to ``{update: model state dict}``: the final
    weights of every non-primary shadow (at ``max_steps``) and every snapshot of
    ``training.ema_snapshot_updates``. The primary final weights stay in ``checkpoint``.
    """

    prediction: np.ndarray
    cell_ids: np.ndarray
    history: list
    checkpoint: dict | None
    diagnostics: dict
    shadow_predictions: dict[str, np.ndarray] = field(default_factory=dict)
    ema_states: dict[str, dict[int, dict]] = field(default_factory=dict)


def ema_eval_decays(training: dict) -> tuple[float, ...]:
    """Optional EMA shadows scored from the same run; they never touch optimization.

    ``training.ema_decay`` stays the primary checkpoint and must be one of the
    distinct shadow decays, so its shadow prediction is the primary prediction.
    """
    values = training.get("ema_eval_decays")
    if values is None:
        return ()
    if not isinstance(values, list) or not values:
        raise ValueError("training.ema_eval_decays must be a nonempty list")
    decays = tuple(validate_trace_graph_ema_decay(value) for value in values)
    primary = validate_trace_graph_ema_decay(training.get("ema_decay"))
    if None in decays or len(set(decays)) != len(decays) or primary not in decays:
        raise ValueError("training.ema_eval_decays must be distinct decays including ema_decay")
    return decays


def ema_snapshot_updates(training: dict) -> tuple[int, ...]:
    """Optional updates after which every EMA average is saved; they never touch fitting."""
    values = training.get("ema_snapshot_updates")
    if values is None:
        return ()
    if (
        not isinstance(values, list)
        or not values
        or any(isinstance(v, bool) or not isinstance(v, int) for v in values)
        or len(set(values)) != len(values)
        or not all(1 <= v < training["max_steps"] for v in values)
        or validate_trace_graph_ema_decay(training.get("ema_decay")) is None
    ):
        raise ValueError(
            "training.ema_snapshot_updates must be distinct updates in [1, max_steps) of an EMA run"
        )
    return tuple(sorted(values))


def shadow_key(decay: float) -> str:
    """Stable text key of one EMA shadow, e.g. ``0.9995``."""
    return repr(float(decay))


def initialize_model(config):
    """Seed before construction, identically for both ReGSI geometries."""
    torch.manual_seed(config["model_seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["model_seed"])
    method = config["method_id"]
    if method == "ccnet5d_grid":
        return CCNet5D(**config["model"])
    if method == "nersi_real":
        return NersiTrace(**config["model"])
    if method in ("regsi_real", "regsi_grid"):
        return RelationalTraceGraphInterpolator(**config["model"])
    raise ValueError("method has no neural model")


def model_parameter_count(config: dict, canvas_length: int) -> int:
    """Parameters of the neural model as constructed for this time canvas."""
    model = initialize_model(canvas_method_config(config, canvas_length))
    return sum(parameter.numel() for parameter in model.parameters())


def model_state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def graph_data(inputs, config):
    """Derive domain, node and edge geometry from the selected geometry table only."""
    g = inputs.geometry
    ids = g.cell_id.to_numpy()
    observed = np.isin(ids, inputs.observed_cells)
    domain = build_trace_graph_domain(
        trace_ids=ids,
        source_xy_m=g[["source_x_m", "source_y_m"]].to_numpy(),
        receiver_xy_m=g[["receiver_x_m", "receiver_y_m"]].to_numpy(),
        ffids=(g.source_grid_line * inputs.spatial_shape[1] + g.source_grid_point).to_numpy(),
        observed_mask=observed,
        time_s=inputs.time_s,
        inputs_lock=inputs.binding,
    )
    preprocessing = build_poc_trace_graph_preprocessing(
        domain, amplitude_scale=inputs.global_rms, **config["geometry_features"]
    )
    training_domain = build_trace_graph_domain(
        trace_ids=ids[observed],
        source_xy_m=domain.source_xy_m[observed],
        receiver_xy_m=domain.receiver_xy_m[observed],
        ffids=domain.ffids[observed],
        observed_mask=np.ones(observed.sum(), dtype=bool),
        array_rows=np.arange(observed.sum()),
        time_s=inputs.time_s,
        inputs_lock=inputs.binding,
    )
    training = C3PocTraceGraphTrainingData(
        training_domain, inputs.read_observed(ids[observed]), preprocessing
    )
    graph = config["graph"]
    relations = graph["relations"]
    scales = [
        [relations[name][a], relations[name][b]]
        for name, a, b in (
            ("source", "source_scale_m", "receiver_scale_m"),
            ("receiver", "source_scale_m", "receiver_scale_m"),
            ("cmp", "midpoint_scale_m", "offset_vector_scale_m"),
            ("offset_azimuth", "midpoint_scale_m", "offset_vector_scale_m"),
        )
    ]
    settings = TraceGraphSettings(
        relation_scales_m=scales,
        neighbors_per_relation=graph["neighbors_per_relation"],
        radius=graph["max_normalized_distance"],
        candidate_chunk_size=graph["candidate_chunk_size"],
        topology=graph["topology"],
        neighbor_search=graph["neighbor_search"],
    )
    return domain, training, settings


def canvas_method_config(config: dict, canvas_length: int) -> dict:
    """Copy of ``config`` whose NeRSI decoder emits the whole aligned canvas."""
    resolved = deepcopy(config)
    if resolved["method_id"] == "nersi_real":
        resolved["model"]["time_sample_count"] = int(canvas_length)
    return resolved


def run_method(inputs: ForgeModelInputs, config: dict, reporter=print) -> ForgeMethodResult:
    """Run in model amplitude units, then apply the preparation's restoration RMS.

    With ``config["alignment"]`` every method fits and predicts on the common
    geometry-only moveout canvas; predictions return to the original samples by
    the exact inverse shift before amplitude restoration.
    EMA shadows are restored exactly like the primary prediction.
    """
    model_inputs = inputs.model_space()
    canvas = None
    if config.get("alignment") is None:
        result = _run_method(model_inputs, config, reporter)
    else:
        aligned, canvas = align_model_inputs(model_inputs, config["alignment"])
        result = _run_method(aligned, canvas_method_config(config, canvas.length), reporter)
        result.diagnostics["alignment_canvas_sample_count"] = canvas.length
    result.prediction = _physical_prediction(result.prediction, result.cell_ids, canvas, inputs)
    decays = ema_eval_decays(config.get("training", {}))
    if decays:
        shadows = {
            key: _physical_prediction(value, result.cell_ids, canvas, inputs)
            for key, value in result.shadow_predictions.items()
        }
        shadows[shadow_key(config["training"]["ema_decay"])] = result.prediction
        result.shadow_predictions = {shadow_key(d): shadows[shadow_key(d)] for d in decays}
    if inputs.test_rms is not None:
        result.diagnostics.update(
            amplitude_scaling=inputs.binding.get("amplitude_scaling", "observed_trace_rms_idw"),
            trace_scale_hash=inputs.binding["trace_scale_hash"],
        )
    return result


def _physical_prediction(prediction, cell_ids, canvas, inputs):
    """Exact inverse moveout shift, then the preparation's restoration RMS."""
    if canvas is not None:
        prediction = unalign_predictions(prediction, cell_ids, canvas)
    if inputs.test_rms is not None:
        prediction = (prediction.astype(np.float64) * inputs.test_rms[:, None]).astype(np.float32)
    return prediction


def _run_method(inputs: ForgeModelInputs, config: dict, reporter) -> ForgeMethodResult:
    """No raw amplitude path or evaluator is reachable through the input object."""
    method = config["method_id"]
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if method in ("pocs_grid", "drr_grid"):
        volume = inputs.volume()
        observed = (volume.values.astype(np.float64) / inputs.global_rms).astype(np.float32)
        execution = config.get("execution", {"backend": "sequential"})
        batched = execution["backend"] == "batched"
        if method == "pocs_grid":
            interpolate = interpolate_pocs_volume_batched if batched else interpolate_pocs_volume
            options = {"device": "cpu"} if batched else {}
            result = interpolate(
                observed, volume.observed_trace_mask, **config["algorithm"], **options
            )
        else:
            interpolate = interpolate_drr_volume_batched if batched else interpolate_drr_volume
            options = {"device": "cpu", "workers": execution.get("workers", 1)} if batched else {}
            result = interpolate(
                observed,
                volume.observed_trace_mask,
                inputs.time_s,
                **config["algorithm"],
                **options,
            )
        prediction = result.values.reshape(len(inputs.time_s), -1)[:, inputs.test_cells].T
        prediction = (prediction.astype(np.float64) * inputs.global_rms).astype(np.float32)
        return ForgeMethodResult(
            prediction, inputs.test_cells.copy(), [], None, {"block_count": result.block_count}
        )
    model = initialize_model(config)
    initialization_hash = model_state_hash(model)
    parameter_count = sum(p.numel() for p in model.parameters())
    tr = config["training"]
    decays = ema_eval_decays(tr)
    snapshots = ema_snapshot_updates(tr)
    started = perf_counter()
    if method == "ccnet5d_grid":
        fit = _fit_ccnet
    elif method == "nersi_real":
        fit = _fit_nersi
    elif method in ("regsi_real", "regsi_grid"):
        fit = _fit_regsi
    else:
        raise ValueError("unsupported MVP method")
    prediction, history, shadows, records, states = fit(
        model, inputs, config, reporter, decays, snapshots
    )
    checkpoint = {
        "method_id": method,
        "model_config": config["model"],
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "steps_completed": tr["max_steps"],
        "selection_rule": config["checkpoint_rule"],
        "initialization_hash": initialization_hash,
        "model_seed": config["model_seed"],
        "input_binding": inputs.binding,
    }
    if config.get("alignment") is not None:
        checkpoint["alignment"] = dict(config["alignment"])
    records["runtime_training_seconds"] = records.pop("training_end") - started
    return ForgeMethodResult(
        prediction,
        inputs.test_cells.copy(),
        history,
        checkpoint,
        {
            "parameter_count": parameter_count,
            "initialization_hash": initialization_hash,
            **records,
        },
        shadows,
        states,
    )


def _training_end(device) -> float:
    """Wall clock after queued device work of the fit has finished."""
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    return perf_counter()


def _window_loss(history, max_steps):
    """Mean loss over the last recorded updates covering ``LOSS_WINDOW_UPDATES``.

    Each history row carries the mean loss since the previous row, so the
    weighted mean is exact over the reported update count.
    """
    first = max_steps - min(LOSS_WINDOW_UPDATES, max_steps)
    previous, total, count = 0, 0.0, 0
    for row in history:
        span = row["step"] - previous
        if row["step"] > first:
            total += row["loss"] * span
            count += span
        previous = row["step"]
    return {
        "final_training_loss_window_mean": total / count,
        "final_training_loss_window_updates": count,
    }


def _shadow_predictions(model, decays, primary, load, predict):
    """Predict every non-primary shadow, then restore the primary weights.

    Also returns each non-primary shadow's weights as predicted (CPU state dicts).
    """
    shadows, states, started = {}, {}, perf_counter()
    if len(decays) > 1:
        primary_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        for decay in decays:
            if decay != primary:
                load(decay)
                shadows[shadow_key(decay)] = predict()
                states[shadow_key(decay)] = _cpu_state(model.state_dict())
        model.load_state_dict(primary_state)
    return shadows, {"runtime_shadow_prediction_seconds": perf_counter() - started}, states


def _cpu_state(state):
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def _ema_states(final_states, snapshots, max_steps):
    """Decay key -> {update: state}: non-primary final shadows and every snapshot.

    ``snapshots`` maps an update to ``{decay: state}``.
    """
    states = {key: {max_steps: state} for key, state in final_states.items()}
    for update, by_decay in snapshots.items():
        for decay, state in by_decay.items():
            states.setdefault(shadow_key(decay), {})[update] = state
    return states


def _average_snapshots(averages, updates):
    """``{update: {decay: state}}`` of TraceGraphEMA averages keyed by decay."""
    return {n: {decay: ema.snapshots[n] for decay, ema in averages.items()} for n in updates}


def _fit_ccnet(model, inputs, config, reporter, decays, snapshot_updates):
    device, tr = config["device"], config["training"]
    volume = inputs.volume()
    patch_shape = (volume.values.shape[0], *config["patches"]["shape"][1:])
    source = CCNet5DObservedPatchSource(
        volume,
        amplitude_scale=inputs.global_rms,
        patch_shape=patch_shape,
        inner_mask_fraction=config["patches"]["inner_mask_fraction"],
        placement_seed=config["model_seed"],
        inner_mask_seed=config["model_seed"],
    )
    result = train_ccnet5d_observed_steps(
        model,
        source,
        device=device,
        optimizer_updates=tr["max_steps"],
        learning_rate=tr["learning_rate"],
        report_every_steps=tr["report_interval"],
        loss_name=config["loss"],
        ema_decay=tr.get("ema_decay"),
        optimizer_name=tr["optimizer"],
        weight_decay=tr["weight_decay"],
        supervised_traces_per_update=tr.get("supervised_traces_per_update"),
        ema_eval_decays=decays,
        ema_snapshot_updates=snapshot_updates,
        reporter=reporter,
    )
    records = {"training_end": _training_end(device)}
    history = list(result.history)

    def parameter_state(parameters):
        state = _cpu_state(model.state_dict())
        for (name, _), value in zip(model.named_parameters(), parameters, strict=True):
            state[name] = value.detach().cpu().clone()
        return state

    def predict():
        values = predict_ccnet5d_volume(
            model,
            volume,
            amplitude_rms=inputs.global_rms,
            core_shape=tuple(config["prediction"]["core_shape"]),
            device=device,
        ).values
        return values.reshape(len(inputs.time_s), -1)[:, inputs.test_cells].T.copy()

    @torch.no_grad()
    def load(decay):
        for parameter, average in zip(
            model.parameters(), result.shadow_parameters[decay], strict=True
        ):
            parameter.copy_(average)

    started = perf_counter()
    prediction = predict()
    records["runtime_prediction_seconds"] = perf_counter() - started
    shadows, timing, final_states = _shadow_predictions(
        model, decays, tr.get("ema_decay"), load, predict
    )
    snapshots = {
        update: {decay: parameter_state(values) for decay, values in by_decay.items()}
        for update, by_decay in result.shadow_snapshots.items()
    }
    records.update(
        timing,
        final_training_loss=result.final_loss,
        **_window_loss(history, tr["max_steps"]),
        mean_patches_per_update=source.patches_drawn / tr["max_steps"],
        mean_visible_observed_cells_per_update=(
            source.visible_observed_cells_drawn / tr["max_steps"]
        ),
    )
    states = _ema_states(final_states, snapshots, tr["max_steps"])
    return prediction, history, shadows, records, states


def _fit_regsi(model, inputs, config, reporter, decays, snapshot_updates):
    device, tr = config["device"], config["training"]
    domain, training, settings = graph_data(inputs, config)
    primary = tr.get("ema_decay")
    ema = TraceGraphEMA(primary, snapshot_updates) if primary else None
    extra = {decay: TraceGraphEMA(decay, snapshot_updates) for decay in decays if decay != primary}
    result = train_relational_trace_graph_poc(
        model,
        training,
        graph_settings=settings,
        random_seed=config["model_seed"],
        inner_mask_fraction=tr["inner_mask_fraction"],
        max_steps=tr["max_steps"],
        query_batch_size=tr["query_batch_size"],
        learning_rate=tr["learning_rate"],
        weight_decay=tr["weight_decay"],
        gradient_clip_norm=tr["gradient_clip_norm"],
        report_interval=tr["report_interval"],
        loss=config["loss"],
        mixed_precision=tr["mixed_precision"],
        cudnn_benchmark=False,
        ema=ema,
        ema_shadows=list(extra.values()),
        device=device,
        reporter=reporter,
    )
    records = {"training_end": _training_end(device)}
    history = result.history
    if ema is not None:
        model.load_state_dict(ema.state_dict())
    observed_waveforms = inputs.read_observed(domain.trace_ids[domain.observed_mask])

    def predict():
        return predict_relational_trace_graph(
            model,
            domain,
            training.preprocessing,
            graph_settings=settings,
            query_trace_ids=inputs.test_cells,
            observed_waveforms=observed_waveforms,
            query_batch_size=config["prediction"]["query_batch_size"],
            device=device,
        ).prediction

    started = perf_counter()
    prediction = predict()
    records["runtime_prediction_seconds"] = perf_counter() - started
    shadows, timing, final_states = _shadow_predictions(
        model,
        decays,
        primary,
        lambda decay: model.load_state_dict(extra[decay].state_dict()),
        predict,
    )
    averages = {} if ema is None else {primary: ema}
    snapshots = _average_snapshots(averages | extra, snapshot_updates)
    records.update(
        timing,
        final_training_loss=history[-1]["loss"],
        **_window_loss(history, tr["max_steps"]),
        mean_subgraph_support_node_count=float(
            np.mean([row["subgraph_support_node_count"] for row in history])
        ),
    )
    clip = tr["gradient_clip_norm"]
    if clip is not None:
        norms = np.array([row["gradient_norm"] for row in history])
        records.update(
            gradient_clip_fraction=float(np.mean(norms > clip)),
            mean_gradient_norm=float(norms.mean()),
        )
    states = _ema_states(final_states, snapshots, tr["max_steps"])
    return prediction, history, shadows, records, states


def _fit_nersi(model, inputs, config, reporter, decays, snapshot_updates):
    """Fit observed traces at four unrounded local coordinates; fixed final EMA.

    Extra EMA shadows and snapshots never touch optimization.
    """
    columns = [
        f"{role}_{axis}_continuous" for role in ("source", "receiver") for axis in ("line", "point")
    ]
    table = inputs.geometry.set_index("cell_id")
    coordinates = table[columns].to_numpy() / np.maximum(np.asarray(inputs.spatial_shape) - 1, 1)
    coordinates = torch.from_numpy(coordinates.astype(np.float32))
    positions = {int(cell): i for i, cell in enumerate(table.index)}
    observed_rows = np.array([positions[int(i)] for i in inputs.observed_cells])
    device, tr = config["device"], config["training"]
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=tr["learning_rate"], weight_decay=tr["weight_decay"]
    )
    rng = np.random.default_rng(config["model_seed"])
    primary = tr.get("ema_decay")
    ema = TraceGraphEMA(primary, snapshot_updates) if primary else None
    extra = {decay: TraceGraphEMA(decay, snapshot_updates) for decay in decays if decay != primary}
    history = []
    window_first = tr["max_steps"] - min(LOSS_WINDOW_UPDATES, tr["max_steps"])
    window_losses = []
    for step in range(1, tr["max_steps"] + 1):
        rows = rng.choice(len(inputs.observed_cells), size=tr["batch_size"], replace=False)
        target = torch.from_numpy(
            (inputs.observed_values[rows].astype(np.float64) / inputs.global_rms).astype(np.float32)
        ).to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(coordinates[observed_rows[rows]].to(device))
        loss = masked_trace_relative_mse(prediction, target)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite NeRSI loss")
        loss.backward()
        optimizer.step()
        if ema is not None:
            ema.update(model)
        for average in extra.values():
            average.update(model)
        if step > window_first:
            window_losses.append(float(loss.detach()))
        if step % tr["report_interval"] == 0 or step == tr["max_steps"]:
            value = float(loss.detach())
            history.append({"step": step, "loss": value, "learning_rate": tr["learning_rate"]})
            reporter(f"NeRSI step {step}/{tr['max_steps']}")
    records = {"training_end": _training_end(device)}
    if ema is not None:
        model.load_state_dict(ema.state_dict())
    model.eval()

    def predict():
        output = np.empty((len(inputs.test_cells), len(inputs.time_s)), dtype=np.float32)
        with torch.inference_mode():
            batch = config["prediction"]["batch_size"]
            for start in range(0, len(output), batch):
                rows = [positions[int(i)] for i in inputs.test_cells[start : start + batch]]
                values = model(coordinates[rows].to(device)).cpu().numpy().astype(np.float64)
                output[start : start + batch] = values * inputs.global_rms
        return output

    started = perf_counter()
    output = predict()
    records["runtime_prediction_seconds"] = perf_counter() - started
    shadows, timing, final_states = _shadow_predictions(
        model,
        decays,
        primary,
        lambda decay: model.load_state_dict(extra[decay].state_dict()),
        predict,
    )
    averages = {} if ema is None else {primary: ema}
    snapshots = _average_snapshots(averages | extra, snapshot_updates)
    records.update(
        timing,
        final_training_loss=window_losses[-1],
        final_training_loss_window_mean=float(np.mean(window_losses)),
        final_training_loss_window_updates=len(window_losses),
    )
    states = _ema_states(final_states, snapshots, tr["max_steps"])
    return output, history, shadows, records, states
