"""Fixed-method adapters consuming only compact observed FORGE inputs.

The six MVP methods plus two methods added to a sealed preparation: ``nersi_profile``,
the profile-wise NeRSI of Gao et al. (2025), and ``inr_real``, the implicit trace field
that returns a complete trace from one linear head.
"""

import hashlib
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
import torch

from seis_interp.data.c3_poc_trace_graph import C3PocTraceGraphTrainingData
from seis_interp.data.forge_aligned_inputs import align_model_inputs, unalign_predictions
from seis_interp.data.forge_mvp_artifacts import ForgeModelInputs
from seis_interp.data.trace_graph_domain import build_trace_graph_domain
from seis_interp.forge_mvp_config import canvas_time_stages, profile_side, profile_time_multiple
from seis_interp.models.ccnet5d import CCNet5D
from seis_interp.models.nersi import Nersi
from seis_interp.models.nersi_trace import NersiTrace
from seis_interp.models.relational_trace_graph import RelationalTraceGraphInterpolator
from seis_interp.models.trace_context_field import TraceContextField
from seis_interp.models.trace_coordinate_field import TraceCoordinateField
from seis_interp.processing.drr_batched import interpolate_drr_volume_batched
from seis_interp.processing.drr_windows import interpolate_drr_volume
from seis_interp.processing.forge_nersi_profiles import (
    ExactTraceProfileSampler,
    nersi_profile_layout,
)
from seis_interp.processing.pocs_batched import interpolate_pocs_volume_batched
from seis_interp.processing.pocs_windows import interpolate_pocs_volume
from seis_interp.processing.trace_field_neighbors import nearest_observed_positions
from seis_interp.processing.trace_graph_preprocessing import build_poc_trace_graph_preprocessing
from seis_interp.processing.trace_graph_settings import TraceGraphSettings
from seis_interp.training.ccnet5d_observed_patches import CCNet5DObservedPatchSource
from seis_interp.training.ccnet5d_observed_training import train_ccnet5d_observed_steps
from seis_interp.training.ccnet5d_prediction import predict_ccnet5d_volume
from seis_interp.training.exact_resume import object_state, restore_objects
from seis_interp.training.mixed_precision import mixed_precision_dtype
from seis_interp.training.nersi_loss_support import recorded_trace_samples
from seis_interp.training.nersi_profile_regularization import (
    ProfileCoordinateJitter,
    normalized_nuclear_norm,
    validate_nuclear_norm_weight,
)
from seis_interp.training.neural_diagnostics import NeuralTrainingDiagnostics
from seis_interp.training.phase_progress import PhaseProgress, ProgressWriter
from seis_interp.training.prediction_warmup import (
    warm_ccnet,
    warm_context_field,
    warm_coordinates,
    warm_graph,
)
from seis_interp.training.relational_trace_graph_poc_trainer import (
    build_training_neighbor_table,
    train_relational_trace_graph_poc,
)
from seis_interp.training.relational_trace_graph_prediction import (
    predict_relational_trace_graph,
    predict_relational_trace_graph_states,
)
from seis_interp.training.trace_graph_ema import TraceGraphEMA, validate_trace_graph_ema_decay
from seis_interp.training.trace_graph_execution import (
    torch_cpu_threads,
    validate_trace_graph_execution,
)
from seis_interp.training.trace_relative_loss import masked_trace_loss, masked_trace_relative_mse

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
    if method == "nersi_profile":
        return Nersi(**config["model"])
    if method == "inr_real":
        return TraceCoordinateField(**config["model"])
    if method == "cinr_real":
        return TraceContextField(**config["model"])
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
    """Copy of ``config`` whose trace decoder emits the whole aligned canvas."""
    resolved = deepcopy(config)
    if resolved["method_id"] in ("nersi_real", "inr_real", "cinr_real"):
        resolved["model"]["time_sample_count"] = int(canvas_length)
    if resolved["method_id"] == "nersi_profile":
        normalize = resolved.pop("canvas_time_upsampling", False)
        if not isinstance(normalize, bool):
            raise ValueError("canvas_time_upsampling must be boolean")
        stages = canvas_time_stages(canvas_length) if normalize else 0
        if stages:
            resolved["model"].update(
                time_upsample_channels=[resolved["model"]["latent_channels"]] * stages,
                time_kernel_size=3,
            )
        resolved["model"]["profile_shape"][0] = profile_side(
            canvas_length, profile_time_multiple(resolved["model"])
        )
    return resolved


def run_method(
    inputs: ForgeModelInputs, config: dict, reporter=print, *, checkpoint_path=None, recovery=None
) -> ForgeMethodResult:
    """Run in model amplitude units, then apply the preparation's restoration RMS.

    With ``config["alignment"]`` every method fits and predicts on the common
    geometry-only moveout canvas; predictions return to the original samples by
    the exact inverse shift before amplitude restoration.
    EMA shadows are restored exactly like the primary prediction.
    A CCNet checkpoint path saves its final training weights before prediction.
    """
    if checkpoint_path is not None and config["method_id"] != "ccnet5d_grid":
        raise ValueError("pre-prediction checkpoint saving requires CCNet5D")
    options = {"checkpoint_path": checkpoint_path} if checkpoint_path is not None else {}
    if recovery is not None:
        options["recovery"] = recovery
    model_inputs = inputs.model_space()
    progress = PhaseProgress(reporter)
    canvas = None
    if config.get("alignment") is None:
        resolved = config
        if config.get("method_id") in ("nersi_real", "inr_real", "cinr_real", "nersi_profile"):
            resolved = canvas_method_config(config, len(model_inputs.time_s))
        result = _run_method(model_inputs, resolved, reporter, **options)
    else:
        aligned, canvas = align_model_inputs(model_inputs, config["alignment"], progress=progress)
        if config["method_id"] == "nersi_profile" and config["training"].get("loss_support_mask"):
            options["loss_support"] = (
                canvas.max_delay_samples - canvas.delays_for(aligned.observed_cells),
                canvas.time_sample_count,
            )
        result = _run_method(
            aligned, canvas_method_config(config, canvas.length), reporter, **options
        )
        result.diagnostics["alignment_canvas_sample_count"] = canvas.length
    result.prediction = _physical_prediction(
        result.prediction, result.cell_ids, canvas, inputs, progress=progress
    )
    decays = ema_eval_decays(config.get("training", {}))
    if decays:
        shadows = {
            key: _physical_prediction(value, result.cell_ids, canvas, inputs, progress=progress)
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


def _physical_prediction(prediction, cell_ids, canvas, inputs, *, progress=None):
    """Exact inverse moveout shift, then the preparation's restoration RMS."""
    if canvas is not None:
        prediction = unalign_predictions(prediction, cell_ids, canvas, progress=progress)
    if inputs.test_rms is not None:
        prediction = (prediction.astype(np.float64) * inputs.test_rms[:, None]).astype(np.float32)
    if progress is not None:
        progress(f"restored amplitudes {len(cell_ids)} cells")
    return prediction


def _run_method(
    inputs: ForgeModelInputs,
    config: dict,
    reporter,
    *,
    checkpoint_path=None,
    loss_support=None,
    recovery=None,
) -> ForgeMethodResult:
    """No raw amplitude path or evaluator is reachable through the input object."""
    method = config["method_id"]
    benchmark = config.get("execution", {}).get("cudnn_benchmark", False)
    if not isinstance(benchmark, bool) or (benchmark and method != "ccnet5d_grid"):
        raise ValueError("execution.cudnn_benchmark is a CCNet-only boolean")
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = not benchmark
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
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
    elif method == "inr_real":
        fit = _fit_inr
    elif method == "cinr_real":
        fit = _fit_context_field
    elif method == "nersi_profile":
        fit = _fit_nersi_profile
    elif method in ("regsi_real", "regsi_grid"):
        fit = _fit_regsi
    else:
        raise ValueError("unsupported MVP method")
    fit_options = {} if recovery is None else {"recovery": recovery}
    if loss_support is not None:
        fit_options["loss_support"] = loss_support
    if checkpoint_path is not None:

        def save_training_checkpoint():
            with open(checkpoint_path, "wb") as stream:
                torch.save(
                    _method_checkpoint(model, inputs, config, initialization_hash),
                    ProgressWriter(stream, PhaseProgress(reporter), "saving training checkpoint"),
                )

        fit_options["before_prediction"] = save_training_checkpoint
    prediction, history, shadows, records, states = fit(
        model, inputs, config, reporter, decays, snapshots, **fit_options
    )
    checkpoint = _method_checkpoint(model, inputs, config, initialization_hash)
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


def _method_checkpoint(model, inputs, config, initialization_hash):
    checkpoint = {
        "method_id": config["method_id"],
        "model_config": config["model"],
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "steps_completed": config["training"]["max_steps"],
        "selection_rule": config["checkpoint_rule"],
        "initialization_hash": initialization_hash,
        "model_seed": config["model_seed"],
        "input_binding": inputs.binding,
    }
    if config.get("alignment") is not None:
        checkpoint["alignment"] = dict(config["alignment"])
    return checkpoint


def _reserved_memory(device, phase):
    if not str(device).startswith("cuda"):
        return {}
    return {
        f"cuda_reserved_{phase}_bytes": int(torch.cuda.memory_reserved(device)),
        f"cuda_max_reserved_{phase}_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


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
                shadows[shadow_key(decay)] = predict(f"ema {shadow_key(decay)}")
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


def _fit_ccnet(
    model,
    inputs,
    config,
    reporter,
    decays,
    snapshot_updates,
    *,
    before_prediction=None,
    recovery=None,
):
    device, tr = config["device"], config["training"]
    volume = inputs.volume()
    patch_shape = (volume.values.shape[0], *config["patches"]["shape"][1:])
    placement_seed, mask_seed, supervision_seed = np.random.SeedSequence(
        config["model_seed"]
    ).spawn(3)
    source = CCNet5DObservedPatchSource(
        volume,
        amplitude_scale=inputs.global_rms,
        patch_shape=patch_shape,
        inner_mask_fraction=config["patches"]["inner_mask_fraction"],
        placement_seed=placement_seed,
        inner_mask_seed=mask_seed,
        supervision_seed=supervision_seed,
    )
    warm_ccnet(model, volume.values.shape, config, reporter=reporter)
    warm_memory = _reserved_memory(device, "after_warmup")
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
        mixed_precision=tr["mixed_precision"],
        reuse_patch=config.get("execution", {}).get("reuse_patch", False),
        spatial_query_chunk_size=config.get("execution", {}).get("spatial_query_chunk_size"),
        cache_patch_gradients=config.get("execution", {}).get("cache_patch_gradients", False),
        recovery=recovery,
        prefetch=config.get("execution", {}).get("patch_prefetch", True),
        profile_first_update=torch.device(device).type == "cuda",
        reporter=reporter,
    )
    records = {
        "training_end": _training_end(device),
        **warm_memory,
        **_reserved_memory(device, "before_prediction"),
    }
    if result.time_conv_tensor_cores is not None:
        records["conv_kernels_first_update"] = list(result.conv_kernels_first_update)
        records["time_conv_tensor_cores"] = result.time_conv_tensor_cores
        if not result.time_conv_tensor_cores:
            reporter("WARNING: CCNet5D time convolutions did not all use Tensor Core kernels")
    history = list(result.history)
    if before_prediction is not None:
        before_prediction()
        reporter("CCNet training checkpoint saved before prediction")

    def parameter_state(parameters):
        state = _cpu_state(model.state_dict())
        for (name, _), value in zip(model.named_parameters(), parameters, strict=True):
            state[name] = value.detach().cpu().clone()
        return state

    def predict(phase="primary"):
        core_shape = tuple(config["prediction"]["core_shape"])
        if config.get("execution", {}).get("full_trace_prediction", False):
            core_shape = (volume.values.shape[0], *core_shape[1:])
        values = predict_ccnet5d_volume(
            model,
            volume,
            amplitude_rms=inputs.global_rms,
            core_shape=core_shape,
            device=device,
            mixed_precision=config["prediction"].get("mixed_precision", "off"),
            reporter=reporter,
            phase=phase,
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
        **result.diagnostics,
        final_training_loss=result.final_loss,
        **_window_loss(history, tr["max_steps"]),
        mean_patches_per_update=source.patches_drawn / tr["max_steps"],
        mean_visible_observed_cells_per_update=(
            source.visible_observed_cells_drawn / tr["max_steps"]
        ),
        supervised_traces_per_update=result.supervised_trace_presentations / tr["max_steps"],
        effective_inner_fraction_per_patch=source.effective_inner_fractions,
        rng_streams="SeedSequence(model_seed).spawn(3): placement, mask, supervision",
        cudnn_benchmark=config.get("execution", {}).get("cudnn_benchmark", False),
    )
    states = _ema_states(final_states, snapshots, tr["max_steps"])
    records.update(_reserved_memory(device, "after_prediction"))
    return prediction, history, shadows, records, states


def _fit_regsi(model, inputs, config, reporter, decays, snapshot_updates, *, recovery=None):
    """Optional ``training.execution`` switches only change how the fit is executed."""
    device, tr = config["device"], config["training"]
    execution = validate_trace_graph_execution(
        tr.get("execution"), neighbor_search=config["graph"]["neighbor_search"], device=device
    )
    domain, training, settings = graph_data(inputs, config)
    table = build_training_neighbor_table(training, settings) if execution.neighbor_table else None
    primary = tr.get("ema_decay")
    ema = TraceGraphEMA(primary, snapshot_updates) if primary else None
    extra = {decay: TraceGraphEMA(decay, snapshot_updates) for decay in decays if decay != primary}
    diagnostics = NeuralTrainingDiagnostics(tr["max_steps"], config["loss"])
    warm_graph(
        model,
        domain,
        training.preprocessing,
        settings,
        inputs.test_cells,
        config["prediction"]["query_batch_size"],
        device,
        table,
        reporter=reporter,
    )
    warm_memory = _reserved_memory(device, "after_warmup")
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
        execution=execution,
        neighbor_table=table,
        diagnostics=diagnostics,
        recovery=recovery,
    )
    records = {
        "training_end": _training_end(device),
        **warm_memory,
        **_reserved_memory(device, "before_prediction"),
    }
    history = result.history
    if ema is not None:
        model.load_state_dict(ema.state_dict())
    observed_waveforms = inputs.read_observed(domain.trace_ids[domain.observed_mask])
    options = {
        "graph_settings": settings,
        "query_trace_ids": inputs.test_cells,
        "observed_waveforms": observed_waveforms,
        "query_batch_size": config["prediction"]["query_batch_size"],
        "device": device,
    }

    def predict(phase="primary"):
        return predict_relational_trace_graph(
            model, domain, training.preprocessing, reporter=reporter, phase=phase, **options
        ).prediction

    with torch_cpu_threads(execution.cpu_threads):
        if execution.shared_prediction:
            prediction, shadows, timing, final_states = _shared_regsi_predictions(
                model,
                domain,
                training,
                options | {"neighbor_table": table, "reporter": reporter},
                decays,
                primary,
                extra,
            )
            records["runtime_prediction_seconds"] = timing.pop("runtime_primary_seconds")
        else:
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
        **diagnostics.records(),
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
    records.update(_reserved_memory(device, "after_prediction"))
    return prediction, history, shadows, records, states


def _shared_regsi_predictions(model, domain, training, options, decays, primary, extra):
    """The primary and every non-primary shadow from one graph per batch.

    Same predictions and shadow states as ``_shadow_predictions``; ``model`` keeps
    the primary weights. The shadows' share of the runtime is their forward passes.
    """
    started = perf_counter()
    others = [decay for decay in decays if decay != primary] if len(decays) > 1 else []
    states = [model.state_dict()] + [extra[decay].state_dict() for decay in others]
    predicted = predict_relational_trace_graph_states(
        model,
        states,
        domain,
        training.preprocessing,
        phases=["primary", *(f"ema {shadow_key(d)}" for d in others)],
        **options,
    )
    total = perf_counter() - started
    shadow_seconds = sum(item.diagnostics["timings"]["forward_seconds"] for item in predicted[1:])
    shadows = {
        shadow_key(d): item.prediction for d, item in zip(others, predicted[1:], strict=True)
    }
    final_states = {
        shadow_key(d): _cpu_state(state) for d, state in zip(others, states[1:], strict=True)
    }
    timing = {
        "runtime_primary_seconds": total - shadow_seconds,
        "runtime_shadow_prediction_seconds": shadow_seconds,
    }
    return predicted[0].prediction, shadows, timing, final_states


def _weight_decay_after(training):
    """Optional one-time switch of Adam's weight decay, applied before that step's update."""
    schedule = training.get("weight_decay_after")
    if schedule is None:
        return None
    step, value = schedule.get("step"), schedule.get("value")
    if (
        set(schedule) != {"step", "value"}
        or isinstance(step, bool)
        or not isinstance(step, int)
        or not 1 <= step <= training["max_steps"]
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(value)
        or value < 0
    ):
        raise ValueError(
            "training.weight_decay_after needs an integer step in [1, max_steps] and a "
            "finite value >= 0"
        )
    return {"step": step, "value": float(value)}


def _fit_nersi(model, inputs, config, reporter, decays, snapshot_updates, *, recovery=None):
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
    decay_after = _weight_decay_after(tr)
    history = []
    window_first = tr["max_steps"] - min(LOSS_WINDOW_UPDATES, tr["max_steps"])
    window_losses = []
    warm_coordinates(
        model, (config["prediction"]["batch_size"], coordinates.shape[1]), device, reporter=reporter
    )
    warm_memory = _reserved_memory(device, "after_warmup")
    objects = {"ema": ema} | {str(key): value for key, value in extra.items()}
    state = {} if recovery is None else recovery.load(model, optimizer)
    if state:
        restore_objects(objects, state["objects"])
        rng.bit_generator.state = state["sampler"]
        history, window_losses = state["history"], state["window_losses"]
    for step in range(state.get("step", 0) + 1, tr["max_steps"] + 1):
        rows = rng.choice(len(inputs.observed_cells), size=tr["batch_size"], replace=False)
        target = torch.from_numpy(
            (inputs.observed_values[rows].astype(np.float64) / inputs.global_rms).astype(np.float32)
        ).to(device)
        if decay_after is not None and step == decay_after["step"]:
            for group in optimizer.param_groups:
                group["weight_decay"] = decay_after["value"]
        optimizer.zero_grad(set_to_none=True)
        prediction = model(coordinates[observed_rows[rows]].to(device))
        loss = masked_trace_relative_mse(prediction, target)
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite NeRSI loss at step {step}")
        loss.backward()
        optimizer.step()
        if ema is not None:
            ema.update(model)
        for average in extra.values():
            average.update(model)
        if step > window_first:
            window_losses.append(float(loss.detach()))
        reporting = step % tr["report_interval"] == 0 or step == tr["max_steps"]
        if reporting or (recovery is not None and recovery.record_all_steps):
            value = float(loss.detach())
            history.append({"step": step, "loss": value, "learning_rate": tr["learning_rate"]})
        if recovery is not None and (recovery.due(step) or step == tr["max_steps"]):
            recovery.save(
                step,
                model,
                optimizer,
                {
                    "objects": object_state(objects),
                    "sampler": rng.bit_generator.state,
                    "history": history,
                    "window_losses": window_losses,
                },
                force=True,
                phase="before_prediction" if step == tr["max_steps"] else "training",
            )
        if reporting:
            reporter(f"NeRSI step {step}/{tr['max_steps']} minibatch_loss {value:.6g}")
    records = {
        "training_end": _training_end(device),
        **warm_memory,
        **_reserved_memory(device, "before_prediction"),
    }
    if ema is not None:
        model.load_state_dict(ema.state_dict())
    model.eval()

    def predict(phase="primary"):
        progress = PhaseProgress(reporter)
        output = np.empty((len(inputs.test_cells), len(inputs.time_s)), dtype=np.float32)
        with torch.inference_mode():
            batch = config["prediction"]["batch_size"]
            for start in range(0, len(output), batch):
                rows = [positions[int(i)] for i in inputs.test_cells[start : start + batch]]
                values = model(coordinates[rows].to(device)).cpu().numpy().astype(np.float64)
                output[start : start + batch] = values * inputs.global_rms
                progress(
                    f"predicting {min(start + batch, len(output))}/{len(output)} cells ({phase})"
                )
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
    records.update(_reserved_memory(device, "after_prediction"))
    return output, history, shadows, records, states


def _fit_inr(model, inputs, config, reporter, decays, snapshot_updates, *, recovery=None):
    """Fit the implicit trace field at four unrounded local coordinates; fixed final EMA.

    One update draws ``training.batch_size`` complete observed traces without
    replacement and scores the declared ``loss`` on them. The field returns the
    whole trace from one linear head, so no output sample is discarded. Extra EMA
    shadows and snapshots never touch optimization.
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
    diagnostics = NeuralTrainingDiagnostics(tr["max_steps"], config["loss"])
    model.to(device)
    if tr["optimizer"] not in ("adam", "adamw"):
        raise ValueError("the trace field optimizer must be adam or adamw")
    optimizer_type = torch.optim.Adam if tr["optimizer"] == "adam" else torch.optim.AdamW
    optimizer = optimizer_type(
        model.parameters(), lr=tr["learning_rate"], weight_decay=tr["weight_decay"]
    )
    rng = np.random.default_rng(config["model_seed"])
    primary = tr.get("ema_decay")
    ema = TraceGraphEMA(primary, snapshot_updates) if primary else None
    extra = {decay: TraceGraphEMA(decay, snapshot_updates) for decay in decays if decay != primary}
    history, window_losses = [], []
    window_first = tr["max_steps"] - min(LOSS_WINDOW_UPDATES, tr["max_steps"])
    warm_coordinates(
        model, (config["prediction"]["batch_size"], coordinates.shape[1]), device, reporter=reporter
    )
    warm_memory = _reserved_memory(device, "after_warmup")
    objects = {"ema": ema, "diagnostics": diagnostics} | {
        str(key): value for key, value in extra.items()
    }
    state = {} if recovery is None else recovery.load(model, optimizer)
    if state:
        restore_objects(objects, state["objects"])
        rng.bit_generator.state = state["sampler"]
        history, window_losses = state["history"], state["window_losses"]
    for step in range(state.get("step", 0) + 1, tr["max_steps"] + 1):
        rows = rng.choice(len(inputs.observed_cells), size=tr["batch_size"], replace=False)
        target = torch.from_numpy(
            (inputs.observed_values[rows].astype(np.float64) / inputs.global_rms).astype(np.float32)
        ).to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(coordinates[observed_rows[rows]].to(device))
        loss = (
            masked_trace_relative_mse(prediction, target)
            if config["loss"] == "masked_trace_relative_mse"
            else masked_trace_loss(prediction, target, loss_name=config["loss"])
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite trace field loss at step {step}")
        diagnostics.observe(step, prediction, target, loss)
        loss.backward()
        diagnostics.gradients(model)
        optimizer.step()
        if ema is not None:
            ema.update(model)
        for average in extra.values():
            average.update(model)
        if step > window_first:
            window_losses.append(float(loss.detach()))
        reporting = step % tr["report_interval"] == 0 or step == tr["max_steps"]
        if reporting or (recovery is not None and recovery.record_all_steps):
            value = float(loss.detach())
            history.append({"step": step, "loss": value, "learning_rate": tr["learning_rate"]})
        if recovery is not None and (recovery.due(step) or step == tr["max_steps"]):
            recovery.save(
                step,
                model,
                optimizer,
                {
                    "objects": object_state(objects),
                    "sampler": rng.bit_generator.state,
                    "history": history,
                    "window_losses": window_losses,
                },
                force=True,
                phase="before_prediction" if step == tr["max_steps"] else "training",
            )
        if reporting:
            reporter(f"trace field step {step}/{tr['max_steps']} minibatch_loss {value:.6g}")
    records = {
        "training_end": _training_end(device),
        **warm_memory,
        **_reserved_memory(device, "before_prediction"),
    }
    if ema is not None:
        model.load_state_dict(ema.state_dict())
    model.eval()

    def predict(phase="primary"):
        progress = PhaseProgress(reporter)
        output = np.empty((len(inputs.test_cells), len(inputs.time_s)), dtype=np.float32)
        with torch.inference_mode():
            batch = config["prediction"]["batch_size"]
            for start in range(0, len(output), batch):
                rows = [positions[int(i)] for i in inputs.test_cells[start : start + batch]]
                values = model(coordinates[rows].to(device)).cpu().numpy().astype(np.float64)
                output[start : start + batch] = values * inputs.global_rms
                progress(
                    f"predicting {min(start + batch, len(output))}/{len(output)} cells ({phase})"
                )
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
        **diagnostics.records(),
        final_training_loss=window_losses[-1],
        final_training_loss_window_mean=float(np.mean(window_losses)),
        final_training_loss_window_updates=len(window_losses),
    )
    states = _ema_states(final_states, snapshots, tr["max_steps"])
    records.update(_reserved_memory(device, "after_prediction"))
    return output, history, shadows, records, states


def _fit_context_field(model, inputs, config, reporter, decays, snapshot_updates, *, recovery=None):
    """Fit observed traces from their nearest other observed traces; fixed final EMA.

    Each update draws ``training.batch_size`` complete observed traces as queries and
    reads ``context.neighbor_count`` other observed traces as their context, so a query
    is never reconstructed from its own amplitudes. A neighbourhood follows the four
    normalized local coordinates alone and is selected once, before any fitting.
    """
    columns = [
        f"{role}_{axis}_continuous" for role in ("source", "receiver") for axis in ("line", "point")
    ]
    table = inputs.geometry.set_index("cell_id")
    coordinates = table[columns].to_numpy() / np.maximum(np.asarray(inputs.spatial_shape) - 1, 1)
    positions = {int(cell): i for i, cell in enumerate(table.index)}
    observed_rows = np.array([positions[int(cell)] for cell in inputs.observed_cells])
    query_rows = np.array([positions[int(cell)] for cell in inputs.test_cells])
    device, tr = config["device"], config["training"]
    count = config["context"]["neighbor_count"]
    progress = PhaseProgress(reporter)
    started = perf_counter()
    observed_context = nearest_observed_positions(coordinates, observed_rows, observed_rows, count)
    progress("observed neighbourhoods selected")
    query_context = nearest_observed_positions(coordinates, observed_rows, query_rows, count)
    progress("query neighbourhoods selected")
    neighbor_seconds = perf_counter() - started
    observed_coordinates = torch.from_numpy(coordinates[observed_rows].astype(np.float32)).to(
        device
    )
    query_coordinates = torch.from_numpy(coordinates[query_rows].astype(np.float32)).to(device)
    observed_values = torch.from_numpy(
        (inputs.observed_values.astype(np.float64) / inputs.global_rms).astype(np.float32)
    ).to(device)
    observed_context = torch.from_numpy(observed_context).to(device)
    query_context = torch.from_numpy(query_context).to(device)
    diagnostics = NeuralTrainingDiagnostics(tr["max_steps"], config["loss"])
    model.to(device)
    if tr["optimizer"] not in ("adam", "adamw"):
        raise ValueError("the trace context field optimizer must be adam or adamw")
    optimizer_type = torch.optim.Adam if tr["optimizer"] == "adam" else torch.optim.AdamW
    optimizer = optimizer_type(
        model.parameters(), lr=tr["learning_rate"], weight_decay=tr["weight_decay"]
    )
    rng = np.random.default_rng(config["model_seed"])
    primary = tr.get("ema_decay")
    ema = TraceGraphEMA(primary, snapshot_updates) if primary else None
    extra = {decay: TraceGraphEMA(decay, snapshot_updates) for decay in decays if decay != primary}
    history, window_losses = [], []
    window_first = tr["max_steps"] - min(LOSS_WINDOW_UPDATES, tr["max_steps"])
    warm_context_field(
        model,
        (config["prediction"]["batch_size"], count, len(inputs.time_s)),
        device,
        reporter=reporter,
    )
    warm_memory = _reserved_memory(device, "after_warmup")
    objects = {"ema": ema, "diagnostics": diagnostics} | {
        str(key): value for key, value in extra.items()
    }
    state = {} if recovery is None else recovery.load(model, optimizer)
    if state:
        restore_objects(objects, state["objects"])
        rng.bit_generator.state = state["sampler"]
        history, window_losses = state["history"], state["window_losses"]
    for step in range(state.get("step", 0) + 1, tr["max_steps"] + 1):
        rows = torch.from_numpy(
            rng.choice(len(observed_rows), size=tr["batch_size"], replace=False)
        ).to(device)
        neighbors = observed_context[rows]
        optimizer.zero_grad(set_to_none=True)
        prediction = model(
            observed_coordinates[rows],
            observed_values[neighbors],
            observed_coordinates[neighbors],
        )
        target = observed_values[rows]
        loss = (
            masked_trace_relative_mse(prediction, target)
            if config["loss"] == "masked_trace_relative_mse"
            else masked_trace_loss(prediction, target, loss_name=config["loss"])
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite trace context field loss at step {step}")
        diagnostics.observe(step, prediction, target, loss)
        loss.backward()
        diagnostics.gradients(model)
        optimizer.step()
        if ema is not None:
            ema.update(model)
        for average in extra.values():
            average.update(model)
        if step > window_first:
            window_losses.append(float(loss.detach()))
        reporting = step % tr["report_interval"] == 0 or step == tr["max_steps"]
        if reporting or (recovery is not None and recovery.record_all_steps):
            value = float(loss.detach())
            history.append({"step": step, "loss": value, "learning_rate": tr["learning_rate"]})
        if recovery is not None and (recovery.due(step) or step == tr["max_steps"]):
            recovery.save(
                step,
                model,
                optimizer,
                {
                    "objects": object_state(objects),
                    "sampler": rng.bit_generator.state,
                    "history": history,
                    "window_losses": window_losses,
                },
                force=True,
                phase="before_prediction" if step == tr["max_steps"] else "training",
            )
        if reporting:
            reporter(f"context field step {step}/{tr['max_steps']} minibatch_loss {value:.6g}")
    records = {
        "training_end": _training_end(device),
        "context_neighbor_count": count,
        "runtime_neighbor_selection_seconds": neighbor_seconds,
        **warm_memory,
        **_reserved_memory(device, "before_prediction"),
    }
    if ema is not None:
        model.load_state_dict(ema.state_dict())
    model.eval()

    def predict(phase="primary"):
        phase_progress = PhaseProgress(reporter)
        output = np.empty((len(inputs.test_cells), len(inputs.time_s)), dtype=np.float32)
        with torch.inference_mode():
            batch = config["prediction"]["batch_size"]
            for start in range(0, len(output), batch):
                stop = min(start + batch, len(output))
                neighbors = query_context[start:stop]
                values = (
                    model(
                        query_coordinates[start:stop],
                        observed_values[neighbors],
                        observed_coordinates[neighbors],
                    )
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
                output[start:stop] = values * inputs.global_rms
                phase_progress(f"predicting {stop}/{len(output)} cells ({phase})")
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
        **diagnostics.records(),
        final_training_loss=window_losses[-1],
        final_training_loss_window_mean=float(np.mean(window_losses)),
        final_training_loss_window_updates=len(window_losses),
    )
    states = _ema_states(final_states, snapshots, tr["max_steps"])
    records.update(_reserved_memory(device, "after_prediction"))
    return output, history, shadows, records, states


def _fit_nersi_profile(
    model, inputs, config, reporter, decays, snapshot_updates, *, loss_support=None, recovery=None
):
    """Fit whole profiles at three unrounded coordinates; exactly k loss traces per update.

    A profile fixes a source cell and a receiver line (``nersi_profile_layout``);
    the decoder's receiver-point columns are the grid points. Each update decodes
    only the drawn profiles holding a selected trace. Fixed final EMA; extra EMA
    shadows and snapshots never touch optimization.

    Optional parameter-free regularizers: ``training.nuclear_norm_weight`` adds that
    weight times the normalized nuclear norm of the decoded profiles to the loss;
    ``training.coordinate_jitter_cells`` perturbs the inputs of the decoded training
    profiles (never prediction). History ``loss`` is the data term only.
    """
    layout = nersi_profile_layout(inputs.geometry, inputs.spatial_shape)
    coordinates = torch.from_numpy(layout.coordinates.astype(np.float32))
    profiles, columns = layout.locate(inputs.observed_cells)
    time_count = len(inputs.time_s)
    device, tr = config["device"], config["training"]
    if not isinstance(tr.get("loss_support_mask", False), bool):
        raise ValueError("training.loss_support_mask must be boolean")
    diagnostics = NeuralTrainingDiagnostics(tr["max_steps"], config["loss"])
    sampler = ExactTraceProfileSampler(
        profiles, tr["supervised_traces_per_update"], config["model_seed"]
    )
    model.to(device)
    if tr["optimizer"] not in ("adam", "adamw"):
        raise ValueError("profile NeRSI optimizer must be adam or adamw")
    optimizer_type = torch.optim.Adam if tr["optimizer"] == "adam" else torch.optim.AdamW
    optimizer = optimizer_type(
        model.parameters(), lr=tr["learning_rate"], weight_decay=tr["weight_decay"]
    )
    primary = tr.get("ema_decay")
    ema = TraceGraphEMA(primary, snapshot_updates) if primary else None
    extra = {decay: TraceGraphEMA(decay, snapshot_updates) for decay in decays if decay != primary}
    nuclear_weight = validate_nuclear_norm_weight(tr.get("nuclear_norm_weight", 0.0))
    jitter = (
        None
        if tr.get("coordinate_jitter_cells") is None
        else ProfileCoordinateJitter(
            tr["coordinate_jitter_cells"], inputs.spatial_shape[:3], config["model_seed"]
        )
    )

    amp_dtype = mixed_precision_dtype(tr["mixed_precision"], device)

    def decode(profile_coordinates):
        """``(profile, receiver point, time)`` at the given profile input coordinates."""
        with (
            torch.autocast(torch.device(device).type, dtype=amp_dtype)
            if amp_dtype is not None and model.training
            else nullcontext()
        ):
            output = model(profile_coordinates.to(device)).float()
        return output[:, 0, :time_count, : layout.receiver_points].transpose(1, 2)

    history, window_losses, decoded = [], [], 0
    window_first = tr["max_steps"] - min(LOSS_WINDOW_UPDATES, tr["max_steps"])
    warm_coordinates(
        model, (config["prediction"]["profile_batch_size"], 3), device, reporter=reporter
    )
    warm_memory = _reserved_memory(device, "after_warmup")
    objects = {"ema": ema, "sampler": sampler, "jitter": jitter, "diagnostics": diagnostics} | {
        str(key): value for key, value in extra.items()
    }
    state = {} if recovery is None else recovery.load(model, optimizer)
    if state:
        restore_objects(objects, state["objects"])
        history, window_losses, decoded = state["history"], state["window_losses"], state["decoded"]
    for step in range(state.get("step", 0) + 1, tr["max_steps"] + 1):
        rows = sampler.sample()
        selected, inverse = np.unique(profiles[rows], return_inverse=True)
        decoded += len(selected)
        target = torch.from_numpy(
            (inputs.observed_values[rows].astype(np.float64) / inputs.global_rms).astype(np.float32)
        ).to(device)
        optimizer.zero_grad(set_to_none=True)
        profile_inputs = coordinates[selected]
        if jitter is not None:
            profile_inputs = jitter(profile_inputs)
        decoded_profiles = decode(profile_inputs)
        prediction = decoded_profiles[
            torch.from_numpy(inverse).to(device), torch.from_numpy(columns[rows]).to(device)
        ]
        if loss_support is not None:
            starts, samples = loss_support
            prediction, target = recorded_trace_samples(prediction, target, starts[rows], samples)
        loss = (
            masked_trace_relative_mse(prediction, target)
            if config["loss"] == "masked_trace_relative_mse"
            else masked_trace_loss(prediction, target, loss_name=config["loss"])
        )
        objective = loss
        if nuclear_weight:
            penalty = normalized_nuclear_norm(decoded_profiles)
            objective = loss + nuclear_weight * penalty
        if not torch.isfinite(objective):
            raise RuntimeError(f"nonfinite NeRSI loss at step {step}")
        diagnostics.observe(step, prediction, target, loss)
        objective.backward()
        diagnostics.gradients(model)
        optimizer.step()
        if ema is not None:
            ema.update(model)
        for average in extra.values():
            average.update(model)
        if step > window_first:
            window_losses.append(float(loss.detach()))
        reporting = step % tr["report_interval"] == 0 or step == tr["max_steps"]
        if reporting or (recovery is not None and recovery.record_all_steps):
            value = float(loss.detach())
            row = {"step": step, "loss": value, "learning_rate": tr["learning_rate"]}
            if nuclear_weight:
                row["nuclear_norm"] = float(penalty.detach())
            history.append(row)
        if recovery is not None and (recovery.due(step) or step == tr["max_steps"]):
            recovery.save(
                step,
                model,
                optimizer,
                {
                    "objects": object_state(objects),
                    "history": history,
                    "window_losses": window_losses,
                    "decoded": decoded,
                },
                force=True,
                phase="before_prediction" if step == tr["max_steps"] else "training",
            )
        if reporting:
            reporter(f"NeRSI profile step {step}/{tr['max_steps']} minibatch_loss {value:.6g}")
    records = {
        "training_end": _training_end(device),
        **warm_memory,
        **_reserved_memory(device, "before_prediction"),
    }
    if ema is not None:
        model.load_state_dict(ema.state_dict())
    model.eval()
    test_profiles, test_columns = layout.locate(inputs.test_cells)
    unique, inverse = np.unique(test_profiles, return_inverse=True)

    def predict(phase="primary"):
        progress = PhaseProgress(reporter)
        output = np.empty((len(inputs.test_cells), time_count), dtype=np.float32)
        batch = config["prediction"]["profile_batch_size"]
        with torch.inference_mode():
            for start in range(0, len(unique), batch):
                values = decode(coordinates[unique[start : start + batch]])
                values = values.cpu().numpy().astype(np.float64)
                rows = np.flatnonzero((inverse >= start) & (inverse < start + batch))
                output[rows] = values[inverse[rows] - start, test_columns[rows]] * inputs.global_rms
                progress(
                    f"predicting {min(start + batch, len(unique))}/{len(unique)} profiles ({phase})"
                )
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
        **diagnostics.records(),
        final_training_loss=window_losses[-1],
        final_training_loss_window_mean=float(np.mean(window_losses)),
        final_training_loss_window_updates=len(window_losses),
        mean_profiles_drawn_per_update=sampler.profiles_drawn / tr["max_steps"],
        mean_profiles_decoded_per_update=decoded / tr["max_steps"],
        loss_support_samples_per_trace=time_count if loss_support is None else loss_support[1],
    )
    states = _ema_states(final_states, snapshots, tr["max_steps"])
    records.update(_reserved_memory(device, "after_prediction"))
    return output, history, shadows, records, states
