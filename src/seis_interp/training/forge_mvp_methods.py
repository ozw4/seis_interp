"""Six fixed-method adapters consuming only compact observed FORGE inputs."""

import hashlib
from dataclasses import dataclass

import numpy as np
import torch

from seis_interp.data.c3_poc_trace_graph import C3PocTraceGraphTrainingData
from seis_interp.data.forge_mvp_artifacts import ForgeModelInputs
from seis_interp.data.trace_graph_domain import build_trace_graph_domain
from seis_interp.models.ccnet5d import CCNet5D
from seis_interp.models.nersi_trace import NersiTrace
from seis_interp.models.relational_trace_graph import RelationalTraceGraphInterpolator
from seis_interp.processing.drr_windows import interpolate_drr_volume
from seis_interp.processing.pocs_windows import interpolate_pocs_volume
from seis_interp.processing.trace_graph_preprocessing import build_poc_trace_graph_preprocessing
from seis_interp.processing.trace_graph_settings import TraceGraphSettings
from seis_interp.training.ccnet5d_observed_patches import CCNet5DObservedPatchSource
from seis_interp.training.ccnet5d_observed_training import train_ccnet5d_observed_steps
from seis_interp.training.ccnet5d_prediction import predict_ccnet5d_volume
from seis_interp.training.relational_trace_graph_poc_trainer import train_relational_trace_graph_poc
from seis_interp.training.relational_trace_graph_prediction import predict_relational_trace_graph
from seis_interp.training.trace_graph_ema import TraceGraphEMA
from seis_interp.training.trace_relative_loss import masked_trace_relative_mse


@dataclass
class ForgeMethodResult:
    prediction: np.ndarray
    cell_ids: np.ndarray
    history: list
    checkpoint: dict | None
    diagnostics: dict


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


def run_method(inputs: ForgeModelInputs, config: dict, reporter=print) -> ForgeMethodResult:
    """No raw amplitude path or evaluator is reachable through the input object."""
    method, device = config["method_id"], config["device"]
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if method in ("pocs_grid", "drr_grid"):
        volume = inputs.volume()
        observed = (volume.values.astype(np.float64) / inputs.global_rms).astype(np.float32)
        if method == "pocs_grid":
            result = interpolate_pocs_volume(
                observed, volume.observed_trace_mask, **config["algorithm"]
            )
        else:
            result = interpolate_drr_volume(
                observed, volume.observed_trace_mask, inputs.time_s, **config["algorithm"]
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
    if method == "ccnet5d_grid":
        volume = inputs.volume()
        source = CCNet5DObservedPatchSource(
            volume,
            amplitude_scale=inputs.global_rms,
            patch_shape=tuple(config["patches"]["shape"]),
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
            optimizer_name=tr["optimizer"],
            weight_decay=tr["weight_decay"],
            reporter=reporter,
        )
        history = list(result.history)
        values = predict_ccnet5d_volume(
            model,
            volume,
            amplitude_rms=inputs.global_rms,
            core_shape=tuple(config["prediction"]["core_shape"]),
            device=device,
        ).values
        prediction = values.reshape(len(inputs.time_s), -1)[:, inputs.test_cells].T.copy()
    elif method == "nersi_real":
        prediction, history = _fit_nersi(model, inputs, config, reporter)
    elif method in ("regsi_real", "regsi_grid"):
        domain, training, settings = graph_data(inputs, config)
        ema = TraceGraphEMA(tr["ema_decay"]) if tr.get("ema_decay") else None
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
            device=device,
            reporter=reporter,
        )
        history = result.history
        if ema is not None:
            model.load_state_dict(ema.state_dict())
        prediction = predict_relational_trace_graph(
            model,
            domain,
            training.preprocessing,
            graph_settings=settings,
            query_trace_ids=inputs.test_cells,
            observed_waveforms=inputs.read_observed(domain.trace_ids[domain.observed_mask]),
            query_batch_size=config["prediction"]["query_batch_size"],
            device=device,
        ).prediction
    else:
        raise ValueError("unsupported MVP method")
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
    return ForgeMethodResult(
        prediction,
        inputs.test_cells.copy(),
        history,
        checkpoint,
        {"parameter_count": parameter_count, "initialization_hash": initialization_hash},
    )


def _fit_nersi(model, inputs, config, reporter):
    """Fit observed traces at four unrounded local coordinates; fixed final EMA."""
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
    ema = TraceGraphEMA(tr["ema_decay"]) if tr.get("ema_decay") else None
    history = []
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
        if step % tr["report_interval"] == 0 or step == tr["max_steps"]:
            history.append(
                {"step": step, "loss": float(loss.detach()), "learning_rate": tr["learning_rate"]}
            )
            reporter(f"NeRSI step {step}/{tr['max_steps']}")
    if ema is not None:
        model.load_state_dict(ema.state_dict())
    model.eval()
    output = np.empty((len(inputs.test_cells), len(inputs.time_s)), dtype=np.float32)
    with torch.inference_mode():
        batch = config["prediction"]["batch_size"]
        for start in range(0, len(output), batch):
            rows = [positions[int(i)] for i in inputs.test_cells[start : start + batch]]
            values = model(coordinates[rows].to(device)).cpu().numpy().astype(np.float64)
            output[start : start + batch] = values * inputs.global_rms
    return output, history
