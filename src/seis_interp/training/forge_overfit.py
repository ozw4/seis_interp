"""Fixed observed-only batches for diagnosing FORGE neural optimization."""

from dataclasses import dataclass

import numpy as np
import torch

from seis_interp.processing.trace_graph_geometry import compute_trace_graph_geometry
from seis_interp.processing.trace_graph_subgraphs import build_trace_graph_subgraph
from seis_interp.training.c3_poc_trace_graph_episodes import (
    PocTraceGraphEpisodeGenerator,
    PocTraceGraphEpisodeLabelReader,
    build_poc_trace_graph_context_source,
)
from seis_interp.training.ccnet5d_observed_patches import CCNet5DObservedPatchSource
from seis_interp.training.forge_mvp_methods import graph_data, initialize_model, model_state_hash
from seis_interp.training.trace_graph_ema import TraceGraphEMA
from seis_interp.training.trace_relative_loss import masked_trace_relative_mse

NEURAL_METHODS = ("ccnet5d_grid", "nersi_real", "regsi_real", "regsi_grid")


@dataclass
class FixedForgeBatch:
    method: str
    model_input: object
    target: torch.Tensor
    target_cells: np.ndarray
    context_cells: np.ndarray
    target_mask: torch.Tensor | None = None

    def predict(self, model):
        if self.method.startswith("regsi_"):
            return model(self.model_input)[0]
        prediction = model(self.model_input)
        if self.method == "ccnet5d_grid":
            return prediction[:, 0].movedim(1, -1)[self.target_mask]
        return prediction


def fixed_forge_batch(inputs, config):
    """Freeze the first native training batch; labels and context belong to O only."""
    inputs = inputs.model_space()
    method, device = config["method_id"], config["device"]
    if method == "ccnet5d_grid":
        source = CCNet5DObservedPatchSource(
            inputs.volume(),
            amplitude_scale=inputs.global_rms,
            patch_shape=tuple(config["patches"]["shape"]),
            inner_mask_fraction=config["patches"]["inner_mask_fraction"],
            placement_seed=config["model_seed"],
            inner_mask_seed=config["model_seed"],
        )
        patch = source.sample()
        cells = np.arange(np.prod(inputs.spatial_shape)).reshape(inputs.spatial_shape)[
            patch.patch_slices[1:]
        ]
        mask = torch.from_numpy(patch.pseudo_target_mask[None]).to(device)
        target = torch.from_numpy(patch.pseudo_target[None]).to(device).movedim(1, -1)[mask]
        return FixedForgeBatch(
            method,
            torch.from_numpy(patch.model_input[None, None]).to(device),
            target,
            cells[patch.pseudo_target_mask],
            cells[patch.visible_observed_mask],
            mask,
        )
    if method == "nersi_real":
        rows = np.random.default_rng(config["model_seed"]).choice(
            len(inputs.observed_cells), size=config["training"]["batch_size"], replace=False
        )
        cells = inputs.observed_cells[rows]
        columns = [
            f"{role}_{axis}_continuous"
            for role in ("source", "receiver")
            for axis in ("line", "point")
        ]
        coordinates = inputs.geometry.set_index("cell_id").loc[cells, columns].to_numpy()
        coordinates /= np.maximum(np.asarray(inputs.spatial_shape) - 1, 1)
        target = (inputs.read_observed(cells).astype(np.float64) / inputs.global_rms).astype(
            np.float32
        )
        return FixedForgeBatch(
            method,
            torch.from_numpy(coordinates.astype(np.float32)).to(device),
            torch.from_numpy(target).to(device),
            cells,
            np.empty(0, dtype=np.int64),
        )
    if method not in ("regsi_real", "regsi_grid"):
        raise ValueError("overfit diagnostic requires a neural method")
    _, training, settings = graph_data(inputs, config)
    episode = PocTraceGraphEpisodeGenerator(
        training,
        random_seed=config["model_seed"],
        missing_fraction=config["training"]["inner_mask_fraction"],
    ).next_episode()
    cells = next(episode.query_batches(config["training"]["query_batch_size"]))
    domain = training.domain
    positions = {int(cell): row for row, cell in enumerate(domain.trace_ids)}
    rows = [positions[int(cell)] for cell in cells]
    geometry = compute_trace_graph_geometry(
        domain.source_xy_m,
        domain.receiver_xy_m,
        azimuth_min_offset_m=training.preprocessing.azimuth_min_offset_m,
    )
    queries = compute_trace_graph_geometry(
        domain.source_xy_m[rows],
        domain.receiver_xy_m[rows],
        azimuth_min_offset_m=training.preprocessing.azimuth_min_offset_m,
    )
    plan = build_trace_graph_subgraph(
        queries,
        cells,
        geometry,
        domain.trace_ids,
        episode.visible_mask,
        rounds=config["model"]["message_passing_rounds"],
        **settings.subgraph_kwargs(),
    )
    batch = build_poc_trace_graph_context_source(training, episode, device=device).inputs(plan)
    labels = PocTraceGraphEpisodeLabelReader(training, episode, device=device).read(cells)
    context = plan.trace_ids[batch.observed_mask.cpu().numpy()]
    return FixedForgeBatch(method, batch, labels, cells, context)


def prediction_metrics(prediction, target):
    """Training-set fit only, in model amplitude units; no test target access."""
    p, y = prediction.detach().double(), target.detach().double()
    error = (p - y).square().sum(dim=1)
    energy = y.square().sum(dim=1)
    pc, yc = p - p.mean(dim=1, keepdim=True), y - y.mean(dim=1, keepdim=True)
    denominator = pc.norm(dim=1) * yc.norm(dim=1)
    valid = denominator > 0
    correlations = ((pc * yc).sum(dim=1)[valid] / denominator[valid]).cpu().numpy()
    return {
        "relative_mse": float((error / torch.where(energy > 0, energy, y.shape[1])).mean()),
        "prediction_rms": float(p.square().mean().sqrt()),
        "target_rms": float(y.square().mean().sqrt()),
        "median_correlation": float(np.median(correlations)) if len(correlations) else None,
        "undefined_correlation_count": int((~valid).sum()),
    }


def overfit_forge_batch(inputs, config, *, steps, report_interval=100, reporter=print):
    """Repeat one frozen batch with the original optimizer, loss and initialization."""
    if steps < 1 or report_interval < 1:
        raise ValueError("steps and report_interval must be positive")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = initialize_model(config).to(config["device"])
    initial_hash = model_state_hash(model)
    batch = fixed_forge_batch(inputs, config)
    if not set(batch.target_cells).issubset(inputs.observed_cells):
        raise ValueError("overfit targets must belong to outer observed")
    if (
        not set(batch.context_cells).issubset(inputs.observed_cells)
        or np.intersect1d(batch.target_cells, batch.context_cells).size
    ):
        raise ValueError("overfit context must be observed and exclude targets")
    tr = config["training"]
    optimizer_class = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[tr["optimizer"]]
    optimizer = optimizer_class(
        model.parameters(), lr=tr["learning_rate"], weight_decay=tr["weight_decay"]
    )
    ema = TraceGraphEMA(tr["ema_decay"]) if tr.get("ema_decay") else None
    with torch.no_grad():
        initial = prediction_metrics(batch.predict(model), batch.target)
    history = []
    model.train()
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction = batch.predict(model)
        loss = masked_trace_relative_mse(prediction, batch.target)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite overfit loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            tr.get("gradient_clip_norm") or float("inf"),
            error_if_nonfinite=True,
        )
        optimizer.step()
        if ema is not None:
            ema.update(model)
        if step == 1 or step % report_interval == 0 or step == steps:
            record = {
                "step": step,
                "loss_before_update": float(loss.detach()),
                "gradient_norm": float(norm),
            }
            history.append(record)
            reporter(
                f"{config['method_id']} {step}/{steps}: loss={record['loss_before_update']:.8g}"
            )
    model.eval()
    with torch.no_grad():
        raw = batch.predict(model).detach().cpu()
    checkpoint = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    predictions = {"raw": raw.numpy()}
    metrics = {"initial": initial, "raw": prediction_metrics(raw, batch.target.cpu())}
    if ema is not None:
        model.load_state_dict(ema.state_dict())
        with torch.no_grad():
            averaged = batch.predict(model).detach().cpu()
        predictions["ema"] = averaged.numpy()
        metrics["ema"] = prediction_metrics(averaged, batch.target.cpu())
    return {
        "metrics": metrics,
        "history": history,
        "target_cells": batch.target_cells,
        "context_cells": batch.context_cells,
        "target": batch.target.detach().cpu().numpy(),
        "predictions": predictions,
        "checkpoint": checkpoint,
        "initialization_hash": initial_hash,
    }
