"""Allocate prediction workspaces before training, retaining the CUDA cache."""

from contextlib import contextmanager, nullcontext

import numpy as np
import torch

from seis_interp.data.masked_trace_source import assemble_masked_trace_graph_inputs
from seis_interp.processing.ccnet5d_tiles import iter_ccnet5d_tiles
from seis_interp.processing.trace_graph_geometry import compute_trace_graph_geometry
from seis_interp.processing.trace_graph_subgraphs import FixedTraceGraphSubgraphBuilder
from seis_interp.training.exact_resume import restore_rng, rng_state


@contextmanager
def prediction_mode(model, device):
    state, training = rng_state(), model.training
    model.to(device).eval()
    try:
        with torch.no_grad():
            yield
    finally:
        model.train(training)
        restore_rng(state)


def warm_coordinates(model, shape, device):
    if torch.device(device).type != "cuda":
        return
    with prediction_mode(model, device):
        model(torch.zeros(shape, device=device))


def warm_ccnet(model, shape, config):
    device = config["device"]
    if torch.device(device).type != "cuda":
        return
    core = tuple(config["prediction"]["core_shape"])
    if config.get("execution", {}).get("full_trace_prediction", False):
        core = (shape[0], *core[1:])
    radius = model.halo_radius
    if core[0] >= shape[0]:
        radius = len(model.network) * (model.kernel_size // 2)
    maximum = max(
        (
            tuple(s.stop - s.start for s in tile.input_slices)
            for tile in iter_ccnet5d_tiles(shape, core, halo_radius=radius)
        ),
        key=lambda value: np.prod(value),
    )
    amp = config["prediction"].get("mixed_precision", "off") == "bf16"
    with (
        prediction_mode(model, device),
        torch.autocast("cuda", dtype=torch.bfloat16) if amp else nullcontext(),
    ):
        model(torch.zeros((1, 1, *maximum), device=device))


def warm_graph(model, domain, preprocessing, settings, query_ids, batch_size, device, table):
    if torch.device(device).type != "cuda":
        return
    geometry = compute_trace_graph_geometry(
        domain.source_xy_m,
        domain.receiver_xy_m,
        azimuth_min_offset_m=preprocessing.azimuth_min_offset_m,
    )
    builder = FixedTraceGraphSubgraphBuilder(
        geometry,
        domain.trace_ids,
        domain.observed_mask,
        neighbor_table=table,
        **settings.subgraph_kwargs(),
    )
    positions = {int(value): i for i, value in enumerate(domain.trace_ids)}
    maximum = None
    for start in range(0, len(query_ids), batch_size):
        ids = query_ids[start : start + batch_size]
        rows = [positions[int(value)] for value in ids]
        queries = compute_trace_graph_geometry(
            domain.source_xy_m[rows],
            domain.receiver_xy_m[rows],
            azimuth_min_offset_m=preprocessing.azimuth_min_offset_m,
        )
        plan = builder.build(queries, ids, rounds=model.message_passing_rounds)
        if maximum is None or (len(plan.trace_ids), plan.edge_index.shape[1]) > (
            len(maximum.trace_ids),
            maximum.edge_index.shape[1],
        ):
            maximum = plan
    if maximum is None:
        return
    batch = assemble_masked_trace_graph_inputs(
        maximum,
        observed_trace_ids=domain.trace_ids[domain.observed_mask],
        observed_waveforms=np.zeros(
            (int(domain.observed_mask.sum()), len(domain.time_s)), np.float32
        ),
        time_s=domain.time_s,
        preprocessing=preprocessing,
        device=device,
    )
    with prediction_mode(model, device):
        model(batch)
