"""Preserve C3 method settings while adapting FORGE boundaries and amplitude units."""

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace

import numpy as np
import torch

from seis_interp.data.forge_aligned_inputs import align_model_inputs, unalign_predictions
from seis_interp.models.relational_trace_graph import TRACE_GRAPH_MEMORY_OPTIONS
from seis_interp.training.c3_region_methods import run_c3_region_method
from seis_interp.training.forge_mvp_methods import run_method


def pad_ccnet_source_lines(inputs, config):
    """Add unknown outer source lines; first-axis extension preserves every cell ID."""
    required = config["patches"]["shape"][1]
    if any(
        width > length
        for width, length in zip(
            config["patches"]["shape"][2:], inputs.spatial_shape[1:], strict=True
        )
    ):
        raise ValueError("only the outer source-line boundary may need CCNet padding")
    shape = (max(required, inputs.spatial_shape[0]), *inputs.spatial_shape[1:])
    return replace(inputs, spatial_shape=shape)


def run_forge_c3_method(inputs, config, reporter=print, *, checkpoint_path=None, recovery=None):
    """Use unchanged model hyperparameters, with optional lossless activation offloading."""
    config = deepcopy(config)
    method = config["method_id"]
    if checkpoint_path is not None and method != "ccnet5d_grid":
        raise ValueError("pre-prediction checkpoint saving requires CCNet5D")
    memory = config.get("execution", {}).get("model_memory_options", {})
    if memory:
        if method != "regsi_real" or set(memory) - set(TRACE_GRAPH_MEMORY_OPTIONS):
            raise ValueError("only ReGSI memory execution options may override the model")
        config["model"].update(memory)
    if method == "ccnet5d_grid":
        inputs = pad_ccnet_source_lines(inputs, config)
    offload = config.get("execution", {}).get("saved_tensors_on_cpu", False)
    context = (
        torch.autograd.graph.save_on_cpu(pin_memory=True)
        if offload and config["device"].startswith("cuda:")
        else nullcontext()
    )
    if method != "drr_grid":
        with context:
            options = {"checkpoint_path": checkpoint_path} if checkpoint_path is not None else {}
            if recovery is not None:
                options["recovery"] = recovery
            return run_method(inputs, config, reporter, **options)
    model_inputs = inputs.model_space()
    canvas = None
    if config.get("alignment") is not None:
        model_inputs, canvas = align_model_inputs(model_inputs, config["alignment"])
    result = run_c3_region_method(model_inputs, config, reporter)
    if canvas is not None:
        result.prediction = unalign_predictions(result.prediction, result.cell_ids, canvas)
        result.diagnostics["alignment_canvas_sample_count"] = canvas.length
    if inputs.test_rms is not None:
        result.prediction = (
            result.prediction.astype(np.float64) * inputs.test_rms[:, None]
        ).astype(np.float32)
    return result
