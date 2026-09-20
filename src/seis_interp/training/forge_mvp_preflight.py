"""Bounded observed-only shape/backward checks; never produce quality metrics."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import torch

from seis_interp.processing.ccnet5d_tiles import iter_ccnet5d_tiles
from seis_interp.processing.drr import interpolate_drr_frequency_slice
from seis_interp.processing.pocs import interpolate_pocs_frequency_slice
from seis_interp.training.ccnet5d_observed_patches import CCNet5DObservedPatchSource
from seis_interp.training.ccnet5d_observed_training import train_ccnet5d_observed_steps
from seis_interp.training.forge_mvp_methods import initialize_model, model_state_hash, run_method


def check_method_shapes(inputs, config):
    """One kernel slice or one diagnostic update with the exact full time axis.

    Diagnostic weights and predictions are discarded. Settings are copied, and
    the diagnostic step count is never written back to the frozen method config.
    This checks one batch, not a complete-run memory/runtime upper bound.
    """
    method = config["method_id"]
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if method in ("pocs_grid", "drr_grid"):
        volume = inputs.volume()
        algorithm = config["algorithm"]
        shape = (
            algorithm["window_shape"][1:]
            if method == "pocs_grid"
            else algorithm["spatial_window_shape"]
        )
        slices = tuple(
            slice(0, min(a, b)) for a, b in zip(shape, inputs.spatial_shape, strict=True)
        )
        mask = volume.observed_trace_mask[slices]
        # A predetermined non-DC coefficient, from O only, tests actual kernel shape.
        values = volume.values[(slice(None), *slices)].astype(np.float64) / inputs.global_rms
        frequency = np.fft.rfft(values, axis=0, norm="ortho")[1]
        if method == "pocs_grid":
            prediction = interpolate_pocs_frequency_slice(
                frequency,
                mask,
                **{k: algorithm[k] for k in ("n_iterations", "threshold_start", "threshold_end")},
            )
        else:
            prediction = interpolate_drr_frequency_slice(
                frequency,
                mask,
                **{k: algorithm[k] for k in ("rank", "damping_power", "n_iterations")},
            )
        if not np.isfinite(prediction).all():
            raise ValueError("nonfinite kernel preflight")
        return {
            "scope": "one spatial frequency slice at configured window shape and iterations",
            "spatial_shape": list(prediction.shape),
            "parameter_count": 0,
        }
    diagnostic = deepcopy(config)
    diagnostic["training"]["max_steps"] = 1
    diagnostic["training"]["report_interval"] = 1
    if method != "ccnet5d_grid":
        # Full O pool, with a predetermined first prediction batch of query geometry.
        count = config["prediction"].get("query_batch_size", config["prediction"].get("batch_size"))
        test = inputs.test_cells[:count]
        selected = inputs.geometry.cell_id.isin(np.concatenate((inputs.observed_cells, test)))
        reduced = replace(inputs, geometry=inputs.geometry.loc[selected].copy(), test_cells=test)
        result = run_method(reduced, diagnostic, reporter=lambda message: None)
        if not np.isfinite(result.prediction).all() or result.prediction.shape[1] != len(
            inputs.time_s
        ):
            raise ValueError("neural preflight prediction failed")
        return {
            "scope": "one full-time diagnostic update and first query batch; weights discarded",
            "prediction_shape": list(result.prediction.shape),
            **result.diagnostics,
        }
    volume = inputs.volume()
    model = initialize_model(config)
    initial = model_state_hash(model)
    source = CCNet5DObservedPatchSource(
        volume,
        amplitude_scale=inputs.global_rms,
        patch_shape=tuple(config["patches"]["shape"]),
        inner_mask_fraction=config["patches"]["inner_mask_fraction"],
        placement_seed=config["model_seed"],
        inner_mask_seed=config["model_seed"],
    )
    tr = config["training"]
    train_ccnet5d_observed_steps(
        model,
        source,
        device=config["device"],
        optimizer_updates=1,
        learning_rate=tr["learning_rate"],
        report_every_steps=1,
        loss_name=config["loss"],
        optimizer_name=tr["optimizer"],
        weight_decay=tr["weight_decay"],
    )
    tile = next(
        iter(
            iter_ccnet5d_tiles(
                volume.values.shape,
                tuple(config["prediction"]["core_shape"]),
                halo_radius=model.halo_radius,
            )
        )
    )
    values = (volume.values[tile.input_slices].astype(np.float64) / inputs.global_rms).astype(
        np.float32
    )
    with torch.inference_mode():
        prediction = model(torch.from_numpy(values)[None, None].to(config["device"]))
        if not torch.isfinite(prediction).all():
            raise ValueError("nonfinite CCNet preflight")
    return {
        "scope": "one full-time diagnostic update and first halo tile; weights discarded",
        "prediction_shape": list(prediction.shape),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "initialization_hash": initial,
    }
