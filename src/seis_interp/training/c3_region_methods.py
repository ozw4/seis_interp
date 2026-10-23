"""The region experiment uses shared neural models and overlap-specific DRR."""

import numpy as np

from seis_interp.processing.drr_batched import interpolate_drr_volume_batched
from seis_interp.processing.drr_grid import resolve_drr_grid_config
from seis_interp.processing.drr_nonoverlapping import interpolate_drr_nonoverlapping
from seis_interp.training.forge_mvp_methods import ForgeMethodResult, run_method


def run_c3_region_method(inputs, config, reporter=print, *, recovery=None):
    if config["method_id"] != "drr_grid":
        options = {} if recovery is None else {"recovery": recovery}
        return run_method(inputs, config, reporter, **options)
    config = resolve_drr_grid_config(config)
    algorithm = config["algorithm"].copy()
    backend = algorithm.pop("backend")
    interpolate = interpolate_drr_volume_batched
    if backend == "drr_nonoverlapping":
        interpolate = interpolate_drr_nonoverlapping
        algorithm.pop("overlap_policy")
        algorithm.pop("boundary_rule")
    volume = inputs.volume()
    values = (volume.values.astype(np.float64) / inputs.global_rms).astype(np.float32)
    reconstruction = interpolate(
        values,
        volume.observed_trace_mask,
        inputs.time_s,
        **algorithm,
        workers=config["execution"]["workers"],
        batch_matrix_limit=config["execution"].get("batch_matrix_limit", 2_000_000),
        reporter=reporter,
    )
    prediction = reconstruction.values.reshape(len(inputs.time_s), -1)[:, inputs.test_cells].T
    prediction = (prediction.astype(np.float64) * inputs.global_rms).astype(np.float32)
    return ForgeMethodResult(
        prediction,
        inputs.test_cells.copy(),
        [],
        None,
        {
            "block_count": reconstruction.block_count,
            "empty_block_count": reconstruction.empty_block_count,
            "uncovered_trace_count": reconstruction.uncovered_trace_count,
        },
    )
