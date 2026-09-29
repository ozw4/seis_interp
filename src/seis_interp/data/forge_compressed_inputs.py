"""Observed-only FORGE inputs with one reversible common amplitude transform."""

import numpy as np

from seis_interp.data.forge_mvp_artifacts import ForgeModelInputs
from seis_interp.processing.amplitude_compression import (
    observed_percentile_scale,
    transform_amplitudes,
)


def compressed_model_inputs(
    geometry,
    mask,
    observed_values,
    spatial_shape,
    time_s,
    mode,
    percentile,
    binding,
    *,
    scale_multiplier=1.0,
):
    """Accept no held-out amplitudes or per-trace scales at the training boundary."""
    if not np.array_equal(geometry.cell_id, mask.cell_id):
        raise ValueError("geometry and mask rows must align")
    observed = mask.split.eq("observed").to_numpy()
    if observed_values.shape != (int(observed.sum()), len(time_s)):
        raise ValueError("expected observed rows only, in mask order")
    if (
        isinstance(scale_multiplier, bool)
        or not np.isfinite(scale_multiplier)
        or scale_multiplier <= 0
    ):
        raise ValueError("scale_multiplier must be finite and positive")
    scale = observed_percentile_scale(observed_values, percentile) * scale_multiplier
    inputs = ForgeModelInputs(
        geometry=geometry,
        observed_cells=mask.loc[observed, "cell_id"].to_numpy(),
        test_cells=mask.loc[~observed, "cell_id"].to_numpy(),
        observed_values=transform_amplitudes(observed_values, scale, mode),
        spatial_shape=tuple(spatial_shape),
        time_s=time_s,
        global_rms=1.0,
        binding=binding,
    )
    return inputs, scale
