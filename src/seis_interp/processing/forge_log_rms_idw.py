"""Non-oracle FORGE trace scales: observed log-RMS IDW on the projected grid."""

import numpy as np

from seis_interp.processing.trace_rms_idw import interpolate_trace_rms_idw

# The projected-grid settings accepted by ``forge_trace_scaling.prepare_trace_scales``.
LOG_RMS_IDW_SETTINGS = {
    "geometry": "projected_grid",
    "radius": 2,
    "power": 2.0,
    "distance": "physical",
    "restoration": "multiply_once",
}


def log_rms_idw_scales(
    observed_values, observed_cells, query_cells, spatial_shape, grids, settings
) -> tuple[np.ndarray, tuple[float, ...]]:
    """Return ``exp(IDW(log RMS))`` of observed traces at ``query_cells``.

    Only observed waveforms are read. Normalized IDW weights make the
    interpolation affine-equivariant, so the log field is shifted to be
    strictly positive, passed through the shared trace-RMS IDW stencil (whose
    single-sample RMS is the value itself), and shifted back.
    """
    if settings != LOG_RMS_IDW_SETTINGS:
        raise ValueError("unsupported FORGE log-RMS IDW settings")
    observed_cells = np.asarray(observed_cells)
    query_cells = np.asarray(query_cells)
    if observed_values.ndim != 2 or len(observed_values) != len(observed_cells):
        raise ValueError("observed waveforms must match their cell IDs")
    if np.isin(query_cells, observed_cells).any():
        raise ValueError("query cells must be missing traces")
    rms = np.empty(len(observed_values), dtype=np.float64)
    for start in range(0, len(observed_values), 128):
        chunk = observed_values[start : start + 128].astype(np.float64)
        rms[start : start + 128] = np.sqrt(np.mean(np.square(chunk), axis=1))
    if not np.isfinite(rms).all() or np.any(rms <= 0):
        raise ValueError("observed trace RMS must be finite and positive")
    log_rms = np.log(rms)
    shift = 1.0 - float(log_rms.min())
    size = int(np.prod(spatial_shape))
    field = np.zeros(size, dtype=np.float64)
    mask = np.zeros(size, dtype=bool)
    field[observed_cells], mask[observed_cells] = log_rms + shift, True
    spacing = tuple(
        float(grids[role][axis])
        for role in ("source", "receiver")
        for axis in ("across_spacing_m", "along_spacing_m")
    )
    interpolated = interpolate_trace_rms_idw(
        field.reshape((1, *spatial_shape)),
        mask.reshape(tuple(spatial_shape)),
        radius=settings["radius"],
        power=settings["power"],
        axis_scales=spacing,
    ).ravel()
    scales = np.exp(interpolated[query_cells] - shift)
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("IDW trace RMS must be finite and positive")
    return scales, spacing
