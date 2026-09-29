"""Observed trace RMS and common projected-grid IDW scales for FORGE."""

import numpy as np
import pandas as pd

from seis_interp.processing.trace_rms_idw import interpolate_trace_rms_idw


def prepare_oracle_trace_scales(observed_values, observed_cells, test_values, test_cells):
    """Keep authorized test RMS as restoration scalars, separate from model inputs."""
    tables = []
    for values, cells, source in (
        (observed_values, observed_cells, "observed"),
        (test_values, test_cells, "oracle_test"),
    ):
        if values.ndim != 2 or len(values) != len(cells) or not values.shape[1]:
            raise ValueError("oracle RMS waveforms must match their cell IDs")
        rms = np.empty(len(values), dtype=np.float64)
        for start in range(0, len(values), 128):
            rms[start : start + 128] = np.sqrt(
                np.mean(np.square(values[start : start + 128].astype(np.float64)), axis=1)
            )
        if not np.isfinite(rms).all() or np.any(rms <= 0):
            raise ValueError("oracle trace RMS must be finite and positive")
        tables.append(pd.DataFrame({"cell_id": cells, "rms": rms, "scale_source": source}))
    result = pd.concat(tables, ignore_index=True).sort_values("cell_id").reset_index(drop=True)
    if result.cell_id.duplicated().any():
        raise ValueError("oracle trace RMS requires disjoint unique cell IDs")
    return result


def prepare_trace_scales(values, observed_cells, eligible_cells, spatial_shape, grids, settings):
    """Interpolate scalar RMS only; no missing waveform or real coordinate is accepted."""
    if settings != {
        "geometry": "projected_grid",
        "radius": 2,
        "power": 2.0,
        "distance": "physical",
        "restoration": "multiply_once",
    }:
        raise ValueError("unsupported FORGE trace RMS IDW settings")
    rms = np.sqrt(np.mean(np.square(values.astype(np.float64)), axis=1))
    if not np.isfinite(rms).all() or np.any(rms <= 0):
        raise ValueError("observed trace RMS must be finite and positive")
    field = np.zeros(int(np.prod(spatial_shape)), dtype=np.float64)
    mask = np.zeros(len(field), dtype=bool)
    field[observed_cells], mask[observed_cells] = rms, True
    spacing = tuple(
        grids[role][axis]
        for role in ("source", "receiver")
        for axis in ("across_spacing_m", "along_spacing_m")
    )
    scales = interpolate_trace_rms_idw(
        field.reshape((1, *spatial_shape)),
        mask.reshape(spatial_shape),
        radius=settings["radius"],
        power=settings["power"],
        axis_scales=spacing,
    ).ravel()
    selected = scales[eligible_cells]
    if not np.isfinite(selected).all() or np.any(selected <= 0):
        raise ValueError("IDW trace RMS must be finite and positive")
    return pd.DataFrame(
        {
            "cell_id": eligible_cells,
            "rms": selected,
            "scale_source": np.where(mask[eligible_cells], "observed", "idw"),
        }
    ), spacing
