"""Derived per-trace energy and descriptive metrics, without model execution."""

import numpy as np
import pandas as pd

from seis_interp.evaluation.forge_fair_grid_statistics import scale_invariant_relative_mse
from seis_interp.evaluation.trace_snr import summarize_trace_snr


def waveform_columns(prediction, target):
    p, y = prediction.astype(np.float64), target.astype(np.float64)
    if p.shape != y.shape or p.ndim != 2 or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("prediction and target must be matching finite trace arrays")
    energy = np.square(y).sum(axis=1)
    error = np.square(p - y).sum(axis=1)
    centred_p, centred_y = p - p.mean(axis=1, keepdims=True), y - y.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(centred_p, axis=1) * np.linalg.norm(centred_y, axis=1)
    correlation = np.divide(
        (centred_p * centred_y).sum(axis=1),
        denominator,
        out=np.full(len(y), np.nan),
        where=denominator > 0,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = 10 * (np.log10(energy) - np.log10(error))
    return {
        "prediction_energy": np.square(p).sum(axis=1),
        "target_energy": energy,
        "error_energy": error,
        "relative_mse": error / np.where(energy > 0, energy, y.shape[1]),
        "correlation": np.clip(correlation, -1, 1),
        "snr_db": snr,
    }


def derived_traces(directory, saved, source, scales=None):
    """Recompute from saved physical predictions/targets in bounded chunks.

    FORGE physical predictions already include oracle RMS. Replacing that
    restoration multiplies by k_i = observed-log-RMS-IDW / oracle RMS once.
    Shadow predictions are not saved and cannot supply these columns.
    """
    prediction = np.load(directory / "prediction.npy", mmap_mode="r", allow_pickle=False)
    cells = np.load(directory / "cell_ids.npy", allow_pickle=False)
    if len(np.unique(cells)) != len(cells) or prediction.shape[0] != len(cells):
        raise ValueError("prediction identity mismatch")
    indices = pd.Index(cells).get_indexer(saved.cell_id)
    if np.any(indices < 0):
        raise ValueError("saved prediction misses scoring cells")
    target, positions, samples = source
    if prediction.shape[1] != samples:
        raise ValueError("prediction sample count differs from targets")
    result = saved.copy().reset_index(drop=True)
    scale_rows = None if scales is None else scales.set_index("cell_id").loc[saved.cell_id]
    chunks = []
    for start in range(0, len(saved), 128):
        selected = slice(start, start + 128)
        y = np.ascontiguousarray(
            target[positions.loc[saved.cell_id.iloc[selected]].to_numpy(), :samples]
        )
        p = np.asarray(prediction[indices[selected]])
        columns = waveform_columns(p, y)
        for key in ("target_energy", "error_energy", "relative_mse", "correlation"):
            if not np.allclose(
                columns[key], saved[key].iloc[selected], rtol=1e-9, atol=1e-12, equal_nan=True
            ):
                raise ValueError(f"saved {key} disagrees with waveforms: {directory}")
        if scale_rows is not None:
            scale = scale_rows.iloc[selected]
            for key in ("oracle_rms", "idw_rms", "idw_scale_k_i"):
                columns[key] = scale[key].to_numpy()
            columns["scale_invariant_relative_mse"] = scale_invariant_relative_mse(p, y)
            non_oracle = (p.astype(np.float64) * scale.idw_scale_k_i.to_numpy()[:, None]).astype(
                np.float32
            )
            columns.update(
                {f"idw_{key}": value for key, value in waveform_columns(non_oracle, y).items()}
            )
        chunks.append(pd.DataFrame(columns))
    derived = pd.concat(chunks, ignore_index=True)
    for key in derived:
        result[key] = derived[key].to_numpy()
    return result


def trace_summary(table, prefix=""):
    if not len(table):
        raise ValueError("empty scoring population")
    snr = table[f"{prefix}snr_db"].to_numpy()
    snr_summary = summarize_trace_snr(snr)
    energy, error = (
        float(table[f"{prefix}{key}"].sum()) for key in ("target_energy", "error_energy")
    )
    nmse = error / energy if energy > 0 else None
    finite = np.isfinite(snr).all()
    return {
        "trace_count": len(table),
        "relative_mse": float(table[f"{prefix}relative_mse"].mean()),
        "global_nmse": nmse,
        "global_snr_db": float(-10 * np.log10(nmse)) if nmse is not None and nmse > 0 else None,
        "median_correlation": float(table[f"{prefix}correlation"].median())
        if table[f"{prefix}correlation"].notna().any()
        else None,
        **snr_summary,
        **{
            f"trace_snr_p{q}": float(np.percentile(snr, q)) if finite else None
            for q in (10, 50, 90)
        },
        "fraction_below_0_db": float(np.mean(snr < 0)) if not np.isnan(snr).any() else None,
    }


def population_summaries(table, dataset):
    rows = []
    for population in ("all_eligible",) if dataset == "c3" else ("clean_target", "all_eligible"):
        selected = table if population == "all_eligible" else table.loc[table.clean_target]
        rows.append(
            {
                "population": population,
                "restoration": "physical" if dataset == "c3" else "trace_oracle",
                **trace_summary(selected),
            }
        )
        if dataset == "forge":
            rows.append(
                {
                    "population": population,
                    "restoration": "non_oracle_log_rms_idw",
                    **trace_summary(selected, "idw_"),
                }
            )
            rows.append(
                {
                    "population": population,
                    "restoration": "scale_invariant",
                    "trace_count": len(selected),
                    "relative_mse": float(selected.scale_invariant_relative_mse.mean()),
                }
            )
    return rows
