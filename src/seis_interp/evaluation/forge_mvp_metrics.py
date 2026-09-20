"""Original-trace FORGE evaluation, independent recomputation, and paired effects."""

import numpy as np
import pandas as pd
import torch

from seis_interp.training.trace_relative_loss import masked_trace_relative_mse


def validate_predictions(prediction, prediction_cells, mask, samples):
    expected = mask.loc[mask.split.eq("test"), "cell_id"].to_numpy()
    ids = np.asarray(prediction_cells)
    if len(np.unique(ids)) != len(ids) or not np.array_equal(np.sort(ids), np.sort(expected)):
        raise ValueError("prediction must cover each original test cell exactly once")
    if (
        prediction.shape != (len(ids), samples)
        or prediction.dtype != np.float32
        or not np.isfinite(prediction).all()
    ):
        raise ValueError("prediction shape/dtype/finite gate failed")
    return pd.Index(ids).get_indexer(expected)


def evaluate_traces(prediction, target, index):
    """Original float32 amplitudes; retain every test row and QC membership."""
    if (
        prediction.shape != target.shape
        or len(index) != len(target)
        or not np.isfinite(target).all()
    ):
        raise ValueError("evaluation target shape/finite mismatch")
    rows = []
    canonical_sums = {"all_eligible": 0.0, "clean_target": 0.0}
    for start in range(0, len(target), 128):
        p, y = (
            prediction[start : start + 128].astype(np.float64),
            target[start : start + 128].astype(np.float64),
        )
        error_energy = np.square(p - y).sum(axis=1)
        target_energy = np.square(y).sum(axis=1)
        # Independent implementation of the exact existing zero-energy convention.
        relative = error_energy / np.where(target_energy > 0, target_energy, y.shape[1])
        py, yy = p - p.mean(axis=1, keepdims=True), y - y.mean(axis=1, keepdims=True)
        divisor = np.linalg.norm(py, axis=1) * np.linalg.norm(yy, axis=1)
        correlation = np.divide(
            (py * yy).sum(axis=1), divisor, out=np.full(len(y), np.nan), where=divisor > 0
        )
        chunk = index.iloc[start : start + len(y)].copy().reset_index(drop=True)
        chunk["relative_mse"] = relative
        chunk["error_energy"] = error_energy
        chunk["target_energy"] = target_energy
        chunk["correlation"] = np.clip(correlation, -1, 1)
        chunk["prediction_all_zero"] = np.all(p == 0, axis=1)
        chunk["prediction_constant"] = np.ptp(p, axis=1) == 0
        rows.append(chunk)
        for name, selected in (
            ("all_eligible", np.ones(len(y), dtype=bool)),
            ("clean_target", chunk.clean_target.to_numpy()),
        ):
            if selected.any():
                value = masked_trace_relative_mse(
                    torch.from_numpy(prediction[start : start + len(y)][selected]),
                    torch.from_numpy(target[start : start + len(y)][selected]),
                )
                canonical_sums[name] += float(value) * int(selected.sum())
    per_trace = pd.concat(rows, ignore_index=True)
    metrics = summarize_metrics(per_trace)
    for name, summary in metrics.items():
        if not summary["trace_count"]:
            raise ValueError("evaluation domain is empty")
        canonical = canonical_sums[name] / summary["trace_count"]
        if not np.isclose(canonical, summary["masked_trace_relative_mse"], rtol=1e-10, atol=1e-12):
            raise ValueError("independent relative-MSE recomputation failed")
        summary["masked_trace_relative_mse"] = canonical
    return metrics, per_trace


def summarize_metrics(rows):
    result = {}
    for name, selected in (("all_eligible", rows), ("clean_target", rows.loc[rows.clean_target])):
        error = float(selected.error_energy.sum())
        energy = float(selected.target_energy.sum())
        correlations = selected.correlation.dropna().to_numpy()
        nmse = error / energy if energy > 0 else None
        result[name] = {
            "trace_count": len(selected),
            "masked_trace_relative_mse": float(selected.relative_mse.mean()),
            "global_nmse": nmse,
            "snr_db": float(-10 * np.log10(nmse)) if nmse is not None and nmse > 0 else None,
            "snr_status": "finite"
            if nmse is not None and nmse > 0
            else "perfect_prediction"
            if energy > 0
            else "zero_target_energy",
            "correlation": {
                name: float(np.percentile(correlations, q)) if len(correlations) else None
                for name, q in (("p05", 5), ("p25", 25), ("median", 50), ("p75", 75), ("p95", 95))
            },
            "undefined_correlation_count": int(selected.correlation.isna().sum()),
            "zero_prediction_count": int(selected.prediction_all_zero.sum()),
            "constant_prediction_count": int(selected.prediction_constant.sum()),
        }
    return result


def verify_saved_metrics(metrics, per_trace):
    """Recompute summaries from the saved per-trace artifact without model code."""
    recomputed = summarize_metrics(per_trace)
    for domain in recomputed:
        for key, expected in recomputed[domain].items():
            actual = metrics[domain][key]
            if isinstance(expected, dict):
                for quantile in expected:
                    _equal_metric(actual[quantile], expected[quantile])
            else:
                _equal_metric(actual, expected)


def verify_metrics_from_waveforms(metrics, prediction, target, clean):
    """Second pass from waveform arrays, independent of the per-trace artifact."""
    for domain, selected in (
        ("all_eligible", np.ones(len(target), dtype=bool)),
        ("clean_target", np.asarray(clean)),
    ):
        ids = np.flatnonzero(selected)
        relative, correlations = [], []
        total_error = total_energy = 0.0
        for start in range(0, len(ids), 128):
            rows = ids[start : start + 128]
            y, p = target[rows].astype(np.float64), prediction[rows].astype(np.float64)
            residual = p - y
            errors = np.einsum("ij,ij->i", residual, residual)
            energies = np.einsum("ij,ij->i", y, y)
            total_error += float(errors.sum())
            total_energy += float(energies.sum())
            relative.extend(errors / np.where(energies > 0, energies, y.shape[1]))
            for teacher, estimate in zip(y, p, strict=True):
                if np.ptp(teacher) > 0 and np.ptp(estimate) > 0:
                    correlations.append(float(np.corrcoef(teacher, estimate)[0, 1]))
        _equal_metric(metrics[domain]["masked_trace_relative_mse"], float(np.mean(relative)))
        nmse = total_error / total_energy if total_energy else None
        _equal_metric(metrics[domain]["global_nmse"], nmse)
        _equal_metric(
            metrics[domain]["snr_db"],
            float(-10 * np.log10(nmse)) if nmse is not None and nmse > 0 else None,
        )
        for name, q in (("p05", 5), ("p25", 25), ("median", 50), ("p75", 75), ("p95", 95)):
            expected = float(np.percentile(correlations, q)) if correlations else None
            _equal_metric(metrics[domain]["correlation"][name], expected)


def _equal_metric(actual, expected):
    if isinstance(expected, (int, float)):
        if actual is None or not np.isclose(actual, expected, rtol=1e-10, atol=1e-12):
            raise ValueError("saved metric recomputation mismatch")
    elif actual != expected:
        raise ValueError("saved metric recomputation mismatch")


def paired_geometry_effects(real, grid):
    """Match by original identity; difference = real error minus grid error."""
    columns = [
        "trace_id",
        "cell_id",
        "clean_target",
        "source_projection_distance_m",
        "receiver_projection_distance_m",
    ]
    left = real[columns + ["relative_mse"]].rename(columns={"relative_mse": "error_regsi_real"})
    right = grid[columns + ["relative_mse"]].rename(columns={"relative_mse": "error_regsi_grid"})
    pair = left.merge(right, on=columns, validate="one_to_one", how="inner")
    if len(pair) != len(real) or len(pair) != len(grid):
        raise ValueError("ReGSI paired evaluation trace mismatch")
    pair["error_difference"] = pair.error_regsi_real - pair.error_regsi_grid
    pair["source_projection_quartile"] = pd.qcut(
        pair.source_projection_distance_m, 4, labels=[1, 2, 3, 4]
    ).astype(int)
    summaries, quartiles = {}, []
    overall_real, overall_grid = summarize_metrics(real), summarize_metrics(grid)
    for domain, rows in (("all_eligible", pair), ("clean_target", pair.loc[pair.clean_target])):
        summaries[domain] = {
            "trace_count": len(rows),
            "relative_mse_difference": float(rows.error_difference.mean()),
            "fraction_real_lower_error": float(rows.error_difference.lt(0).mean()),
        }
        for metric in ("global_nmse", "snr_db"):
            a, b = overall_real[domain][metric], overall_grid[domain][metric]
            summaries[domain][f"{metric}_difference"] = (
                a - b if a is not None and b is not None else None
            )
        a = overall_real[domain]["correlation"]["median"]
        b = overall_grid[domain]["correlation"]["median"]
        summaries[domain]["median_correlation_difference"] = (
            a - b if a is not None and b is not None else None
        )
        for quartile, group in rows.groupby("source_projection_quartile", sort=True):
            quartiles.append(
                {
                    "domain": domain,
                    "quartile": int(quartile),
                    "trace_count": len(group),
                    "source_projection_min_m": float(group.source_projection_distance_m.min()),
                    "source_projection_max_m": float(group.source_projection_distance_m.max()),
                    "relative_mse_real": float(group.error_regsi_real.mean()),
                    "relative_mse_grid": float(group.error_regsi_grid.mean()),
                }
            )
    return pair.sort_values("cell_id"), summaries, pd.DataFrame(quartiles)
