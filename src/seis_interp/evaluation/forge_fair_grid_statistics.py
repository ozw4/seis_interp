"""Pure statistics for the pre-registered FORGE fair-grid final evaluation.

Every function takes arrays or tables and returns values; nothing here reads
files, targets, or configurations. Paired differences are always
``left error - right error``, so a negative value favours the left method.
"""

import math

import numpy as np
import pandas as pd

# The six pre-declared contrasts (plan section 2.4), in their declared order.
FAIR_GRID_CONTRASTS = (
    ("regsi_grid_vs_ccnet5d_grid", "regsi_grid", "ccnet5d_grid", "headline_matched_geometry"),
    ("regsi_real_vs_nersi_real", "regsi_real", "nersi_real", "headline_matched_geometry"),
    ("regsi_real_vs_regsi_grid", "regsi_real", "regsi_grid", "geometry_effect"),
    ("regsi_grid_vs_pocs_grid", "regsi_grid", "pocs_grid", "classical_baseline"),
    ("regsi_grid_vs_drr_grid", "regsi_grid", "drr_grid", "classical_baseline"),
    ("regsi_real_vs_ccnet5d_grid", "regsi_real", "ccnet5d_grid", "includes_geometry_difference"),
)


def seed_mean_sd(values) -> tuple[float, float | None]:
    """Mean and sample standard deviation (ddof=1) over seeds; sd is None for one seed."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("seed values must be a nonempty finite vector")
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else None


def seed_average_errors(per_seed_errors) -> np.ndarray:
    """Average per-trace errors over seeds; every seed must score the same rows."""
    stacked = np.asarray(per_seed_errors, dtype=np.float64)
    if stacked.ndim != 2 or not stacked.shape[0] or not np.isfinite(stacked).all():
        raise ValueError("per-seed errors must be a finite [seed, trace] array")
    return stacked.mean(axis=0)


def scale_invariant_relative_mse(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Relative MSE after the per-trace least-squares scalar ``<p, y> / <p, p>``.

    A zero prediction keeps scalar 0. The denominator follows the primary
    metric's zero-energy convention (target energy, or the sample count).
    """
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must be matching [trace, sample] arrays")
    result = np.empty(len(target), dtype=np.float64)
    for start in range(0, len(target), 128):
        p = prediction[start : start + 128].astype(np.float64)
        y = target[start : start + 128].astype(np.float64)
        power = np.einsum("ij,ij->i", p, p)
        scale = np.divide(np.einsum("ij,ij->i", p, y), power, out=np.zeros(len(y)), where=power > 0)
        residual = scale[:, None] * p - y
        energy = np.einsum("ij,ij->i", y, y)
        result[start : start + len(y)] = np.einsum("ij,ij->i", residual, residual) / np.where(
            energy > 0, energy, y.shape[1]
        )
    if not np.isfinite(result).all():
        raise ValueError("scale-invariant error must be finite")
    return result


def error_summary(errors) -> dict:
    """Mean and median of per-trace errors over one domain."""
    errors = np.asarray(errors, dtype=np.float64)
    if errors.ndim != 1 or not len(errors) or not np.isfinite(errors).all():
        raise ValueError("errors must be a nonempty finite vector")
    return {
        "trace_count": len(errors),
        "mean": float(errors.mean()),
        "median": float(np.median(errors)),
    }


def cluster_bootstrap_weights(cluster_count: int, resamples: int, seed: int) -> np.ndarray:
    """Multiplicity of each cluster in each resample: ``[resample, cluster]``.

    Each resample draws ``cluster_count`` clusters with replacement from
    ``numpy.random.default_rng(seed)``; the same weights serve every contrast
    of one family so the contrasts are resampled jointly.
    """
    for value in (cluster_count, resamples):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError("cluster and resample counts must be positive integers")
    rng = np.random.default_rng(seed)
    return rng.multinomial(cluster_count, np.full(cluster_count, 1.0 / cluster_count), resamples)


def cluster_bootstrap_mean(values, cluster_ids, clusters, weights) -> dict:
    """Trace-mean of ``values`` with a cluster (source-cell) bootstrap.

    The resampled statistic is the mean over every trace of every drawn
    cluster. The CI is the percentile 2.5/97.5 interval; the two-sided p-value
    is null-centred: ``(1 + #{|t* - t| >= |t|}) / (B + 1)``.
    """
    values = np.asarray(values, dtype=np.float64)
    cluster_ids = np.asarray(cluster_ids)
    clusters = np.asarray(clusters)
    if values.shape != cluster_ids.shape or values.ndim != 1 or not len(values):
        raise ValueError("values and cluster IDs must be matching nonempty vectors")
    if not np.isfinite(values).all() or len(np.unique(clusters)) != len(clusters):
        raise ValueError("values must be finite and clusters unique")
    if weights.ndim != 2 or weights.shape[1] != len(clusters):
        raise ValueError("bootstrap weights must be [resample, cluster]")
    position = pd.Index(clusters).get_indexer(cluster_ids)
    if np.any(position < 0) or len(np.unique(position)) != len(clusters):
        raise ValueError("every trace needs a listed cluster and every cluster a trace")
    sums = np.bincount(position, weights=values, minlength=len(clusters))
    counts = np.bincount(position, minlength=len(clusters)).astype(np.float64)
    estimate = float(values.mean())
    resampled = (weights @ sums) / (weights @ counts)
    exceed = np.abs(resampled - estimate) >= abs(estimate)
    low, high = np.percentile(resampled, [2.5, 97.5])
    return {
        "estimate": estimate,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "bootstrap_p_value": float((1 + exceed.sum()) / (len(resampled) + 1)),
        "bootstrap_resamples": len(resampled),
        "cluster_count": len(clusters),
    }


def holm_adjust(p_values) -> np.ndarray:
    """Holm step-down adjusted p-values in the input order, capped at 1."""
    p = np.asarray(p_values, dtype=np.float64)
    if p.ndim != 1 or not len(p) or np.any((p < 0) | (p > 1)) or not np.isfinite(p).all():
        raise ValueError("p-values must be a nonempty vector in [0, 1]")
    order = np.argsort(p, kind="mergesort")
    stepped = np.minimum(1.0, (len(p) - np.arange(len(p))) * p[order])
    adjusted = np.empty_like(p)
    adjusted[order] = np.maximum.accumulate(stepped)
    return adjusted


def wilcoxon_signed_rank(differences) -> dict:
    """Two-sided Wilcoxon signed-rank test, normal approximation.

    Zero differences are discarded (Wilcoxon's method), tied magnitudes get
    average ranks with the tie-corrected variance, and no continuity
    correction is applied. Traces are treated as independent, so this is a
    secondary check next to the cluster bootstrap.
    """
    d = np.asarray(differences, dtype=np.float64)
    if d.ndim != 1 or not np.isfinite(d).all():
        raise ValueError("differences must be a finite vector")
    d = d[d != 0]
    n = len(d)
    if not n:
        return {"nonzero_count": 0, "w_plus": 0.0, "z": 0.0, "p_value": 1.0}
    magnitude = np.abs(d)
    order = np.argsort(magnitude, kind="mergesort")
    _, first, ties = np.unique(magnitude[order], return_index=True, return_counts=True)
    ranks = np.empty(n)
    ranks[order] = np.repeat(first + (ties + 1) / 2.0, ties)
    w_plus = float(ranks[d > 0].sum())
    mean = n * (n + 1) / 4.0
    variance = n * (n + 1) * (2 * n + 1) / 24.0 - float(np.sum(ties**3 - ties)) / 48.0
    z = (w_plus - mean) / math.sqrt(variance) if variance > 0 else 0.0
    return {
        "nonzero_count": n,
        "w_plus": w_plus,
        "z": float(z),
        "p_value": float(math.erfc(abs(z) / math.sqrt(2.0))),
    }


def paired_contrasts(errors, cluster_ids, weights, contrasts=FAIR_GRID_CONTRASTS) -> pd.DataFrame:
    """Bootstrap CI, Holm-adjusted bootstrap p and Wilcoxon for each paired contrast.

    ``errors`` maps method -> seed-averaged per-trace error on one common row
    order; ``weights`` come from ``cluster_bootstrap_weights`` over the sorted
    unique ``cluster_ids``. Holm adjustment runs over the listed contrasts.
    """
    clusters = np.unique(cluster_ids)
    rows = []
    for name, left, right, role in contrasts:
        a, b = (np.asarray(errors[m], dtype=np.float64) for m in (left, right))
        if a.shape != b.shape:
            raise ValueError(f"{name}: paired errors must cover the same traces")
        difference = a - b
        wilcoxon = wilcoxon_signed_rank(difference)
        rows.append(
            {
                "contrast": name,
                "left_method": left,
                "right_method": right,
                "role": role,
                "trace_count": len(difference),
                "left_mean_error": float(a.mean()),
                "right_mean_error": float(b.mean()),
                **cluster_bootstrap_mean(difference, cluster_ids, clusters, weights),
                "median_difference": float(np.median(difference)),
                "fraction_left_lower_error": float(np.mean(difference < 0)),
                "wilcoxon_z": wilcoxon["z"],
                "wilcoxon_p_value": wilcoxon["p_value"],
                "wilcoxon_nonzero_count": wilcoxon["nonzero_count"],
            }
        )
    table = pd.DataFrame(rows)
    table.insert(
        table.columns.get_loc("bootstrap_p_value") + 1,
        "holm_p_value",
        holm_adjust(table.bootstrap_p_value.to_numpy()),
    )
    return table


def expected_best_curve(scores, lower_is_better: bool = True) -> pd.DataFrame:
    """Expected best score of ``n`` configurations drawn with replacement (Dodge et al. 2019).

    With ascending scores ``v_(1..N)`` and minimisation, the best of ``n`` draws
    equals ``v_(i)`` with probability ``((N-i+1)/N)^n - ((N-i)/N)^n``; ties need
    no special case because tied positions share one value. Returns rows
    ``n = 1..N`` with the expectation and standard deviation.
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("scores must be a nonempty finite vector")
    values = np.sort(values) if lower_is_better else np.sort(values)[::-1]
    total = len(values)
    remaining = (total - np.arange(total + 1)) / total
    rows = []
    for n in range(1, total + 1):
        probability = remaining[:-1] ** n - remaining[1:] ** n
        mean = float(probability @ values)
        variance = max(float(probability @ np.square(values)) - mean**2, 0.0)
        rows.append({"configurations": n, "expected_best": mean, "sd": math.sqrt(variance)})
    return pd.DataFrame(rows)
