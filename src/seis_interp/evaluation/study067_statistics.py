"""Seed-balanced, paired source-cell inference for the registered common grid.

Unlike the older fair-grid evaluation, each seed has its own test rows. We
resample source cells once, compute a trace-weighted mean within each seed,
then give the three seed means equal weight. Positive SNR differences favour
the left method; negative relMSE differences favour it.
"""

import math

import numpy as np
import pandas as pd

from seis_interp.evaluation.forge_fair_grid_statistics import holm_adjust, seed_mean_sd


def select_candidate(records):
    """Minimum validation relMSE, then ID; diverged candidates rank last."""
    if not records:
        raise ValueError("S1 has no candidates")
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("duplicate S1 candidate IDs")
    for row in records:
        if row["status"] not in ("completed", "diverged"):
            raise ValueError(f"unfinished S1 candidate: {row['id']}")
        if row["status"] == "completed" and not np.isfinite(row["validation_score"]):
            raise ValueError("completed S1 candidate needs a finite validation score")
    return min(
        records,
        key=lambda r: (
            r["status"] == "diverged",
            r["validation_score"] if r["status"] == "completed" else math.inf,
            r["id"],
        ),
    )


def validation_free_mask(trace_ids, validation_sets):
    """Exclude the union of *all* S1 validation trace identities, including pilots."""
    used = set().union(*(set(ids) for ids in validation_sets))
    return ~pd.Index(trace_ids).isin(used)


def paired_t_interval(seed_differences):
    """Two-sided 95% paired t interval for exactly three seeds (df=2)."""
    values = np.asarray(seed_differences, dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("paired t interval requires three finite seed differences")
    mean, sd = seed_mean_sd(values)
    # Exact quantile for t_2: F(t) = 1/2 + t/(2 sqrt(t^2 + 2)).
    critical = math.sqrt(2) * 0.95 / math.sqrt(1 - 0.95**2)
    half = critical * sd / math.sqrt(3)
    return {"t_df": 2, "t_ci95_low": mean - half, "t_ci95_high": mean + half}


def clustered_sums(values, cluster_ids, clusters):
    values = np.asarray(values, dtype=np.float64)
    ids = np.asarray(cluster_ids)
    clusters = np.asarray(clusters)
    if values.ndim != 1 or values.shape != ids.shape or not len(values):
        raise ValueError("values and cluster IDs must be matching nonempty vectors")
    if not np.isfinite(values).all() or len(np.unique(clusters)) != len(clusters):
        raise ValueError("values must be finite and clusters unique")
    position = pd.Index(clusters).get_indexer(ids)
    if np.any(position < 0):
        raise ValueError("trace has an unlisted source cell")
    return (
        np.bincount(position, weights=values, minlength=len(clusters)),
        np.bincount(position, minlength=len(clusters)).astype(np.float64),
    )


def bootstrap_seed_means(per_seed, cluster_ids, clusters, weights):
    """Return [replicate, seed], allowing seed-specific rows and absent cells."""
    weights = np.asarray(weights)
    if (
        weights.ndim != 2
        or weights.shape[1] != len(clusters)
        or not len(weights)
        or np.any(weights < 0)
        or not np.isfinite(weights).all()
        or not np.all(weights.sum(axis=1) == len(clusters))
    ):
        raise ValueError("weights must be source-cell multinomial multiplicities")
    if len(per_seed) != len(cluster_ids) or not len(per_seed):
        raise ValueError("each seed needs values and source-cell IDs")
    result = []
    for values, ids in zip(per_seed, cluster_ids, strict=True):
        sums, counts = clustered_sums(values, ids, clusters)
        denominator = weights @ counts
        if np.any(denominator == 0):
            raise ValueError("a bootstrap replicate has no test traces for a seed")
        result.append((weights @ sums) / denominator)
    return np.column_stack(result)


def paired_seed_bootstrap(per_seed, cluster_ids, clusters, weights):
    """Percentile CI and null-centred two-sided p, matching fair-grid's convention.

    p = (1 + count(abs(delta* - delta) >= abs(delta))) / (B + 1).
    No seed intersection or pooling: each seed's own test set enters intact.
    """
    draws = bootstrap_seed_means(per_seed, cluster_ids, clusters, weights).mean(axis=1)
    means = [float(np.mean(values)) for values in per_seed]
    estimate = float(np.mean(means))
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "estimate": estimate,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "bootstrap_p_value": float(
            (1 + (np.abs(draws - estimate) >= abs(estimate)).sum()) / (len(draws) + 1)
        ),
        "bootstrap_resamples": len(draws),
        "cluster_count": len(clusters),
        **{f"delta_seed_{i + 1}": value for i, value in enumerate(means)},
        **paired_t_interval(means),
    }


def contrast_table(
    tests, contrasts, seeds, clusters, weights, *, cluster_column, validation_sets=()
):
    """Paired SNR and relMSE contrasts, retaining exact identity checks."""
    rows = []
    for baseline in contrasts:
        by_metric = {metric: [] for metric in ("snr_db", "relative_mse")}
        ids, counts = [], []
        for seed in seeds:
            if tests["regsi_real", seed] is None or tests[baseline, seed] is None:
                ids.append(None)
                counts.append(None)
                for metric in by_metric:
                    by_metric[metric].append(None)
                continue
            left, right = (
                tests[method, seed].set_index("cell_id", verify_integrity=True).sort_index()
                for method in ("regsi_real", baseline)
            )
            if not left.index.equals(right.index):
                raise ValueError(f"{baseline}, seed {seed}: unpaired test cells")
            for key in ("trace_id", cluster_column, "clean_target"):
                if not left[key].equals(right[key]):
                    raise ValueError(f"{baseline}, seed {seed}: paired {key} differs")
            keep = validation_free_mask(left.trace_id, validation_sets)
            left, right = left.loc[keep], right.loc[keep]
            ids.append(left[cluster_column].to_numpy())
            counts.append(len(left))
            for metric in by_metric:
                by_metric[metric].append(left[metric].to_numpy() - right[metric].to_numpy())
        for metric, differences in by_metric.items():
            identity = {
                "contrast": f"regsi_real_minus_{baseline}",
                "baseline": baseline,
                "metric": metric,
                "higher_is_better": metric == "snr_db",
                **{f"seed_{i + 1}": seed for i, seed in enumerate(seeds)},
                **{f"trace_count_seed_{i + 1}": n for i, n in enumerate(counts)},
            }
            if any(d is None for d in differences):
                rows.append(
                    {
                        **identity,
                        "status": "unavailable: a selected evaluation seed diverged",
                        "direction": "算出不能",
                        "passes_best_condition": False,
                        "estimate": np.nan,
                        "ci95_low": np.nan,
                        "ci95_high": np.nan,
                        "bootstrap_p_value": np.nan,
                        "bootstrap_resamples": len(weights),
                        "cluster_count": len(clusters),
                        "t_df": 2,
                        "t_ci95_low": np.nan,
                        "t_ci95_high": np.nan,
                        **{
                            f"delta_seed_{i + 1}": float(d.mean()) if d is not None else np.nan
                            for i, d in enumerate(differences)
                        },
                    }
                )
                continue
            statistics = paired_seed_bootstrap(differences, ids, clusters, weights)
            low, high = statistics["ci95_low"], statistics["ci95_high"]
            direction = (
                "区別不能"
                if low <= 0 <= high
                else "ReGSI 優位"
                if (low > 0 if metric == "snr_db" else high < 0)
                else "逆"
            )
            rows.append(
                {
                    **identity,
                    "status": "available",
                    "direction": direction,
                    **statistics,
                    "passes_best_condition": metric == "snr_db"
                    and low > 0
                    and all(float(np.mean(d)) > 0 for d in differences),
                }
            )
    return pd.DataFrame(rows)


def adjust_families(table, families):
    """Holm across both datasets, independently for the two registered families."""
    table = table.copy()
    table["family"] = ""
    table["holm_p_value"] = np.nan
    table["holm_status"] = ""
    for family, metric in families.items():
        keep = table.metric.eq(metric)
        if int(keep.sum()) != 8:
            raise ValueError(f"{family}: expected all eight contrasts before Holm adjustment")
        table.loc[keep, "family"] = family
        if table.loc[keep, "bootstrap_p_value"].notna().all():
            table.loc[keep, "holm_p_value"] = holm_adjust(table.loc[keep, "bootstrap_p_value"])
            table.loc[keep, "holm_status"] = "available"
        else:
            # The plan gives no missing-contrast rule. Do not redefine the family,
            # invent a p-value, or silently perform a smaller Holm correction.
            table.loc[keep, "holm_status"] = "unavailable: incomplete eight-contrast family"
    return table


def best_claims(table):
    rows = []
    for dataset, group in table.groupby("dataset", sort=False):
        snr = group.loc[group.metric.eq("snr_db")]
        rel = group.loc[group.metric.eq("relative_mse")]
        best = len(snr) == 4 and bool(snr.passes_best_condition.all())
        if best:
            wording = "SNR では最良"
            if rel.direction.eq("逆").any() and rel.direction.eq("区別不能").any():
                wording += "、relMSE では区別不能／逆"
            elif rel.direction.eq("逆").any():
                wording += "、relMSE では逆"
            elif rel.direction.eq("区別不能").any():
                wording += "、relMSE では区別不能"
            else:
                wording += "、relMSE でも優位"
        else:
            wording = "ReGSI が最良という条件を満たさない"
        rows.append({"dataset": dataset, "regsi_best_snr": best, "wording": wording})
    return pd.DataFrame(rows)
