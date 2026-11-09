"""Paired contrast of two methods scored in separate experiments.

``study067_statistics.contrast_table`` fixes the proposed method on the left and
expects one experiment per dataset. A later study that re-takes a single arm needs
the same statistics for an explicit pair whose two sides come from different
experiments, so only the pairing and the labels change here; the estimator, the
interval, the p-value convention and the validation-free mask are reused.
"""

import numpy as np
import pandas as pd

from seis_interp.evaluation.forge_fair_grid_statistics import holm_adjust
from seis_interp.evaluation.study067_statistics import paired_seed_bootstrap, validation_free_mask

METRICS = ("snr_db", "relative_mse")


def paired_method_contrast(
    left_tests,
    right_tests,
    *,
    left,
    right,
    seeds,
    clusters,
    weights,
    cluster_column,
    validation_sets=(),
):
    """One row per metric for ``left`` minus ``right``, on each seed's own test set.

    ``left_tests`` and ``right_tests`` map a seed to that seed's per-trace table,
    or to ``None`` when the seed diverged. The pairing is exact: for every seed the
    two tables must carry the same cells, trace identities, clusters and
    clean-target flags, so two studies' assignments are checked, not assumed.
    Seeds are never pooled or intersected.
    """
    counts, cluster_ids = [], []
    differences = {metric: [] for metric in METRICS}
    for seed in seeds:
        pair = (left_tests.get(seed), right_tests.get(seed))
        if any(table is None for table in pair):
            counts.append(None)
            cluster_ids.append(None)
            for metric in METRICS:
                differences[metric].append(None)
            continue
        tables = [table.set_index("cell_id", verify_integrity=True).sort_index() for table in pair]
        if not tables[0].index.equals(tables[1].index):
            raise ValueError(f"seed {seed}: unpaired test cells")
        for key in ("trace_id", cluster_column, "clean_target"):
            if not tables[0][key].equals(tables[1][key]):
                raise ValueError(f"seed {seed}: paired {key} differs")
        keep = validation_free_mask(tables[0].trace_id, validation_sets)
        tables = [table.loc[keep] for table in tables]
        counts.append(len(tables[0]))
        cluster_ids.append(tables[0][cluster_column].to_numpy())
        for metric in METRICS:
            differences[metric].append(tables[0][metric].to_numpy() - tables[1][metric].to_numpy())

    rows = []
    for metric, values in differences.items():
        identity = {
            "contrast": f"{left}_minus_{right}",
            "left": left,
            "right": right,
            "metric": metric,
            "higher_is_better": metric == "snr_db",
            **{f"seed_{i + 1}": seed for i, seed in enumerate(seeds)},
            **{f"trace_count_seed_{i + 1}": count for i, count in enumerate(counts)},
        }
        if any(value is None for value in values):
            rows.append(
                {
                    **identity,
                    "status": "unavailable: a selected evaluation seed diverged",
                    "direction": "算出不能",
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
                        f"delta_seed_{i + 1}": (
                            float(value.mean()) if value is not None else np.nan
                        )
                        for i, value in enumerate(values)
                    },
                }
            )
            continue
        statistics = paired_seed_bootstrap(values, cluster_ids, clusters, weights)
        low, high = statistics["ci95_low"], statistics["ci95_high"]
        favours_left = low > 0 if metric == "snr_db" else high < 0
        rows.append(
            {
                **identity,
                "status": "available",
                "direction": (
                    "区別不能"
                    if low <= 0 <= high
                    else f"{left} 優位"
                    if favours_left
                    else f"{right} 優位"
                ),
                **statistics,
            }
        )
    return pd.DataFrame(rows)


def adjust_declared_families(table, families):
    """Holm within each declared family of ``{name: {contrast, metric}}``.

    A family whose rows are not all available keeps its p-values unadjusted: the
    declaration gives no missing-contrast rule, so the family is neither redefined
    nor silently corrected at a smaller size.
    """
    table = table.copy()
    for column, value in (("family", ""), ("holm_p_value", np.nan), ("holm_status", "")):
        table[column] = value
    for family, selector in families.items():
        keep = table.contrast.eq(selector["contrast"]) & table.metric.eq(selector["metric"])
        if not int(keep.sum()):
            raise ValueError(f"{family}: no rows match the declared contrast and metric")
        table.loc[keep, "family"] = family
        if table.loc[keep, "bootstrap_p_value"].notna().all():
            table.loc[keep, "holm_p_value"] = holm_adjust(table.loc[keep, "bootstrap_p_value"])
            table.loc[keep, "holm_status"] = "available"
        else:
            table.loc[keep, "holm_status"] = "unavailable: incomplete declared family"
    descriptive = table.family.eq("")
    table.loc[descriptive, "holm_status"] = "descriptive: no multiplicity adjustment"
    return table
