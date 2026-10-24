"""Descriptive S1 search diagnostics; none changes the registered selection."""

import numpy as np
import pandas as pd

from seis_interp.evaluation.forge_fair_grid_statistics import expected_best_curve
from seis_interp.evaluation.study067_statistics import clustered_sums


def ranking_summary(records):
    table = pd.DataFrame(records)
    successful = table.status.eq("completed")
    rel = table.validation_score.where(successful, np.inf).rank(method="average")
    snr = (-table.validation_snr_db).where(successful, np.inf).rank(method="average")
    rho = float(rel.corr(snr)) if rel.nunique() > 1 and snr.nunique() > 1 else None
    best = table.loc[successful].sort_values(["validation_snr_db", "id"], ascending=[False, True])
    return {"spearman_rho": rho, "best_snr_id": None if best.empty else best.iloc[0].id}


def search_curve(records, metric):
    """Exact empirical best-of-k with replacement, retaining divergence mass.

    No finite penalty is invented for divergence. The unconditional expected
    best is infinite if all k draws can diverge; a separately labelled finite
    conditional expectation supplies the plottable curve.
    """
    key = "validation_score" if metric == "relative_mse" else "validation_snr_db"
    values = np.asarray([r[key] for r in records if r["status"] == "completed"], dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("search curve needs finite completed S1 scores")
    total, valid = len(records), len(values)
    lower = metric == "relative_mse"
    if valid == total:
        result = expected_best_curve(values, lower_is_better=lower)
        result["conditional_expected_best"] = result.expected_best
        result["probability_all_diverged"] = 0.0
    else:
        values = np.sort(values) if lower else np.sort(values)[::-1]
        remaining = (total - np.arange(valid + 1)) / total
        rows = []
        for k in range(1, total + 1):
            mass = remaining[:-1] ** k - remaining[1:] ** k
            failure = ((total - valid) / total) ** k
            conditional = mass / mass.sum()
            mean = float(conditional @ values)
            sd = float(np.sqrt(max(0, conditional @ values**2 - mean**2)))
            rows.append(
                {
                    "configurations": k,
                    "expected_best": np.inf if lower else -np.inf,
                    "conditional_expected_best": mean,
                    "sd": sd,
                    "probability_all_diverged": failure,
                }
            )
        result = pd.DataFrame(rows)
    result["metric"] = metric
    result["scored_configurations"] = total
    result["diverged_configurations"] = total - valid
    return result


def winners_curse(validation, records, metric, clusters, weights, cluster_column):
    """Cluster-bootstrap S1 optimism: select in bag, evaluate out of bag.

    For relMSE, optimism is OOB - in-bag; for SNR it is in-bag - OOB.
    These are descriptive validation-only estimates, not new decision rules.
    Diverged configurations remain rank-last and cannot win a finite draw.
    """
    completed = sorted((r for r in records if r["status"] == "completed"), key=lambda r: r["id"])
    sums, counts = [], []
    identities = None
    for row in completed:
        traces = validation[row["id"]].sort_values("cell_id")
        current = traces[["cell_id", "trace_id", cluster_column]]
        if identities is not None and not current.reset_index(drop=True).equals(identities):
            return {
                "metric": metric,
                "status": "unavailable: S1 validation sets differ",
                "scored_configurations": len(records),
            }
        identities = current.reset_index(drop=True)
        numerator, denominator = clustered_sums(traces[metric], traces[cluster_column], clusters)
        sums.append(numerator)
        counts.append(denominator)
    sums, counts = np.column_stack(sums), np.column_stack(counts)
    denominator = weights @ counts
    if np.any(denominator == 0):
        raise ValueError("empty in-bag validation bootstrap")
    inbag = (weights @ sums) / denominator
    out_weights = (weights == 0).astype(float)
    out_count = out_weights @ counts
    oob = np.divide(
        out_weights @ sums, out_count, out=np.full_like(inbag, np.nan), where=out_count > 0
    )
    winner = np.argmin(inbag, axis=1) if metric == "relative_mse" else np.argmax(inbag, axis=1)
    index = np.arange(len(weights))
    chosen_in, chosen_out = inbag[index, winner], oob[index, winner]
    valid = np.isfinite(chosen_out)
    optimism = (chosen_out - chosen_in) if metric == "relative_mse" else (chosen_in - chosen_out)
    return {
        "metric": metric,
        "status": "available" if valid.any() else "unavailable: empty OOB",
        "scored_configurations": len(records),
        "diverged_configurations": len(records) - len(completed),
        "bootstrap_inbag_mean": float(chosen_in[valid].mean()) if valid.any() else None,
        "bootstrap_oob_mean": float(chosen_out[valid].mean()) if valid.any() else None,
        "optimism_mean": float(optimism[valid].mean()) if valid.any() else None,
        "replicates_with_oob": int(valid.sum()),
        "replicates_without_oob": int((~valid).sum()),
        "definition": "S1 source-cell bootstrap selection in bag; evaluation on undrawn cells",
    }
