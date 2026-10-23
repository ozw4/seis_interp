"""FORGE physical-domain metrics and equal-weight, sample-SD seed summaries."""

import numpy as np
import pandas as pd

from seis_interp.evaluation.forge_mvp_metrics import evaluate_traces, verify_saved_metrics
from seis_interp.evaluation.seed_completion import seed_completion
from seis_interp.evaluation.trace_snr import summarize_trace_snr, trace_snr_db


def score_forge_grid(prediction, target, rows, *, progress=None):
    metrics, traces = evaluate_traces(prediction, target, rows, progress=progress)
    verify_saved_metrics(metrics, traces)
    snr = trace_snr_db(target, prediction)
    traces["snr_db"] = snr
    for population, selected in (
        ("all_eligible", np.ones(len(rows), dtype=bool)),
        ("clean_target", rows.clean_target.to_numpy()),
    ):
        metrics[population].update(summarize_trace_snr(snr[selected]))
        metrics[population]["rmse"] = float(
            np.sqrt(
                np.mean(
                    (prediction[selected].astype(np.float64) - target[selected].astype(np.float64))
                    ** 2
                )
            )
        )
    return metrics, traces


def forge_seed_metric_rows(records):
    for record in records:
        for domain, populations in record["metrics"].items():
            for population, values in populations.items():
                flat = {}
                for metric, value in values.items():
                    if isinstance(value, dict):
                        flat.update({f"{metric}_{key}": item for key, item in value.items()})
                    else:
                        flat[metric] = value
                yield {
                    "method_id": record["method_id"],
                    "trace_seed": record["trace_seed"],
                    "configuration_id": record["id"],
                    "domain": domain,
                    "population": population,
                    **flat,
                }


def summarize_forge_grid_seeds(records, *, expected_seeds=None):
    completed, details = seed_completion(records, "trace_seed", expected_seeds)
    table = pd.DataFrame(forge_seed_metric_rows(completed))
    rows = []
    metrics = [
        "masked_trace_relative_mse",
        "global_nmse",
        "snr_db",
        "mean_trace_snr_db",
        "rmse",
        *[f"correlation_{q}" for q in ("p05", "p25", "median", "p75", "p95")],
    ]
    for domain, population in (
        (d, p) for d in ("validation", "test") for p in ("all_eligible", "clean_target")
    ):
        group = (
            table.loc[table.domain.eq(domain) & table.population.eq(population)]
            if len(table)
            else table
        )
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float) if metric in group else np.array([])
            valid = len(values) == 3 and np.isfinite(values).all()
            rows.append(
                {
                    "domain": domain,
                    "population": population,
                    "metric": metric,
                    "mean": float(values.mean()) if valid else None,
                    "sample_std": float(values.std(ddof=1)) if valid else None,
                    "seed_count": len(values),
                    **details,
                }
            )
    return pd.DataFrame(rows)
