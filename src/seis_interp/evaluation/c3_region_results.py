"""Physical-amplitude scoring and descriptive statistics across split seeds."""

import numpy as np
import pandas as pd

from seis_interp.evaluation.forge_mvp_metrics import evaluate_traces, verify_saved_metrics
from seis_interp.evaluation.seed_completion import seed_completion
from seis_interp.evaluation.trace_snr import trace_snr_db


def score_c3_region(prediction, target, rows):
    metrics, traces = evaluate_traces(prediction, target, rows)
    verify_saved_metrics(metrics, traces)
    summary = metrics["all_eligible"]
    snr = trace_snr_db(target, prediction)
    finite = np.isfinite(snr)
    summary.update(
        rmse=float(
            np.sqrt(np.mean((prediction.astype(np.float64) - target.astype(np.float64)) ** 2))
        ),
        mean_trace_snr_db=float(snr.mean()) if finite.all() else None,
        nonfinite_trace_snr_count=int((~finite).sum()),
        zero_target_count=int(np.all(target == 0, axis=1).sum()),
    )
    return summary, traces


def summarize_c3_region_seeds(records: list[dict], *, expected_seeds=None):
    """All three additional seeds have equal weight; no winning seed is selected."""
    seed_column = "trace_seed" if records and "trace_seed" in records[0] else "region_seed"
    completed, details = seed_completion(records, seed_column, expected_seeds)
    rows = []
    for domain in ("validation", "test"):
        for metric in (
            "masked_trace_relative_mse",
            "global_nmse",
            "snr_db",
            "mean_trace_snr_db",
            "rmse",
        ):
            values = [record["metrics"][domain][metric] for record in completed]
            valid = len(values) == 3 and all(
                value is not None and np.isfinite(value) for value in values
            )
            rows.append(
                {
                    "domain": domain,
                    "metric": metric,
                    "mean": float(np.mean(values)) if valid else None,
                    "sample_std": float(np.std(values, ddof=1)) if valid else None,
                    "seed_count": len(values),
                    **details,
                }
            )
    return pd.DataFrame(rows)
