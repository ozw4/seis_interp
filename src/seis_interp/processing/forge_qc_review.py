"""Fixed numeric exclusions and diagnostic review of DC and amplitude anomalies."""

import numpy as np
import pandas as pd

KEYS = ["source_file", "trace_index"]


def build_qc_review(table: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Keep all rows; only zero/nonzero-constant candidates are newly excluded.

    Review flags are diagnostic, never additional rejection decisions. The AC
    amplitude comparison uses the same shot/offset reference as the raw RMS.
    """
    if table.duplicated(KEYS).any():
        raise ValueError("duplicate trace keys")
    if set(config["fixed_exclusions"]) != {"all_zero", "nonzero_constant"}:
        raise ValueError("fixed exclusions must be all_zero and nonzero_constant")
    if table.loc[table.header_eligible, "nonfinite_count"].gt(0).any():
        raise ValueError("nonfinite candidate needs a separate exclusion decision")
    out = table.copy()
    out["fixed_qc_excluded"] = out.header_eligible & out.constant
    out["exclusion_reason"] = np.select(
        [~out.header_eligible, out.all_zero, out.constant],
        ["header_excluded", "all_zero", "nonzero_constant"],
        default="",
    )
    out["eligible_after_fixed_qc"] = out.header_eligible & ~out.fixed_qc_excluded
    keep = out.eligible_after_fixed_qc
    if not out.loc[keep, "numerically_usable"].all():
        raise ValueError("retained rows disagree with numeric audit")
    out["std_to_rms"] = (out["std"] / out.rms).where(keep)
    out["near_constant_review"] = keep & out.std_to_rms.lt(
        config["near_constant_review_std_to_rms"]
    )
    out["offset_bin"] = np.floor(out.offset_m / config["offset_bin_width_m"])
    baseline = out["std"].where(keep)
    groups = [out.source_file, out.offset_bin]
    median = baseline.groupby(groups).transform("median")
    count = baseline.groupby(groups).transform("count")
    out["ac_reference_count"] = count
    out["ac_rms_to_shot_offset_median"] = (out["std"] / median).where(
        keep & count.ge(config["minimum_reference_traces"]) & median.gt(0)
    )
    out["amplitude_direction"] = np.select(
        [
            keep & out.rms_to_shot_offset_median.gt(config["amplitude_review_ratio"]),
            keep & out.rms_to_shot_offset_median.lt(1 / config["amplitude_review_ratio"]),
        ],
        ["high", "low"],
        default="within_or_unassessed",
    )
    out["review_pending"] = keep & (out.waveform_status.eq("review") | out.near_constant_review)
    return out


def review_distributions(table: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Threshold sensitivity on the retained domain; station rates use own counts."""
    c = table[table.eligible_after_fixed_qc]
    rows = []
    for metric, thresholds, direction in [
        ("std_to_rms", config["variation_thresholds"], "lt"),
        ("dc_to_rms", config["dc_thresholds"], "gt"),
        ("rms_to_shot_offset_median", config["amplitude_thresholds"], "gt"),
        ("ac_rms_to_shot_offset_median", config["amplitude_thresholds"], "gt"),
    ]:
        for threshold in thresholds:
            values = c[metric]
            selected = values.lt(threshold) if direction == "lt" else values.gt(threshold)
            rows.append(
                {
                    "metric": metric,
                    "comparison": direction,
                    "threshold": threshold,
                    "count": int(selected.sum()),
                    "assessed_count": int(values.notna().sum()),
                    "retained_count": len(c),
                }
            )
            if "median" in metric:
                rows.append(
                    {
                        "metric": metric,
                        "comparison": "lt",
                        "threshold": 1 / threshold,
                        "count": int(values.lt(1 / threshold).sum()),
                        "assessed_count": int(values.notna().sum()),
                        "retained_count": len(c),
                    }
                )
    stations = (
        c.groupby(["receiver_line", "receiver_point"])
        .agg(
            retained_count=("trace_index", "size"),
            dc_count=("dc_review", "sum"),
            near_constant_count=("near_constant_review", "sum"),
            amplitude_count=("amplitude_review", "sum"),
            x_m=("receiver_x_m", "first"),
            y_m=("receiver_y_m", "first"),
        )
        .reset_index()
    )
    for flag in ["dc", "near_constant", "amplitude"]:
        stations[flag + "fraction"] = stations[flag + "_count"] / stations.retained_count
    return pd.DataFrame(rows), stations


def select_review_examples(table: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Choose central and extreme cases in declared strata, with stable tie order."""
    c = table[table.eligible_after_fixed_qc].sort_values(KEYS)
    q, d, a = c.std_to_rms, c.dc_to_rms, c.rms_to_shot_offset_median
    near = config["near_constant_review_std_to_rms"]
    limit = config["amplitude_review_ratio"]
    strata = [
        ("near_constant", q.lt(near), "std_to_rms", True),
        ("low_variation", q.ge(near) & q.lt(0.1), "std_to_rms", True),
        ("dc_0p1_0p2", d.gt(0.1) & d.le(0.2), "dc_to_rms", False),
        ("dc_0p2_0p5", d.gt(0.2) & d.le(0.5), "dc_to_rms", False),
        ("dc_above_0p5", d.gt(0.5) & q.ge(0.1), "dc_to_rms", False),
        ("amplitude_high_dc", a.gt(limit) & d.gt(0.9), "rms_to_shot_offset_median", False),
        ("amplitude_high_ac", a.gt(limit) & d.le(0.9), "rms_to_shot_offset_median", False),
        ("amplitude_low", a.lt(1 / limit), "rms_to_shot_offset_median", True),
    ]
    selected = []
    for name, mask, metric, low_extreme in strata:
        group = c[mask].sort_values([metric, *KEYS])
        if group.empty:
            continue
        for label, position in [
            ("central", len(group) // 2),
            ("extreme", 0 if low_extreme else len(group) - 1),
        ]:
            row = group.iloc[position].to_dict()
            row.update(review_stratum=name, selection=label, stratum_count=len(group))
            selected.append(row)
    return pd.DataFrame(
        selected, columns=[*c.columns, "review_stratum", "selection", "stratum_count"]
    )


def select_review_controls(gather: pd.DataFrame, target: pd.Series, count: int) -> pd.DataFrame:
    """Nearest unflagged points on the same receiver line and shot; no imputation."""
    pool = gather[
        gather.eligible_after_fixed_qc
        & ~gather.review_pending
        & gather.source_file.eq(target.source_file)
        & gather.receiver_line.eq(target.receiver_line)
        & gather.trace_index.ne(target.trace_index)
    ].copy()
    pool["point_distance"] = abs(pool.receiver_point - target.receiver_point)
    return pool.sort_values(["point_distance", "trace_index"]).head(count)


def summarize_qc_review(table: pd.DataFrame) -> dict:
    c = table[table.eligible_after_fixed_qc]
    return {
        "trace_count": len(table),
        "header_candidate_count": int(table.header_eligible.sum()),
        "fixed_exclusions": table.loc[table.fixed_qc_excluded, "exclusion_reason"]
        .value_counts()
        .to_dict(),
        "retained_count": len(c),
        "review_pending_count": int(c.review_pending.sum()),
        "near_constant_review_count": int(c.near_constant_review.sum()),
        "dc_review_count": int(c.dc_review.sum()),
        "amplitude_direction_counts": c.amplitude_direction.value_counts().to_dict(),
        "raw_amplitude_unassessed_count": int(c.rms_to_shot_offset_median.isna().sum()),
        "review_overlap": c.groupby(["near_constant_review", "dc_review", "amplitude_review"])
        .size()
        .rename("count")
        .reset_index()
        .to_dict("records"),
        "quantiles": {
            name: {
                str(q): (float(v) if pd.notna(v) else None)
                for q, v in c[name].quantile([0, 0.001, 0.01, 0.5, 0.99, 0.999, 1]).items()
            }
            for name in [
                "std_to_rms",
                "dc_to_rms",
                "rms_to_shot_offset_median",
                "ac_rms_to_shot_offset_median",
            ]
        },
        "benchmark_adoption": "pending; retained is not a final clean-data mask",
    }


def measure_review_waveform(values: np.ndarray, dt: float) -> dict:
    """Describe selected traces in stored units, including quantization and drift."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all() or dt <= 0:
        raise ValueError("expected one finite waveform and positive interval")
    window = max(1, min(len(x), round(0.5 / dt)))
    return {
        "unique_sample_count": len(np.unique(x)),
        "peak_to_peak": float(np.ptp(x)),
        "centered_peak": float(np.max(np.abs(x - x.mean()))),
        "first_half_second_mean": float(x[:window].mean()),
        "last_half_second_mean": float(x[-window:].mean()),
    }
