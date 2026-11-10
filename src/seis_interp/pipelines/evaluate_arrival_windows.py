"""Re-score saved predictions on either side of the declared pre-arrival window.

No inference and no new prediction: every run already stores its physical test
prediction, so this reads those arrays and the same targets the run was scored
against. Each method is checked back against its own saved per-trace scores before
any windowed number is reported, so a path or ordering mistake cannot pass silently.

This is a post-hoc secondary measurement, added after the studies were scored. It
never feeds selection, which is closed on validation relative MSE.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from seis_interp.data.forge_headers import sha256_file
from seis_interp.data.forge_mvp_artifacts import write_json
from seis_interp.evaluation.arrival_window_metrics import (
    ARRIVAL_WINDOWS,
    MINIMUM_WINDOW_SAMPLES,
    arrival_window_ends,
    windowed_waveform_columns,
)
from seis_interp.evaluation.study067_trace_metrics import waveform_columns

SNR_TOLERANCE_DB = 1e-6
CHUNK = 2048


def method_prediction(experiment, method, seed):
    """The directory holding one scored prediction, in either run layout."""
    whole_shot = experiment / "method_runs" / method
    seeded = whole_shot / "evaluation" / f"seed_{seed}"
    directory = seeded if seed is not None else whole_shot
    if not (directory / "prediction.npy").is_file():
        raise ValueError(f"{method}: no saved prediction under {directory}")
    return directory


def run_inputs(repo, experiment, contract):
    """Targets in manifest order, their offsets, and the trace roles."""
    dataset = contract["dataset_id"]
    if dataset == "seg_c3_na":
        sources = yaml.safe_load((experiment / "study/inputs.yaml").read_text())
        manifest = pd.read_parquet(experiment / "input/trace_manifest.parquet")
        geometry = pd.read_parquet(experiment / "input/geometry.parquet")
        positions = manifest.set_index("cell_id", verify_integrity=True).array_row
        values = np.load(repo / sources["amplitudes"], mmap_mode="r", allow_pickle=False)
    else:
        manifest = pd.read_parquet(experiment / "input/manifest.parquet")
        geometry = pd.read_parquet(experiment / "input/real_geometry.parquet")
        positions = pd.Series(np.arange(len(manifest)), index=manifest.cell_id)
        values = np.load(experiment / "input/waveforms.npy", mmap_mode="r", allow_pickle=False)
    offsets = geometry.set_index("cell_id", verify_integrity=True).offset_m
    return values, positions, offsets


def scored_traces(directory):
    """The run's own saved per-trace scores for every trace it predicted.

    A grid run predicts the validation traces as well as the test traces and scores
    them in separate files; a whole-shot run has test traces only. Both are read so
    that the acceptance check covers every predicted trace, not a subset.
    """
    columns = {"cell_id", "error_energy", "target_energy", "clean_target"}
    tables = []
    for domain in ("test", "validation"):
        path = directory / f"{domain}_per_trace.parquet"
        if not path.is_file():
            continue
        table = pd.read_parquet(path)
        if not columns.issubset(table.columns):
            raise ValueError(f"{path}: saved per-trace scores are missing {columns}")
        tables.append(table[sorted(columns)].assign(scored_domain=domain))
    if not tables:
        raise ValueError(f"{directory}: no saved per-trace scores to check against")
    return pd.concat(tables, ignore_index=True).set_index("cell_id", verify_integrity=True)


def evaluate_method(repo, experiment, contract, method, seed, window):
    """Windowed per-trace columns for one method, checked against its saved scores."""
    directory = method_prediction(experiment, method, seed)
    cells = np.load(directory / "cell_ids.npy", allow_pickle=False)
    prediction = np.load(directory / "prediction.npy", mmap_mode="r", allow_pickle=False)
    if len(cells) != len(prediction):
        raise ValueError(f"{method}: prediction and cell_ids disagree")
    values, positions, offsets = run_inputs(repo, experiment, contract)
    samples = contract["time_sample_count"]
    ends = arrival_window_ends(
        offsets.reindex(cells).to_numpy(),
        intercept_s=window["intercept_s"],
        velocity_m_per_s=window["velocity_m_per_s"],
        sample_interval_s=contract["sample_interval_us"] / 1e6,
        sample_count=samples,
        minimum_samples=MINIMUM_WINDOW_SAMPLES,
    )
    saved = scored_traces(directory).reindex(cells)
    rows, checks = [], []
    for start in range(0, len(cells), CHUNK):
        block = slice(start, start + CHUNK)
        predicted = np.asarray(prediction[block], dtype=np.float64)
        target = np.asarray(
            values[positions.reindex(cells[block]).to_numpy(), :samples], dtype=np.float64
        )
        whole = waveform_columns(predicted, target)
        windowed = windowed_waveform_columns(predicted, target, ends[block])
        rows.append(pd.DataFrame({"cell_id": cells[block], **windowed}))
        with np.errstate(divide="ignore", invalid="ignore"):
            reference = 10 * (
                np.log10(saved.target_energy.to_numpy()[block])
                - np.log10(saved.error_energy.to_numpy()[block])
            )
        finite = np.isfinite(reference) & np.isfinite(whole["snr_db"])
        checks.append(np.max(np.abs(whole["snr_db"][finite] - reference[finite]), initial=0.0))
    table = pd.concat(rows, ignore_index=True)
    if saved.clean_target.isna().any():
        raise ValueError(f"{method}: predicted traces are missing from the saved scores")
    table["clean_target"] = saved.clean_target.to_numpy().astype(bool)
    table["scored_domain"] = saved.scored_domain.to_numpy()
    difference = float(max(checks))
    if difference > SNR_TOLERANCE_DB:
        raise ValueError(f"{method}: recomputed SNR differs from the saved scores by {difference}")
    return table, {
        "method": method,
        "prediction": str(directory.relative_to(repo)),
        "prediction_sha256": sha256_file(directory / "prediction.npy"),
        "trace_count": int(len(table)),
        "max_abs_snr_db_difference": difference,
        "window_samples_min": int(ends.min()),
        "window_samples_median": float(np.median(ends)),
        "window_samples_max": int(ends.max()),
    }


def _safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else float("nan")


def _statistic(values, name):
    """Mean or median of a column that is entirely undefined on a silent reference."""
    return float(getattr(values, name)()) if values.notna().any() else float("nan")


def summarize(table, population):
    """Equal-weight trace summary of one method on one population.

    Only test traces are summarized: a grid run also predicts its validation traces,
    and those took part in selection.
    """
    rows = table[table.scored_domain.eq("test")]
    rows = rows if population == "all_eligible" else rows[rows.clean_target]
    return {
        "population": population,
        "trace_count": int(len(rows)),
        "mean_post_snr_db": float(rows.post_snr_db.mean()),
        "median_post_snr_db": float(rows.post_snr_db.median()),
        # Mean of the per-trace ratio, matching the studies' masked_trace_relative_mse,
        # with the pooled ratio beside it because the two weight traces differently.
        "mean_post_relative_mse": float(rows.post_relative_mse.mean()),
        "pooled_post_relative_mse": float(
            rows.post_error_energy.sum() / rows.post_target_energy.sum()
        ),
        "mean_pre_energy_ratio": _statistic(rows.pre_energy_ratio, "mean"),
        "median_pre_energy_ratio": _statistic(rows.pre_energy_ratio, "median"),
        "median_pre_excess_ratio": _statistic(rows.pre_excess_ratio, "median"),
        "median_pre_target_share": _statistic(rows.pre_target_share, "median"),
        # Synthetic data can be exactly silent before the arrival, which leaves the
        # excess ratio undefined; the share of whole-trace energy stays defined there.
        "pooled_pre_excess_ratio": _safe_ratio(
            rows.pre_prediction_energy.sum(), rows.pre_target_energy.sum()
        ),
        "pooled_pre_energy_ratio": _safe_ratio(
            rows.pre_prediction_energy.sum(),
            rows.post_target_energy.sum() + rows.pre_target_energy.sum(),
        ),
    }


def evaluate_arrival_windows(repo, experiment, output, *, methods=None, seed=None):
    """Write per-trace and summary windowed scores for one run's methods."""
    repo, experiment = Path(repo).resolve(), Path(experiment).resolve()
    output = Path(output).resolve()
    if (repo / "runs").resolve() in output.parents:
        raise ValueError("analysis output must be outside runs/")
    contract = yaml.safe_load((experiment / "study/config.yaml").read_text())
    window = ARRIVAL_WINDOWS[contract["dataset_id"]]
    names = list(methods or contract["methods"])
    output.mkdir(parents=True, exist_ok=False)
    tables, sources, summaries = [], [], []
    for method in names:
        table, source = evaluate_method(repo, experiment, contract, method, seed, window)
        tables.append(table.assign(method_id=method))
        sources.append(source)
        for population in ("clean_target", "all_eligible"):
            summaries.append({"method_id": method, **summarize(table, population)})
    pd.concat(tables, ignore_index=True).to_parquet(output / "per_trace.parquet", index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "summary.csv", index=False)
    write_json(
        output / "manifest.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "experiment": str(experiment.relative_to(repo)),
            "study_id": contract["study_id"],
            "dataset_id": contract["dataset_id"],
            "display_seed": seed,
            "post_hoc_secondary_metric": True,
            "used_for_selection": False,
            "new_inference": False,
            "window_rule": "t < intercept_s + offset / velocity_m_per_s, declared from "
            "observed traces only and never from a first-break pick",
            "window": dict(window),
            "minimum_window_samples": MINIMUM_WINDOW_SAMPLES,
            "acceptance_rule": f"recomputed whole-trace SNR reproduces each run's saved "
            f"per-trace scores within {SNR_TOLERANCE_DB} dB",
            "zero_prediction_caveat": "a zero prediction leaks nothing and scores best on "
            "pre_energy_ratio; read it beside post_snr_db",
            "methods": sources,
            "artifacts": {
                name: sha256_file(output / name) for name in ("per_trace.parquet", "summary.csv")
            },
        },
    )
    return output


def read_summary(output):
    """Convenience reader for a written summary."""
    return pd.read_csv(Path(output) / "summary.csv"), json.loads(
        (Path(output) / "manifest.json").read_text()
    )
