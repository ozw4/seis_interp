"""Test-only scoring after every fixed periodic-shot method has a final outcome."""

import fcntl
import json
from itertools import combinations

import numpy as np
import pandas as pd
import yaml

from seis_interp.data.forge_headers import sha256_file
from seis_interp.data.forge_mvp_artifacts import verify_hashes, write_json
from seis_interp.data.forge_mvp_run_records import utc_now
from seis_interp.data.forge_trace_experiment import read_forge_trace_targets
from seis_interp.evaluation.forge_mvp_metrics import evaluate_traces, validate_predictions
from seis_interp.evaluation.trace_snr import summarize_trace_snr, trace_snr_db
from seis_interp.processing.forge_shot_mask import random_shot_support


def finalize_forge_shot_experiment(experiment):
    """Return False until all methods finish; verify every success before any target read."""
    workers = experiment / "workers"
    workers.mkdir(exist_ok=True)
    with (workers / "summary.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        seal = json.loads((experiment / "experiment.json").read_text())
        contract = yaml.safe_load((experiment / "study/config.yaml").read_text())
        if (
            contract.get("mask_family") not in ("periodic_shot", "random_shot")
            or contract["validation_holdout"] is not None
        ):
            raise ValueError("test-only finalization requires a whole-shot contract")
        records = {}
        for method in seal["methods"]:
            path = experiment / "method_runs" / method / "manifest.json"
            if not path.is_file():
                return False
            record = json.loads(path.read_text())
            if (
                record["status"] not in ("predicted", "completed", "failed")
                or "ended_at" not in record
            ):
                return False
            records[method] = record
        verify_hashes(experiment, seal["artifacts"])
        partition = pd.read_parquet(experiment / "input/partition.parquet")
        mask = partition.assign(split=np.where(partition.role.eq("train"), "observed", "test"))
        for method, record in records.items():
            if record["status"] == "failed":
                continue
            output = experiment / "method_runs" / method
            binding = record["input_binding"]
            if (
                binding["experiment_hash"] != sha256_file(experiment / "experiment.json")
                or binding["partition_hash"] != seal["artifacts"]["input/partition.parquet"]
            ):
                raise ValueError("methods must share the sealed experiment and acquisition mask")
            if "prediction_finalized_at" not in record:
                raise ValueError("scoring requires a finalized prediction")
            verify_hashes(output, record["artifacts"])
            validate_predictions(
                np.load(output / "prediction.npy", mmap_mode="r"),
                np.load(output / "cell_ids.npy"),
                mask,
                contract["time_sample_count"],
            )
        for method, record in records.items():
            if record["status"] != "predicted":
                continue
            output = experiment / "method_runs" / method
            rows, target = read_forge_trace_targets(experiment, "test")
            cells = np.load(output / "cell_ids.npy")
            prediction = np.load(output / "prediction.npy", mmap_mode="r")[
                pd.Index(cells).get_indexer(rows.cell_id)
            ]
            metrics, traces = evaluate_traces(prediction, target, rows)
            traces["trace_snr_db"] = trace_snr_db(target, prediction)
            for population in ("clean_target", "all_eligible"):
                selected = (
                    traces if population == "all_eligible" else traces.loc[traces.clean_target]
                )
                metrics[population].update(summarize_trace_snr(selected.trace_snr_db.to_numpy()))
            traces.to_parquet(output / "test_per_trace.parquet", index=False)
            record.update(
                status="completed",
                metrics={"test": metrics},
                test_scoring_calls=1,
                scored_at=utc_now(),
            )
            record["artifacts"]["test_per_trace.parquet"] = sha256_file(
                output / "test_per_trace.parquet"
            )
            temporary = output / "manifest.tmp"
            write_json(temporary, record)
            temporary.replace(output / "manifest.json")
            for path in output.iterdir():
                if path.is_file():
                    path.chmod(0o444)
            state_path = workers / f"{method}.json"
            if state_path.exists():
                state = json.loads(state_path.read_text())
                state.update(status="completed", phase="completed", metrics=record["metrics"])
                temporary = state_path.with_suffix(".tmp")
                write_json(temporary, state)
                temporary.replace(state_path)
        write_shot_tables(experiment, records, contract)
        return True


def score_rows(rows):
    """The same per-trace means, with explicit nonfinite SNR status and no clipping."""
    if rows.empty:
        return {
            "trace_count": 0,
            "masked_trace_relative_mse": None,
            "mean_trace_snr_db": None,
            "mean_trace_snr_status": "empty",
        }
    return {
        "trace_count": len(rows),
        "masked_trace_relative_mse": float(rows.relative_mse.mean()),
        **summarize_trace_snr(rows.trace_snr_db.to_numpy()),
    }


def write_shot_tables(experiment, records, contract):
    """Keep five methods (including NA failures), shot identities and edge support strata."""
    partition = pd.read_parquet(experiment / "input/partition.parquet")
    test = partition.loc[partition.role.eq("test")]
    columns = ["source_cell", "source_grid_line", "source_grid_point"]
    columns += [name for name in ("source_line", "source_point") if name in test]
    shots = (
        test.groupby(columns, sort=True)
        .agg(
            eligible_trace_count=("cell_id", "size"),
            clean_target_trace_count=("clean_target", "sum"),
        )
        .reset_index()
    )
    points = contract["spatial_shape"][1]
    if contract.get("mask_family") == "random_shot":
        shots = shots.merge(
            random_shot_support(partition, contract["spatial_shape"]),
            on=["source_cell", "source_grid_line", "source_grid_point"],
            validate="one_to_one",
        )
    else:
        shots["shot_support"] = np.where(
            shots.source_grid_point.isin([0, points - 1]), "edge", "internal"
        )
    summary, strata = [], []
    for method in contract["methods"]:
        record = records[method]
        rows = (
            pd.read_parquet(experiment / "method_runs" / method / "test_per_trace.parquet")
            if record["status"] == "completed"
            else None
        )
        base = {
            "method_id": method,
            "status": record["status"],
            "failure_reason": record.get("failure_reason"),
            **{
                key: record.get(key)
                for key in (
                    "runtime_seconds",
                    "peak_cpu_rss_bytes",
                    "peak_cuda_allocated_bytes",
                    "peak_cuda_reserved_bytes",
                )
            },
        }
        for population in ("clean_target", "all_eligible"):
            selected = (
                None
                if rows is None
                else rows.loc[rows.clean_target]
                if population == "clean_target"
                else rows
            )
            score = (
                score_rows(selected)
                if selected is not None
                else {
                    "trace_count": int(test.clean_target.sum())
                    if population == "clean_target"
                    else len(test),
                    "mean_trace_snr_db": None,
                    "masked_trace_relative_mse": None,
                }
            )
            summary.append({**base, "population": population, **score})
            for support in ("internal", "edge"):
                cells = shots.loc[shots.shot_support.eq(support), "source_cell"]
                subset = (
                    selected.loc[selected.source_cell.isin(cells)] if selected is not None else None
                )
                strata.append(
                    {
                        **base,
                        "population": population,
                        "shot_support": support,
                        "shot_count": len(cells),
                        **(
                            score_rows(subset)
                            if subset is not None
                            else {"mean_trace_snr_db": None, "masked_trace_relative_mse": None}
                        ),
                    }
                )
            for metric in ("mean_trace_snr_db", "masked_trace_relative_mse"):
                shots[f"{method}_{population}_{metric}"] = np.nan
            if selected is not None:
                for source, group in selected.groupby("source_cell", sort=True):
                    values = score_rows(group)
                    for metric in ("mean_trace_snr_db", "masked_trace_relative_mse"):
                        shots.loc[
                            shots.source_cell.eq(source), f"{method}_{population}_{metric}"
                        ] = values[metric]
        shots[f"{method}_status"] = record["status"]
        shots[f"{method}_failure_reason"] = record.get("failure_reason")
    pd.DataFrame(summary).to_csv(experiment / "summary.csv", index=False, na_rep="NA")
    pd.DataFrame(strata).to_csv(experiment / "shot_support.csv", index=False, na_rep="NA")
    shots.to_csv(experiment / "per_shot.csv", index=False, na_rep="NA")
    write_shot_averages_and_differences(experiment, shots, contract["methods"])
    write_json(
        experiment / "outcomes.json",
        {
            "all_methods_completed": all(r["status"] == "completed" for r in records.values()),
            "methods": {
                m: {
                    k: r.get(k)
                    for k in ("status", "failure_reason", "prediction_finalized_at", "scored_at")
                }
                for m, r in records.items()
            },
        },
    )


def write_shot_averages_and_differences(experiment, shots, methods):
    """Equal-weight shot means and differences on the same original shot identities."""
    means, differences = [], []
    for population in ("clean_target", "all_eligible"):
        for support in ("all", "internal", "edge"):
            subset = shots if support == "all" else shots.loc[shots.shot_support.eq(support)]
            for method in methods:
                row = {
                    "method_id": method,
                    "population": population,
                    "shot_support": support,
                    "shot_count": len(subset),
                }
                for metric in ("mean_trace_snr_db", "masked_trace_relative_mse"):
                    values = subset[f"{method}_{population}_{metric}"].to_numpy(dtype=float)
                    row[f"equal_weight_shot_mean_{metric}"] = (
                        float(values.mean()) if len(values) and np.isfinite(values).all() else None
                    )
                means.append(row)
        for left, right in combinations(methods, 2):
            for _, shot in shots.iterrows():
                row = {
                    key: shot[key]
                    for key in (
                        "source_cell",
                        "source_grid_line",
                        "source_grid_point",
                        "shot_support",
                    )
                }
                row.update(population=population, method_left=left, method_right=right)
                for metric in ("mean_trace_snr_db", "masked_trace_relative_mse"):
                    row[f"left_minus_right_{metric}"] = (
                        shot[f"{left}_{population}_{metric}"]
                        - shot[f"{right}_{population}_{metric}"]
                    )
                differences.append(row)
    pd.DataFrame(means).to_csv(experiment / "shot_equal_weight.csv", index=False, na_rep="NA")
    pd.DataFrame(differences).to_csv(
        experiment / "paired_shot_differences.csv", index=False, na_rep="NA"
    )
