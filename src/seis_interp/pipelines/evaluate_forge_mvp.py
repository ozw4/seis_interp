"""Evaluation-only access to the artificial FORGE test amplitudes."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from seis_interp.data.forge_headers import sha256_file
from seis_interp.data.forge_mvp_artifacts import (
    read_selected_traces,
    verify_hashes,
    verify_mvp_implementation,
    write_json,
)
from seis_interp.data.forge_mvp_run_records import utc_now
from seis_interp.evaluation.forge_mvp_metrics import (
    evaluate_traces,
    paired_geometry_effects,
    validate_predictions,
    verify_metrics_from_waveforms,
    verify_saved_metrics,
)
from seis_interp.processing.forge_mvp_contract import METHODS
from seis_interp.visualization.forge_mvp import plot_projection_errors, plot_sections


def evaluate_forge_mvp(repo: Path, preparation: Path, output: Path):
    seal = json.loads((preparation / "preparation_manifest.json").read_text())
    verify_hashes(preparation, seal["artifacts"])
    verify_mvp_implementation(repo, seal["implementation_hashes"])
    contract = json.loads((preparation / "input/input_contract.json").read_text())
    matrix = json.loads((output / "matrix_manifest.json").read_text())
    if matrix["preparation_hash"] != sha256_file(
        preparation / "preparation_manifest.json"
    ) or matrix["methods"] != list(METHODS):
        raise ValueError("execution matrix preparation mismatch")
    manifest = pd.read_parquet(preparation / "input/m1_trace_manifest.parquet")
    mask = pd.read_parquet(preparation / "masks/m1_random80_mvp_v1.parquet")
    test = mask.split.eq("test").to_numpy()
    records, valid, summaries, per_trace = {}, {}, [], {}
    # Inspect the whole matrix before inspecting predictions or opening targets.
    for method in METHODS:
        path = output / "runs" / method
        record = (
            json.loads((path / "run_manifest.json").read_text())
            if (path / "run_manifest.json").is_file()
            else {"status": "failed", "failure_reason": "worker exited without a manifest"}
        )
        records[method] = record
    findings = []
    # Validate every available manifest before loading prediction arrays.
    for method, record in records.items():
        if record["status"] != "predicted":
            if record["status"] == "running":
                record.update(
                    status="failed", failure_reason="worker terminated before prediction completed"
                )
            continue
        for key in (
            "mask_hash",
            "mapping_hash",
            "input_hashes",
            "global_rms",
            "test_trace_count",
            "clean_test_trace_count",
        ):
            if record[key] != seal[key]:
                raise ValueError(f"{method}: shared input mismatch for {key}")
        if (
            record["git_commit"] != matrix["git_commit"]
            or record["preparation_hash"] != matrix["preparation_hash"]
        ):
            raise ValueError("mixed implementation/preparation runs")
        expected_config = sha256_file(preparation / "configs" / f"{method}.yaml")
        if record["config_hash"] != expected_config:
            raise ValueError("method config differs from frozen preparation")
    for method, record in records.items():
        if record["status"] != "predicted":
            continue
        path = output / "runs" / method
        artifacts = {
            "prediction.npy": record["prediction_hash"],
            "prediction_index.parquet": record["prediction_index_hash"],
            "resolved_config.yaml": record["config_hash"],
        }
        if record["checkpoint_hash"]:
            artifacts["checkpoint/final.pt"] = record["checkpoint_hash"]
        verify_hashes(path, artifacts)
        prediction = np.load(path / "prediction.npy", mmap_mode="r")
        index = pd.read_parquet(path / "prediction_index.parquet")
        expected_index = mask.loc[test].reset_index(drop=True)
        if not index.drop(columns="prediction_row").equals(expected_index) or not np.array_equal(
            index.prediction_row, np.arange(len(index))
        ):
            raise ValueError("prediction original-trace inverse mapping mismatch")
        order = validate_predictions(
            prediction, index.cell_id.to_numpy(), mask, contract["time_sample_count"]
        )
        if np.all(np.ptp(prediction, axis=1) == 0):
            findings.append({"method": method, "issue": "all test predictions are constant"})
            continue
        valid[method] = prediction[order]
    if len(valid) != len(METHODS):
        return _write_incomplete_result(output, matrix, records, valid, findings)
    a, b = records["regsi_real"], records["regsi_grid"]
    if (
        a["initialization_hash"] != b["initialization_hash"]
        or a["parameter_count"] != b["parameter_count"]
    ):
        raise ValueError("ReGSI initial state or parameter count differs")
    raw = json.loads((preparation / "input/raw_input_hashes.json").read_text())
    verify_hashes(repo / raw["dataset_root"], raw["hashes"])
    target = read_selected_traces(
        repo / raw["dataset_root"],
        manifest.loc[test],
        contract["time_sample_count"],
        contract["sample_interval_us"],
    )
    for name in ("aggregate", "figures", "tables"):
        (output / name).mkdir(exist_ok=False)
    for method in METHODS:
        record = records[method]
        row = {
            "method": method,
            "geometry": method.rsplit("_", 1)[1],
            "status": record["status"],
            "runtime_seconds": record.get("runtime_seconds"),
            "peak_cpu_rss_bytes": record.get("peak_cpu_rss_bytes"),
            "peak_cuda_allocated_bytes": record.get("peak_cuda_allocated_bytes"),
            "failure_reason": record.get("failure_reason"),
        }
        if method in valid:
            metrics, traces = evaluate_traces(
                valid[method], target, mask.loc[test].reset_index(drop=True)
            )
            path = output / "runs" / method
            write_json(path / "metrics.json", metrics)
            traces.to_parquet(path / "metrics_by_trace.parquet", index=False)
            verify_saved_metrics(
                json.loads((path / "metrics.json").read_text()),
                pd.read_parquet(path / "metrics_by_trace.parquet"),
            )
            verify_metrics_from_waveforms(
                metrics, valid[method], target, mask.loc[test, "clean_target"]
            )
            record.update(
                status="evaluated",
                metrics_hash=sha256_file(path / "metrics.json"),
                metrics_by_trace_hash=sha256_file(path / "metrics_by_trace.parquet"),
                independent_metrics_verified=True,
            )
            write_json(path / "run_manifest.json", record)
            primary = metrics["clean_target"]
            row.update(
                status="evaluated",
                primary_relative_mse=primary["masked_trace_relative_mse"],
                global_nmse=primary["global_nmse"],
                snr_db=primary["snr_db"],
                median_correlation=primary["correlation"]["median"],
            )
            per_trace[method] = traces
        summaries.append(row)
    frame = pd.DataFrame(summaries)
    frame.to_parquet(output / "aggregate/mvp_metrics.parquet", index=False)
    frame.to_csv(output / "tables/mvp_metrics.csv", index=False)
    frame[
        [
            "method",
            "runtime_seconds",
            "peak_cpu_rss_bytes",
            "peak_cuda_allocated_bytes",
            "status",
            "failure_reason",
        ]
    ].to_parquet(output / "aggregate/runtime_summary.parquet", index=False)
    if "regsi_real" in per_trace and "regsi_grid" in per_trace:
        pair, effect, quartiles = paired_geometry_effects(
            per_trace["regsi_real"], per_trace["regsi_grid"]
        )
        pair.to_parquet(output / "aggregate/regsi_paired_effects.parquet", index=False)
        quartiles.to_parquet(output / "aggregate/projection_quartiles.parquet", index=False)
        write_json(output / "aggregate/regsi_effect_summary.json", effect)
        plot_projection_errors(quartiles, output / "figures")
    if valid:
        truth = np.empty((len(mask), contract["time_sample_count"]), dtype=np.float32)
        truth[test] = target
        truth[~test] = np.load(preparation / "input/observed_waveforms.npy", mmap_mode="r")
        # Only materialize the two predeclared sections, not six complete cubes.
        selected = manifest.source_cell.eq(
            contract["visualization"]["source_cell"]
        ) | manifest.receiver_cell.eq(contract["visualization"]["receiver_cell"])
        section_manifest, section_mask = (
            manifest.loc[selected].reset_index(drop=True),
            mask.loc[selected].reset_index(drop=True),
        )
        predictions = {}
        for method, prediction in valid.items():
            section = truth[selected].copy()
            test_positions = pd.Index(mask.loc[test, "cell_id"]).get_indexer(
                section_mask.loc[section_mask.split.eq("test"), "cell_id"]
            )
            section[section_mask.split.eq("test")] = prediction[test_positions]
            predictions[method] = section
        plot_sections(
            section_manifest,
            section_mask,
            truth[selected],
            predictions,
            contract,
            output / "figures",
        )
    result = {
        "status": "accepted",
        "findings": findings,
        "preliminary": True,
        "mask_count": 1,
        "model_seed_count": 1,
        "statistical_claims": False,
        "completed_methods": list(valid),
        "end_time": utc_now(),
        "preparation_hash": matrix["preparation_hash"],
        "git_commit": matrix["git_commit"],
        "artifacts": {
            str(p.relative_to(output)): sha256_file(p)
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
    }
    write_json(output / "aggregate/final_mvp_manifest.json", result)
    return result


def _write_incomplete_result(output, matrix, records, valid, findings):
    """Record readiness and runtime only, without loading targets or producing metrics."""
    (output / "aggregate").mkdir(exist_ok=False)
    reasons = {finding["method"]: finding["issue"] for finding in findings}
    rows = [
        {
            "method": method,
            "status": record["status"],
            "prediction_ready": method in valid,
            "runtime_seconds": record.get("runtime_seconds"),
            "peak_cpu_rss_bytes": record.get("peak_cpu_rss_bytes"),
            "peak_cuda_allocated_bytes": record.get("peak_cuda_allocated_bytes"),
            "failure_reason": record.get("failure_reason") or reasons.get(method),
        }
        for method, record in records.items()
    ]
    pd.DataFrame(rows).to_parquet(output / "aggregate/runtime_summary.parquet", index=False)
    result = {
        "status": "incomplete",
        "findings": findings,
        "preliminary": True,
        "mask_count": 1,
        "model_seed_count": 1,
        "statistical_claims": False,
        "completed_methods": [],
        "valid_methods": list(valid),
        "evaluation_started": False,
        "test_amplitudes_read": 0,
        "methods": rows,
        "end_time": utc_now(),
        "preparation_hash": matrix["preparation_hash"],
        "git_commit": matrix["git_commit"],
        "artifacts": {
            str(path.relative_to(output)): sha256_file(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
        },
    }
    write_json(output / "aggregate/final_mvp_manifest.json", result)
    return result
