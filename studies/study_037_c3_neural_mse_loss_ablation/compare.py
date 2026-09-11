"""Verify the three completed MSE runs against frozen Stage-1 evidence and print CSV."""

import argparse
import csv
import sys
from copy import deepcopy
from pathlib import Path

from seis_interp.configuration import load_resolved_config
from seis_interp.data.file_checksums import file_sha256
from seis_interp.evaluation.c3_poc_comparison import read_poc_json, validate_poc_run_artifacts

STUDY = Path(__file__).resolve().parent
BASELINE = STUDY.parent / "study_036_c3_random80_observed_only_poc"
METHODS = ("nersi", "ccnet5d", "relational_trace_graph")


def comparison_rows(run_root: Path) -> list[dict[str, object]]:
    """Read run records and hash artifacts; never load target or prediction amplitudes."""
    frozen = read_poc_json(BASELINE / "stage_1_baseline.lock.json")
    baseline_run = (BASELINE / frozen["run_directory"]).resolve()
    rows = []
    for method in METHODS:
        for filename in (
            "config.resolved.yaml",
            "inputs.lock.json",
            "metadata.json",
            "metrics.json",
            "prediction.npy",
            "final.pt",
        ):
            relative = f"{method}/{filename}"
            if file_sha256(baseline_run / relative) != frozen["run_file_sha256"][relative]:
                raise ValueError(f"Frozen baseline artifact changed: {relative}")
        baseline, old_lock = validate_poc_run_artifacts(method, baseline_run / method)
        output = run_root / method
        current, lock = validate_poc_run_artifacts(method, output)
        if lock != frozen["inputs_lock"] or old_lock != lock:
            raise ValueError(f"{method}: full input lock differs from frozen Stage-1")
        expected = deepcopy(load_resolved_config(baseline_run / method / "config.resolved.yaml"))
        expected["training"]["loss"] = "masked_trace_mse"
        if load_resolved_config(output / "config.resolved.yaml") != expected:
            raise ValueError(f"{method}: resolved config differs beyond training.loss")
        metadata = read_poc_json(output / "metadata.json")
        loss_names = [metadata["loss_or_native_objective"]]
        for section in (metadata["method_details"], metadata["training_or_reconstruction"]):
            if "loss" in section:
                loss_names.append(section["loss"])
        if any(loss != "masked_trace_mse" for loss in loss_names):
            raise ValueError(f"{method}: inconsistent recorded loss")
        if current["optimizer_updates"] != 5000:
            raise ValueError(f"{method}: full budget was not completed")
        row = {
            "method": method,
            "baseline_snr_status": baseline["snr_status"],
            "mse_snr_status": current["snr_status"],
        }
        for metric in ("snr_db", "rmse", "mean_trace_relative_mse"):
            row[f"baseline_{metric}"] = baseline[metric]
            row[f"mse_{metric}"] = current[metric]
            row[f"delta_{metric}"] = (
                None
                if baseline[metric] is None or current[metric] is None
                else current[metric] - baseline[metric]
            )
        for field in (
            "training_or_reconstruction_seconds",
            "prediction_seconds",
            "cuda_max_memory_allocated_bytes",
        ):
            row[field] = current[field]
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    rows = comparison_rows(args.run_root)
    writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
    writer.writeheader()
    for row in rows:
        formatted = {}
        for key, value in row.items():
            if isinstance(value, float):
                text = f"{value:.4f}"
                formatted[key] = "approximately 0" if value != 0 and float(text) == 0 else text
            else:
                formatted[key] = value
        writer.writerow(formatted)


if __name__ == "__main__":
    main()
