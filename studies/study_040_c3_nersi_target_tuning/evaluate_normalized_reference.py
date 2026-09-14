"""Score immutable trace-normalized NeRSI runs in the normalized reference domain."""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from seis_interp import run_records
from seis_interp.configuration import load_resolved_config
from seis_interp.data.c3_poc_inputs import load_c3_random80_poc_inputs
from seis_interp.data.file_checksums import file_sha256
from seis_interp.evaluation.c3_poc_comparison import validate_poc_run_artifacts
from seis_interp.evaluation.normalized_trace_reference import evaluate_normalized_trace_reference
from seis_interp.training.c3_volume_nersi_data import build_c3_volume_nersi_data
from seis_interp.training.nersi_checkpoints import (
    load_fixed_step_nersi_checkpoint,
    validate_fixed_step_nersi_checkpoint_input_binding,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_records.check_new_output_directory(args.output)
    study = Path(__file__).resolve().parent
    paths = {
        k + "_dir": (study / v).resolve()
        for k, v in yaml.safe_load((study / "inputs.yaml").read_text())["paths"].items()
    }
    results = []
    lock = None
    frozen_lock = json.loads(
        (
            study.parent / "study_037_c3_neural_mse_loss_ablation/stage_1b_results.lock.json"
        ).read_text()
    )["inputs_lock"]
    for run in args.run:
        physical, current = validate_poc_run_artifacts("nersi", run)
        if current != frozen_lock:
            raise ValueError("reference inputs differ from frozen baseline")
        if lock is not None and current != lock:
            raise ValueError("normalized comparison inputs differ")
        lock = current
        config = load_resolved_config(run / "config.resolved.yaml")
        if config["training"]["amplitude_scaling"] != "observed_trace_rms_idw":
            raise ValueError("reference requires trace-RMS-normalized training")
        inputs = load_c3_random80_poc_inputs(config=config, **paths)
        if current != inputs.inputs_lock:
            raise ValueError("source run differs from verified inputs")
        checkpoint = load_fixed_step_nersi_checkpoint(run / "final.pt")
        data = build_c3_volume_nersi_data(
            inputs.observed_volume,
            amplitude_scale=checkpoint.amplitude_scale,
            trace_amplitude_scale=checkpoint.trace_amplitude_scale,
            time_alignment=checkpoint.time_alignment,
        )
        validate_fixed_step_nersi_checkpoint_input_binding(checkpoint, current, data)
        metrics = evaluate_normalized_trace_reference(
            np.load(run / "prediction.npy", mmap_mode="r", allow_pickle=False),
            inputs.observed_volume,
            checkpoint.trace_amplitude_scale,
            interim_dir=paths["interim_dir"],
            volume_metadata=inputs.volume_metadata,
        )
        results.append(
            {
                "run": str(run),
                "source_artifacts": physical,
                "config_sha256": file_sha256(run / "config.resolved.yaml"),
                "metadata_sha256": file_sha256(run / "metadata.json"),
                "metrics": metrics,
            }
        )
        print(run.name, metrics["evaluation_target"], flush=True)
    snrs = [r["metrics"]["evaluation_target"]["snr_db"] for r in results]
    summary = {
        "kind": "normalized_trace_reference_experiment",
        "independent_evaluation": False,
        "training_executed": False,
        "created_at_utc": run_records.utc_timestamp(),
        "inputs_lock": lock,
        "runs": results,
        "target_snr_db": 15.0,
        "target_reached": any(v is not None and v > 15 for v in snrs)
        or any(
            r["metrics"]["evaluation_target"]["snr_status"] == "perfect_reconstruction"
            for r in results
        ),
        "physical_goal_unchanged": True,
        "implementation_sha256": {
            str(path): file_sha256(path)
            for path in (
                Path(__file__),
                Path("src/seis_interp/evaluation/normalized_trace_reference.py"),
            )
        },
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
