"""Evaluate O-only IDW amplitude correction of an immutable NeRSI prediction."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from seis_interp import run_records
from seis_interp.c3_poc_run_records import validate_poc_prediction
from seis_interp.configuration import load_resolved_config
from seis_interp.data.c3_poc_inputs import load_c3_random80_poc_inputs
from seis_interp.data.file_checksums import file_sha256
from seis_interp.evaluation.c3_poc_comparison import validate_poc_run_artifacts
from seis_interp.evaluation.c3_volume_metrics import evaluate_c3_volume_prediction
from seis_interp.processing.trace_rms_idw import (
    interpolate_trace_rms_idw,
    match_prediction_trace_rms,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_records.check_new_output_directory(args.output)
    provenance = run_records.current_git_metadata()
    started_at = run_records.utc_timestamp()
    source, lock = validate_poc_run_artifacts("nersi", args.source_run)
    config = load_resolved_config(args.source_run / "config.resolved.yaml")
    study = Path(__file__).resolve().parent
    paths = yaml.safe_load((study / "inputs.yaml").read_text())["paths"]
    paths = {f"{key}_dir": (study / value).resolve() for key, value in paths.items()}
    inputs = load_c3_random80_poc_inputs(config=config, **paths)
    if inputs.inputs_lock != lock:
        raise ValueError("source prediction and current inputs have different full locks")
    observed = inputs.observed_volume
    prediction = np.load(args.source_run / "prediction.npy", allow_pickle=False)
    validate_poc_prediction(prediction, observed)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "inputs.lock.json").write_text(
        json.dumps(lock, indent=2, allow_nan=False) + "\n"
    )
    candidates = [
        {"radius": radius, "power": power, "axis_scales": scales}
        for radius, power, scales in (
            (1, 2, [1, 1, 1, 1]),
            (2, 2, [1, 1, 1, 1]),
            (2, 4, [1, 1, 1, 1]),
            (3, 2, [1, 1, 1, 1]),
            (2, 2, [4, 1, 1, 1]),
            (2, 4, [4, 1, 1, 1]),
        )
    ]
    rows = []
    for index, options in enumerate(candidates):
        directory = args.output / f"candidate_{index:02d}"
        directory.mkdir()
        started = time.perf_counter()
        record = {"options": options, "operation": "match_prediction_trace_rms"}
        try:
            field = interpolate_trace_rms_idw(
                observed.values, observed.observed_trace_mask, **options
            )
            corrected = match_prediction_trace_rms(prediction, field, observed.observed_trace_mask)
            validate_poc_prediction(corrected, observed)
            record["postprocessing_seconds"] = time.perf_counter() - started
            metrics = evaluate_c3_volume_prediction(
                corrected,
                observed,
                interim_dir=paths["interim_dir"],
                volume_metadata=inputs.volume_metadata,
                target_coverage_mask=observed.evaluation_target_trace_mask,
            )
            if metrics["observed_max_abs_error"] != 0:
                raise ValueError("postprocessing changed observed amplitudes")
            np.save(directory / "prediction.npy", corrected, allow_pickle=False)
            record.update(
                status="success",
                metrics=metrics,
                prediction_path=str(directory / "prediction.npy"),
                prediction_sha256=file_sha256(directory / "prediction.npy"),
            )
            print(options, metrics["evaluation_target"]["snr_db"], flush=True)
        except ValueError as error:
            record.update(
                status="failed", error_type=type(error).__name__, error_message=str(error)
            )
            print(options, str(error), flush=True)
        record["end_to_end_seconds"] = time.perf_counter() - started
        (directory / "metadata.json").write_text(
            json.dumps(record, indent=2, allow_nan=False) + "\n"
        )
        rows.append(record)
    summary = {
        "kind": "target_selected_NeRSI_IDW_amplitude_postprocessing",
        "started_at_utc": started_at,
        **provenance,
        "source_run": str(args.source_run),
        "source_artifacts": source,
        "source_config_sha256": file_sha256(args.source_run / "config.resolved.yaml"),
        "source_metadata_sha256": file_sha256(args.source_run / "metadata.json"),
        "implementation_sha256": {
            str(path): file_sha256(path)
            for path in (Path(__file__), Path("src/seis_interp/processing/trace_rms_idw.py"))
        },
        "training_executed": False,
        "independent_evaluation": False,
        "target_amplitudes_used_for_IDW": False,
        "checkpoint": "unchanged source checkpoint plus recorded postprocessing",
        "runs": rows,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
