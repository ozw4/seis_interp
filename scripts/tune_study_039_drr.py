"""Run the study-039 target-tuned pilot grid using the existing DRR implementation."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml

from seis_interp import run_records
from seis_interp.configuration import load_resolved_config
from seis_interp.data.c3_poc_inputs import load_c3_random80_poc_inputs
from seis_interp.evaluation.drr_parameter_scan import score_drr_target_regions

STUDY = Path("studies/study_039_c3_drr_target_tuning")
REFERENCE = Path("studies/study_036_c3_random80_observed_only_poc/formal/drr.yaml")
INPUTS = {
    "interim_dir": Path("data/interim/c3_na/all_ffids"),
    "processed_dir": Path("data/processed/c3_na/study_029_c3_amplitude_qc/partition"),
    **{
        f"{name}_dir": Path("data/processed/c3_na/study_029_c3_amplitude_qc")
        / folder
        / "c3_benchmark_test_random_trace_80_seed42"
        for name, folder in (("mask", "masks"), ("case", "cases"), ("volume", "volumes"))
    },
}


def run_candidate(job):
    name, parameters, output, protocol = job
    started = time.perf_counter()
    inputs = load_c3_random80_poc_inputs(config=load_resolved_config(REFERENCE), **INPUTS)
    volume = inputs.observed_volume
    regions = [
        tuple(
            slice(start, start + length)
            for start, length in zip(origin, protocol["region_shape"], strict=True)
        )
        for origin in protocol["region_origins"]
    ]
    result = score_drr_target_regions(
        volume,
        interim_dir=INPUTS["interim_dir"],
        volume_metadata=inputs.volume_metadata,
        parameters=parameters,
        regions=regions,
    )
    record = {
        "candidate": name,
        "parameters": parameters,
        "target_pilot": result,
        "seconds": time.perf_counter() - started,
        "inputs_lock": inputs.inputs_lock,
        "git": run_records.current_git_metadata(),
    }
    (output / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
    return name, result["snr_db"], record["seconds"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=STUDY / "pilot.yaml")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    protocol = yaml.safe_load(args.protocol.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "protocol.yaml").write_text(yaml.safe_dump(protocol, sort_keys=False))
    jobs = [
        (name, params, args.output, protocol) for name, params in protocol["candidates"].items()
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(run_candidate, jobs):
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
