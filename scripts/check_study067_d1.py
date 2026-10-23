"""Check one completed D-1 production pilot against its frozen scratch reference."""

import argparse
import json
import math
from pathlib import Path


def check(experiment):
    reference = json.loads((experiment / "study/d1_reference.json").read_text())
    record = json.loads((experiment / "method_runs/drr_grid/S1/a02_b02/manifest.json").read_text())
    if record["status"] != "completed" or record["test_scoring_calls"] != 0:
        raise ValueError("acceptance requires a completed validation-only run")
    metrics = record["metrics"]["validation"]
    if reference["dataset"] == "forge":
        metrics = metrics["clean_target"]
    actual = metrics["mean_trace_snr_db"]
    difference = actual - reference["snr"]
    passed = math.isfinite(actual) and abs(difference) <= 0.01
    return {
        "experiment": str(experiment),
        "scratch_label": reference["label"],
        "expected_snr_db": reference["snr"],
        "production_snr_db": actual,
        "difference_db": difference,
        "tolerance_db": 0.01,
        "passed": passed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.experiment)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
