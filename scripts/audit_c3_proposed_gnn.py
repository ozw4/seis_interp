#!/usr/bin/env python3
"""Audit explicit completed C3 GNN native runs with fixed engineering tolerances."""

import argparse
import json
import sys
from pathlib import Path

from seis_interp.evaluation.trace_graph_run_audit import audit_c3_proposed_gnn


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--prediction-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-suite-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        report = audit_c3_proposed_gnn(
            suite=args.suite,
            case=args.case,
            training_run=args.training_run,
            prediction_run=args.prediction_run,
            output=args.output,
            expected_suite_sha256=args.expected_suite_sha256,
        )
    except (OSError, ValueError) as error:
        print(f"audit input error: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    return 0 if report["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
