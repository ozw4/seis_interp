"""Fixed random-80 PoC input checking and independent method execution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def add_poc_commands(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register input checking and the five-method comparison runner."""
    poc = subparsers.add_parser("poc", help="Check inputs and run the fixed random-80 PoC.")
    commands = poc.add_subparsers(dest="poc_command", required=True)
    check = commands.add_parser("check", help="Verify inputs without training or evaluation.")
    for name in ("interim", "processed", "mask", "case", "volume"):
        check.add_argument(f"--{name}", type=Path, required=True)
    check.add_argument("--json", action="store_true", help="Print the input summary as JSON.")
    check.set_defaults(handler=_check)
    run_all = commands.add_parser("run-all", help="Run each PoC method in a separate process.")
    for name in ("interim", "processed", "mask", "case", "volume"):
        run_all.add_argument(f"--{name}", type=Path, required=True)
    for name in ("pocs", "drr", "nersi", "ccnet5d", "gnn"):
        run_all.add_argument(f"--{name}-config", type=Path, required=True)
    run_all.add_argument("--output", type=Path, required=True, help="A new comparison directory.")
    run_all.add_argument(
        "--json", action="store_true", help="Print the comparison summary as JSON."
    )
    run_all.set_defaults(handler=_run_all)


def _check(args: argparse.Namespace) -> int:
    from seis_interp.pipelines.check_c3_poc import check_c3_poc_inputs

    try:
        summary = check_c3_poc_inputs(
            interim_dir=args.interim,
            processed_dir=args.processed,
            mask_dir=args.mask,
            case_dir=args.case,
            volume_dir=args.volume,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"poc check failed: {error}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    else:
        print(f"Benchmark case: {summary['case_id']}")
        print(f"Volume shape: {summary['shape']}")
        print(f"Observed traces: {summary['observed_trace_count']}")
        print(f"Target traces: {summary['target_trace_count']}")
        print(f"Observed global RMS: {summary['observed_global_rms']}")
    return 0


def _run_all(args: argparse.Namespace) -> int:
    from seis_interp.pipelines.run_c3_random80_poc import run_c3_random80_poc

    try:
        summary = run_c3_random80_poc(
            interim_dir=args.interim,
            processed_dir=args.processed,
            mask_dir=args.mask,
            case_dir=args.case,
            volume_dir=args.volume,
            pocs_config=args.pocs_config,
            drr_config=args.drr_config,
            nersi_config=args.nersi_config,
            ccnet5d_config=args.ccnet5d_config,
            gnn_config=args.gnn_config,
            output_dir=args.output,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"poc run-all failed: {error}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    else:
        print(f"Comparison status: {summary['status']}")
        print(f"Comparison valid: {summary['comparison_valid']}")
        for run in summary["runs"]:
            print(f"{run['method']}: {run['status']} ({run['output_directory']})")
            if run.get("error_message"):
                print(f"  {run['error_type']}: {run['error_message']}")
        print(f"Summary JSON: {args.output / 'summary.json'}")
        print(f"Summary CSV: {args.output / 'summary.csv'}")
    if summary.get("error_message"):
        print(f"poc run-all failed: {summary['error_message']}", file=sys.stderr)
    return 0 if summary["status"] == "success" else 1
