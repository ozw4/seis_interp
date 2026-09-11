"""Read-only PoC input check command."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def add_poc_commands(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the lightweight PoC checker."""
    poc = subparsers.add_parser("poc", help="Check the fixed random-80 PoC inputs.")
    commands = poc.add_subparsers(dest="poc_command", required=True)
    check = commands.add_parser("check", help="Verify inputs without training or evaluation.")
    for name in ("interim", "processed", "mask", "case", "volume"):
        check.add_argument(f"--{name}", type=Path, required=True)
    check.add_argument("--json", action="store_true", help="Print the input summary as JSON.")
    check.set_defaults(handler=_check)


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
