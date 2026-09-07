"""The ``interpolate`` commands for non-training interpolation methods."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping
from numbers import Real
from pathlib import Path


def _interpolate_pocs(args: argparse.Namespace) -> int:
    from seis_interp.pipelines.interpolate_pocs import interpolate_pocs_run

    progress_reporter = _print_progress_to_stderr if args.json else None
    try:
        summary = interpolate_pocs_run(
            config_path=args.config,
            interim_dir=args.interim,
            processed_dir=args.processed,
            mask_dir=args.mask,
            case_dir=args.case,
            volume_dir=args.volume,
            output_dir=args.output,
            progress_reporter=progress_reporter,
        )
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as error:
        print(f"interpolate pocs failed: {error}", file=sys.stderr)
        return 1

    _print_warnings_to_stderr(summary)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    else:
        target = summary["evaluation_target"]
        if not isinstance(target, Mapping):
            raise ValueError("evaluation_target summary must be a mapping")
        print(f"Output directory: {args.output}")
        print(f"Method: {summary['method']}")
        print(f"Benchmark case: {summary['case_id']}")
        print(f"Benchmark volume: {summary['volume_id']}")
        print(f"Target global S/N: {_format_snr(target)}")
        print(f"Target RMSE: {target['rmse']}")
        print(f"Observed maximum absolute error: {summary['observed_max_abs_error']}")
        print(f"Uncovered samples: {summary['uncovered_sample_count']}")
    return 0


def _format_snr(target: Mapping[str, object]) -> str:
    status = target.get("snr_status")
    if status == "finite":
        value = target.get("snr_db")
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            return "unavailable (invalid finite S/N)"
        return f"{value} dB"
    if status == "perfect_reconstruction":
        return "perfect reconstruction"
    if status == "undefined_zero_reference":
        return "undefined (zero reference energy)"
    return f"unavailable ({status})"


def _print_progress_to_stderr(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _print_warnings_to_stderr(summary: Mapping[str, object]) -> None:
    warnings = summary.get("warnings", [])
    if not isinstance(warnings, list):
        return
    for warning in warnings:
        print(f"Warning: {warning}", file=sys.stderr)


def add_interpolate_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register interpolation command parsers."""
    interpolate = subparsers.add_parser(
        "interpolate",
        help="Run non-training interpolation methods.",
    )
    interpolate_commands = interpolate.add_subparsers(
        dest="interpolate_command",
        required=True,
    )
    pocs = interpolate_commands.add_parser(
        "pocs",
        help="Run CPU Fourier POCS-5D on a verified C3 volume.",
    )
    pocs.add_argument("--config", type=Path, required=True, help="Study configuration YAML.")
    pocs.add_argument("--interim", type=Path, required=True, help="Interim trace dataset.")
    pocs.add_argument("--processed", type=Path, required=True, help="Prepared split dataset.")
    pocs.add_argument("--mask", type=Path, required=True, help="Interpolation mask artifact.")
    pocs.add_argument("--case", type=Path, required=True, help="Benchmark case artifact.")
    pocs.add_argument("--volume", type=Path, required=True, help="C3 volume-index artifact.")
    pocs.add_argument("--output", type=Path, required=True, help="Run output directory.")
    pocs.add_argument("--json", action="store_true", help="Print metrics as JSON.")
    pocs.set_defaults(handler=_interpolate_pocs)
