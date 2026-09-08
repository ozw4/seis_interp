"""The ``interpolate`` commands for benchmark-volume interpolation methods."""

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

    _print_run_summary(summary, args.output, json_output=args.json)
    return 0


def _interpolate_drr(args: argparse.Namespace) -> int:
    from seis_interp.pipelines.interpolate_drr import interpolate_drr_run

    progress_reporter = _print_progress_to_stderr if args.json else None
    try:
        summary = interpolate_drr_run(
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
        print(f"interpolate drr failed: {error}", file=sys.stderr)
        return 1

    _print_run_summary(summary, args.output, json_output=args.json)
    return 0


def _interpolate_siren(args: argparse.Namespace) -> int:
    from seis_interp.pipelines.interpolate_siren import interpolate_siren_run

    try:
        summary = interpolate_siren_run(
            config_path=args.config,
            interim_dir=args.interim,
            processed_dir=args.processed,
            mask_dir=args.mask,
            case_dir=args.case,
            volume_dir=args.volume,
            output_dir=args.output,
            device_override=args.device,
            progress_reporter=_print_progress_to_stderr,
        )
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as error:
        print(f"interpolate siren failed: {error}", file=sys.stderr)
        return 1

    _print_run_summary(summary, args.output, json_output=args.json)
    return 0


def _interpolate_ccnet5d(args: argparse.Namespace) -> int:
    from seis_interp.pipelines.interpolate_ccnet5d import interpolate_ccnet5d_run

    try:
        summary = interpolate_ccnet5d_run(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            interim_dir=args.interim,
            processed_dir=args.processed,
            mask_dir=args.mask,
            case_dir=args.case,
            volume_dir=args.volume,
            output_dir=args.output,
            device_override=args.device,
            progress_reporter=_print_progress_to_stderr,
        )
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as error:
        print(f"interpolate ccnet5d failed: {error}", file=sys.stderr)
        return 1

    _print_run_summary(summary, args.output, json_output=args.json)
    return 0


def _print_run_summary(
    summary: Mapping[str, object], output_directory: Path, *, json_output: bool
) -> None:
    _print_warnings_to_stderr(summary)
    if json_output:
        print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    else:
        target = summary["evaluation_target"]
        if not isinstance(target, Mapping):
            raise ValueError("evaluation_target summary must be a mapping")
        print(f"Output directory: {output_directory}")
        print(f"Method: {summary['method']}")
        print(f"Benchmark case: {summary['case_id']}")
        print(f"Benchmark volume: {summary['volume_id']}")
        print(f"Target global S/N: {_format_snr(target)}")
        print(f"Target RMSE: {target['rmse']}")
        print(f"Observed maximum absolute error: {summary['observed_max_abs_error']}")
        if "uncovered_trace_count" in summary:
            print(f"Uncovered traces: {summary['uncovered_trace_count']}")
        print(f"Uncovered samples: {summary['uncovered_sample_count']}")


def _interpolate_relational_trace_graph(args: argparse.Namespace) -> int:
    from seis_interp.pipelines.interpolate_relational_trace_graph import (
        interpolate_relational_trace_graph_run,
    )

    try:
        summary = interpolate_relational_trace_graph_run(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            interim_dir=args.interim,
            processed_dir=args.processed,
            mask_dir=args.mask,
            case_dir=args.case,
            volume_dir=args.volume,
            output_dir=args.output,
            device_override=args.device,
            progress_reporter=_print_progress_to_stderr,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"interpolate relational-trace-graph failed: {error}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    else:
        print(f"Output directory: {args.output}")
        print(f"Benchmark case: {summary['case_id']}")
        print(f"Target global S/N: {_format_snr(summary['evaluation_target'])}")
        print(f"Target RMSE: {summary['evaluation_target']['rmse']:.4f}")
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


def _add_c3_volume_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True, help="Study configuration YAML.")
    parser.add_argument("--interim", type=Path, required=True, help="Interim trace dataset.")
    parser.add_argument("--processed", type=Path, required=True, help="Prepared split dataset.")
    parser.add_argument("--mask", type=Path, required=True, help="Interpolation mask artifact.")
    parser.add_argument("--case", type=Path, required=True, help="Benchmark case artifact.")
    parser.add_argument("--volume", type=Path, required=True, help="C3 volume-index artifact.")
    parser.add_argument("--output", type=Path, required=True, help="Run output directory.")
    parser.add_argument("--json", action="store_true", help="Print metrics as JSON.")


def add_interpolate_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register interpolation command parsers."""
    interpolate = subparsers.add_parser(
        "interpolate",
        help="Run interpolation methods on verified benchmark volumes.",
    )
    interpolate_commands = interpolate.add_subparsers(
        dest="interpolate_command",
        required=True,
    )
    pocs = interpolate_commands.add_parser(
        "pocs",
        help="Run CPU Fourier POCS-5D on a verified C3 volume.",
    )
    _add_c3_volume_run_arguments(pocs)
    pocs.set_defaults(handler=_interpolate_pocs)
    drr = interpolate_commands.add_parser(
        "drr",
        help="Run CPU damped rank-reduction 5D on a verified C3 volume.",
    )
    _add_c3_volume_run_arguments(drr)
    drr.set_defaults(handler=_interpolate_drr)
    siren = interpolate_commands.add_parser(
        "siren",
        help="Fit a per-volume SIREN on observed C3 samples and interpolate missing traces.",
    )
    _add_c3_volume_run_arguments(siren)
    siren.add_argument("--device", help="Override the configured training device for this run.")
    siren.set_defaults(handler=_interpolate_siren)
    ccnet5d = interpolate_commands.add_parser(
        "ccnet5d", help="Interpolate a verified C3 volume with a frozen CCNet5D checkpoint."
    )
    _add_c3_volume_run_arguments(ccnet5d)
    ccnet5d.add_argument(
        "--checkpoint", type=Path, required=True, help="Pretrained CCNet5D checkpoint."
    )
    ccnet5d.add_argument("--device", help="Override the configured inference device for this run.")
    ccnet5d.set_defaults(handler=_interpolate_ccnet5d)
    relational = interpolate_commands.add_parser(
        "relational-trace-graph", help="Predict a native case or volume with a frozen trace graph."
    )
    for option, help_text in (
        ("checkpoint", "Trained relational trace graph checkpoint."),
        ("config", "Frozen inference configuration YAML."),
        ("interim", "Interim trace dataset."),
        ("processed", "Prepared split dataset."),
        ("mask", "Interpolation mask artifact."),
        ("case", "Benchmark case artifact."),
        ("output", "New run output directory."),
    ):
        relational.add_argument(f"--{option}", type=Path, required=True, help=help_text)
    relational.add_argument("--volume", type=Path, help="Optional C3 volume-index selection.")
    relational.add_argument("--device", help="Override prediction.device for this environment.")
    relational.add_argument("--json", action="store_true", help="Print metrics as strict JSON.")
    relational.set_defaults(handler=_interpolate_relational_trace_graph)
