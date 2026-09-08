"""Thin CLI entry points for fixed C3 QC, preparation, and suite verification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def add_c3_benchmark_commands(subparsers: argparse._SubParsersAction) -> None:
    for name, handler in (("qc-c3-geometry", _geometry), ("qc-c3-crop", _crop)):
        command = subparsers.add_parser(name, help="Inspect the fixed C3 benchmark without a mask.")
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--input", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--json", action="store_true")
        if name == "qc-c3-crop":
            command.add_argument("--plots", action="store_true")
        command.set_defaults(handler=handler)
    command = subparsers.add_parser(
        "prepare-c3-benchmark",
        help="Plan C3 benchmark preparation; use --execute to write artifacts.",
    )
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--inputs", type=Path, required=True)
    command.add_argument("--output", type=Path)
    command.add_argument("--execute", action="store_true")
    command.add_argument("--case-id", action="append", dest="case_ids")
    command.add_argument("--plots", action="store_true")
    command.add_argument("--json", action="store_true")
    command.set_defaults(handler=_prepare)
    command = subparsers.add_parser(
        "verify-c3-benchmark",
        help="Recompute hashes and verify every fixed C3 suite artifact read-only.",
    )
    command.add_argument("--input", type=Path, required=True)
    command.add_argument("--json", action="store_true")
    command.set_defaults(handler=_verify)


def _emit(args: argparse.Namespace, result: dict) -> int:
    print(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) if args.json else str(result)
    )
    return (
        0 if result.get("status") not in ("partial", "failed") and result.get("ready", True) else 1
    )


def _geometry(args: argparse.Namespace) -> int:
    from seis_interp.configuration import load_resolved_config
    from seis_interp.pipelines.qc_c3 import qc_c3_geometry
    from seis_interp.processing.c3_benchmark_contract import validate_c3_benchmark_contract

    try:
        config = load_resolved_config(args.config)
        validate_c3_benchmark_contract(config)
        return _emit(args, qc_c3_geometry(args.input, args.output))
    except (OSError, ValueError) as error:
        print(f"data qc-c3-geometry failed: {error}", file=sys.stderr)
        return 1


def _crop(args: argparse.Namespace) -> int:
    from seis_interp.configuration import load_resolved_config
    from seis_interp.data.c3_benchmark_artifacts import load_c3_geometry_inputs
    from seis_interp.pipelines.qc_c3 import qc_c3_crop
    from seis_interp.processing.c3_benchmark_contract import validate_c3_benchmark_contract
    from seis_interp.processing.c3_geometry_qc import summarize_c3_geometry

    try:
        config = load_resolved_config(args.config)
        validate_c3_benchmark_contract(config)
        table, times = load_c3_geometry_inputs(args.input)
        _, geometry = summarize_c3_geometry(
            table, times, requested_inclusive=(25, 40), time_range=(0, 384)
        )
        return _emit(
            args,
            qc_c3_crop(
                args.input,
                args.output,
                source_line_range=tuple(geometry["sail_lines"]["index_range"]),
                time_range=(0, 384),
                spatial_lengths=(32, 8, 32),
                explicit_ranges=config["benchmark_volume"]["selection"],
                plots=args.plots,
            ),
        )
    except (OSError, ValueError) as error:
        print(f"data qc-c3-crop failed: {error}", file=sys.stderr)
        return 1


def _prepare(args: argparse.Namespace) -> int:
    from seis_interp.pipelines.prepare_c3_benchmark import prepare_c3_benchmark

    try:
        result = prepare_c3_benchmark(
            args.config,
            args.inputs,
            output_dir=args.output,
            execute=args.execute,
            case_ids=args.case_ids,
            plots=args.plots,
            progress_reporter=lambda message: print(message, file=sys.stderr),
        )
        return _emit(args, result)
    except (OSError, ValueError, AssertionError) as error:
        print(f"data prepare-c3-benchmark failed: {error}", file=sys.stderr)
        return 1


def _verify(args: argparse.Namespace) -> int:
    from seis_interp.data.c3_benchmark_suite import verify_c3_benchmark_suite
    from seis_interp.data.file_checksums import file_sha256

    try:
        suite = verify_c3_benchmark_suite(args.input)
        return _emit(
            args,
            {
                "status": "verified",
                "case_count": len(suite["cases"]),
                "manifest": str(args.input / "benchmark_suite.json"),
                "sha256": file_sha256(args.input / "benchmark_suite.json"),
            },
        )
    except (OSError, ValueError, AssertionError, KeyError, TypeError) as error:
        print(f"data verify-c3-benchmark failed: {error}", file=sys.stderr)
        return 1
