"""Command-line entry points for repository diagnostics."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _command_version(command: str) -> dict[str, str | bool | None]:
    executable = shutil.which(command)
    if executable is None:
        return {"available": False, "path": None, "version": None}

    completed = subprocess.run(
        [executable, "--version"],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    output = (completed.stdout or completed.stderr).strip()
    return {
        "available": completed.returncode == 0,
        "path": executable,
        "version": output or None,
    }


def _torch_environment() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {
            "available": False,
            "version": None,
            "cuda_available": False,
            "cuda_version": None,
            "device_count": 0,
            "devices": [],
        }

    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    devices = [torch.cuda.get_device_name(index) for index in range(device_count)]
    return {
        "available": True,
        "version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "device_count": device_count,
        "devices": devices,
    }


def collect_environment() -> dict[str, Any]:
    data_root = Path(os.environ.get("SEIS_INTERP_DATA_ROOT", "/home/dcuser/data"))
    return {
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "packages": {
            name: _distribution_version(name)
            for name in ("numpy", "PyYAML", "segyio", "pandas", "pyarrow", "matplotlib")
        },
        "torch": _torch_environment(),
        "commands": {
            "codex": _command_version("codex"),
            "claude": _command_version("claude"),
            "gh": _command_version("gh"),
        },
        "data_root": {
            "path": str(data_root),
            "exists": data_root.exists(),
            "readable": os.access(data_root, os.R_OK) if data_root.exists() else False,
        },
    }


def _print_human_readable(report: dict[str, Any]) -> None:
    python = report["python"]
    print(f"Python: {python['version']} ({python['executable']})")

    torch = report["torch"]
    if torch["available"]:
        print(
            "PyTorch: "
            f"{torch['version']} | CUDA available={torch['cuda_available']} "
            f"| devices={torch['device_count']}"
        )
        for index, device in enumerate(torch["devices"]):
            print(f"  GPU {index}: {device}")
    else:
        print("PyTorch: not installed")

    print("Packages:")
    for name, version in report["packages"].items():
        print(f"  {name}: {version or 'not installed'}")

    print("Commands:")
    for name, metadata in report["commands"].items():
        status = metadata["version"] if metadata["available"] else "not available"
        print(f"  {name}: {status}")

    data_root = report["data_root"]
    print(
        "Data root: "
        f"{data_root['path']} | exists={data_root['exists']} | readable={data_root['readable']}"
    )


def _doctor(args: argparse.Namespace) -> int:
    report = collect_environment()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_human_readable(report)

    if not args.strict:
        return 0

    commands_ready = all(report["commands"][name]["available"] for name in ("codex", "claude"))
    return 0 if commands_ready and report["data_root"]["readable"] else 1


def _prepare_c3_shot(args: argparse.Namespace) -> int:
    # Imported here so that `doctor` keeps working without the data and segy extras.
    from seis_interp.pipelines.prepare_c3 import prepare_c3_complete_shot
    from seis_interp.processing.ffid_selection import DEFAULT_EXPECTED_TRACE_COUNT

    expected_traces = args.expected_traces
    if expected_traces is None:
        expected_traces = DEFAULT_EXPECTED_TRACE_COUNT

    try:
        summary = prepare_c3_complete_shot(
            input_path=args.input,
            output_dir=args.output,
            ffid=args.ffid,
            expected_trace_count=expected_traces,
            dataset_id=args.dataset_id,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        print(f"prepare-c3-shot failed: {error}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"Source file: {args.input}")
        print(f"Selected FFID: {summary['selection']['ffid']}")
        print(f"Traces: {summary['trace_count']}")
        print(f"Samples per trace: {summary['sample_count']}")
        print(f"Sample interval: {summary['sample_interval_s']} s")
        print(f"Output directory: {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="seis-interp")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Inspect the development environment.")
    doctor.add_argument("--json", action="store_true", help="Print JSON output.")
    doctor.add_argument(
        "--strict",
        action="store_true",
        help="Fail when AI CLIs or the configured data root are unavailable.",
    )
    doctor.set_defaults(handler=_doctor)

    prepare = subparsers.add_parser(
        "prepare-c3-shot",
        help="Write one complete SEG C3 NA shot as an interim trace dataset.",
    )
    prepare.add_argument("--input", type=Path, required=True, help="Input SEG-Y file.")
    prepare.add_argument("--output", type=Path, required=True, help="Output directory.")
    prepare.add_argument(
        "--ffid",
        type=int,
        default=None,
        help="FFID to select. Defaults to the smallest complete FFID.",
    )
    prepare.add_argument(
        "--expected-traces",
        type=int,
        default=None,
        help="Trace count that marks an FFID as complete (default: 544).",
    )
    prepare.add_argument(
        "--dataset-id",
        type=str,
        default="seg_c3_na",
        help="Dataset identifier stored in dataset.json.",
    )
    prepare.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the generated files in an existing output directory.",
    )
    prepare.add_argument("--json", action="store_true", help="Print the summary as JSON.")
    prepare.set_defaults(handler=_prepare_c3_shot)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
