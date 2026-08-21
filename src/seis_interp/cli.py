"""Command-line entry points for repository diagnostics and data management."""

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
from urllib.error import URLError

from seis_interp.data.data_root import DataRootError
from seis_interp.data.seg_c3_na import (
    DATASET_ID,
    DataIntegrityError,
    ManifestError,
    default_manifest_path,
    download_seg_c3_na,
    verify_seg_c3_na,
)


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


def _download_data(args: argparse.Namespace) -> int:
    try:
        lock_path = download_seg_c3_na(
            args.manifest,
            args.data_root,
            force=args.force,
            resume=not args.no_resume,
            timeout_s=args.timeout_s,
        )
    except (DataRootError, DataIntegrityError, ManifestError, OSError, URLError, ValueError) as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1

    print(f"Download lock written to: {lock_path}")
    return 0


def _verify_data(args: argparse.Namespace) -> int:
    try:
        results = verify_seg_c3_na(args.manifest, args.data_root)
    except (DataRootError, DataIntegrityError, ManifestError, OSError, ValueError) as exc:
        print(f"Verification failed: {exc}", file=sys.stderr)
        return 1

    for result in results:
        print(f"{result.status.upper():>17}  {result.name}: {result.detail}")
    return 0 if all(result.ok for result in results) else 1


def _add_data_commands(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    data = subparsers.add_parser("data", help="Acquire and verify external datasets.")
    data_commands = data.add_subparsers(dest="data_command", required=True)

    download = data_commands.add_parser("download", help="Download an external dataset.")
    download.add_argument("dataset", choices=(DATASET_ID,))
    download.add_argument(
        "--manifest",
        type=Path,
        default=default_manifest_path(),
        help="Path to the tracked dataset manifest.",
    )
    download.add_argument(
        "--data-root",
        type=Path,
        help="Override SEIS_INTERP_DATA_ROOT for this command.",
    )
    download.add_argument(
        "--force",
        action="store_true",
        help="Delete completed and partial files before downloading.",
    )
    download.add_argument(
        "--no-resume",
        action="store_true",
        help="Discard partial files instead of issuing HTTP Range requests.",
    )
    download.add_argument(
        "--timeout-s",
        type=float,
        default=60.0,
        help="Per-request network timeout in seconds.",
    )
    download.set_defaults(handler=_download_data)

    verify = data_commands.add_parser("verify", help="Verify an external dataset.")
    verify.add_argument("dataset", choices=(DATASET_ID,))
    verify.add_argument(
        "--manifest",
        type=Path,
        default=default_manifest_path(),
        help="Path to the tracked dataset manifest.",
    )
    verify.add_argument(
        "--data-root",
        type=Path,
        help="Override SEIS_INTERP_DATA_ROOT for this command.",
    )
    verify.set_defaults(handler=_verify_data)


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

    _add_data_commands(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
