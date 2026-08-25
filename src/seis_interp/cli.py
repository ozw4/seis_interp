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
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.error import URLError

from seis_interp.configuration import (
    REPOSITORY_ROOT,
    ConfigurationError,
    get_required_config_value,
    load_resolved_config,
    repository_relative_config_source,
)
from seis_interp.data.data_root import DataRootError
from seis_interp.data.seg_c3_na import (
    DATASET_ID,
    DataIntegrityError,
    ManifestError,
    default_manifest_path,
    download_seg_c3_na,
    verify_seg_c3_na,
)
from seis_interp.data.seg_c3_na_inspection import (
    DEFAULT_SAMPLE_TRACE_COUNT,
    SegyInspectionError,
    format_seg_c3_na_inspection,
    inspect_seg_c3_na,
    seg_c3_na_inspection_ok,
    seg_c3_na_inspection_to_dict,
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


def _inspect_data(args: argparse.Namespace) -> int:
    try:
        inspection = inspect_seg_c3_na(
            args.manifest,
            args.data_root,
            sample_trace_count=args.sample_traces,
        )
    except (DataRootError, ManifestError, SegyInspectionError, OSError, ValueError) as exc:
        print(f"Inspection failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(seg_c3_na_inspection_to_dict(inspection), indent=2, sort_keys=True))
    else:
        print(format_seg_c3_na_inspection(inspection))
    return 0 if seg_c3_na_inspection_ok(inspection) else 1


def _prepare_c3_shot(args: argparse.Namespace) -> int:
    # Imported here so that `doctor` keeps working without the data and segy extras.
    from seis_interp.pipelines.prepare_c3 import (
        C3_COMPLETE_SHOT_TRACE_COUNT,
        prepare_c3_complete_shot,
    )

    expected_traces = args.expected_traces
    if expected_traces is None:
        expected_traces = C3_COMPLETE_SHOT_TRACE_COUNT

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
        print(f"data prepare-c3-shot failed: {error}", file=sys.stderr)
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


def _prepare_baseline(args: argparse.Namespace) -> int:
    # Imported here so that unrelated commands keep working without the data extras.
    from seis_interp.pipelines.prepare_baseline import (
        AMPLITUDE_NORMALIZATION_METHOD,
        COORDINATE_NORMALIZATION_METHOD,
        prepare_baseline_dataset,
    )

    try:
        config = load_resolved_config(args.config, repository_root=REPOSITORY_ROOT)
        study_config = config.get("study")
        if isinstance(study_config, Mapping) and "random_seed" in study_config:
            raise ConfigurationError("study.random_seed is not supported; use project.random_seed")

        random_seed = _config_value_or_override(
            args.random_seed,
            config,
            "project.random_seed",
        )
        holdout_fraction = _config_value_or_override(
            args.holdout_fraction,
            config,
            "sampling.random_trace_holdout_fraction",
        )
        validation_fraction = _config_value_or_override(
            args.validation_fraction_of_holdout,
            config,
            "sampling.validation_fraction_of_holdout",
        )
        coordinate_normalization = _required_supported_config_value(
            config,
            "normalization.coordinates",
            COORDINATE_NORMALIZATION_METHOD,
        )
        amplitude_normalization = _required_supported_config_value(
            config,
            "normalization.amplitude",
            AMPLITUDE_NORMALIZATION_METHOD,
        )
        config_source = repository_relative_config_source(
            args.config,
            repository_root=REPOSITORY_ROOT,
        )
    except (OSError, ValueError) as error:
        print(f"data prepare-baseline failed: {error}", file=sys.stderr)
        return 1

    try:
        summary = prepare_baseline_dataset(
            interim_dir=args.input,
            output_dir=args.output,
            holdout_fraction=holdout_fraction,
            validation_fraction_of_holdout=validation_fraction,
            random_seed=random_seed,
            coordinate_normalization=coordinate_normalization,
            amplitude_normalization=amplitude_normalization,
            config_source=config_source,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        print(f"data prepare-baseline failed: {error}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        split_counts = summary["split_counts"]
        print(f"Configuration: {summary['config_source']}")
        print(f"Input dataset: {args.input}")
        print(f"Output directory: {args.output}")
        print(f"Traces: {summary['trace_count']}")
        print(f"Train traces: {split_counts['train']}")
        print(f"Validation traces: {split_counts['validation']}")
        print(f"Test traces: {split_counts['test']}")
    return 0


def _train_siren(args: argparse.Namespace) -> int:
    from seis_interp.pipelines.train_siren import CHECKPOINT_RELATIVE_PATH, train_siren_run

    try:
        summary = train_siren_run(
            config_path=args.config,
            interim_dir=args.interim,
            processed_dir=args.processed,
            output_dir=args.output,
            device_override=args.device,
        )
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as error:
        print(f"train siren failed: {error}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"Output directory: {args.output}")
        print(f"Best epoch: {summary['best_epoch']}")
        print(
            f"Best validation median trace S/N: {summary['best_validation_median_trace_snr_db']} dB"
        )
        print(f"Global validation S/N at best epoch: {summary['best_validation_global_snr_db']} dB")
        print(f"Epochs completed: {summary['epochs_completed']}")
        print(f"Stopped early: {summary['stopped_early']}")
        print(f"Checkpoint: {args.output / CHECKPOINT_RELATIVE_PATH}")
    return 0


def _config_value_or_override(
    override: object | None,
    config: Mapping[str, object],
    dotted_path: str,
) -> object:
    if override is not None:
        return override
    return get_required_config_value(config, dotted_path)


def _required_supported_config_value(
    config: Mapping[str, object],
    dotted_path: str,
    supported_value: str,
) -> str:
    value = get_required_config_value(config, dotted_path)
    if value != supported_value:
        raise ConfigurationError(f"{dotted_path} must be {supported_value!r}, got {value!r}")
    return supported_value


def _add_data_commands(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    data = subparsers.add_parser(
        "data", help="Acquire, verify, inspect, and prepare external datasets."
    )
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

    inspect = data_commands.add_parser("inspect", help="Inspect SEG-Y structure and content.")
    inspect.add_argument("dataset", choices=(DATASET_ID,))
    inspect.add_argument(
        "--manifest",
        type=Path,
        default=default_manifest_path(),
        help="Path to the tracked dataset manifest.",
    )
    inspect.add_argument(
        "--data-root",
        type=Path,
        help="Override SEIS_INTERP_DATA_ROOT for this command.",
    )
    inspect.add_argument(
        "--sample-traces",
        type=int,
        default=DEFAULT_SAMPLE_TRACE_COUNT,
        help="Number of evenly spaced traces sampled per SEG-Y file.",
    )
    inspect.add_argument("--json", action="store_true", help="Print JSON output.")
    inspect.set_defaults(handler=_inspect_data)

    prepare = data_commands.add_parser(
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
        help="Trace count that marks an FFID as complete (default: 544, a complete C3 NA shot).",
    )
    prepare.add_argument(
        "--dataset-id",
        type=str,
        default=DATASET_ID,
        help="Dataset identifier stored in dataset.json.",
    )
    prepare.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the generated files in an existing output directory.",
    )
    prepare.add_argument("--json", action="store_true", help="Print the summary as JSON.")
    prepare.set_defaults(handler=_prepare_c3_shot)

    prepare_baseline = data_commands.add_parser(
        "prepare-baseline",
        help="Create trace splits and normalization metadata from an interim dataset.",
    )
    prepare_baseline.add_argument(
        "--input", type=Path, required=True, help="Input interim trace dataset directory."
    )
    prepare_baseline.add_argument(
        "--output", type=Path, required=True, help="Output processed dataset directory."
    )
    prepare_baseline.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Study configuration YAML to resolve, including its extends chain.",
    )
    prepare_baseline.add_argument(
        "--holdout-fraction",
        type=float,
        help="Override sampling.random_trace_holdout_fraction.",
    )
    prepare_baseline.add_argument(
        "--validation-fraction-of-holdout",
        type=float,
        help="Override sampling.validation_fraction_of_holdout.",
    )
    prepare_baseline.add_argument(
        "--random-seed",
        type=int,
        help="Override project.random_seed for deterministic trace-level splitting.",
    )
    prepare_baseline.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace files generated by this pipeline in an existing output directory.",
    )
    prepare_baseline.add_argument("--json", action="store_true", help="Print the summary as JSON.")
    prepare_baseline.set_defaults(handler=_prepare_baseline)


def _add_train_commands(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    train = subparsers.add_parser("train", help="Train coordinate-based interpolation models.")
    train_commands = train.add_subparsers(dest="train_command", required=True)
    siren = train_commands.add_parser("siren", help="Train a SIREN on prepared trace splits.")
    siren.add_argument("--config", type=Path, required=True, help="Study configuration YAML.")
    siren.add_argument("--interim", type=Path, required=True, help="Interim trace dataset.")
    siren.add_argument("--processed", type=Path, required=True, help="Prepared split dataset.")
    siren.add_argument("--output", type=Path, required=True, help="Run output directory.")
    siren.add_argument("--device", help="Override training.device for this environment.")
    siren.add_argument("--json", action="store_true", help="Print metrics as JSON.")
    siren.set_defaults(handler=_train_siren)


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
    _add_train_commands(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
