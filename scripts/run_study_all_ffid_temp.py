"""Run an overwriteable survey-wide scratch experiment.

Training writes to a unique staging directory. A successful run replaces
``runs/study_all_ffid_temp/current``; a failed run leaves the previous result intact.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from uuid import uuid4

from seis_interp.pipelines.train_siren import train_siren_run

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "studies" / "study_all_ffid_temp" / "config.yaml"
INTERIM_DIRECTORY = REPOSITORY_ROOT / "data" / "interim" / "c3_na" / "all_ffids"
PROCESSED_DIRECTORY = (
    REPOSITORY_ROOT
    / "data"
    / "processed"
    / "c3_na"
    / "all_ffids_per_ffid_random_split_amplitude_qc"
)
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "runs" / "study_all_ffid_temp" / "current"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--device", help="Override training.device without editing the config.")
    return parser


def run(*, config_path: Path, device_override: str | None = None) -> dict[str, object]:
    """Train once and install the successful result as the current scratch run."""
    output_parent = OUTPUT_DIRECTORY.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    _validate_current_output(OUTPUT_DIRECTORY)
    staging_directory = output_parent / f".{OUTPUT_DIRECTORY.name}-{uuid4().hex}.staging"

    try:
        metrics = train_siren_run(
            config_path=Path(config_path),
            interim_dir=INTERIM_DIRECTORY,
            processed_dir=PROCESSED_DIRECTORY,
            output_dir=staging_directory,
            device_override=device_override,
        )
        _replace_current_output(staging_directory, OUTPUT_DIRECTORY)
    except Exception:
        _discard_generated_directory(staging_directory)
        raise
    return metrics


def _replace_current_output(staging_directory: Path, output_directory: Path) -> None:
    if not staging_directory.is_dir() or staging_directory.is_symlink():
        raise RuntimeError(f"training did not create a staging directory: {staging_directory}")
    _validate_current_output(output_directory)

    backup_directory: Path | None = None
    if output_directory.exists():
        backup_directory = output_directory.parent / (
            f".{output_directory.name}-{uuid4().hex}.backup"
        )
        output_directory.replace(backup_directory)

    try:
        staging_directory.replace(output_directory)
    except Exception:
        if backup_directory is not None:
            backup_directory.replace(output_directory)
        raise

    if backup_directory is not None:
        shutil.rmtree(backup_directory)


def _validate_current_output(output_directory: Path) -> None:
    if output_directory.is_symlink() or (
        output_directory.exists() and not output_directory.is_dir()
    ):
        raise FileExistsError(
            f"scratch output path must be a directory, not a file or symlink: {output_directory}"
        )


def _discard_generated_directory(directory: Path) -> None:
    if directory.is_symlink() or directory.is_file():
        directory.unlink()
    elif directory.is_dir():
        shutil.rmtree(directory)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        metrics = run(config_path=args.config, device_override=args.device)
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as error:
        print(f"study_all_ffid_temp run failed: {error}", file=sys.stderr)
        return 1

    print(f"Best epoch: {metrics['best_epoch']}")
    if "amplitude_scaling" in metrics:
        print(f"Amplitude scaling: {metrics['amplitude_scaling']}")
    trace_quality = metrics.get("trace_quality")
    if isinstance(trace_quality, dict):
        print(f"Excluded traces: {trace_quality['excluded_trace_count']}")
    if metrics.get("validation_metric_domain") == "oracle_per_trace_unit_rms":
        print("Validation metric domain: oracle per-trace unit RMS")
        validation_label = "Best oracle-normalized validation global S/N"
    else:
        validation_label = "Best validation global S/N"
    print(f"{validation_label}: {metrics['best_validation_global_snr_db']} dB")
    print(f"Optimizer steps: {metrics['global_steps']}")
    print(f"Replaced output: {OUTPUT_DIRECTORY.relative_to(REPOSITORY_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
