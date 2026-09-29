"""Run, check or summarize a scratch FORGE NeRSI experiment from a YAML config.

Each run trains NeRSI on the sealed preparation named in ``inputs.yaml`` and
writes an immutable ``runs/study_forge_nersi_temp/<UTC>_<sha12>_<name>/``
directory; ``summary.csv`` beside it is the overwriteable cross-run table.
"""

import os

# Deterministic cuBLAS workspaces must be configured before torch initializes CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

from seis_interp.pipelines.run_forge_nersi_experiment import (  # noqa: E402
    check_forge_nersi_experiment,
    format_check_report,
    format_experiment_summary,
    run_forge_nersi_experiment,
    write_experiment_summary,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "studies" / "study_forge_nersi_temp" / "config.yaml"
DEFAULT_INPUTS_PATH = REPOSITORY_ROOT / "studies" / "study_forge_nersi_temp" / "inputs.yaml"
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "runs" / "study_forge_nersi_temp"


def _print_progress(message: str) -> None:
    print(message, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS_PATH)
    parser.add_argument("--device", help="Override the config device (cpu or cuda:<index>).")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="Validate and report without training, reading test data or creating files.",
    )
    mode.add_argument(
        "--summarize",
        action="store_true",
        help="Rebuild and print summary.csv from existing run records without training.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.summarize:
            summary = write_experiment_summary(OUTPUT_DIRECTORY)
            print(format_experiment_summary(summary))
        elif args.check:
            report = check_forge_nersi_experiment(
                REPOSITORY_ROOT, args.config, args.inputs, device_override=args.device
            )
            print(format_check_report(report))
        else:
            run_forge_nersi_experiment(
                REPOSITORY_ROOT,
                args.config,
                args.inputs,
                OUTPUT_DIRECTORY,
                device_override=args.device,
                reporter=_print_progress,
            )
    except (
        FileNotFoundError,
        FileExistsError,
        OSError,
        RuntimeError,
        ValueError,
        KeyError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
