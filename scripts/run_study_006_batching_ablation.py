"""Run Study 006 exact-coverage versus random-replacement batching ablation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from seis_interp.pipelines.batching_ablation import run_batching_ablation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--interim", type=Path, required=True)
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_batching_ablation(
            config_path=args.config,
            interim_dir=args.interim,
            processed_dir=args.processed,
            output_root=args.output_root,
            device_override=args.device,
        )
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as error:
        print(f"Study 006 batching ablation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
