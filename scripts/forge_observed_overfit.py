"""Run an observed-only fixed-batch FORGE diagnostic."""

import argparse
from pathlib import Path

from seis_interp.pipelines.overfit_forge import run_forge_observed_overfit
from seis_interp.training.forge_overfit import NEURAL_METHODS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--methods", nargs="+", choices=NEURAL_METHODS, default=NEURAL_METHODS)
    args = parser.parse_args()
    run_forge_observed_overfit(
        Path(__file__).resolve().parents[1],
        args.preparation,
        output=args.output,
        steps=args.steps,
        device=args.device,
        methods=args.methods,
    )


if __name__ == "__main__":
    main()
