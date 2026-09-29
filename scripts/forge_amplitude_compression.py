"""Start one observed-only linear/asinh FORGE amplitude comparison."""

import argparse
from pathlib import Path

from seis_interp.pipelines.run_forge_amplitude_compression import run_forge_amplitude_compression


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("linear", "asinh"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--scale-multiplier", type=float, default=1.0)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    run_forge_amplitude_compression(
        repo,
        repo / "studies/study_055_forge_amplitude_compression",
        args.variant,
        device=args.device,
        steps=args.steps,
        scale_multiplier=args.scale_multiplier,
    )


if __name__ == "__main__":
    main()
