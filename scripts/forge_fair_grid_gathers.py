"""Export publication gather comparisons from an evaluated FORGE fair grid."""

import argparse
from pathlib import Path

from seis_interp.pipelines.visualize_forge_fair_grid import visualize_forge_fair_grid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7311)
    parser.add_argument("--dpi", type=int, default=1200)
    args = parser.parse_args()
    print(
        visualize_forge_fair_grid(
            Path(__file__).resolve().parents[1],
            args.evaluation.resolve(),
            args.output_dir.resolve(),
            seed=args.seed,
            dpi=args.dpi,
        )
    )


if __name__ == "__main__":
    main()
