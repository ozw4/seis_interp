"""Export study_067 central gathers from the selected cells' saved seed predictions."""

import argparse
from pathlib import Path

from seis_interp.pipelines.export_common_grid_gathers import export_common_grid_gathers

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c3", type=Path, required=True)
    parser.add_argument("--forge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7321)
    parser.add_argument("--dpi", type=int, default=1200)
    args = parser.parse_args()
    experiments = {"c3": args.c3.resolve(), "forge": args.forge.resolve()}
    print(export_common_grid_gathers(ROOT, experiments, args.output, seed=args.seed, dpi=args.dpi))


if __name__ == "__main__":
    main()
