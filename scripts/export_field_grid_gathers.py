"""Export the common-grid gathers with the conditional field as the implicit panel."""

import argparse
from pathlib import Path

from seis_interp.pipelines.export_field_grid_gathers import export_field_grid_gathers
from seis_interp.visualization.common_grid_gathers import FIELD_DISPLAY_METHODS

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for dataset in ("c3", "forge"):
        parser.add_argument(f"--{dataset}", type=Path, required=True, help="run of the four runs")
        parser.add_argument(
            f"--{dataset}-field", type=Path, required=True, help="run supplying cinr_real"
        )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7321)
    parser.add_argument("--dpi", type=int, default=1200)
    args = parser.parse_args()
    experiments = {
        dataset: {
            method: (field if method == "cinr_real" else shared).resolve()
            for method in FIELD_DISPLAY_METHODS
        }
        for dataset, shared, field in (
            ("c3", args.c3, args.c3_field),
            ("forge", args.forge, args.forge_field),
        )
    }
    print(export_field_grid_gathers(ROOT, experiments, args.output, seed=args.seed, dpi=args.dpi))


if __name__ == "__main__":
    main()
