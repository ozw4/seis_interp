"""Prepare a sealed C3 fair-grid run in the layout of a FORGE preparation."""

import argparse
from pathlib import Path

from seis_interp.pipelines.prepare_c3_fair_grid import prepare_c3_fair_grid

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True, help="studies/<C3 fair-grid study>")
    parser.add_argument(
        "--validation-holdout",
        action="store_true",
        help="hold out the study's validation traces from the outer observed set",
    )
    args = parser.parse_args()
    print(
        prepare_c3_fair_grid(
            Path(__file__).resolve().parents[1],
            args.study.resolve(),
            validation_holdout=args.validation_holdout,
        )
    )
