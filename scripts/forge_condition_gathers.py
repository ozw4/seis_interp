"""Render fixed source/receiver gathers for one saved ReGSI-real condition."""

import argparse
from pathlib import Path

from seis_interp.pipelines.visualize_forge_condition_gathers import (
    visualize_forge_condition_gathers,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    visualize_forge_condition_gathers(
        Path(__file__).resolve().parents[1], args.run.resolve(), device=args.device
    )


if __name__ == "__main__":
    main()
