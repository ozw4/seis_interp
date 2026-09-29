"""Render per-trace RMS normalized sections from an evaluated FORGE MVP run."""

import argparse
from pathlib import Path

from seis_interp.pipelines.visualize_forge_mvp import visualize_forge_mvp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    print(visualize_forge_mvp(Path(__file__).resolve().parents[1], args.run.resolve()))


if __name__ == "__main__":
    main()
