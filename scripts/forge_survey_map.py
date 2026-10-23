"""Show a FORGE analysis subset in the full source/receiver acquisition geometry."""

import argparse
from pathlib import Path

from seis_interp.pipelines.visualize_forge_geometry import visualize_forge_survey


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--geometry-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    print(
        visualize_forge_survey(
            Path(__file__).resolve().parents[1],
            args.preparation.resolve(),
            args.geometry_run.resolve(),
            args.output_dir.resolve(),
            dpi=args.dpi,
        )
    )


if __name__ == "__main__":
    main()
