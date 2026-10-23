"""Export actual FORGE source/receiver geometry and the mask of a fixed source."""

import argparse
from pathlib import Path

from seis_interp.pipelines.visualize_forge_geometry import visualize_forge_geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-cell", type=int, default=112)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    print(
        visualize_forge_geometry(
            Path(__file__).resolve().parents[1],
            args.preparation.resolve(),
            args.output_dir.resolve(),
            source_cell=args.source_cell,
            dpi=args.dpi,
        )
    )


if __name__ == "__main__":
    main()
