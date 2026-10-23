"""Re-seal a sealed FORGE preparation for a study that adds methods on identical inputs."""

import argparse
from pathlib import Path

from seis_interp.pipelines.derive_forge_preparation import derive_forge_preparation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True, help="sealed parent preparation")
    parser.add_argument("--study", type=Path, required=True, help="studies/<study adding methods>")
    parser.add_argument("--method", action="append", required=True, help="added method id")
    args = parser.parse_args()
    print(
        derive_forge_preparation(
            Path(__file__).resolve().parents[1],
            args.parent.resolve(),
            args.study.resolve(),
            args.method,
        )
    )
