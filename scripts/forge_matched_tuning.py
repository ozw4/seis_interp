"""Run a recorded matched-budget method search on a sealed FORGE preparation."""

import argparse
from pathlib import Path

from seis_interp.pipelines.tune_forge_matched import tune_forge_matched

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", help="execute on this device instead of the yaml's device")
    args = parser.parse_args()
    print(
        tune_forge_matched(
            Path(__file__).resolve().parents[1],
            args.preparation.resolve(),
            args.config.resolve(),
            device=args.device,
        )
    )
