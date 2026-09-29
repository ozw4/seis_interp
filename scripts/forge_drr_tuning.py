"""Run a recorded DRR parameter search against a completed FORGE experiment."""

import argparse
from pathlib import Path

from seis_interp.pipelines.tune_forge_classical import tune_forge_classical

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(
        tune_forge_classical(
            Path(__file__).resolve().parents[1],
            args.run.resolve(),
            args.config.resolve(),
            method_id="drr_grid",
        )
    )
