"""Score every fair-grid final FORGE prediction on the mask-B test set in one pass."""

import argparse
from pathlib import Path

from seis_interp.pipelines.evaluate_forge_fair_grid import evaluate_forge_fair_grid

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(evaluate_forge_fair_grid(Path(__file__).resolve().parents[1], args.config.resolve()))
