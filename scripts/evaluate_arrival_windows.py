"""Re-score one run's saved predictions on either side of the declared arrival window."""

import argparse
from pathlib import Path

from seis_interp.pipelines.evaluate_arrival_windows import evaluate_arrival_windows

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", action="append", help="defaults to the run's methods")
    parser.add_argument("--seed", type=int, help="evaluation seed for grid-layout runs")
    args = parser.parse_args()
    print(
        evaluate_arrival_windows(
            ROOT, args.experiment, args.output, methods=args.method, seed=args.seed
        )
    )


if __name__ == "__main__":
    main()
