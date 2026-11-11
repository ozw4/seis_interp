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
    parser.add_argument(
        "--window-shift-samples",
        type=int,
        default=0,
        help="move every window end by this many samples; negative pulls it earlier",
    )
    args = parser.parse_args()
    print(
        evaluate_arrival_windows(
            ROOT,
            args.experiment,
            args.output,
            methods=args.method,
            seed=args.seed,
            shift_samples=args.window_shift_samples,
        )
    )


if __name__ == "__main__":
    main()
