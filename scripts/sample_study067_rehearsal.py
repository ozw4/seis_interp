"""Sample structural GPU/process evidence until both rehearsal supervisors complete."""

import argparse
from pathlib import Path

from seis_interp.training.rehearsal_observations import sample_until_complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--c3", type=Path, required=True)
    parser.add_argument("--forge", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=10)
    args = parser.parse_args()
    if not 0 < args.interval < float("inf"):
        parser.error("interval must be positive and finite")
    sample_until_complete(
        args.repo, [args.c3.resolve(), args.forge.resolve()], args.out, args.interval
    )


if __name__ == "__main__":
    main()
