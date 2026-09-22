"""Prepare data or run one ReGSI-real condition-search experiment."""

import argparse
from pathlib import Path

from seis_interp.pipelines.search_forge_conditions import (
    prepare_forge_conditions,
    run_forge_condition,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--preparation", type=Path, required=True)
    prepare.add_argument(
        "--factors", type=int, nargs="+", choices=(1, 2, 8, 16), default=(1, 8, 16)
    )
    run = commands.add_parser("run")
    run.add_argument("--cache", type=Path, required=True)
    run.add_argument("--factor", type=int, choices=(1, 2, 8, 16), default=8)
    run.add_argument("--missing-fraction", type=float, default=0.1)
    run.add_argument(
        "--normalization", choices=("trace_oracle", "observed_global"), default="trace_oracle"
    )
    run.add_argument("--steps", type=int, default=5000)
    run.add_argument("--batch-size", type=int, default=32)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--inner-fraction", type=float, default=0.1)
    run.add_argument("--learning-rate", type=float, default=0.001)
    run.add_argument("--time-shift", type=int, default=0)
    run.add_argument("--all-test", action="store_true")
    args = vars(parser.parse_args())
    action = args.pop("action")
    repo = Path(__file__).resolve().parents[1]
    if action == "prepare":
        prepare_forge_conditions(repo, args["preparation"].resolve(), factors=args["factors"])
    else:
        args["cache"] = args["cache"].resolve()
        run_forge_condition(repo, **args)


if __name__ == "__main__":
    main()
