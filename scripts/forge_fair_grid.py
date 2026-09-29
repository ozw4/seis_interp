"""Plan, track and close the stages of the pre-registered study_060 fair grid search."""

import argparse
from pathlib import Path

from seis_interp.pipelines.fair_grid_stage import (
    fair_grid_status,
    load_fair_grid_context,
    load_grid,
    plan_fair_grid_stage,
    select_fair_grid,
)

REPO = Path(__file__).resolve().parents[1]
GRID = REPO / "studies/study_060_forge_fair_grid/grid.yaml"


def _context(args):
    grid, _ = load_grid(args.grid)
    return load_fair_grid_context(
        REPO, grid, args.validation_preparation.resolve(), args.final_preparation.resolve()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, default=GRID)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="write one stage's search yamls")
    select = commands.add_parser("select", help="write the selection and final-run yamls")
    for command in (plan, select):
        command.add_argument("--method", required=True)
        command.add_argument("--validation-preparation", type=Path, required=True)
        command.add_argument("--final-preparation", type=Path, required=True)
    plan.add_argument("--stage", required=True)
    for command in (plan, select):
        command.add_argument(
            "--devices",
            help="comma-separated execution devices assigned round-robin (e.g. cuda:0,cuda:1)",
        )
    commands.add_parser("status", help="print every planned run, its status and scores")
    args = parser.parse_args()
    grid_path = args.grid.resolve()
    devices = tuple(args.devices.split(",")) if getattr(args, "devices", None) else None
    if args.command == "plan":
        print(
            plan_fair_grid_stage(
                REPO, grid_path, args.method, args.stage, _context(args), devices=devices
            )
        )
    elif args.command == "select":
        print(select_fair_grid(REPO, grid_path, args.method, _context(args), devices=devices))
    else:
        print(fair_grid_status(REPO, grid_path).to_string(index=False))
