"""Prepare or execute the fixed C3-settings FORGE comparison."""

import argparse
from pathlib import Path

from seis_interp.data.forge_trace_experiment import prepare_forge_trace_experiment
from seis_interp.pipelines.run_forge_trace_experiment import (
    launch_forge_trace_experiment,
    run_forge_trace_method,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--study", type=Path, required=True)
    launch = commands.add_parser("launch")
    worker = commands.add_parser("worker")
    for command in (launch, worker):
        command.add_argument("--experiment", type=Path, required=True)
    worker.add_argument("--method", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if args.command == "prepare":
        print(prepare_forge_trace_experiment(repo, args.study.resolve()), flush=True)
    elif args.command == "launch":
        print(launch_forge_trace_experiment(repo, args.experiment.resolve()), flush=True)
    else:
        run_forge_trace_method(repo, args.experiment.resolve(), args.method)
