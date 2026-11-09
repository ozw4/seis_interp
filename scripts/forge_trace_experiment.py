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
    prepare.add_argument("--observed-phase", type=int, choices=(0, 1))
    launch = commands.add_parser("launch")
    worker = commands.add_parser("worker")
    score = commands.add_parser("score")
    figures = commands.add_parser("figures")
    for command in (launch, worker, score, figures):
        command.add_argument("--experiment", type=Path, required=True)
    figures.add_argument("--output", type=Path, required=True)
    worker.add_argument("--method", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if args.command == "prepare":
        print(
            prepare_forge_trace_experiment(
                repo, args.study.resolve(), observed_phase=args.observed_phase
            ),
            flush=True,
        )
    elif args.command == "launch":
        print(launch_forge_trace_experiment(repo, args.experiment.resolve()), flush=True)
    elif args.command == "worker":
        run_forge_trace_method(repo, args.experiment.resolve(), args.method)
    elif args.command == "score":
        from seis_interp.evaluation.forge_shot_results import finalize_forge_shot_experiment

        print(finalize_forge_shot_experiment(args.experiment.resolve()), flush=True)
    else:
        from seis_interp.visualization.forge_shot_comparison import export_forge_shot_results

        print(
            export_forge_shot_results(args.experiment.resolve(), args.output.resolve()), flush=True
        )
