"""Prepare, run or inspect the fixed C3 25+3 experiment."""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--study", type=Path, required=True)
    prepare.add_argument("--check-only", action="store_true")
    for name in ("run", "worker", "pilot", "status"):
        command = commands.add_parser(name)
        command.add_argument("--experiment", type=Path, required=True)
        if name == "run":
            command.add_argument("--only-dead-workers", action="store_true")
        if name in ("worker", "pilot"):
            command.add_argument("--method", required=True)
        if name == "worker":
            command.add_argument("--shard", default="0/1", help="disjoint worker i/n (zero based)")
        if name == "pilot":
            command.add_argument("--candidate", required=True, help="S1 cell ID or all")
        if name != "status":
            command.add_argument("--repo", type=Path, default=Path.cwd())
    prepare.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    if args.command in ("worker", "pilot") and args.method in ("pocs_grid", "drr_grid"):
        current_nice = os.getpriority(os.PRIO_PROCESS, 0)
        if current_nice < 10:
            os.nice(10 - current_nice)
    if args.command == "prepare":
        if args.check_only:
            from seis_interp.data.experiment_import_coverage import check_grid_prepare

            check_grid_prepare(args.repo.resolve(), args.study.resolve(), "c3")
            return
        from seis_interp.data.c3_region_inputs import prepare_c3_region_experiment

        print(prepare_c3_region_experiment(args.repo.resolve(), args.study.resolve()))
    elif args.command == "status":
        workers = args.experiment / "workers"
        for path in sorted(workers.glob("*.json")):
            if not path.name.endswith("_selection.json"):
                print(json.dumps(json.loads(path.read_text()), ensure_ascii=False))
    elif args.command == "pilot":
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        from seis_interp.pipelines.run_c3_region_experiment import execute_c3_region_pilot

        execute_c3_region_pilot(
            args.repo.resolve(), args.experiment.resolve(), args.method, args.candidate
        )
    elif args.command == "worker":
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        from seis_interp.pipelines.run_c3_region_experiment import run_c3_region_worker

        run_c3_region_worker(
            args.repo.resolve(), args.experiment.resolve(), args.method, shard=args.shard
        )
    else:
        from seis_interp.pipelines.run_c3_region_experiment import supervise_c3_region_experiment

        supervise_c3_region_experiment(
            args.repo.resolve(),
            args.experiment.resolve(),
            Path(__file__).resolve(),
            **({"only_dead_workers": True} if args.only_dead_workers else {}),
        )


if __name__ == "__main__":
    main()
