"""Launch a detached supervisor and workers entirely from a prepared source snapshot."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def launch(repo, experiment, dataset, *, only_dead_workers=False, dry_run=False):
    repo, experiment = repo.resolve(), experiment.resolve()
    name = "c3_region" if dataset == "c3" else "forge_grid"
    script = experiment / "source/scripts" / f"run_{name}_experiment.py"
    if not script.is_file() or not (experiment / "source/src/seis_interp").is_dir():
        raise ValueError("prepared experiment has no frozen runner/source")
    command = [
        "nohup",
        sys.executable,
        str(script),
        "run",
        "--repo",
        str(repo),
        "--experiment",
        str(experiment),
    ]
    if only_dead_workers:
        command.append("--only-dead-workers")
    environment = os.environ | {
        "PYTHONPATH": str(experiment / "source/src"),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    info = {
        "command": command,
        "pythonpath": environment["PYTHONPATH"],
        "log": str(experiment / "supervisor.out"),
        "start_new_session": True,
    }
    if not dry_run:
        with (experiment / "supervisor.out").open("a") as stream:
            process = subprocess.Popen(
                command,
                cwd=repo,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        info["pid"] = process.pid
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("c3", "forge"))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--only-dead-workers", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            launch(
                args.repo,
                args.experiment,
                args.dataset,
                only_dead_workers=args.only_dead_workers,
                dry_run=args.dry_run,
            )
        )
    )


if __name__ == "__main__":
    main()
