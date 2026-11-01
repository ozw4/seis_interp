"""Launch a prepared sensitivity on GPU0 through the existing frozen pilot launcher."""

import argparse
import re
import subprocess
from pathlib import Path

from launch_study067_pilots import PYTHON, REPO, start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=("sa_forge_mask010", "sb_forge_fp32", "sa_c3_mask010_seeds"))
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--production", type=Path)
    parser.add_argument("--log", default="sensitivity.log")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    if Path(args.log).name != args.log:
        parser.error("--log must be a filename within the sensitivity experiment")
    c3 = args.arm == "sa_c3_mask010_seeds"
    if c3 and args.production is None:
        parser.error("C3 requires --production for the paired input and barrier checks")
    if c3:
        script = experiment / "study/run_seeds.py"
        arguments = ["--production", str(args.production.resolve())]
    else:
        script = experiment / "source/scripts/run_forge_grid_experiment.py"
        arguments = ["pilot", "--method", "ccnet5d_grid", "--candidate", "a03_b02"]
    if not script.is_file() or not (experiment / "source/src/seis_interp").is_dir():
        parser.error("experiment has no frozen sensitivity runner/source")
    command = [
        "nice",
        "-n",
        "19",
        str(PYTHON),
        str(script),
        *arguments,
        "--repo",
        str(REPO),
        "--experiment",
        str(experiment),
    ]
    # Resolve physical index 0 before CUDA starts. A UUID also prevents CUDA's
    # enumeration order from redirecting a numeric selector to physical GPU1.
    gpu = "0"
    if not args.dry_run:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--id=0", "--query-gpu=uuid", "--format=csv,noheader"], text=True
        ).strip()
        if not re.fullmatch(r"GPU-[0-9a-fA-F-]+", gpu):
            raise ValueError("physical GPU0 did not resolve to a complete GPU UUID")
    start(command, experiment, args.log, gpu=gpu, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
