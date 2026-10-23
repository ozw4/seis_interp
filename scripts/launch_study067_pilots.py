"""Launch the prepared C-1, N-1, D-1 or D-2 pilots; --dry-run never starts a job."""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PYTHON = REPO / ".venv/bin/python"
RUN_PREFIX = "runs/study_067_forge_c3_common_grid/"
PREPARED = {
    "c1_c3_tk3": RUN_PREFIX + "20261022T202519313159Z_05c92e79ce80_traces",
    "c1_c3_tk9": RUN_PREFIX + "20261022T202537703605Z_05c92e79ce80_traces",
    "c1_forge_tk3": RUN_PREFIX + "20261022T202533191991Z_05c92e79ce80_forge",
    "c1_forge_tk9": RUN_PREFIX + "20261022T202604622878Z_05c92e79ce80_forge",
    "n1_jitter_canvas": RUN_PREFIX + "20261022T183518698062Z_05c92e79ce80_forge",
    "n1_jitter_support": RUN_PREFIX + "20261022T183524117130Z_05c92e79ce80_forge",
    "n1_no_jitter_canvas": RUN_PREFIX + "20261022T183529511220Z_05c92e79ce80_forge",
    "n1_no_jitter_support": RUN_PREFIX + "20261022T183534795056Z_05c92e79ce80_forge",
    "d1_forge_d7": RUN_PREFIX + "20261022T183540163875Z_05c92e79ce80_forge",
    "d1_forge_d7_rerun": RUN_PREFIX + "20261023T003937042544Z_05c92e79ce80_forge",
    "d1_c3_d7": RUN_PREFIX + "20261022T183550980564Z_05c92e79ce80_traces",
    "d1_forge_d5b": RUN_PREFIX + "20261022T183556619725Z_05c92e79ce80_forge",
    "d2_forge": RUN_PREFIX + "20261022T183601578711Z_05c92e79ce80_forge",
    "d2_c3": RUN_PREFIX + "20261022T183612495378Z_05c92e79ce80_traces",
}
D1_PILOTS = ("d1_forge_d7_rerun", "d1_c3_d7", "d1_forge_d5b")


def pending_d1_pilots():
    names = []
    for name in D1_PILOTS:
        manifest = REPO / PREPARED[name] / "method_runs/drr_grid/S1/a02_b02/manifest.json"
        if manifest.exists() and json.loads(manifest.read_text())["status"] == "completed":
            continue
        names.append(name)
    return names


def environment(experiment, gpu=""):
    return os.environ | {
        "CUDA_VISIBLE_DEVICES": gpu,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONPATH": str(experiment / "source/src"),
        "PYTHONUNBUFFERED": "1",
    }


def pilot_command(name):
    experiment = REPO / PREPARED[name]
    runner = "run_c3_region_experiment.py" if "_c3_" in name else "run_forge_grid_experiment.py"
    if name.startswith("c1_"):
        method = "ccnet5d_grid"
        candidate = "a05_b02" if "_c3_" in name else "a03_b02"
    else:
        method, candidate = (
            ("nersi_profile", "a02_b04") if name.startswith("n1_") else ("drr_grid", "a02_b02")
        )
    command = [
        str(PYTHON),
        str(experiment / "source/scripts" / runner),
        "pilot",
        "--repo",
        str(REPO),
        "--experiment",
        str(experiment),
        "--method",
        method,
        "--candidate",
        candidate,
    ]
    return command if name.startswith(("c1_", "n1_")) else ["nice", "-n", "10", *command]


def start(command, experiment, log_name, *, gpu="", dry_run=False):
    info = {"command": command, "experiment": str(experiment), "log": str(experiment / log_name)}
    if dry_run:
        env = environment(experiment, gpu)
        declared = {
            key: env[key]
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "PYTORCH_CUDA_ALLOC_CONF",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "PYTHONPATH",
                "PYTHONUNBUFFERED",
            )
        }
        print(json.dumps(info | {"environment": declared, "dry_run": True}))
        return None
    with (experiment / log_name).open("x") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO,
            env=environment(experiment, gpu),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print(json.dumps(info | {"pid": process.pid, "gpu_uuid": gpu}), flush=True)
    return process


def pinned_process_alive(item):
    try:
        fields = Path(f"/proc/{item['pid']}/stat").read_text().rsplit(") ", 1)[1].split()
        return fields[0] != "Z" and int(fields[19]) == item["process_start_time"]
    except FileNotFoundError:
        return False


def check_n1_gpu(gpu):
    import yaml

    if not re.fullmatch(r"GPU-[0-9a-fA-F-]+", gpu):
        raise ValueError("supply the full physical GPU UUID (GPU-...)")
    selected = (
        subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu}",
                "--query-gpu=uuid,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        .strip()
        .split(",")
    )
    if selected[0].strip() != gpu:
        raise ValueError("supply the complete UUID, not an index or UUID prefix")
    required_gib = 0
    pinned = []
    for name, path in PREPARED.items():
        if not name.startswith("n1_"):
            continue
        contract = yaml.safe_load((REPO / path / "study/config.yaml").read_text())
        resources = contract["cuda_execution"]
        required_gib += (
            resources["minimum_free_memory_gib"]["nersi_profile"]
            + resources["admission_margin_gib"]
        )
        pinned.extend(
            item
            for items in resources["excluded_devices_while_pids_alive"].values()
            for item in items
            if pinned_process_alive(item)
        )
    if float(selected[1]) < required_gib * 1024:
        raise ValueError(f"N-1 batch requires {required_gib:g} GiB free on {gpu}")
    if pinned:
        applications = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
            text=True,
        )
        busy = {
            int(fields[1])
            for line in applications.splitlines()
            if len(fields := [part.strip() for part in line.split(",")]) == 2 and fields[0] == gpu
        }
        if any(item["pid"] in busy for item in pinned):
            raise ValueError("the selected physical GPU has a live pinned legacy worker")


def check_c1_gpus(gpu_c3, gpu_forge):
    import yaml

    for gpu in dict.fromkeys((gpu_c3, gpu_forge)):
        if not re.fullmatch(r"GPU-[0-9a-fA-F-]+", gpu):
            raise ValueError("supply the full physical GPU UUID (GPU-...)")
        selected = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--id={gpu}",
                    "--query-gpu=uuid,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
            .strip()
            .split(",")
        )
        if selected[0].strip() != gpu:
            raise ValueError("supply the complete UUID, not an index or UUID prefix")
        required_gib = 0
        pinned = []
        for name, path in PREPARED.items():
            if not name.startswith("c1_"):
                continue
            assigned = gpu_c3 if "_c3_" in name else gpu_forge
            if assigned != gpu:
                continue
            contract = yaml.safe_load((REPO / path / "study/config.yaml").read_text())
            resources = contract["cuda_execution"]
            candidate = "a05_b02" if "_c3_" in name else "a03_b02"
            required_gib += (
                resources["minimum_free_memory_gib"]["ccnet5d_grid"][candidate]
                + resources["admission_margin_gib"]
            )
            pinned.extend(
                item
                for items in resources["excluded_devices_while_pids_alive"].values()
                for item in items
                if pinned_process_alive(item)
            )
        if float(selected[1]) < required_gib * 1024:
            raise ValueError(f"C-1 batch requires {required_gib:g} GiB free on {gpu}")
        if pinned:
            applications = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=gpu_uuid,pid",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
            busy = {
                int(fields[1])
                for line in applications.splitlines()
                if len(fields := [part.strip() for part in line.split(",")]) == 2
                and fields[0] == gpu
            }
            if any(item["pid"] in busy for item in pinned):
                raise ValueError("the selected physical GPU has a live pinned legacy worker")


def d2_commands(cap):
    for dataset in ("forge", "c3"):
        experiment = REPO / PREPARED[f"d2_{dataset}"]
        for candidate in ("a05_b05", "a05_b04"):
            yield (
                experiment,
                candidate,
                [
                    "nice",
                    "-n",
                    "10",
                    str(PYTHON),
                    str(experiment / "study/measure_drr.py"),
                    "--experiment",
                    str(experiment),
                    "--dataset",
                    dataset,
                    "--candidate",
                    candidate,
                    "--address-space-cap-gib",
                    str(cap),
                    "--output",
                    str(experiment / "d2_measurements" / candidate),
                ],
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot", choices=("c1", "n1", "d1", "d2", "check-d1"))
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--gpu-uuid-c3")
    parser.add_argument("--gpu-uuid-forge")
    parser.add_argument("--address-space-cap-gib", type=float, default=24)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sequential-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.pilot == "n1" and not args.gpu_uuid:
        parser.error("N-1 requires --gpu-uuid GPU-...")
    if args.pilot == "c1" and not (args.gpu_uuid_c3 and args.gpu_uuid_forge):
        parser.error("C-1 requires --gpu-uuid-c3 GPU-... and --gpu-uuid-forge GPU-...")
    if not 0 < args.address_space_cap_gib < float("inf"):
        parser.error("the memory cap must be positive and finite")
    if args.pilot == "check-d1":
        from check_study067_d1 import check

        results = [check(REPO / PREPARED[name]) for name in D1_PILOTS]
        print(json.dumps(results, indent=2))
        raise SystemExit(0 if all(item["passed"] for item in results) else 1)
    if args.pilot in ("c1", "n1", "d1"):
        names = (
            pending_d1_pilots()
            if args.pilot == "d1"
            else [name for name in PREPARED if name.startswith(args.pilot + "_")]
        )
        # Check all destinations before starting the first detached process.
        for name in names:
            experiment = REPO / PREPARED[name]
            if (experiment / f"{args.pilot}.log").exists():
                raise FileExistsError(f"pilot already launched: {experiment}")
        if args.pilot == "n1" and not args.dry_run:
            check_n1_gpu(args.gpu_uuid)
        if args.pilot == "c1" and not args.dry_run:
            check_c1_gpus(args.gpu_uuid_c3, args.gpu_uuid_forge)
        for name in names:
            gpu = (
                (args.gpu_uuid_c3 if "_c3_" in name else args.gpu_uuid_forge)
                if args.pilot == "c1"
                else args.gpu_uuid or ""
            )
            start(
                pilot_command(name),
                REPO / PREPARED[name],
                f"{args.pilot}.log",
                gpu=gpu,
                dry_run=args.dry_run,
            )
        return
    if args.dry_run:
        for experiment, candidate, command in d2_commands(args.address_space_cap_gib):
            start(command, experiment, f"d2_{candidate}.log", dry_run=True)
        return
    if args.sequential_child:
        failed = False
        for experiment, candidate, command in d2_commands(args.address_space_cap_gib):
            process = start(command, experiment, f"d2_{candidate}.log")
            failed |= process.wait() != 0
        raise SystemExit(1 if failed else 0)
    available = (
        int(
            next(
                line.split()[1]
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:")
            )
        )
        * 1024
    )
    if args.address_space_cap_gib * 2**30 > available * 0.8:
        parser.error("cap exceeds 80% of current available RAM; choose a smaller cap")
    command = [
        str(PYTHON),
        str(Path(__file__).resolve()),
        "d2",
        "--sequential-child",
        "--address-space-cap-gib",
        str(args.address_space_cap_gib),
    ]
    start(command, REPO / PREPARED["d2_forge"], "d2_batch.log")


if __name__ == "__main__":
    main()
