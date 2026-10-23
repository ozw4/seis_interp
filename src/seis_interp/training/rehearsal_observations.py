"""Read-only operator sampling of rehearsal GPU and process evidence."""

import json
import os
import subprocess
import time
from pathlib import Path

from seis_interp.data.forge_headers import sha256_file
from seis_interp.data.forge_mvp_run_records import utc_now
from seis_interp.training.worker_processes import process_is_alive


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _process_sample(pid):
    try:
        root = Path(f"/proc/{pid}")
        fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
        environment = dict(
            item.split("=", 1) for item in (root / "environ").read_text().split("\0") if "=" in item
        )
        return {
            "pid": pid,
            "start_ticks": int(fields[19]),
            "cpu_ticks": int(fields[11]) + int(fields[12]),
            "nice": int(fields[16]),
            "pythonpath": environment.get("PYTHONPATH"),
            "cmdline": (root / "cmdline").read_text().replace("\0", " "),
        }
    except (OSError, ValueError, IndexError):
        return None


def _gpu_samples():
    """A failed NVML query loses one GPU sample, not the process/barrier evidence."""
    try:
        devices = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
        gpus = []
        for line in devices.splitlines():
            index, uuid, total, used, free, utilization = [
                value.strip() for value in line.split(",")
            ]
            gpus.append(
                {
                    "device": f"cuda:{int(index)}",
                    "uuid": uuid,
                    "total_bytes": int(total) * 2**20,
                    "used_bytes": int(used) * 2**20,
                    "free_bytes": int(free) * 2**20,
                    "utilization": float(utilization),
                }
            )
        if not gpus:
            raise ValueError("empty nvidia-smi output")
        return gpus, None
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        return [], f"{type(error).__name__}: {error}"


def _slot_samples(repo):
    return [
        item | {"slot_file": path.name}
        for path in (repo / "runs/.gpu_admission").glob("slot_*.json")
        if (item := read_json(path)) is not None
        and process_is_alive(item.get("pid", -1), item.get("process_start_time"))
    ]


def sample_rehearsal(repo, experiments):
    slots_before = _slot_samples(repo)
    gpus, gpu_error = _gpu_samples()
    # The admission monitor publishes every second. Allow a full poll after
    # nvidia-smi so allocator ramps cannot leave both slot reads behind it.
    time.sleep(1.5)
    slots_after = _slot_samples(repo)
    observation = {
        "at": utc_now(),
        "monotonic": time.monotonic(),
        "clock_ticks": os.sysconf("SC_CLK_TCK"),
        "gpus": gpus,
        "gpu_error": gpu_error,
        "slots_before": slots_before,
        "slots_after": slots_after,
        "experiments": {},
    }
    for experiment in experiments:
        workers = experiment / "workers"
        states = {
            path.stem: state
            for path in workers.glob("*.json")
            if (state := read_json(path)) is not None and "status" in state
        }
        identities = {state.get("pid") for state in states.values()}
        identities.update(
            child["pid"] for state in states.values() for child in state.get("child_processes", [])
        )
        marker = workers / "s1_barrier.json"
        summaries = {path.name: sha256_file(path) for path in experiment.glob("*.csv")}
        observation["experiments"][str(experiment.resolve())] = {
            "states": states,
            "processes": [
                value for pid in identities if pid and (value := _process_sample(pid)) is not None
            ],
            "barrier_sha256": sha256_file(marker) if marker.exists() else None,
            "summary_sha256": summaries,
        }
    return observation


def sample_until_complete(repo, experiments, output, interval=10.0):
    for experiment in experiments:
        seal = read_json(experiment / "experiment.json")
        if seal is None or seal.get("rehearsal") is not True:
            raise ValueError("sampling requires sealed rehearsal experiments")
    with output.open("x") as stream:
        while True:
            sample = sample_rehearsal(repo, experiments)
            stream.write(json.dumps(sample, sort_keys=True) + "\n")
            stream.flush()
            if all(
                value["states"].get("supervisor", {}).get("status") == "completed"
                for value in sample["experiments"].values()
            ):
                return
            time.sleep(interval)
