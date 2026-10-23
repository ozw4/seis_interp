"""Durable, monotonic permission to read held-out evaluation targets."""

import fcntl
import json
import os

from seis_interp.data.forge_mvp_run_records import utc_now
from seis_interp.training.run_recovery import S1BarrierError, write_run_manifest
from seis_interp.training.worker_processes import worker_failure


def s1_barrier_ready(experiment, sources):
    workers = experiment / "workers"
    workers.mkdir(exist_ok=True)
    marker = workers / "s1_barrier.json"
    with (workers / "s1_barrier.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if marker.exists():
            return True
        ready = True
        for method, source in sources.items():
            path = source / "workers" / f"{method}.json"
            if not path.exists():
                ready = False
                continue
            state = json.loads(path.read_text())
            failure = worker_failure(state)
            if failure is not None:
                raise S1BarrierError(f"{method} failed: {failure}")
            ready &= state.get("s1_completed", False) and state.get("status") != "retry_pending"
        if ready:
            write_run_manifest(
                marker, {"passed": True, "methods": list(sources), "passed_at": utc_now()}
            )
            with marker.open("rb") as stream:
                os.fsync(stream.fileno())
            descriptor = os.open(workers, os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        return bool(ready)
