"""Restart exited infrastructure workers without sending process signals."""

import fcntl
import json
import os

from seis_interp.training.run_recovery import failure_kind
from seis_interp.training.worker_processes import (
    WorkerLockBusy,
    process_is_alive,
    write_worker_state,
)


def refuse_live_workers(directory, names):
    """Check the whole launch before starting anything or rewriting worker state."""
    for name in names:
        path = directory / f"{name}.json"
        state = json.loads(path.read_text()) if path.exists() else {}
        if process_is_alive(state.get("pid", -1), state.get("process_start_time")):
            raise RuntimeError(f"refusing duplicate live worker: {name}")
        with path.with_suffix(".lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f"refusing duplicate live worker: {name}") from error


class ObservedWorker:
    """Monitor an existing worker by kernel identity without becoming its parent."""

    def __init__(self, path, state):
        self.path = path
        self.pid = state.get("pid", -1)
        self.start_time = state.get("process_start_time")

    def poll(self):
        if process_is_alive(self.pid, self.start_time):
            return None
        state = json.loads(self.path.read_text()) if self.path.exists() else {}
        return 0 if state.get("status") == "completed" else 1


def start_dead_workers(directory, names, start):
    """Adopt live workers unchanged; start missing/dead retryable workers only."""
    processes = {}
    for name in names:
        path = directory / f"{name}.json"
        state = json.loads(path.read_text()) if path.exists() else {}
        observed = ObservedWorker(path, state)
        if (
            process_is_alive(observed.pid, observed.start_time)
            or state.get("status") == "completed"
        ):
            processes[name] = observed
            continue
        # Reuse the bounded restart path, including its worker lock check.
        try:
            process = restart_exited_worker(directory, name, start)
        except WorkerLockBusy:
            # The supervisor will poll this identity and retry within its lock grace.
            process = None
        processes[name] = process if process is not None else observed
    return processes


def restart_exited_worker(directory, method, start):
    path = directory / f"{method}.json"
    state = json.loads(path.read_text()) if path.exists() else {}
    if not _worker_can_restart(state) or process_is_alive(
        state.get("pid", -1), state.get("process_start_time")
    ):
        return None
    with path.with_suffix(".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise WorkerLockBusy(path.with_suffix(".lock"), state) from error
        if not _mark_worker_restart(directory, method, path, state):
            return None
    return start(method)


def _worker_can_restart(state):
    if state.get("status") == "completed":
        return False
    return (
        state.get("status") != "failed"
        or failure_kind(RuntimeError(state.get("failure_reason", ""))) == "infrastructure"
    )


def _mark_worker_restart(directory, method, path, state):
    if not _worker_can_restart(state):
        return False
    counter = directory / f"{method}_retries.json"
    previous = json.loads(counter.read_text()) if counter.exists() else {}
    run = [
        state.get(key) for key in ("phase", "active_run", "completed_s1", "completed_evaluation")
    ]
    retries = previous.get("retries", 0) if previous.get("run") == run else 0
    if retries >= 2:
        return False
    write_worker_state(counter, {"run": run, "retries": retries + 1})
    seal = directory.parent / "experiment.json"
    if seal.exists() and json.loads(seal.read_text()).get("rehearsal") is True:
        from seis_interp.data.forge_mvp_run_records import utc_now

        event = {
            "worker": method,
            "run": run,
            "retries": retries + 1,
            "previous_pid": state.get("pid"),
            "at": utc_now(),
        }
        descriptor = os.open(
            directory / "rehearsal_restarts.jsonl", os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644
        )
        try:
            os.write(descriptor, (json.dumps(event) + "\n").encode())
        finally:
            os.close(descriptor)
    state.update(status="retry_pending")
    write_worker_state(path, state)
    return True
