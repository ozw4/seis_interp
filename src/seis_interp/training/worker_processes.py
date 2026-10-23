"""Process identity and failure detection for experiment workers."""

import fcntl
import json
import os
import time
import uuid
from pathlib import Path

from seis_interp.data.forge_mvp_artifacts import write_json

WORKER_LOCK_GRACE_SECONDS = 120.0


class WorkerLockBusy(BlockingIOError):
    """A dead worker's lock is still held, possibly by an orphaned fork."""

    def __init__(self, path, state):
        super().__init__(f"dead worker lock is busy: {path}")
        self.path = path
        self.state = state


def lock_holder_pids(path):
    """Best-effort holders from Linux fdinfo; ignore other opens of this inode."""
    try:
        target = path.stat()
    except OSError:
        return []
    holders = set()
    try:
        processes = list(Path("/proc").iterdir())
    except OSError:
        return []
    for process in processes:
        if not process.name.isdigit():
            continue
        try:
            for descriptor in (process / "fd").iterdir():
                try:
                    stat = descriptor.stat()
                    if (stat.st_dev, stat.st_ino) != (target.st_dev, target.st_ino):
                        continue
                    info = (process / "fdinfo" / descriptor.name).read_text()
                    if any("FLOCK" in line and "WRITE" in line for line in info.splitlines()):
                        holders.add(int(process.name))
                except OSError:
                    continue
        except OSError:
            continue
    return sorted(holders)


def restart_worker_processes(processes, outcomes, restart, pending):
    """Retry busy dead-worker locks on later polls without spending restart credits."""
    for method, code in outcomes.items():
        if code is None:
            continue
        try:
            replacement = restart(method)
        except WorkerLockBusy as error:
            identity = (error.state.get("pid"), error.state.get("process_start_time"))
            previous_identity, since = pending.setdefault(method, (identity, time.monotonic()))
            if identity != previous_identity:
                since = time.monotonic()
                pending[method] = identity, since
            if time.monotonic() - since < WORKER_LOCK_GRACE_SECONDS:
                outcomes[method] = None
                continue
            reason = (
                f"worker exited with code {code}; restart lock grace expired "
                f"after {WORKER_LOCK_GRACE_SECONDS:g}s: "
                f"{error.path}; holder PIDs from /proc: {lock_holder_pids(error.path)}"
            )
            state = error.state | {"status": "failed", "failure_reason": reason, "exit_code": code}
            _write_worker_exit(error.path.with_suffix(".json"), state)
            outcomes[method] = code or 1
            continue
        pending.pop(method, None)
        if replacement is not None:
            processes[method] = replacement
            outcomes[method] = None


def _write_worker_exit(path, state):
    temporary = path.with_suffix(f".supervisor.{uuid.uuid4().hex}.tmp")
    try:
        write_json(temporary, state)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def source_root():
    import seis_interp

    return str(Path(seis_interp.__file__).resolve().parents[1])


def process_stat(pid):
    """Read state and kernel start ticks, including names containing parentheses."""
    try:
        fields = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[0], int(fields[19])
    except (FileNotFoundError, ProcessLookupError, ValueError, IndexError):
        return None


def process_start_time(pid):
    stat = process_stat(pid)
    return stat[1] if stat is not None else None


def process_is_alive(pid, start_time):
    stat = process_stat(pid)
    return stat is not None and stat[0] not in ("Z", "X") and stat[1] == start_time


def worker_failure(state):
    if state["status"] == "failed":
        return state.get("failure_reason", "worker failed")
    if state.get("restartable") and state["status"] in ("running", "retry_pending"):
        return None
    if state["status"] in ("running", "queued") and not process_is_alive(
        state.get("pid", -1), state.get("process_start_time")
    ):
        return "worker process is missing, zombie, or its PID was reused"
    return None


def record_worker_exits(directory, processes, outcomes):
    """Publish unreported exits without replacing a live process's state."""
    failed = False
    for method, code in outcomes.items():
        if code is None:
            continue
        path = directory / f"{method}.json"
        directory.mkdir(parents=True, exist_ok=True)
        # Keep a replacement worker from starting between the check and publication.
        with path.with_suffix(".lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            state = json.loads(path.read_text()) if path.exists() else {}
            if state.get("status") == "failed":
                failed = True
                continue
            if code == 0 and state.get("status") == "completed":
                continue
            # Reaped children are dead; a live recorded identity belongs to another process.
            if state and process_is_alive(state.get("pid", -1), state.get("process_start_time")):
                continue
            reason = f"worker exited with code {code}"
            if not state:
                reason += "; missing worker state file"
            else:
                reason += f"; worker state was {state.get('status', 'missing')}"
            state.update(
                method_id=method,
                pid=processes[method].pid,
                status="failed",
                failure_reason=reason,
                exit_code=code,
            )
            _write_worker_exit(path, state)
            failed = True
    return failed


def wait_for_workers(processes, poll=1.0, *, directory=None, restart=None):
    """Reap every exited worker; return on the first failure without signalling."""
    pending = {}
    while True:
        outcomes = {method: process.poll() for method, process in processes.items()}
        if restart is not None:
            restart_worker_processes(processes, outcomes, restart, pending)
        failed = directory is not None and record_worker_exits(directory, processes, outcomes)
        if failed or any(code is not None and code != 0 for code in outcomes.values()):
            return outcomes, False
        if all(code is not None for code in outcomes.values()):
            return outcomes, True
        time.sleep(poll)


def write_worker_state(path, state):
    """Keep the latest admission summary when the worker flushes in-memory state."""
    if path.exists():
        previous = json.loads(path.read_text())
        if previous.get("pid") == state.get("pid") and "last_admission" in previous:
            state["last_admission"] = previous["last_admission"]
    temporary = path.with_suffix(".tmp")
    write_json(temporary, state)
    temporary.replace(path)
    seal = path.parent.parent / "experiment.json"
    if (
        "status" in state
        and seal.exists()
        and json.loads(seal.read_text()).get("rehearsal") is True
    ):
        from seis_interp.data.forge_mvp_run_records import utc_now

        event = state | {"worker": path.stem, "observed_at": utc_now()}
        descriptor = os.open(
            path.parent / "rehearsal_events.jsonl", os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644
        )
        try:
            os.write(descriptor, (json.dumps(event, sort_keys=True) + "\n").encode())
        finally:
            os.close(descriptor)


def _worker_path(experiment, method):
    for path in sorted((experiment / "workers").glob(f"{method}*.json")):
        state = json.loads(path.read_text())
        if state.get("pid") == os.getpid() and "status" in state:
            return path
    return experiment / "workers" / f"{method}.json"


def report_admission(experiment, method, candidate_id, stage, details):
    """Restore the running phase immediately after GPU admission."""
    path = _worker_path(experiment, method)
    if path.exists():
        state = json.loads(path.read_text())
        if state.get("pid") == os.getpid():
            state.update(
                phase=stage,
                active_run=candidate_id,
                last_admission={
                    "run": candidate_id,
                    "device": details["device"],
                    "wait_seconds": details["wait_seconds"],
                },
            )
            state.pop("resource_wait", None)
            # This is the new admission; do not carry forward the previous summary.
            temporary = path.with_suffix(".tmp")
            write_json(temporary, state)
            temporary.replace(path)


def report_admission_wait(experiment, method, candidate_id, details):
    """A waiting candidate has no run directory; expose its wait in worker state."""
    path = _worker_path(experiment, method)
    if path.exists():
        state = json.loads(path.read_text())
        if state.get("pid") == os.getpid():
            state.update(phase="waiting_for_gpu", active_run=candidate_id, resource_wait=details)
            write_worker_state(path, state)
    print(f"{method} {candidate_id}: waiting for GPU: {details}", flush=True)


def report_resource_wait(experiment, method, name, reason, *, log=True):
    path = _worker_path(experiment, method)
    state = json.loads(path.read_text()) if path.exists() else {}
    if state.get("pid") != os.getpid():
        path = experiment / "workers" / f"{method}_process_{os.getpid()}.json"
        state = {
            "pid": os.getpid(),
            "process_start_time": process_start_time(os.getpid()),
            "method_id": method,
            "status": "running",
            "source_root": source_root(),
        }
    path.parent.mkdir(exist_ok=True)
    state.update(
        phase="waiting_for_resource",
        active_run=name,
        resource_wait={"reason": reason, "reported_at": time.time()},
    )
    write_worker_state(path, state)
    if log:
        print(f"{method} {name}: ResourceWait: {reason}", flush=True)
    return path


def clear_resource_wait(path, stage):
    if path is None or not path.exists():
        return
    state = json.loads(path.read_text())
    if state.get("pid") == os.getpid():
        state.pop("resource_wait", None)
        state["phase"] = stage
        write_worker_state(path, state)
