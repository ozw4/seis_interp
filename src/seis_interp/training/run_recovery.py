"""Bounded run retries, progress watchdogs and admission observations."""

import fcntl
import gc
import json
import math
import os
import shutil
import subprocess
import threading
import time
from collections import deque
from functools import wraps

import numpy as np
import torch

from seis_interp.data.forge_mvp_artifacts import write_json


def write_run_manifest(path, record):
    temporary = path.with_suffix(".tmp")
    write_json(temporary, record)
    temporary.replace(path)


class ResourceWait(OSError):
    """A resource precondition has not been met; no run attempt was made."""


class S1BarrierError(RuntimeError):
    """A peer failure is an experiment precondition, not this run's failure."""


class RestartRunProcess(RuntimeError):
    """The current CUDA context must be discarded before the next attempt."""


def failure_kind(error):
    if isinstance(error, S1BarrierError):
        return "failed"
    message = str(error).lower()
    if isinstance(error, FloatingPointError) or any(
        term in message for term in ("nonfinite", "non-finite", "must be finite")
    ):
        return "diverged"
    if isinstance(error, (MemoryError, OSError, TimeoutError, RestartRunProcess)) or any(
        term in message
        for term in (
            "out of memory",
            "cuda error",
            "driver",
            "device-side",
            "worker exited",
            "brokenprocesspool",
            "terminated abruptly",
            "connection reset",
            "watchdog",
            "illegal memory access",
            "unspecified launch failure",
        )
    ):
        return "infrastructure"
    return "failed"


def requires_process_restart(error):
    message = str(error).lower()
    return (
        isinstance(error, RestartRunProcess)
        or any(
            term in message
            for term in (
                "cuda error",
                "device-side",
                "illegal memory access",
                "unspecified launch failure",
            )
        )
        and "out of memory" not in message
    )


def require_finite_metrics(metrics):
    """Nulls carry the existing undefined-metric conventions; NaN/inf are divergence."""
    for value in metrics.values():
        if isinstance(value, dict):
            require_finite_metrics(value)
        elif isinstance(value, (int, float)) and not math.isfinite(value):
            raise FloatingPointError("non-finite evaluation metric")


def archive_attempt(output):
    attempts = output / "attempts"
    attempts.mkdir(exist_ok=True)
    number = max((int(path.name) for path in attempts.iterdir()), default=0) + 1
    destination = attempts / str(number)
    destination.mkdir()
    for path in list(output.iterdir()):
        if path.name not in ("attempts", "retry_state.json"):
            path.rename(destination / path.name)
    record = json.loads((destination / "manifest.json").read_text())
    if number <= 2:
        for name in ("resume.pt", "prediction_result.pt"):
            checkpoint = destination / name
            if checkpoint.exists():
                temporary = output / ".resume.copy"
                shutil.copyfile(checkpoint, temporary)
                temporary.replace(output / name)
        if (output / "prediction_result.pt").exists():
            write_run_manifest(output / "manifest.json", record | {"status": "awaiting_barrier"})
    else:
        record.update(
            status="failed",
            retry_exhausted=True,
            artifacts={
                f"attempts/{number}/{name}": digest
                for name, digest in record.get("artifacts", {}).items()
            },
        )
        write_run_manifest(output / "manifest.json", record)
    write_run_manifest(
        output / "retry_state.json",
        {
            "failed_attempts": number,
            "next_retry_at": time.time() + min(5.0 * 2 ** (number - 1), 300.0),
        },
    )
    return number


def _record_failed_attempt(output, candidate, seed, stage, error):
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    record = (
        json.loads(path.read_text())
        if path.exists()
        else {
            "id": candidate["id"],
            "method_id": candidate["config"]["method_id"],
            "stage": stage,
            "trace_seed": seed,
            "artifacts": {},
        }
    )
    record.update(
        status="failed",
        failure_kind="infrastructure",
        failure_reason=f"{type(error).__name__}: {error}",
    )
    write_run_manifest(path, record)
    if archive_attempt(output) > 2:
        raise RuntimeError(f"automatic retry limit exhausted: {output}") from error


def retry_run(execute):
    """Fence a run and count failures even when execution never rewrites its manifest."""

    @wraps(execute)
    def run(repo, experiment, candidate, seed, stage, **kwargs):
        from seis_interp.training.worker_processes import clear_resource_wait, report_resource_wait

        method = candidate["config"]["method_id"]
        name = candidate["id"] if stage == "S1" else f"seed_{seed}"
        output = experiment / "method_runs" / method / stage / name
        output.parent.mkdir(parents=True, exist_ok=True)
        wait_reported = None
        wait_path = None
        try:
            with output.with_suffix(".lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                while True:
                    path = output / "manifest.json"
                    previous = json.loads(path.read_text()) if path.exists() else {}
                    if previous.get("retry_exhausted"):
                        raise RuntimeError(f"automatic retry limit exhausted: {output}")
                    if previous.get("status") in ("failed", "running"):
                        reason = previous.get("failure_reason", "worker exited during an attempt")
                        if (
                            previous["status"] == "running"
                            or previous.get("failure_kind") == "infrastructure"
                        ):
                            _record_failed_attempt(
                                output, candidate, seed, stage, RuntimeError(reason)
                            )
                    counter = output / "retry_state.json"
                    if counter.exists() and previous.get("status") not in ("completed", "diverged"):
                        delay = json.loads(counter.read_text())["next_retry_at"] - time.time()
                        if delay > 0:
                            time.sleep(delay)
                    try:
                        result = execute(repo, experiment, candidate, seed, stage, **kwargs)
                        clear_resource_wait(wait_path, stage)
                        return result
                    except ResourceWait as error:
                        now = time.monotonic()
                        log_wait = wait_reported is None or now - wait_reported >= 60.0
                        wait_path = report_resource_wait(
                            experiment, method, name, str(error), log=log_wait
                        )
                        if log_wait:
                            wait_reported = now
                        time.sleep(5.0)
                        continue
                    except Exception as error:
                        if failure_kind(error) != "infrastructure":
                            raise
                        _record_failed_attempt(output, candidate, seed, stage, error)
                        if requires_process_restart(error):
                            raise RestartRunProcess(f"fresh process required: {error}") from error
                    gc.collect()
                    if torch.cuda.is_initialized():
                        torch.cuda.empty_cache()
        finally:
            reservation = candidate.get("admission_reservation")
            if reservation is not None:
                from pathlib import Path

                Path(reservation).unlink(missing_ok=True)

    return run


def check_free_disk(directory, minimum_gib=5.0):
    if not math.isfinite(minimum_gib) or minimum_gib < 0:
        raise ValueError("minimum_free_disk_gib must be nonnegative")
    path = directory
    while not path.exists():
        path = path.parent
    free = shutil.disk_usage(path).free
    if free < minimum_gib * 2**30:
        raise ResourceWait(
            f"insufficient free disk: {free / 2**30:.3f} GiB; need {minimum_gib} GiB"
        )
    return free


def gpu_utilization(device):
    if not str(device).startswith("cuda:"):
        return []
    try:
        return (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=5,
            )
            .strip()
            .splitlines()
        )
    except (OSError, subprocess.SubprocessError):
        return None


class ProgressWatchdog:
    """Bound a resumable run using a generous floor and a high interval quantile.

    Every report contributes, including checkpoint and prediction phase reports.
    Methods without step progress and exact resume use an optional wall limit only.
    """

    def __init__(
        self,
        output,
        expected_seconds=1800.0,
        *,
        floor_seconds=1800.0,
        wall_seconds=None,
        enabled=True,
        exit_process=os._exit,
    ):
        self.output = output
        if not math.isfinite(expected_seconds) or expected_seconds <= 0 or floor_seconds <= 0:
            raise ValueError("expected report interval and floor must be positive")
        if wall_seconds is not None and (not math.isfinite(wall_seconds) or wall_seconds <= 0):
            raise ValueError("wall-clock limit must be positive")
        self.initial = expected_seconds
        self.floor = floor_seconds
        self.wall_seconds = wall_seconds
        self.enabled = enabled or wall_seconds is not None
        self.exit_process = exit_process
        self.last = self.started = time.monotonic()
        self.intervals = deque(maxlen=256)
        self.stopped = threading.Event()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._watch, daemon=True)

    @property
    def threshold(self):
        expected = (
            max(self.initial, float(np.quantile(self.intervals, 0.95)))
            if self.intervals
            else self.initial
        )
        return max(self.floor, 3 * expected)

    def start(self):
        if self.enabled:
            self.thread.start()

    def report(self, message=""):
        with self.lock:
            now = time.monotonic()
            self.intervals.append(max(now - self.last, 0.0))
            self.last = now

    def _watch(self):
        while not self.stopped.wait(1.0):
            with self.lock:
                elapsed = time.monotonic() - (
                    self.started if self.wall_seconds is not None else self.last
                )
                limit = self.wall_seconds if self.wall_seconds is not None else self.threshold
            if elapsed > limit:
                write_json(
                    self.output / "watchdog.json",
                    {
                        "reason": "run exceeded its wall-clock limit"
                        if self.wall_seconds
                        else "progress stalled",
                        "threshold_seconds": limit,
                        "exit_code": 75,
                    },
                )
                self.exit_process(75)
                return

    def close(self):
        self.stopped.set()
        if self.thread.ident is not None:
            self.thread.join()


def run_watchdog(output, config, *, resumable=True):
    execution = config.get("execution", {})
    resumable = resumable and config["method_id"] not in ("pocs_grid", "drr_grid")
    return ProgressWatchdog(
        output,
        execution.get("expected_report_seconds", 1800.0),
        floor_seconds=execution.get("watchdog_floor_seconds", 1800.0),
        enabled=resumable,
        wall_seconds=execution.get("wall_clock_limit_seconds") if not resumable else None,
    )
