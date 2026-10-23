"""Opt-in, repository-wide CUDA admission with reservations for growing jobs."""

import fcntl
import json
import math
import os
import subprocess
import tempfile
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from pathlib import Path

import torch

from seis_interp.data.forge_mvp_artifacts import write_json
from seis_interp.training.run_recovery import ResourceWait, failure_kind, requires_process_restart
from seis_interp.training.worker_processes import process_is_alive, process_start_time


def admission_devices(config, resources):
    device = config["device"]
    devices = (
        device
        if isinstance(device, list)
        else resources["devices"]
        if device == "auto"
        else [device]
    )
    if not devices or not set(devices) <= set(resources["devices"]):
        raise ValueError("requested device is outside the admission device list")
    if any(not device.startswith("cuda:") for device in devices):
        raise ValueError("admission devices must be explicit CUDA device names")
    return devices


def admission_required_bytes(config, resources, candidate_id):
    minimum = resources["minimum_free_memory_gib"][config["method_id"]]
    if isinstance(minimum, dict):
        minimum = minimum[candidate_id]
    if minimum is None:
        raise ValueError(
            f"{config['method_id']} {candidate_id}: minimum_free_memory_gib is unset; "
            "measure with the probe first"
        )
    margin = resources.get("admission_margin_gib", 0.0)
    if not math.isfinite(minimum) or minimum <= 0 or not math.isfinite(margin) or margin < 0:
        raise ValueError("admission memory must be positive and margin nonnegative")
    return math.ceil((minimum + margin) * 2**30)


def validate_admission_capacity(required, totals, label, context_allowance_gib=2.0):
    if not math.isfinite(context_allowance_gib) or context_allowance_gib < 0:
        raise ValueError("context_allowance_gib must be nonnegative and finite")
    capacity = max(totals.values()) - math.ceil(context_allowance_gib * 2**30)
    if required > capacity:
        raise ValueError(
            f"{label}: required {required / 2**30:.3f} GiB exceeds largest allowed device "
            f"usable memory {capacity / 2**30:.3f} GiB "
            f"(total {max(totals.values()) / 2**30:.3f} GiB minus "
            f"context allowance {context_allowance_gib:.3f} GiB)"
        )


def prepare_admission_capacity(resources, candidates):
    """Check declared capacity or nvidia-smi totals without initializing CUDA."""
    if not resources or not resources.get("global_admission", False):
        return
    requests = [
        (
            item,
            admission_devices(item["config"], resources),
            admission_required_bytes(item["config"], resources, item["id"]),
        )
        for item in candidates
        if item["config"]["device"] != "cpu"
    ]
    if not requests:
        return
    declared = resources.get("device_total_memory_gib")
    if declared is not None:
        totals = {device: math.floor(value * 2**30) for device, value in declared.items()}
    else:
        totals = {}
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        for device in resources["devices"]:
            index = int(device.split(":")[1])
            selector = visible.split(",")[index] if visible else str(index)
            value = subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--id={selector}",
                    "--query-gpu=memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip()
            totals[device] = int(value) * 2**20
    for item, devices, required in requests:
        allowed = {device: totals[device] for device in devices}
        validate_admission_capacity(
            required,
            allowed,
            f"{item['config']['method_id']} {item['id']}",
            resources.get("context_allowance_gib", 2.0),
        )


def choose_admission_device(devices, free_bytes, slots, required_bytes):
    """Reserve each live job's room to regrow from its conservative reservation."""
    available = {}
    for device in devices:
        pending = sum(
            max(0, slot["required_bytes"] - slot["current_reserved_bytes"])
            for slot in slots
            if slot["device"] == device
        )
        available[device] = free_bytes[device] - pending
    fitting = [device for device in devices if available[device] >= required_bytes]
    return max(fitting, key=lambda device: available[device]) if fitting else None


def _recent_reserved_minimum(reports, now, reserved_bytes):
    """Keep a lower reservation estimate across cache releases between reports."""
    reports.append((now, reserved_bytes))
    while reports[0][0] < now - 30.0:
        reports.popleft()
    return min(reserved for _, reserved in reports)


def live_admission_slots(directory):
    """PID liveness avoids expiring a running job during a long CUDA operation."""
    result = []
    for path in sorted(directory.glob("slot_*.json")):
        try:
            slot = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(slot, dict) or type(slot.get("pid")) is not int:
            continue
        if not process_is_alive(slot["pid"], slot.get("process_start_time")):
            path.unlink(missing_ok=True)
            continue
        if not all(key in slot for key in ("device", "required_bytes", "current_reserved_bytes")):
            continue
        if not isinstance(slot["device"], str) or any(
            not isinstance(slot[key], (int, float)) or not math.isfinite(slot[key]) or slot[key] < 0
            for key in ("required_bytes", "current_reserved_bytes")
        ):
            continue
        if "reservation" in slot and (
            not isinstance(slot["reservation"], str)
            or type(slot.get("reserved_at_ns", 0)) is not int
        ):
            continue
        if "reservation" in slot and time.monotonic() >= slot.get("retry_hold_until", math.inf):
            path.unlink(missing_ok=True)
            continue
        result.append(slot)
    return result


def write_admission_slot(path, slot):
    """Publish reservations and monitor observations outside the reader glob."""
    descriptor, temporary = tempfile.mkstemp(prefix=".slot_", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary)
    try:
        write_json(temporary, slot)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def admission_predecessors(slots, own):
    """Running jobs count; FIFO priority applies only on the reserved device.

    Counting younger reservations too makes all holders wait for the sum of their
    requests when free memory drops after selection. On other devices every
    reservation counts, so migration cannot take a younger holder's room.
    """
    if own is None or "reservation" not in own:
        return slots
    priority = (own.get("reserved_at_ns", 0), own["reservation"])
    return [
        slot
        for slot in slots
        if slot is not own
        and (
            "reservation" not in slot
            or slot["device"] != own["device"]
            or (slot.get("reserved_at_ns", 0), slot["reservation"]) < priority
        )
    ]


def excluded_admission_devices(exclusions):
    """Exclude devices with live pinned identities, ignoring zombies and reused PIDs."""
    return {
        device
        for device, processes in exclusions.items()
        if any(process_is_alive(item["pid"], item["process_start_time"]) for item in processes)
    }


@contextmanager
def gpu_admission_slot(
    directory,
    devices,
    minimum_free_memory_gib,
    waiting,
    *,
    timeout=60.0,
    poll=1.0,
    admitted=None,
    context_allowance_gib=2.0,
    excluded_devices_while_pids_alive=None,
    reservation=None,
):
    """Serialize startup only; monitor each live job until its run leaves the slot.

    The admission lock is retained until 90% of the declared reservation has been
    reached or the bounded startup timeout expires. The unreached reservation
    remains accounted for after that timeout. A process-local monitor reports
    the minimum CUDA reservation over the last 30 seconds so cache released
    between reports remains reserved for regrowth.
    """
    if not devices or any(not device.startswith("cuda:") for device in devices):
        raise ValueError("admission devices must be explicit CUDA device names")
    if (
        not math.isfinite(minimum_free_memory_gib)
        or minimum_free_memory_gib <= 0
        or timeout <= 0
        or poll <= 0
    ):
        raise ValueError("admission memory and intervals must be positive")
    required = int(minimum_free_memory_gib * 2**30)
    totals = {device: torch.cuda.get_device_properties(device).total_memory for device in devices}
    validate_admission_capacity(required, totals, "GPU admission", context_allowance_gib)
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / "admission.lock").open("a")
    held = False
    slot_path = (
        Path(reservation)
        if reservation
        else directory / f"slot_{os.getpid()}_{uuid.uuid4().hex}.json"
    )
    stopped = threading.Event()
    monitor = None
    own = None
    retry_reservation = None
    selected = None
    wait_started = time.monotonic()
    try:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                held = True
            except BlockingIOError:
                waiting({"reason": "another_job_starting", "devices": devices})
                time.sleep(poll)
                continue
            excluded = excluded_admission_devices(excluded_devices_while_pids_alive or {})
            eligible = [device for device in devices if device not in excluded]
            free = {device: torch.cuda.mem_get_info(device)[0] for device in eligible}
            slots = live_admission_slots(directory)
            own = next(
                (
                    slot
                    for slot in slots
                    if slot.get("reservation") == slot_path.stem.rsplit("_", 1)[-1]
                ),
                None,
            )
            slots = admission_predecessors(slots, own)
            selected = choose_admission_device(eligible, free, slots, required)
            if selected is not None:
                break
            fcntl.flock(lock, fcntl.LOCK_UN)
            held = False
            waiting(
                {
                    "reason": "insufficient_free_memory",
                    "free_memory_bytes": free,
                    "required_bytes": required,
                    "excluded_devices": sorted(excluded.intersection(devices)),
                }
            )
            time.sleep(poll)
        torch.cuda.set_device(selected)
        fraction = required / totals[selected]
        torch.cuda.set_per_process_memory_fraction(fraction, selected)
        torch.cuda.reset_peak_memory_stats(selected)
        reserved_now = int(torch.cuda.memory_reserved(selected))
        slot = {
            "pid": os.getpid(),
            "process_start_time": process_start_time(os.getpid()),
            "device": selected,
            "required_bytes": required,
            "current_reserved_bytes": reserved_now,
            "current_reserved_now_bytes": reserved_now,
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(selected)),
            "total_device_bytes": totals[selected],
            "per_process_memory_fraction": fraction,
            "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            "wait_seconds": time.monotonic() - wait_started,
        }
        if admitted is not None:
            admitted(dict(slot))

        def publish():
            write_admission_slot(slot_path, slot)

        reserved_reports = deque([(time.monotonic(), slot["current_reserved_bytes"])])
        publish()
        started = time.monotonic()

        def observe():
            nonlocal held
            while not stopped.wait(poll):
                slot["current_reserved_now_bytes"] = int(torch.cuda.memory_reserved(selected))
                slot["current_reserved_bytes"] = _recent_reserved_minimum(
                    reserved_reports, time.monotonic(), slot["current_reserved_now_bytes"]
                )
                slot["peak_reserved_bytes"] = max(
                    slot["peak_reserved_bytes"], int(torch.cuda.max_memory_reserved(selected))
                )
                slot["updated_at_monotonic"] = time.monotonic()
                publish()
                if held and (
                    slot["peak_reserved_bytes"] >= 0.9 * required
                    or time.monotonic() - started >= timeout
                ):
                    fcntl.flock(lock, fcntl.LOCK_UN)
                    held = False

        monitor = threading.Thread(target=observe, daemon=True)
        monitor.start()
        yield selected
    except Exception as error:
        if (
            own is not None
            and failure_kind(error) == "infrastructure"
            and not requires_process_restart(error)
        ):
            # Retain queue priority through the in-process backoff, including a
            # resource wait. Repeated waits cannot renew this five-minute bound.
            retry_reservation = own | {
                "pid": os.getpid(),
                "process_start_time": process_start_time(os.getpid()),
                "device": selected or own["device"],
                "current_reserved_bytes": 0,
                "retry_hold_until": own.get("retry_hold_until", time.monotonic() + 300.0)
                if isinstance(error, ResourceWait)
                else time.monotonic() + 300.0,
            }
        raise
    finally:
        stopped.set()
        if monitor is not None:
            monitor.join()
        try:
            if retry_reservation is not None:
                # The retry is published before its caller releases cached memory.
                # Keep the instantaneous evidence separate from the zero regrowth estimate.
                retry_device = retry_reservation["device"]
                retry_reservation["current_reserved_now_bytes"] = int(
                    torch.cuda.memory_reserved(retry_device)
                )
                retry_reservation["peak_reserved_bytes"] = int(
                    torch.cuda.max_memory_reserved(retry_device)
                )
                write_admission_slot(slot_path, retry_reservation)
            else:
                slot_path.unlink(missing_ok=True)
        finally:
            if held:
                fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()


@contextmanager
def candidate_admission(
    repo, config, resources, candidate_id, waiting, *, admitted=None, reservation=None
):
    """Legacy studies retain their existing locks unless global_admission is true."""
    device = config["device"]
    if device == "cpu" or not resources or not resources.get("global_admission", False):
        yield device
        return
    devices = admission_devices(config, resources)
    required = admission_required_bytes(config, resources, candidate_id)
    with gpu_admission_slot(
        repo / "runs/.gpu_admission",
        devices,
        required / 2**30,
        waiting,
        timeout=resources.get("startup_timeout_seconds", 60.0),
        admitted=admitted,
        context_allowance_gib=resources.get("context_allowance_gib", 2.0),
        excluded_devices_while_pids_alive=resources.get("excluded_devices_while_pids_alive"),
        reservation=reservation,
    ) as selected:
        yield selected


def memory_fitting_candidates(
    repo, candidates, resources, *, poll=1.0, without_admission=(), waiting=None
):
    """Choose and reserve atomically; finalized/importable cells need no GPU."""
    remaining = list(candidates)
    wait_reported = None
    if not resources or not resources.get("global_admission", False):
        yield from remaining
        return
    while remaining:
        chosen = next(
            (
                item
                for item in remaining
                if item["id"] in without_admission or item["config"]["device"] == "cpu"
            ),
            None,
        )
        if chosen is not None:
            remaining.remove(chosen)
            yield chosen
            continue
        directory = repo / "runs/.gpu_admission"
        directory.mkdir(parents=True, exist_ok=True)
        reservation = None
        details = {
            "reason": "another_job_starting",
            "remaining": [item["id"] for item in remaining],
        }
        with (directory / "admission.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                slots = live_admission_slots(directory)
                excluded = excluded_admission_devices(
                    resources.get("excluded_devices_while_pids_alive", {})
                )
                details.update(
                    reason="no_fitting_candidate",
                    required_bytes={},
                    free_memory_bytes={},
                    excluded_devices=sorted(excluded),
                )
                for item in remaining:
                    devices = [
                        d for d in admission_devices(item["config"], resources) if d not in excluded
                    ]
                    free = {device: torch.cuda.mem_get_info(device)[0] for device in devices}
                    required = admission_required_bytes(item["config"], resources, item["id"])
                    details["required_bytes"][item["id"]] = required
                    details["free_memory_bytes"].update(free)
                    device = choose_admission_device(devices, free, slots, required)
                    if device is not None:
                        chosen = item
                        identity = uuid.uuid4().hex
                        reservation = directory / f"slot_{os.getpid()}_{identity}.json"
                        write_admission_slot(
                            reservation,
                            {
                                "pid": os.getpid(),
                                "process_start_time": process_start_time(os.getpid()),
                                "device": device,
                                "required_bytes": required,
                                "current_reserved_bytes": 0,
                                "current_reserved_now_bytes": int(
                                    torch.cuda.memory_reserved(device)
                                ),
                                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                                "reservation": identity,
                                "reserved_at_ns": time.monotonic_ns(),
                            },
                        )
                        break
        if chosen is None:
            now = time.monotonic()
            if waiting is not None and (wait_reported is None or now - wait_reported >= 60.0):
                waiting(details)
                wait_reported = now
            time.sleep(poll)
            continue
        remaining.remove(chosen)
        wait_reported = None
        yield chosen | {"admission_reservation": str(reservation)}
