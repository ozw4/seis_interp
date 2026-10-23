"""Serialize runs on each CUDA device and wait for their declared free memory."""

import fcntl
import time
from contextlib import contextmanager

import torch


@contextmanager
def cuda_run_slot(directory, device, minimum_free_memory_gib, waiting):
    """Release the device lock after one run, including when the run fails."""
    if not device.startswith("cuda:"):
        yield
        return
    required = int(minimum_free_memory_gib * 2**30)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"cuda_{int(device.split(':')[1])}.lock"
    with path.open("a") as lock:
        while True:
            acquired = False
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except BlockingIOError:
                waiting({"device": device, "reason": "another_experiment_run"})
                fcntl.flock(lock, fcntl.LOCK_EX)
                acquired = True
            if acquired:
                try:
                    free, _ = torch.cuda.mem_get_info(device)
                    if free >= required:
                        yield
                        return
                    waiting(
                        {
                            "device": device,
                            "reason": "insufficient_free_memory",
                            "free_memory_bytes": free,
                            "required_free_memory_bytes": required,
                        }
                    )
                finally:
                    fcntl.flock(lock, fcntl.LOCK_UN)
            time.sleep(10)
