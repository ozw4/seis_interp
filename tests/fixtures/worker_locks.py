"""Real forked lock holders that exit on their own, without process signals."""

import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from types import SimpleNamespace

_FORKING_WORKER = """
import fcntl
import os
import sys
import time
from pathlib import Path
from seis_interp.training.worker_processes import (
    process_is_alive, process_start_time, write_worker_state,
)

directory, name, delay, admission = sys.argv[1:]
directory = Path(directory)
parent = os.getpid()
start_time = process_start_time(parent)
if admission == 'True':
    from seis_interp.training import gpu_admission
    gpu_admission_slot, torch = gpu_admission.gpu_admission_slot, gpu_admission.torch
    gpu_admission.gpu_memory_info = lambda devices: {d: (100 * 2**30, 120 * 2**30) for d in devices}
    torch.cuda.set_device = lambda _: None
    torch.cuda.set_per_process_memory_fraction = lambda *args: None
    torch.cuda.reset_peak_memory_stats = lambda _: None
    torch.cuda.memory_reserved = lambda _: 0
    torch.cuda.max_memory_reserved = lambda _: 0
    context = gpu_admission_slot(directory, ['cuda:0'], 10, print, timeout=0.2, poll=0.01)
else:
    context = (directory / (name + '.lock')).open('a')
with context as lock:
    if admission != 'True':
        fcntl.flock(lock, fcntl.LOCK_EX)
    write_worker_state(directory / (name + '.json'), {
        'pid': parent, 'process_start_time': start_time, 'status': 'running',
        'method_id': name.split('_shard_')[0], 'phase': 'S1', 'active_run': 'a01_b05',
        'completed_s1': 1, 'completed_evaluation': 0,
    })
    if os.fork() == 0:
        ready = directory / 'child_ready.tmp'
        ready.write_text(str(os.getpid()))
        ready.replace(directory / 'child_ready')
        while process_is_alive(parent, start_time):
            time.sleep(0.01)
        time.sleep(float(delay))
        if admission != 'True':
            lock.close()
        (directory / 'child_released').write_text('done')
        os._exit(0)
    while not (directory / 'exit_parent').exists():
        time.sleep(0.01)
    os._exit(137)
"""


def _wait_for_file(path, timeout):
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"forked lock holder did not write {path}")
        time.sleep(0.01)


@contextmanager
def orphaned_lock_holder(directory, name, *, hold_seconds=2, admission=False):
    directory.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            _FORKING_WORKER,
            str(directory),
            name,
            str(hold_seconds),
            str(admission),
        ],
        env=os.environ | {"CUDA_VISIBLE_DEVICES": ""},
    )
    exit_path = directory / "exit_parent"
    try:
        _wait_for_file(directory / "child_ready", 15)
        yield SimpleNamespace(
            process=process,
            child_pid=int((directory / "child_ready").read_text()),
            exit_path=exit_path,
            state=json.loads((directory / f"{name}.json").read_text()),
        )
    finally:
        exit_path.touch()
        process.wait(timeout=10)
        _wait_for_file(directory / "child_released", hold_seconds + 10)
