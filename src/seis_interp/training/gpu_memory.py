"""CUDA-visible free/total memory without creating CUDA primary contexts."""

import os
import subprocess


def _physical_gpu_memory():
    try:
        import pynvml
    except ImportError:
        pynvml = None
    if pynvml is not None:
        try:
            pynvml.nvmlInit()
            try:
                rows = []
                for index in range(pynvml.nvmlDeviceGetCount()):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                    identity = pynvml.nvmlDeviceGetUUID(handle)
                    if isinstance(identity, bytes):
                        identity = identity.decode()
                    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    rows.append((identity, int(memory.free), int(memory.total)))
                return rows
            finally:
                pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=uuid,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return [
        (identity.strip(), int(free) * 2**20, int(total) * 2**20)
        for line in output.splitlines()
        for identity, free, total in [line.split(",")]
    ]


def gpu_memory_info(devices):
    """Map CUDA indices through CUDA_VISIBLE_DEVICES to physical GPU UUIDs.

    Numeric selectors retain the physical index mapping used by preparation;
    full or unambiguous abbreviated UUID selectors also support reordered GPUs.
    Neither NVML nor nvidia-smi initializes CUDA in the calling process.
    """
    if not devices:
        return {}
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    selectors = visible.split(",") if visible is not None else None
    if selectors == [""]:
        raise ValueError("CUDA_VISIBLE_DEVICES hides all admission devices")
    physical = _physical_gpu_memory()
    result = {}
    for device in devices:
        index = int(device.split(":")[1])
        selector = selectors[index].strip() if selectors is not None else str(index)
        if selector.isdecimal():
            identity = physical[int(selector)][0]
        else:
            matches = [row[0] for row in physical if row[0].startswith(selector)]
            if len(matches) != 1:
                raise ValueError(f"GPU selector {selector!r} does not identify one UUID")
            identity = matches[0]
        result[device] = next((free, total) for uuid, free, total in physical if uuid == identity)
    return result
