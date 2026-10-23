"""Memory telemetry must never initialize CUDA in an admission process."""

import sys
from types import SimpleNamespace

import pytest
import torch

from seis_interp.training import gpu_memory


@pytest.mark.parametrize("backend", ["nvml", "missing_nvml", "failed_nvml"])
@pytest.mark.parametrize("visible", [None, "1,0", "GPU-b,GPU-a"])
def test_memory_telemetry_maps_visible_indices_without_cuda(monkeypatch, backend, visible):
    class NVMLError(Exception):
        pass

    calls = []

    def initialize():
        calls.append("nvml")
        if backend == "failed_nvml":
            raise NVMLError("unavailable")

    fake = SimpleNamespace(
        NVMLError=NVMLError,
        nvmlInit=initialize,
        nvmlShutdown=lambda: calls.append("shutdown"),
        nvmlDeviceGetCount=lambda: 2,
        nvmlDeviceGetHandleByIndex=lambda i: i,
        nvmlDeviceGetUUID=lambda i: [b"GPU-aaaa", b"GPU-bbbb"][i],
        nvmlDeviceGetMemoryInfo=lambda i: SimpleNamespace(free=(i + 1) * 2**20, total=10 * 2**20),
    )
    monkeypatch.setitem(sys.modules, "pynvml", None if backend == "missing_nvml" else fake)

    def query(command, **kwargs):
        assert command == [
            "nvidia-smi",
            "--query-gpu=uuid,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ]
        calls.append("smi")
        return "GPU-aaaa, 1, 10\nGPU-bbbb, 2, 10\n"

    monkeypatch.setattr(gpu_memory.subprocess, "check_output", query)
    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda: pytest.fail("CUDA initialized"))
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    result = gpu_memory.gpu_memory_info(["cuda:0", "cuda:1"])
    free = [1, 2] if visible is None else [2, 1]
    assert result == {f"cuda:{i}": (value * 2**20, 10 * 2**20) for i, value in enumerate(free)}
    assert ("smi" in calls) == (backend != "nvml")
    assert ("shutdown" in calls) == (backend == "nvml")


def test_hidden_devices_never_query_hardware(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(gpu_memory, "_physical_gpu_memory", lambda: pytest.fail("hardware queried"))
    assert gpu_memory.gpu_memory_info([]) == {}
    with pytest.raises(ValueError, match="hides all"):
        gpu_memory.gpu_memory_info(["cuda:0"])


@pytest.mark.parametrize("selector", ["GPU-", "GPU-absent", "-1"])
def test_ambiguous_or_unavailable_gpu_is_rejected(monkeypatch, selector):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", selector)
    monkeypatch.setattr(
        gpu_memory, "_physical_gpu_memory", lambda: [("GPU-a", 1, 10), ("GPU-b", 2, 10)]
    )
    with pytest.raises(ValueError, match="does not identify one UUID"):
        gpu_memory.gpu_memory_info(["cuda:0"])
