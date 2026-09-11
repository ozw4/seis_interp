"""AMP capability validation rejects fallback and checks the requested device."""

from contextlib import contextmanager

import pytest
import torch

from seis_interp.training.mixed_precision import mixed_precision_dtype


def test_bf16_capability_is_checked_on_selected_device(monkeypatch):
    events = []

    @contextmanager
    def device(selected):
        events.append(str(selected))
        yield
        events.append("restored")

    def supported():
        assert events == ["cuda:1"]
        return False

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device", device)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", supported)
    with pytest.raises(ValueError, match="bf16 is unsupported"):
        mixed_precision_dtype("bf16", "cuda:1")


@pytest.mark.parametrize("mode", ["fp16", "bf16"])
def test_cuda_precision_without_cuda_fails_instead_of_disabling_scaler(monkeypatch, mode):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="available CUDA"):
        mixed_precision_dtype(mode, "cuda")


def test_off_does_not_probe_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("off must not probe CUDA"))
    assert mixed_precision_dtype("off", "cpu") is None
