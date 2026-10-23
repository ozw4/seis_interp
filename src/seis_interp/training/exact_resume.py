"""Atomic, update-boundary recovery for long fixed-step neural fits."""

import os
import random
import tempfile
import time
from collections import deque
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch


def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def object_state(objects):
    """Small training accumulators/EMAs/samplers, never input adapters or models."""
    return {name: deepcopy(value.__dict__) for name, value in objects.items() if value is not None}


def restore_objects(objects, state):
    for name, value in state.items():
        objects[name].__dict__.update(value)


def _device_training_state(value, device):
    """Restore live accumulators on device while retaining CPU EMA snapshots."""
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {
            key: item if key == "snapshots" else _device_training_state(item, device)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list, deque)):
        return type(value)(_device_training_state(item, device) for item in value)
    return value


class ExactResume:
    """One recovery file; callers supply every method's local interval state."""

    def __init__(
        self,
        path,
        *,
        updates=2500,
        seconds=1800.0,
        reporter=None,
        checkpoint=None,
        record_all_steps=False,
    ):
        self.path = Path(path)
        self.updates = updates
        self.seconds = seconds
        self.saved_at = time.monotonic()
        self.reporter = reporter
        self.checkpoint = checkpoint
        self.record_all_steps = record_all_steps

    def load(self, model, optimizer):
        if not self.path.exists():
            return {}
        payload = torch.load(self.path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        restore_rng(payload["rng"])
        training = _device_training_state(payload["training"], next(model.parameters()).device)
        if self.reporter is not None:
            from seis_interp.training.rehearsal import state_sha256

            saved = state_sha256(
                {key: payload[key] for key in ("model", "optimizer", "rng", "training")}
            )
            loaded = state_sha256(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "rng": rng_state(),
                    "training": training,
                }
            )
            steps = [
                state["step"].device.type
                for state in optimizer.state.values()
                if isinstance(state.get("step"), torch.Tensor)
            ]
            step_device = ",".join(sorted(set(steps))) or "cpu"
            self.reporter(
                f"resumed step={payload['step']} phase={payload['phase']} "
                f"state_sha256_saved={saved} state_sha256_loaded={loaded} "
                f"adamw_step_device={step_device}"
            )
            if saved != loaded:
                raise RuntimeError("restored training state digest differs from saved state")
        return training

    def due(self, step):
        return step % self.updates == 0 or time.monotonic() - self.saved_at >= self.seconds

    def save(self, step, model, optimizer, state, *, force=False, phase="training"):
        if not force and not self.due(step):
            return
        atomic_torch_save(
            {
                "phase": phase,
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng": rng_state(),
                "training": state | {"step": step},
            },
            self.path,
        )
        self.saved_at = time.monotonic()
        if self.checkpoint is not None:
            self.checkpoint(step, phase)
