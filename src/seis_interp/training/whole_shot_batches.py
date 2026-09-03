"""Whole-shot batch contract and random target batch provider for gather pipelines."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch

from seis_interp.data import whole_shot
from seis_interp.processing.c3_receiver_grid import RECEIVER_X_COUNT, RECEIVER_Y_COUNT
from seis_interp.training import randomness

WholeShotBatch = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


class WholeShotBatchProvider(Protocol):
    """Supply one random whole-shot batch using caller-owned randomness."""

    def __call__(
        self,
        batch_size: int,
        *,
        generator: torch.Generator,
        neighbor_dropout: float,
    ) -> WholeShotBatch: ...


def validated_whole_shot_batch(batch: object, *, batch_size: int) -> WholeShotBatch:
    """Check the six-tensor whole-shot batch contract shared by gather trainers."""
    if not isinstance(batch, tuple) or len(batch) != 6:
        raise TypeError(
            "batch_provider must return six tensors: (neighbors, availability, "
            "source_deltas, target_coordinates, targets, target_availability)"
        )
    names = (
        "neighbors",
        "availability",
        "source_deltas",
        "target_coordinates",
        "targets",
        "target_availability",
    )
    if not all(isinstance(value, torch.Tensor) for value in batch):
        invalid_name = next(
            name
            for name, value in zip(names, batch, strict=True)
            if not isinstance(value, torch.Tensor)
        )
        raise TypeError(f"batch {invalid_name} must be a torch.Tensor")
    neighbors, availability, source_deltas, target_coordinates, targets, target_mask = batch
    if neighbors.ndim != 5 or neighbors.shape[0] != batch_size:
        raise ValueError("batch neighbors must have shape (batch, sources, 8, 68, time)")
    _, source_count, receiver_x, receiver_y, time_count = neighbors.shape
    if (receiver_x, receiver_y) != (RECEIVER_X_COUNT, RECEIVER_Y_COUNT) or time_count < 2:
        raise ValueError("batch neighbors must have shape (batch, sources, 8, 68, time>=2)")
    if availability.shape != (batch_size, source_count, RECEIVER_X_COUNT, RECEIVER_Y_COUNT):
        raise ValueError("batch availability must match neighbor source and receiver dimensions")
    if source_deltas.shape != (batch_size, source_count, 2):
        raise ValueError("batch source_deltas must have shape (batch, sources, 2)")
    if target_coordinates.shape != (batch_size, 2):
        raise ValueError("batch target_coordinates must have shape (batch, 2)")
    expected_targets = (batch_size, RECEIVER_X_COUNT, RECEIVER_Y_COUNT, time_count)
    if targets.shape != expected_targets:
        raise ValueError(f"batch targets must have shape {expected_targets}")
    if target_mask.shape != expected_targets[:3]:
        raise ValueError("batch target_availability must match target receiver dimensions")
    for name, value in (
        ("neighbors", neighbors),
        ("source_deltas", source_deltas),
        ("target_coordinates", target_coordinates),
        ("targets", targets),
    ):
        if not value.is_floating_point():
            raise TypeError(f"batch {name} must have a floating-point dtype")
    if availability.dtype != torch.bool or target_mask.dtype != torch.bool:
        raise TypeError("batch availability tensors must have dtype torch.bool")
    return batch


class RandomWholeShotBatchProvider:
    """Sample whole TRAIN FFIDs and track exact target coverage."""

    def __init__(
        self,
        source: whole_shot.WholeShotTensorSource,
        targets: whole_shot.WholeShotTargets,
        *,
        target_sampling: str,
        target_generator: torch.Generator | None,
    ) -> None:
        self.source = source
        self.targets = targets
        self.target_sampling = target_sampling
        self._target_generator = target_generator
        self._epoch_order: torch.Tensor | None = None
        self._epoch_cursor = 0
        self._seen = np.zeros(targets.ffid_count, dtype=bool)
        self.draw_count = 0
        if target_sampling == randomness.EPOCH_WITHOUT_REPLACEMENT_TARGET_SAMPLING:
            if not isinstance(target_generator, torch.Generator):
                raise TypeError("epoch sampling requires a target_generator")
        elif target_sampling != randomness.WITH_REPLACEMENT_TARGET_SAMPLING:
            raise ValueError(f"unsupported target sampling mode: {target_sampling!r}")

    @property
    def unique_target_count(self) -> int:
        return int(np.count_nonzero(self._seen))

    def __call__(
        self,
        batch_size: int,
        *,
        generator: torch.Generator,
        neighbor_dropout: float,
    ) -> WholeShotBatch:
        if self.target_sampling == randomness.WITH_REPLACEMENT_TARGET_SAMPLING:
            target_indices = torch.randint(
                self.targets.ffid_count,
                (batch_size,),
                generator=generator,
                device=self.source.device,
            )
        else:
            target_indices = self._next_epoch_indices(batch_size)
        target_numpy = target_indices.cpu().numpy()
        self._seen[target_numpy] = True
        self.draw_count += batch_size
        neighbors, availability, source_deltas, target_coordinates = self.source.inputs(
            self.targets,
            target_numpy,
            generator=generator,
            neighbor_dropout=neighbor_dropout,
        )
        return (
            neighbors,
            availability,
            source_deltas,
            target_coordinates,
            self.targets.gathers[target_indices],
            self.targets.availability[target_indices],
        )

    def _next_epoch_indices(self, batch_size: int) -> torch.Tensor:
        if self._target_generator is None:
            raise AssertionError("validated epoch sampler is missing its generator")
        chunks: list[torch.Tensor] = []
        remaining = batch_size
        while remaining:
            if self._epoch_order is None or self._epoch_cursor == self.targets.ffid_count:
                self._epoch_order = torch.randperm(
                    self.targets.ffid_count,
                    generator=self._target_generator,
                    device=self.source.device,
                )
                self._epoch_cursor = 0
            take = min(remaining, self.targets.ffid_count - self._epoch_cursor)
            stop = self._epoch_cursor + take
            chunks.append(self._epoch_order[self._epoch_cursor : stop])
            self._epoch_cursor = stop
            remaining -= take
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)
