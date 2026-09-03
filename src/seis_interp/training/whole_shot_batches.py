"""Random whole-shot target batch provider shared by gather pipelines."""

from __future__ import annotations

import numpy as np
import torch

from seis_interp.data import whole_shot
from seis_interp.training import randomness


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
    ) -> tuple[torch.Tensor, ...]:
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
