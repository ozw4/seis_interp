from __future__ import annotations

import numpy as np
import pytest
import torch

from seis_interp.pipelines.train_neighbor_inpainter import (
    EPOCH_WITHOUT_REPLACEMENT_TARGET_SAMPLING,
    _NeighborTensorSource,
    _RandomTrainBatchProvider,
)


class _FakeTensorSource:
    """Minimal tensor source that exposes target ids and consumes dropout RNG."""

    def __init__(self, trace_count: int) -> None:
        self.train_positions = np.arange(trace_count, dtype=np.int64) + 100
        target_ids = torch.arange(trace_count, dtype=torch.float32)
        self.train_amplitudes = target_ids[:, None].repeat(1, 2)
        self.device = torch.device("cpu")

    def gather(
        self,
        target_positions: np.ndarray,
        *,
        generator: torch.Generator | None = None,
        neighbor_dropout: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = len(target_positions)
        availability = torch.ones(batch_size, 3, dtype=torch.bool)
        if neighbor_dropout > 0.0:
            if generator is None:
                raise AssertionError("test source requires a dropout generator")
            availability &= torch.rand((batch_size, 3), generator=generator) >= neighbor_dropout
        neighbors = torch.zeros(batch_size, 3, 2)
        coordinates = torch.zeros(batch_size, 3)
        return neighbors, availability, coordinates


class _FakeGeometryLookup:
    row_count = 3
    offsets = (
        (1, 0, 0, 0),
        (0, 1, 1, 0),
        (0, 0, 0, 1),
    )

    def neighbor_positions(self, target_positions: np.ndarray) -> np.ndarray:
        return np.tile(np.asarray([0, 1, 2], dtype=np.int64), (len(target_positions), 1))

    def target_coordinates(self, target_positions: np.ndarray) -> np.ndarray:
        return np.zeros((len(target_positions), 4), dtype=np.float64)


def _epoch_provider(trace_count: int, seed: int) -> _RandomTrainBatchProvider:
    return _RandomTrainBatchProvider(  # type: ignore[arg-type]
        _FakeTensorSource(trace_count),
        target_sampling=EPOCH_WITHOUT_REPLACEMENT_TARGET_SAMPLING,
        target_generator=torch.Generator().manual_seed(seed),
    )


def _draw_target_ids(
    provider: _RandomTrainBatchProvider,
    batch_sizes: list[int],
    *,
    dropout_generator: torch.Generator,
    neighbor_dropout: float = 0.0,
) -> tuple[list[int], list[torch.Tensor]]:
    target_ids: list[int] = []
    availability: list[torch.Tensor] = []
    for batch_size in batch_sizes:
        _neighbors, batch_availability, _coordinates, targets = provider(
            batch_size,
            generator=dropout_generator,
            neighbor_dropout=neighbor_dropout,
        )
        target_ids.extend(int(value) for value in targets[:, 0])
        availability.append(batch_availability)
    return target_ids, availability


def test_epoch_sampler_covers_each_target_once_per_epoch_across_batch_wrap() -> None:
    provider = _epoch_provider(trace_count=5, seed=47)

    target_ids, _availability = _draw_target_ids(
        provider,
        [3, 4, 3],
        dropout_generator=torch.Generator().manual_seed(11),
    )

    assert sorted(target_ids[:5]) == list(range(5))
    assert sorted(target_ids[5:10]) == list(range(5))
    assert len(set(target_ids[:5])) == 5
    assert len(set(target_ids[5:10])) == 5
    assert provider.draw_count == 10
    assert provider.unique_target_count == 5


def test_epoch_sampler_sequence_is_deterministic_and_batch_partition_independent() -> None:
    first = _epoch_provider(trace_count=7, seed=53)
    second = _epoch_provider(trace_count=7, seed=53)

    first_ids, _ = _draw_target_ids(
        first,
        [2, 8, 5],
        dropout_generator=torch.Generator().manual_seed(1),
    )
    second_ids, _ = _draw_target_ids(
        second,
        [3, 1, 6, 5],
        dropout_generator=torch.Generator().manual_seed(1),
    )

    assert first_ids == second_ids


def test_epoch_target_sequence_is_independent_of_neighbor_dropout_rng() -> None:
    first = _epoch_provider(trace_count=11, seed=59)
    second = _epoch_provider(trace_count=11, seed=59)
    first_dropout = torch.Generator().manual_seed(13)
    second_dropout = torch.Generator().manual_seed(13)
    torch.rand(101, generator=second_dropout)

    first_ids, _ = _draw_target_ids(
        first,
        [4, 9, 7],
        dropout_generator=first_dropout,
        neighbor_dropout=0.5,
    )
    second_ids, _ = _draw_target_ids(
        second,
        [4, 9, 7],
        dropout_generator=second_dropout,
        neighbor_dropout=0.5,
    )

    assert first_ids == second_ids


def test_default_sampler_preserves_legacy_shared_generator_sequence() -> None:
    source = _FakeTensorSource(trace_count=13)
    provider = _RandomTrainBatchProvider(source)  # type: ignore[arg-type]
    actual_generator = torch.Generator().manual_seed(43)
    expected_generator = torch.Generator().manual_seed(43)

    expected_targets: list[int] = []
    expected_availability: list[torch.Tensor] = []
    for batch_size in (4, 7, 2):
        expected_targets.extend(
            int(value)
            for value in torch.randint(
                13,
                (batch_size,),
                generator=expected_generator,
            )
        )
        expected_availability.append(
            torch.rand((batch_size, 3), generator=expected_generator) >= 0.4
        )

    actual_targets, actual_availability = _draw_target_ids(
        provider,
        [4, 7, 2],
        dropout_generator=actual_generator,
        neighbor_dropout=0.4,
    )

    assert actual_targets == expected_targets
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(
            actual_availability,
            expected_availability,
            strict=True,
        )
    )
    assert torch.equal(actual_generator.get_state(), expected_generator.get_state())


def test_epoch_sampler_requires_a_dedicated_target_generator() -> None:
    with pytest.raises(TypeError, match="target_generator is required"):
        _RandomTrainBatchProvider(  # type: ignore[arg-type]
            _FakeTensorSource(3),
            target_sampling=EPOCH_WITHOUT_REPLACEMENT_TARGET_SAMPLING,
        )


def test_tensor_source_can_exclude_every_neighbor_from_the_target_ffid() -> None:
    amplitudes = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    source = _NeighborTensorSource(
        _FakeGeometryLookup(),  # type: ignore[arg-type]
        train_positions=np.arange(3, dtype=np.int64),
        train_amplitudes=amplitudes,
        device=torch.device("cpu"),
        exclude_target_ffid_neighbors=True,
        ffids_by_position=np.asarray([10, 11, 10], dtype=np.int64),
    )

    neighbors, availability, _coordinates = source.gather(np.asarray([0], dtype=np.int64))

    assert availability.tolist() == [[False, True, False]]
    torch.testing.assert_close(
        neighbors,
        torch.tensor([[[0.0, 0.0], [2.0, 2.0], [0.0, 0.0]]]),
    )
