from __future__ import annotations

import numpy as np
import pytest
import torch

from seis_interp.training.whole_shot_batches import RandomWholeShotBatchProvider

SOURCE_GATHER_COUNT = 2


class _FakeTensorSource:
    """Minimal source that exposes device and consumes dropout RNG like production."""

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.drawn_indices: list[np.ndarray] = []
        self.keep_masks: list[torch.Tensor] = []

    def inputs(
        self,
        targets: _FakeTargets,
        target_indices: np.ndarray,
        *,
        generator: torch.Generator | None = None,
        neighbor_dropout: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = np.asarray(target_indices, dtype=np.int64)
        if neighbor_dropout > 0.0:
            keep = (
                torch.rand(
                    (len(indices), SOURCE_GATHER_COUNT, 1, 1),
                    generator=generator,
                    device=self.device,
                )
                >= neighbor_dropout
            )
        else:
            keep = torch.ones(len(indices), SOURCE_GATHER_COUNT, 1, 1, dtype=torch.bool)
        self.drawn_indices.append(indices.copy())
        self.keep_masks.append(keep.clone())
        neighbors = torch.as_tensor(indices, dtype=torch.float32)[:, None].repeat(
            1, SOURCE_GATHER_COUNT
        )
        availability = keep[:, :, 0, 0]
        source_deltas = torch.zeros(len(indices), SOURCE_GATHER_COUNT, 2)
        target_coordinates = torch.zeros(len(indices), 2)
        return neighbors, availability, source_deltas, target_coordinates


class _FakeTargets:
    def __init__(self, ffid_count: int) -> None:
        self.ffid_count = ffid_count
        self.gathers = torch.arange(ffid_count, dtype=torch.float32)[:, None]
        self.availability = torch.eye(ffid_count, dtype=torch.bool)


def _epoch_provider(
    source: _FakeTensorSource,
    targets: _FakeTargets,
    *,
    seed: int,
) -> RandomWholeShotBatchProvider:
    return RandomWholeShotBatchProvider(
        source,
        targets,
        target_sampling="epoch_without_replacement",
        target_generator=torch.Generator(device=source.device).manual_seed(seed),
    )


def _drawn_sequence(source: _FakeTensorSource) -> list[int]:
    return [int(index) for batch in source.drawn_indices for index in batch]


def test_epoch_sampling_draws_each_ffid_once_per_epoch() -> None:
    source = _FakeTensorSource()
    provider = _epoch_provider(source, _FakeTargets(5), seed=3)
    dropout_generator = torch.Generator(device=source.device).manual_seed(0)

    for _ in range(2):
        provider(5, generator=dropout_generator, neighbor_dropout=0.0)

    sequence = _drawn_sequence(source)
    assert sorted(sequence[:5]) == [0, 1, 2, 3, 4]
    assert sorted(sequence[5:]) == [0, 1, 2, 3, 4]


def test_epoch_sequence_is_independent_of_batch_partition() -> None:
    first_source = _FakeTensorSource()
    first = _epoch_provider(first_source, _FakeTargets(4), seed=9)
    second_source = _FakeTensorSource()
    second = _epoch_provider(second_source, _FakeTargets(4), seed=9)
    dropout_generator = torch.Generator(device=torch.device("cpu")).manual_seed(0)

    for batch_size in (3, 3, 2):
        first(batch_size, generator=dropout_generator, neighbor_dropout=0.0)
    for batch_size in (4, 4):
        second(batch_size, generator=dropout_generator, neighbor_dropout=0.0)

    assert _drawn_sequence(first_source) == _drawn_sequence(second_source)


def test_epoch_target_order_is_independent_of_dropout_rng() -> None:
    first_source = _FakeTensorSource()
    first = _epoch_provider(first_source, _FakeTargets(6), seed=21)
    second_source = _FakeTensorSource()
    second = _epoch_provider(second_source, _FakeTargets(6), seed=21)

    first(6, generator=torch.Generator().manual_seed(1), neighbor_dropout=0.5)
    second(6, generator=torch.Generator().manual_seed(999), neighbor_dropout=0.0)

    assert _drawn_sequence(first_source) == _drawn_sequence(second_source)


def test_with_replacement_keeps_interleaved_rng_sequence() -> None:
    source = _FakeTensorSource()
    targets = _FakeTargets(7)
    provider = RandomWholeShotBatchProvider(
        source,
        targets,
        target_sampling="with_replacement",
        target_generator=None,
    )
    generator = torch.Generator(device=source.device).manual_seed(42)
    provider(3, generator=generator, neighbor_dropout=0.25)
    provider(3, generator=generator, neighbor_dropout=0.25)

    expected_generator = torch.Generator(device=source.device).manual_seed(42)
    for call_index in range(2):
        expected_indices = torch.randint(7, (3,), generator=expected_generator)
        expected_keep = (
            torch.rand((3, SOURCE_GATHER_COUNT, 1, 1), generator=expected_generator) >= 0.25
        )
        np.testing.assert_array_equal(
            source.drawn_indices[call_index],
            expected_indices.numpy(),
        )
        assert torch.equal(source.keep_masks[call_index], expected_keep)


def test_batch_return_order_and_target_alignment() -> None:
    source = _FakeTensorSource()
    targets = _FakeTargets(4)
    provider = _epoch_provider(source, targets, seed=5)

    batch = provider(4, generator=torch.Generator().manual_seed(0), neighbor_dropout=0.0)

    assert len(batch) == 6
    neighbors, availability, source_deltas, target_coordinates, gathers, target_masks = batch
    drawn = source.drawn_indices[0]
    torch.testing.assert_close(neighbors[:, 0], torch.as_tensor(drawn, dtype=torch.float32))
    assert availability.shape == (4, SOURCE_GATHER_COUNT)
    assert source_deltas.shape == (4, SOURCE_GATHER_COUNT, 2)
    assert target_coordinates.shape == (4, 2)
    torch.testing.assert_close(gathers, targets.gathers[torch.as_tensor(drawn)])
    assert torch.equal(target_masks, targets.availability[torch.as_tensor(drawn)])


def test_draw_count_and_unique_target_count() -> None:
    source = _FakeTensorSource()
    provider = RandomWholeShotBatchProvider(
        source,
        _FakeTargets(5),
        target_sampling="with_replacement",
        target_generator=None,
    )
    generator = torch.Generator(device=source.device).manual_seed(0)

    provider(3, generator=generator, neighbor_dropout=0.0)
    provider(3, generator=generator, neighbor_dropout=0.0)

    unique_drawn = {int(index) for batch in source.drawn_indices for index in batch}
    assert provider.draw_count == 6
    assert provider.unique_target_count == len(unique_drawn)


def test_epoch_sampling_requires_target_generator() -> None:
    with pytest.raises(TypeError, match="epoch sampling requires a target_generator"):
        RandomWholeShotBatchProvider(
            _FakeTensorSource(),
            _FakeTargets(4),
            target_sampling="epoch_without_replacement",
            target_generator=None,
        )


def test_unsupported_sampling_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported target sampling mode"):
        RandomWholeShotBatchProvider(
            _FakeTensorSource(),
            _FakeTargets(4),
            target_sampling="round_robin",
            target_generator=None,
        )
