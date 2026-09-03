from __future__ import annotations

import torch

from seis_interp.training.randomness import (
    EPOCH_TARGET_SAMPLING_SEED_OFFSET,
    EPOCH_WITHOUT_REPLACEMENT_TARGET_SAMPLING,
    NEIGHBOR_DROPOUT_SEED_OFFSET,
    TRAINING_AUDIT_SEED_OFFSET,
    WITH_REPLACEMENT_TARGET_SAMPLING,
    seed_global_model_initialization,
    target_sampling_seed,
)


def test_mode_strings_and_seed_offsets_are_fixed() -> None:
    assert WITH_REPLACEMENT_TARGET_SAMPLING == "with_replacement"
    assert EPOCH_WITHOUT_REPLACEMENT_TARGET_SAMPLING == "epoch_without_replacement"
    assert NEIGHBOR_DROPOUT_SEED_OFFSET == 1
    assert TRAINING_AUDIT_SEED_OFFSET == 2
    assert EPOCH_TARGET_SAMPLING_SEED_OFFSET == 3


def test_epoch_mode_uses_dedicated_target_seed() -> None:
    assert target_sampling_seed(100, "epoch_without_replacement") == 103


def test_with_replacement_mode_shares_dropout_seed() -> None:
    assert target_sampling_seed(100, "with_replacement") == 101


def test_model_initialization_seeding_is_reproducible_on_cpu() -> None:
    device = torch.device("cpu")

    seed_global_model_initialization(1234, device=device)
    first = torch.randn(8)
    seed_global_model_initialization(1234, device=device)
    second = torch.randn(8)

    torch.testing.assert_close(first, second)


def test_cpu_path_does_not_require_cuda() -> None:
    seed_global_model_initialization(7, device=torch.device("cpu"))

    assert torch.initial_seed() == 7
