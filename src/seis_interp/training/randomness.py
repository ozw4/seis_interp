"""Shared target-sampling modes, seed offsets, and model-initialization seeding."""

from __future__ import annotations

import torch

WITH_REPLACEMENT_TARGET_SAMPLING = "with_replacement"
EPOCH_WITHOUT_REPLACEMENT_TARGET_SAMPLING = "epoch_without_replacement"

NEIGHBOR_DROPOUT_SEED_OFFSET = 1
TRAINING_AUDIT_SEED_OFFSET = 2
EPOCH_TARGET_SAMPLING_SEED_OFFSET = 3


def target_sampling_seed(random_seed: int, target_sampling: str) -> int:
    """Return the target-draw seed for a pipeline-validated sampling mode."""
    if target_sampling == EPOCH_WITHOUT_REPLACEMENT_TARGET_SAMPLING:
        return random_seed + EPOCH_TARGET_SAMPLING_SEED_OFFSET
    return random_seed + NEIGHBOR_DROPOUT_SEED_OFFSET


def seed_global_model_initialization(random_seed: int, *, device: torch.device) -> None:
    torch.manual_seed(random_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
