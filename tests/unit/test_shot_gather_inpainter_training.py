from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from seis_interp.cli import build_parser
from seis_interp.models.shot_gather_inpainter import (
    RECEIVER_X_COUNT,
    RECEIVER_Y_COUNT,
    ShotGatherInpainter,
)
from seis_interp.pipelines.train_shot_gather_inpainter import (
    _nearest_train_source_indices,
)
from seis_interp.training.shot_gather_inpainter_checkpoints import (
    load_shot_gather_inpainter_checkpoint,
    save_shot_gather_inpainter_checkpoint,
)
from seis_interp.training.shot_gather_inpainter_trainer import (
    _masked_mean_square,
    train_shot_gather_inpainter,
)


def test_checkpoint_round_trip_preserves_constructor_and_selection(tmp_path: Path) -> None:
    model = ShotGatherInpainter(
        width=8,
        temporal_dilations=(1, 2),
        stem_kernel_size=5,
        residual_kernel_size=3,
        distance_epsilon=2.0e-5,
    )
    with torch.no_grad():
        model.head[-1].bias.fill_(0.25)
    path = tmp_path / "best.pt"

    save_shot_gather_inpainter_checkpoint(
        path,
        model,
        best_step=0,
        best_validation_global_snr_db=3.5,
    )
    loaded = load_shot_gather_inpainter_checkpoint(path)

    assert loaded.best_step == 0
    assert loaded.best_validation_global_snr_db == 3.5
    assert loaded.input_feature_schema_version == 1
    assert loaded.input_feature_names == model.input_feature_names
    assert loaded.model.width == 8
    assert loaded.model.temporal_dilations == (1, 2)
    assert loaded.model.distance_epsilon == 2.0e-5
    for expected, actual in zip(
        model.state_dict().values(),
        loaded.model.state_dict().values(),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)


def test_checkpoint_rejects_changed_input_feature_order(tmp_path: Path) -> None:
    model = ShotGatherInpainter(width=8, temporal_dilations=(1,))
    path = tmp_path / "best.pt"
    save_shot_gather_inpainter_checkpoint(
        path,
        model,
        best_step=0,
        best_validation_global_snr_db=3.5,
    )
    payload = torch.load(path, weights_only=True)
    payload["input_feature_schema"]["names"] = list(
        reversed(payload["input_feature_schema"]["names"])
    )
    torch.save(payload, path)

    with pytest.raises(ValueError, match="feature names"):
        load_shot_gather_inpainter_checkpoint(path)


def test_masked_mean_square_ignores_unavailable_receiver_cells() -> None:
    values = torch.zeros(1, RECEIVER_X_COUNT, RECEIVER_Y_COUNT, 2)
    values[0, 0, 0] = torch.tensor([2.0, 4.0])
    values[0, 0, 1] = 1000.0
    mask = torch.zeros(1, RECEIVER_X_COUNT, RECEIVER_Y_COUNT, dtype=torch.bool)
    mask[0, 0, 0] = True

    result = _masked_mean_square(values, mask)

    torch.testing.assert_close(result, torch.tensor(10.0))


def test_nearest_train_sources_exclude_target_and_break_ties_by_ffid() -> None:
    train_ffids = np.asarray([30, 10, 20, 40], dtype=np.int64)
    train_sources = np.asarray([[1.0, 0.0], [-1.0, 0.0], [0.0, 0.0], [3.0, 0.0]])

    indices = _nearest_train_source_indices(
        train_ffids,
        train_sources,
        np.asarray([20, 99]),
        np.asarray([[0.0, 0.0], [0.0, 0.0]]),
        source_gather_count=2,
    )

    assert train_ffids[indices[0]].tolist() == [10, 30]
    assert train_ffids[indices[1]].tolist() == [10, 30]


def test_trainer_writes_best_checkpoint_for_masked_gather_batch(tmp_path: Path) -> None:
    model = ShotGatherInpainter(width=8, temporal_dilations=(1,))
    neighbors = torch.randn(1, 2, RECEIVER_X_COUNT, RECEIVER_Y_COUNT, 3)
    availability = torch.ones(1, 2, RECEIVER_X_COUNT, RECEIVER_Y_COUNT, dtype=torch.bool)
    source_deltas = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]])
    target_coordinates = torch.zeros(1, 2)
    targets = neighbors.mean(dim=1) + 0.1
    target_mask = torch.ones(1, RECEIVER_X_COUNT, RECEIVER_Y_COUNT, dtype=torch.bool)

    def batch_provider(
        batch_size: int,
        *,
        generator: torch.Generator,
        neighbor_dropout: float,
    ) -> tuple[torch.Tensor, ...]:
        assert batch_size == 1
        assert isinstance(generator, torch.Generator)
        assert neighbor_dropout == 0.0
        return (
            neighbors,
            availability,
            source_deltas,
            target_coordinates,
            targets,
            target_mask,
        )

    validation_values = iter((1.0, 1.25))
    result = train_shot_gather_inpainter(
        model,
        batch_provider,
        lambda _model: next(validation_values),
        device="cpu",
        generator=torch.Generator().manual_seed(4),
        checkpoint_path=tmp_path / "best.pt",
        total_steps=1,
        batch_size=1,
        neighbor_dropout=0.0,
        derivative_weight=0.1,
        learning_rate=1.0e-3,
        weight_decay=0.0,
        validation_interval=1,
        use_bfloat16=False,
        training_ffid_count=3,
        training_trace_count=100,
        reporter=lambda _message: None,
    )

    assert result.best_step == 1
    assert result.best_validation_global_snr_db == 1.25
    assert [row["step"] for row in result.history] == [0, 1]
    assert load_shot_gather_inpainter_checkpoint(tmp_path / "best.pt").best_step == 1


def test_trainer_can_keep_step_zero_and_does_not_force_step_one_validation(
    tmp_path: Path,
) -> None:
    model = ShotGatherInpainter(width=8, temporal_dilations=(1,))
    neighbors = torch.randn(1, 2, RECEIVER_X_COUNT, RECEIVER_Y_COUNT, 3)
    availability = torch.ones(1, 2, RECEIVER_X_COUNT, RECEIVER_Y_COUNT, dtype=torch.bool)
    source_deltas = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]])
    target_coordinates = torch.zeros(1, 2)
    targets = neighbors.mean(dim=1) + 0.1
    target_mask = torch.ones(1, RECEIVER_X_COUNT, RECEIVER_Y_COUNT, dtype=torch.bool)
    validation_values = iter((2.0, 1.0))

    result = train_shot_gather_inpainter(
        model,
        lambda _batch_size, *, generator, neighbor_dropout: (
            neighbors,
            availability,
            source_deltas,
            target_coordinates,
            targets,
            target_mask,
        ),
        lambda _model: next(validation_values),
        device="cpu",
        generator=torch.Generator().manual_seed(4),
        checkpoint_path=tmp_path / "best.pt",
        total_steps=2,
        batch_size=1,
        neighbor_dropout=0.0,
        derivative_weight=0.1,
        learning_rate=1.0e-3,
        weight_decay=0.0,
        validation_interval=2,
        use_bfloat16=False,
        training_ffid_count=3,
        training_trace_count=100,
        reporter=lambda _message: None,
    )

    assert result.best_step == 0
    assert result.best_validation_global_snr_db == 2.0
    assert [row["step"] for row in result.history] == [0, 2]
    assert "loss" not in result.history[0]
    assert load_shot_gather_inpainter_checkpoint(tmp_path / "best.pt").best_step == 0


def test_cli_parser_exposes_shot_gather_command_without_overwrite() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "train",
            "shot-gather-inpainter",
            "--config",
            "config.yaml",
            "--interim",
            "interim",
            "--processed",
            "processed",
            "--output",
            "run",
        ]
    )

    assert args.train_command == "shot-gather-inpainter"
    assert not hasattr(args, "overwrite")
