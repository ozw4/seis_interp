from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from seis_interp.data.trace_schema import MODEL_COORDINATE_ORDER
from seis_interp.models.siren import Siren
from seis_interp.processing.normalization import NormalizationParameters
from seis_interp.training.checkpoints import load_siren_checkpoint
from seis_interp.training.model_inputs import to_model_tensors
from seis_interp.training.point_sampler import RandomPointSampler, build_trace_points
from seis_interp.training.trainer import build_loss, train_siren


def _normalization() -> NormalizationParameters:
    return NormalizationParameters(
        coordinate_order=MODEL_COORDINATE_ORDER,
        coordinate_min=(-1.0,) * 6,
        coordinate_max=(1.0,) * 6,
        amplitude_rms=1.0,
    )


def _training_inputs() -> tuple[RandomPointSampler, np.ndarray, np.ndarray]:
    time = np.linspace(-1.0, 1.0, 5, dtype=np.float64)
    spatial = np.array(
        [[-0.5, 0.2, 0.3, 0.0, 1.0], [0.5, -0.2, -0.3, 1.0, 0.0]],
        dtype=np.float64,
    )
    amplitudes = (time[np.newaxis, :] + spatial[:, :1]).astype(np.float32)
    sampler = RandomPointSampler(time, spatial, amplitudes, np.array([0]), random_seed=3)
    validation_coordinates, validation_targets = build_trace_points(
        time, spatial, amplitudes, np.array([1])
    )
    return sampler, validation_coordinates, validation_targets


@pytest.mark.parametrize(
    ("name", "expected_type"),
    [("l1", torch.nn.L1Loss), ("l2", torch.nn.MSELoss)],
)
def test_build_loss_selects_the_configured_objective(
    name: str, expected_type: type[torch.nn.Module]
) -> None:
    assert isinstance(build_loss(name), expected_type)


def test_build_loss_rejects_an_unsupported_name() -> None:
    with pytest.raises(ValueError, match="unsupported loss: huber"):
        build_loss("huber")


def test_trains_and_saves_best_checkpoint_from_whole_trace_points(tmp_path: Path) -> None:
    torch.manual_seed(4)
    model = Siren(hidden_width=8, hidden_layers=1)
    initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
    sampler, coordinates, targets = _training_inputs()
    checkpoint_path = tmp_path / "best.pt"

    result = train_siren(
        model,
        sampler,
        coordinates,
        targets,
        _normalization(),
        device="cpu",
        loss="l1",
        learning_rate=1e-3,
        batch_size=8,
        steps_per_epoch=2,
        max_epochs=3,
        early_stopping_patience=3,
        validation_batch_size=3,
        checkpoint_path=checkpoint_path,
    )

    assert result.epochs_completed == 3
    assert result.global_steps == 6
    assert len(result.history) == 3
    assert [item["global_step"] for item in result.history] == [2, 4, 6]
    assert any(not torch.equal(initial[name], value) for name, value in model.state_dict().items())
    loaded = load_siren_checkpoint(checkpoint_path)
    assert loaded.epoch == result.best_epoch
    assert loaded.validation_snr_db == result.best_validation_snr_db


def test_early_stopping_counts_consecutive_non_improvements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sampler, coordinates, targets = _training_inputs()
    snr_values = iter([1.0, 2.0, 2.0, 1.5])
    monkeypatch.setattr(
        "seis_interp.training.trainer.signal_to_noise_ratio_db",
        lambda reference, prediction: next(snr_values),
    )

    result = train_siren(
        Siren(hidden_width=4, hidden_layers=1),
        sampler,
        coordinates,
        targets,
        _normalization(),
        device="cpu",
        loss="l2",
        learning_rate=1e-3,
        batch_size=2,
        steps_per_epoch=1,
        max_epochs=10,
        early_stopping_patience=2,
        validation_batch_size=20,
        checkpoint_path=tmp_path / "best.pt",
    )

    assert result.stopped_early
    assert result.epochs_completed == 4
    assert result.best_epoch == 2
    assert result.global_steps == 4


def test_history_records_the_configured_loss(tmp_path: Path) -> None:
    torch.manual_seed(7)
    model = Siren(hidden_width=8, hidden_layers=1)
    sampler, coordinates, targets = _training_inputs()
    replayed_sampler, _, _ = _training_inputs()
    batch_coordinates, batch_targets = replayed_sampler.sample(8)
    coordinate_tensor, target_tensor = to_model_tensors(batch_coordinates, batch_targets)
    with torch.no_grad():
        expected_loss = float(torch.nn.MSELoss()(model(coordinate_tensor), target_tensor))

    result = train_siren(
        model,
        sampler,
        coordinates,
        targets,
        _normalization(),
        device="cpu",
        loss="l2",
        learning_rate=1e-3,
        batch_size=8,
        steps_per_epoch=1,
        max_epochs=1,
        early_stopping_patience=1,
        validation_batch_size=3,
        checkpoint_path=tmp_path / "best.pt",
    )

    assert result.history[0]["train_loss"] == pytest.approx(expected_loss)
