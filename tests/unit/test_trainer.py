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

SAMPLES_PER_TRACE = 5


def _normalization() -> NormalizationParameters:
    return NormalizationParameters(
        coordinate_order=MODEL_COORDINATE_ORDER,
        coordinate_min=(-1.0,) * 6,
        coordinate_max=(1.0,) * 6,
        amplitude_rms=1.0,
    )


def _training_inputs() -> tuple[RandomPointSampler, np.ndarray, np.ndarray]:
    time = np.linspace(-1.0, 1.0, SAMPLES_PER_TRACE, dtype=np.float64)
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
        validation_samples_per_trace=SAMPLES_PER_TRACE,
        checkpoint_path=checkpoint_path,
    )

    assert result.epochs_completed == 3
    assert result.global_steps == 6
    assert len(result.history) == 3
    assert [item["global_step"] for item in result.history] == [2, 4, 6]
    assert any(not torch.equal(initial[name], value) for name, value in model.state_dict().items())
    loaded = load_siren_checkpoint(checkpoint_path)
    assert loaded.epoch == result.best_epoch
    assert loaded.validation_median_trace_snr_db == result.best_validation_median_trace_snr_db
    assert loaded.validation_global_snr_db == result.best_validation_global_snr_db


def test_selection_and_early_stopping_follow_the_median_trace_metric(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sampler, coordinates, targets = _training_inputs()
    median_values = iter([1.0, 2.0, 2.0, 1.5])
    global_values = iter([10.0, 5.0, 20.0, 30.0])
    monkeypatch.setattr(
        "seis_interp.training.trainer.median_trace_signal_to_noise_ratio_db",
        lambda reference, prediction: next(median_values),
    )
    monkeypatch.setattr(
        "seis_interp.training.trainer.signal_to_noise_ratio_db",
        lambda reference, prediction: next(global_values),
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
        validation_samples_per_trace=SAMPLES_PER_TRACE,
        checkpoint_path=tmp_path / "best.pt",
    )

    assert result.stopped_early
    assert result.epochs_completed == 4
    assert result.global_steps == 4
    assert result.best_epoch == 2
    assert result.best_validation_median_trace_snr_db == 2.0
    assert result.best_validation_global_snr_db == 5.0
    assert [item["validation_median_trace_snr_db"] for item in result.history] == [
        1.0,
        2.0,
        2.0,
        1.5,
    ]
    assert [item["validation_global_snr_db"] for item in result.history] == [
        10.0,
        5.0,
        20.0,
        30.0,
    ]
    assert all("validation_snr_db" not in item for item in result.history)
    assert load_siren_checkpoint(tmp_path / "best.pt").validation_global_snr_db == 5.0


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
        validation_samples_per_trace=SAMPLES_PER_TRACE,
        checkpoint_path=tmp_path / "best.pt",
    )

    assert result.history[0]["train_loss"] == pytest.approx(expected_loss)


def test_rejects_a_validation_point_count_that_is_not_whole_traces(tmp_path: Path) -> None:
    sampler, coordinates, targets = _training_inputs()

    with pytest.raises(ValueError, match="divisible by validation_samples_per_trace"):
        train_siren(
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
            max_epochs=1,
            early_stopping_patience=1,
            validation_batch_size=20,
            validation_samples_per_trace=SAMPLES_PER_TRACE - 1,
            checkpoint_path=tmp_path / "best.pt",
        )


def test_rejects_checkpoint_scaling_that_does_not_match_the_sampler(tmp_path: Path) -> None:
    sampler, coordinates, targets = _training_inputs()

    with pytest.raises(ValueError, match="must match the RandomPointSampler"):
        train_siren(
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
            max_epochs=1,
            early_stopping_patience=1,
            validation_batch_size=20,
            validation_samples_per_trace=SAMPLES_PER_TRACE,
            checkpoint_path=tmp_path / "best.pt",
            amplitude_scaling="per_trace_rms",
        )
