from __future__ import annotations

import weakref
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch

from seis_interp.data.trace_schema import MODEL_COORDINATE_ORDER
from seis_interp.models.siren import Siren
from seis_interp.processing.normalization import NormalizationParameters
from seis_interp.training.checkpoints import load_siren_checkpoint
from seis_interp.training.complete_trace_trainer import (
    train_siren_by_random_complete_traces,
)
from seis_interp.training.ffid_batches import RandomCompleteTraceBatchSampler


def _normalization() -> NormalizationParameters:
    return NormalizationParameters(
        coordinate_order=MODEL_COORDINATE_ORDER,
        coordinate_min=(-1.0,) * 6,
        coordinate_max=(1.0,) * 6,
        amplitude_rms=1.0,
    )


def _sampler(*, amplitude_scaling: str = "train_global_rms"):
    time = np.asarray([-1.0, 0.0, 1.0], dtype=np.float64)
    spatial = np.arange(30, dtype=np.float64).reshape(6, 5) / 30.0
    amplitudes = np.arange(1, 19, dtype=np.float32).reshape(6, 3) / 10.0
    return RandomCompleteTraceBatchSampler(
        time,
        spatial,
        amplitudes,
        np.arange(6, dtype=np.int64),
        amplitude_rms=1.0,
        random_seed=5,
        amplitude_scaling=amplitude_scaling,
    )


def _scores(values: list[float]):
    iterator: Iterator[float] = iter(values)

    def evaluate(_model: torch.nn.Module) -> float:
        return next(iterator)

    return evaluate


def test_random_complete_trace_trainer_selects_by_global_validation_and_records_training_snr(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "best.pt"
    messages: list[str] = []
    result = train_siren_by_random_complete_traces(
        Siren(hidden_width=8, hidden_layers=1),
        _sampler(amplitude_scaling="per_trace_rms"),
        _scores([1.0, 3.0, 2.0, 1.5]),
        _normalization(),
        device="cpu",
        loss="l2",
        optimizer="adam",
        learning_rate=1.0e-3,
        traces_per_update=2,
        steps_per_epoch=3,
        max_epochs=6,
        early_stopping_patience=2,
        validation_ffid_count=2,
        checkpoint_path=checkpoint,
        amplitude_scaling="per_trace_rms",
        training_evaluator=_scores([10.0, 11.0, 12.0, 13.0]),
        reporter=messages.append,
    )

    assert result.best_epoch == 2
    assert result.best_validation_global_snr_db == 3.0
    assert result.epochs_completed == 4
    assert result.global_steps == 12
    assert result.stopped_early
    assert result.training_trace_count == 6
    assert result.validation_ffid_count == 2
    assert [row["training_global_snr_db"] for row in result.history] == [
        10.0,
        11.0,
        12.0,
        13.0,
    ]
    assert [row["validation_global_snr_db"] for row in result.history] == [
        1.0,
        3.0,
        2.0,
        1.5,
    ]
    assert all("mean_trace_batch_loss" in row for row in result.history)
    loaded = load_siren_checkpoint(checkpoint)
    assert loaded.epoch == 2
    assert loaded.global_step == 6
    assert loaded.validation_median_trace_snr_db is None
    assert loaded.validation_global_snr_db == 3.0
    assert loaded.validation_metric_domain == "oracle_per_trace_unit_rms"
    assert "validation_metric_domain=oracle_per_trace_unit_rms" in messages[0]
    assert "oracle_per_trace_unit_rms_global_snr_db=" in messages[-1]


def test_random_complete_trace_trainer_rejects_more_traces_than_the_pool(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        train_siren_by_random_complete_traces(
            Siren(hidden_width=8, hidden_layers=1),
            _sampler(),
            _scores([1.0]),
            _normalization(),
            device="cpu",
            loss="l2",
            optimizer="adam",
            learning_rate=1.0e-3,
            traces_per_update=7,
            steps_per_epoch=1,
            max_epochs=1,
            early_stopping_patience=1,
            validation_ffid_count=1,
            checkpoint_path=tmp_path / "best.pt",
            reporter=lambda _message: None,
        )


def test_random_complete_trace_trainer_releases_a_batch_before_sampling_the_next(
    tmp_path: Path,
) -> None:
    class LifetimeCheckingSampler:
        training_trace_count = 2
        amplitude_scaling = "train_global_rms"

        def __init__(self) -> None:
            self.previous: (
                tuple[
                    weakref.ReferenceType[np.ndarray],
                    weakref.ReferenceType[np.ndarray],
                ]
                | None
            ) = None
            self.release_checks: list[bool] = []
            self.calls = 0

        def sample(self, _traces_per_update: int) -> tuple[np.ndarray, np.ndarray]:
            if self.previous is not None:
                self.release_checks.append(all(reference() is None for reference in self.previous))
            coordinates = np.full((3, 6), self.calls / 10.0, dtype=np.float64)
            targets = np.full(3, (self.calls + 1) / 10.0, dtype=np.float32)
            self.previous = (weakref.ref(coordinates), weakref.ref(targets))
            self.calls += 1
            return coordinates, targets

    sampler = LifetimeCheckingSampler()
    result = train_siren_by_random_complete_traces(
        Siren(hidden_width=8, hidden_layers=1),
        sampler,  # type: ignore[arg-type]
        _scores([1.0]),
        _normalization(),
        device="cpu",
        loss="l2",
        optimizer="adam",
        learning_rate=1.0e-3,
        traces_per_update=1,
        steps_per_epoch=2,
        max_epochs=1,
        early_stopping_patience=1,
        validation_ffid_count=1,
        checkpoint_path=tmp_path / "best.pt",
        reporter=lambda _message: None,
    )

    assert result.global_steps == 2
    assert sampler.release_checks == [True]
