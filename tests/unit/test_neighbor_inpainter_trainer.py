from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from torch import nn

from seis_interp.training import neighbor_inpainter_trainer
from seis_interp.training.neighbor_inpainter_trainer import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DERIVATIVE_WEIGHT,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NEIGHBOR_DROPOUT,
    DEFAULT_TOTAL_STEPS,
    DEFAULT_VALIDATION_INTERVAL,
    DEFAULT_WEIGHT_DECAY,
    MAX_GRADIENT_NORM,
    MINIMUM_LEARNING_RATE_FACTOR,
    train_neighbor_trace_inpainter,
)


class _TinyInpainter(nn.Module):
    neighbor_count = 2
    width = 8

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.25))

    def forward(
        self,
        neighbors: torch.Tensor,
        availability: torch.Tensor,
        target_coordinates: torch.Tensor,
    ) -> torch.Tensor:
        del availability
        return self.scale * neighbors[:, 0] + target_coordinates[:, :1]


def _batch(batch_size: int = 2) -> tuple[torch.Tensor, ...]:
    neighbors = (
        torch.arange(batch_size * 2 * 4, dtype=torch.float32).reshape(batch_size, 2, 4) / 10.0
    )
    availability = torch.ones(batch_size, 2)
    coordinates = torch.zeros(batch_size, 3)
    targets = neighbors[:, 0] * 0.75
    return neighbors, availability, coordinates, targets


def _score_sequence(values: list[float]) -> tuple[object, list[bool]]:
    iterator: Iterator[float] = iter(values)
    observed_training_modes: list[bool] = []

    def evaluate(model: nn.Module) -> float:
        observed_training_modes.append(model.training)
        return next(iterator)

    return evaluate, observed_training_modes


def test_successful_proxy_defaults_are_frozen() -> None:
    assert DEFAULT_TOTAL_STEPS == 2500
    assert DEFAULT_BATCH_SIZE == 96
    assert DEFAULT_NEIGHBOR_DROPOUT == 0.05
    assert DEFAULT_DERIVATIVE_WEIGHT == 0.1
    assert DEFAULT_LEARNING_RATE == 5.0e-4
    assert DEFAULT_WEIGHT_DECAY == 1.0e-5
    assert DEFAULT_VALIDATION_INTERVAL == 100
    assert MINIMUM_LEARNING_RATE_FACTOR == 0.03
    assert MAX_GRADIENT_NORM == 1.0


def test_trainer_uses_caller_generator_schedule_and_raw_validation_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _TinyInpainter()
    generator = torch.Generator().manual_seed(43)
    provider_calls: list[tuple[int, torch.Generator, float]] = []
    saved: list[tuple[int, float, float]] = []
    reports: list[str] = []

    def provide(
        batch_size: int,
        *,
        generator: torch.Generator,
        neighbor_dropout: float,
    ) -> tuple[torch.Tensor, ...]:
        provider_calls.append((batch_size, generator, neighbor_dropout))
        return _batch(batch_size)

    evaluator, observed_modes = _score_sequence([1.0, 3.0, 2.0, 2.5])

    def record_checkpoint(
        _path: Path,
        saved_model: nn.Module,
        *,
        best_step: int,
        best_validation_global_snr_db: float,
    ) -> None:
        saved.append(
            (
                best_step,
                best_validation_global_snr_db,
                float(saved_model.scale.detach()),  # type: ignore[attr-defined]
            )
        )

    monkeypatch.setattr(
        neighbor_inpainter_trainer,
        "save_neighbor_inpainter_checkpoint",
        record_checkpoint,
    )
    result = train_neighbor_trace_inpainter(
        model,  # type: ignore[arg-type]
        provide,  # type: ignore[arg-type]
        evaluator,  # type: ignore[arg-type]
        device="cpu",
        generator=generator,
        checkpoint_path=tmp_path / "best.pt",
        total_steps=5,
        batch_size=2,
        validation_interval=2,
        training_trace_count=17,
        use_bfloat16=True,
        reporter=reports.append,
    )

    assert [row["step"] for row in result.history] == [1, 2, 4, 5]
    assert [row["validation_global_snr_db"] for row in result.history] == [1.0, 3.0, 2.0, 2.5]
    expected_learning_rates = [
        DEFAULT_LEARNING_RATE
        * (
            MINIMUM_LEARNING_RATE_FACTOR
            + (1.0 - MINIMUM_LEARNING_RATE_FACTOR) * (1.0 + math.cos(math.pi * step / 5)) / 2.0
        )
        for step in (1, 2, 4, 5)
    ]
    assert [row["learning_rate"] for row in result.history] == pytest.approx(
        expected_learning_rates
    )
    assert all(
        math.isfinite(float(row[name]))
        for row in result.history
        for name in ("loss", "mse", "derivative_mse", "learning_rate")
    )
    assert result.best_step == 2
    assert result.best_validation_global_snr_db == 3.0
    assert result.steps_completed == 5
    assert result.training_trace_count == 17
    assert [entry[:2] for entry in saved] == [(1, 1.0), (2, 3.0)]
    assert len(provider_calls) == 5
    assert all(call == (2, generator, DEFAULT_NEIGHBOR_DROPOUT) for call in provider_calls)
    assert observed_modes == [False] * 4
    assert len(reports) == 4
    assert all("oracle_per_trace_unit_rms_global_snr_db" in report for report in reports)


def test_final_step_is_validated_once_when_it_is_on_the_interval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator, _modes = _score_sequence([1.0, 1.0, 1.0])
    saved_steps: list[int] = []
    monkeypatch.setattr(
        neighbor_inpainter_trainer,
        "save_neighbor_inpainter_checkpoint",
        lambda _path, _model, *, best_step, best_validation_global_snr_db: saved_steps.append(
            best_step
        ),
    )

    result = train_neighbor_trace_inpainter(
        _TinyInpainter(),  # type: ignore[arg-type]
        lambda batch_size, *, generator, neighbor_dropout: _batch(batch_size),  # type: ignore[arg-type]
        evaluator,  # type: ignore[arg-type]
        device="cpu",
        generator=torch.Generator().manual_seed(1),
        checkpoint_path=tmp_path / "best.pt",
        total_steps=4,
        batch_size=2,
        validation_interval=2,
        reporter=lambda _message: None,
    )

    assert [row["step"] for row in result.history] == [1, 2, 4]
    assert saved_steps == [1]


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"total_steps": 0}, ValueError, "total_steps"),
        ({"batch_size": True}, ValueError, "batch_size"),
        ({"validation_interval": 0}, ValueError, "validation_interval"),
        ({"learning_rate": 0.0}, ValueError, "learning_rate"),
        ({"learning_rate": math.inf}, ValueError, "learning_rate"),
        ({"weight_decay": -1.0}, ValueError, "weight_decay"),
        ({"neighbor_dropout": -0.1}, ValueError, "neighbor_dropout"),
        ({"neighbor_dropout": 1.0}, ValueError, "neighbor_dropout"),
        ({"derivative_weight": math.nan}, ValueError, "derivative_weight"),
        ({"use_bfloat16": 1}, ValueError, "use_bfloat16"),
        ({"training_trace_count": 0}, ValueError, "training_trace_count"),
        ({"generator": object()}, TypeError, "generator"),
    ],
)
def test_trainer_rejects_invalid_settings(
    tmp_path: Path,
    overrides: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "device": "cpu",
        "generator": torch.Generator().manual_seed(1),
        "checkpoint_path": tmp_path / "best.pt",
        "total_steps": 1,
        "batch_size": 2,
        "reporter": lambda _message: None,
    }
    arguments.update(overrides)

    with pytest.raises(error, match=message):
        train_neighbor_trace_inpainter(
            _TinyInpainter(),  # type: ignore[arg-type]
            lambda batch_size, *, generator, neighbor_dropout: _batch(batch_size),  # type: ignore[arg-type]
            lambda _model: 1.0,
            **arguments,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf])
def test_trainer_rejects_nonfinite_validation_scores(tmp_path: Path, score: float) -> None:
    with pytest.raises(ValueError, match="validation global S/N must be a finite number"):
        train_neighbor_trace_inpainter(
            _TinyInpainter(),  # type: ignore[arg-type]
            lambda batch_size, *, generator, neighbor_dropout: _batch(batch_size),  # type: ignore[arg-type]
            lambda _model: score,
            device="cpu",
            generator=torch.Generator().manual_seed(1),
            checkpoint_path=tmp_path / "best.pt",
            total_steps=1,
            batch_size=2,
            reporter=lambda _message: None,
        )


def test_trainer_rejects_nonfinite_training_loss(tmp_path: Path) -> None:
    def nonfinite_batch(batch_size: int, **_kwargs: object) -> tuple[torch.Tensor, ...]:
        neighbors, availability, coordinates, targets = _batch(batch_size)
        targets[0, 0] = math.nan
        return neighbors, availability, coordinates, targets

    with pytest.raises(RuntimeError, match="non-finite training loss at step 1"):
        train_neighbor_trace_inpainter(
            _TinyInpainter(),  # type: ignore[arg-type]
            nonfinite_batch,  # type: ignore[arg-type]
            lambda _model: 1.0,
            device="cpu",
            generator=torch.Generator().manual_seed(1),
            checkpoint_path=tmp_path / "best.pt",
            total_steps=1,
            batch_size=2,
            reporter=lambda _message: None,
        )


@pytest.mark.parametrize(
    ("batch", "error", "message"),
    [
        ((torch.ones(2, 2, 4),), TypeError, "four-tensor tuple"),
        (
            (torch.ones(2, 2, 1), torch.ones(2, 2), torch.ones(2, 3), torch.ones(2, 1)),
            ValueError,
            "at least two time samples",
        ),
        (
            (torch.ones(2, 2, 4), torch.ones(2, 2), torch.ones(2, 3), torch.ones(2, 3)),
            ValueError,
            "batch targets",
        ),
        (
            (
                torch.ones(2, 2, 4, dtype=torch.int64),
                torch.ones(2, 2),
                torch.ones(2, 3),
                torch.ones(2, 4),
            ),
            TypeError,
            "neighbors.*floating-point",
        ),
    ],
)
def test_trainer_rejects_invalid_batches(
    tmp_path: Path,
    batch: tuple[torch.Tensor, ...],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        train_neighbor_trace_inpainter(
            _TinyInpainter(),  # type: ignore[arg-type]
            lambda _batch_size, *, generator, neighbor_dropout: batch,  # type: ignore[arg-type]
            lambda _model: 1.0,
            device="cpu",
            generator=torch.Generator().manual_seed(1),
            checkpoint_path=tmp_path / "best.pt",
            total_steps=1,
            batch_size=2,
            reporter=lambda _message: None,
        )
