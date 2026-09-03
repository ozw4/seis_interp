"""Loop-control contract tests for the shared whole-shot training loop."""

from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path

import pytest
import torch

from seis_interp.training.whole_shot_training_loop import (
    MINIMUM_LEARNING_RATE_FACTOR,
    WholeShotLoopResult,
    WholeShotStepResult,
    run_whole_shot_training_loop,
)

SOURCES = 2
LEARNING_RATE = 1.0e-2


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))


def _valid_batch() -> tuple[torch.Tensor, ...]:
    return (
        torch.zeros(1, SOURCES, 8, 68, 2),
        torch.ones(1, SOURCES, 8, 68, dtype=torch.bool),
        torch.zeros(1, SOURCES, 2),
        torch.zeros(1, 2),
        torch.zeros(1, 8, 68, 2),
        torch.ones(1, 8, 68, dtype=torch.bool),
    )


class _Recorder:
    """Record every loop-visible callback interaction for one run."""

    def __init__(self, validation_values: Iterable[float]) -> None:
        self._validation_values = iter(validation_values)
        self.provider_calls: list[tuple[int, float]] = []
        self.checkpoint_calls: list[tuple[Path, int, float]] = []
        self.training_step_flags: list[bool] = []
        self.training_step_model_training: list[bool] = []
        self.validation_model_training: list[bool] = []
        self.validation_inference_mode: list[bool] = []
        self.messages: list[str] = []

    def provider(
        self,
        batch_size: int,
        *,
        generator: torch.Generator,
        neighbor_dropout: float,
    ) -> tuple[torch.Tensor, ...]:
        assert isinstance(generator, torch.Generator)
        self.provider_calls.append((batch_size, neighbor_dropout))
        return _valid_batch()

    def evaluator(self, model: torch.nn.Module) -> float:
        self.validation_model_training.append(model.training)
        self.validation_inference_mode.append(torch.is_inference_mode_enabled())
        return next(self._validation_values)

    def training_step(
        self,
        model: _TinyModel,
        batch: tuple[torch.Tensor, ...],
        use_cuda_bfloat16: bool,
    ) -> WholeShotStepResult:
        assert len(batch) == 6
        self.training_step_flags.append(use_cuda_bfloat16)
        self.training_step_model_training.append(model.training)
        loss = torch.square(model.weight).sum()
        metric = loss.detach() * 2.0
        return WholeShotStepResult(
            loss=loss,
            history_metrics=(("metric_a", metric),),
            finite_checks=(("training metric A", metric),),
        )

    def saver(
        self,
        path: Path,
        model: torch.nn.Module,
        *,
        best_step: int,
        best_validation_global_snr_db: float,
    ) -> None:
        assert isinstance(model, torch.nn.Module)
        self.checkpoint_calls.append((path, best_step, best_validation_global_snr_db))

    def report(self, message: str) -> None:
        self.messages.append(message)


def _run(
    recorder: _Recorder,
    tmp_path: Path,
    **overrides: object,
) -> WholeShotLoopResult:
    model = overrides.pop("model", _TinyModel())
    provider = overrides.pop("provider", recorder.provider)
    keywords: dict[str, object] = {
        "training_step": recorder.training_step,
        "checkpoint_saver": recorder.saver,
        "progress_name": "tiny_trainer",
        "progress_metric_labels": (("loss", "loss"), ("metric_a", "metric_a")),
        "device": "cpu",
        "generator": torch.Generator().manual_seed(0),
        "checkpoint_path": tmp_path / "best.pt",
        "total_steps": 2,
        "batch_size": 1,
        "neighbor_dropout": 0.0,
        "learning_rate": LEARNING_RATE,
        "weight_decay": 0.0,
        "validation_interval": 1,
        "use_bfloat16": False,
        "training_ffid_count": 3,
        "training_trace_count": 100,
        "reporter": recorder.report,
    }
    keywords.update(overrides)
    return run_whole_shot_training_loop(model, provider, recorder.evaluator, **keywords)


def test_step_zero_validates_checkpoints_and_reports(tmp_path: Path) -> None:
    recorder = _Recorder([1.0, 0.5, 0.25])
    result = _run(recorder, tmp_path)

    assert recorder.checkpoint_calls[0] == (tmp_path / "best.pt", 0, 1.0)
    assert result.history[0] == {
        "step": 0,
        "learning_rate": LEARNING_RATE,
        "validation_global_snr_db": 1.0,
    }
    assert recorder.messages[0] == ("tiny_trainer 0/2: oracle_per_trace_unit_rms_global_snr_db=1")


def test_final_step_is_validated_outside_the_interval(tmp_path: Path) -> None:
    recorder = _Recorder([1.0, 2.0, 3.0])
    result = _run(recorder, tmp_path, total_steps=3, validation_interval=2)

    assert [row["step"] for row in result.history] == [0, 2, 3]


def test_step_one_is_not_forced_to_validate(tmp_path: Path) -> None:
    recorder = _Recorder([1.0, 2.0])
    result = _run(recorder, tmp_path, total_steps=2, validation_interval=2)

    assert [row["step"] for row in result.history] == [0, 2]


def test_strict_improvement_updates_the_checkpoint(tmp_path: Path) -> None:
    recorder = _Recorder([1.0, 1.0, 2.0])
    result = _run(recorder, tmp_path)

    assert [(step, snr) for _, step, snr in recorder.checkpoint_calls] == [(0, 1.0), (2, 2.0)]
    assert result.best_step == 2
    assert result.best_validation_global_snr_db == 2.0


def test_equal_metric_does_not_overwrite_the_checkpoint(tmp_path: Path) -> None:
    recorder = _Recorder([1.0, 1.0, 1.0])
    result = _run(recorder, tmp_path)

    assert len(recorder.checkpoint_calls) == 1
    assert result.best_step == 0
    assert result.best_validation_global_snr_db == 1.0


def test_scheduler_advances_before_validation_and_ends_at_eta_min(tmp_path: Path) -> None:
    recorder = _Recorder([1.0, 0.5, 0.25, 0.1, 0.05])
    result = _run(recorder, tmp_path, total_steps=4)

    eta_min = LEARNING_RATE * MINIMUM_LEARNING_RATE_FACTOR
    for row in result.history[1:]:
        step = int(row["step"])
        expected = eta_min + (LEARNING_RATE - eta_min) * (1.0 + math.cos(math.pi * step / 4)) / 2.0
        assert row["learning_rate"] == pytest.approx(expected)
    assert result.history[-1]["learning_rate"] == pytest.approx(eta_min)


def test_history_rows_keep_field_order_and_float_values(tmp_path: Path) -> None:
    recorder = _Recorder([1.0, 2.0])
    result = _run(recorder, tmp_path, total_steps=1)

    assert list(result.history[0]) == ["step", "learning_rate", "validation_global_snr_db"]
    row = result.history[1]
    assert list(row) == ["step", "loss", "metric_a", "learning_rate", "validation_global_snr_db"]
    assert isinstance(row["loss"], float)
    assert isinstance(row["metric_a"], float)
    assert row["metric_a"] == pytest.approx(row["loss"] * 2.0)


def test_progress_uses_supplied_labels_in_order(tmp_path: Path) -> None:
    recorder = _Recorder([1.0, 2.0])
    result = _run(
        recorder,
        tmp_path,
        total_steps=1,
        progress_metric_labels=(("loss", "loss"), ("metric_a", "fancy")),
    )

    row = result.history[1]
    assert recorder.messages[1] == (
        f"tiny_trainer 1/1: loss={row['loss']:.8g} fancy={row['metric_a']:.8g} "
        f"learning_rate={row['learning_rate']:.8g} "
        "oracle_per_trace_unit_rms_global_snr_db=2"
    )


def test_model_switches_between_train_and_eval(tmp_path: Path) -> None:
    recorder = _Recorder([1.0, 2.0])
    _run(recorder, tmp_path, total_steps=1)

    assert recorder.training_step_model_training == [True]
    assert recorder.validation_model_training == [False, False]


def test_validation_runs_in_inference_mode(tmp_path: Path) -> None:
    recorder = _Recorder([1.0, 2.0])
    _run(recorder, tmp_path, total_steps=1)

    assert recorder.validation_inference_mode == [True, True]


def test_cpu_device_disables_cuda_bfloat16(tmp_path: Path) -> None:
    recorder = _Recorder([1.0, 2.0])
    _run(recorder, tmp_path, total_steps=1, use_bfloat16=True)

    assert recorder.training_step_flags == [False]


def test_loss_is_finite_checked_before_supplied_checks(tmp_path: Path) -> None:
    recorder = _Recorder([1.0, 2.0])

    def nan_loss_step(
        model: _TinyModel,
        batch: tuple[torch.Tensor, ...],
        use_cuda_bfloat16: bool,
    ) -> WholeShotStepResult:
        nan = model.weight.sum() * float("nan")
        return WholeShotStepResult(
            loss=nan,
            history_metrics=(),
            finite_checks=(("training metric A", nan.detach()),),
        )

    with pytest.raises(FloatingPointError, match="training loss is non-finite at step 1"):
        _run(recorder, tmp_path, training_step=nan_loss_step)


def test_supplied_finite_check_names_are_reported(tmp_path: Path) -> None:
    recorder = _Recorder([1.0, 2.0])

    def nan_metric_step(
        model: _TinyModel,
        batch: tuple[torch.Tensor, ...],
        use_cuda_bfloat16: bool,
    ) -> WholeShotStepResult:
        return WholeShotStepResult(
            loss=torch.square(model.weight).sum(),
            history_metrics=(),
            finite_checks=(("training metric A", torch.tensor(float("nan"))),),
        )

    with pytest.raises(FloatingPointError, match="training metric A is non-finite at step 1"):
        _run(recorder, tmp_path, training_step=nan_metric_step)


def test_supplied_finite_checks_run_in_order(tmp_path: Path) -> None:
    recorder = _Recorder([1.0, 2.0])

    def two_nan_checks_step(
        model: _TinyModel,
        batch: tuple[torch.Tensor, ...],
        use_cuda_bfloat16: bool,
    ) -> WholeShotStepResult:
        nan = torch.tensor(float("nan"))
        return WholeShotStepResult(
            loss=torch.square(model.weight).sum(),
            history_metrics=(),
            finite_checks=(("training first check", nan), ("training second check", nan)),
        )

    with pytest.raises(FloatingPointError, match="training first check is non-finite at step 1"):
        _run(recorder, tmp_path, training_step=two_nan_checks_step)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"total_steps": 0}, "total_steps must be a positive integer"),
        ({"batch_size": 0}, "batch_size must be a positive integer"),
        ({"validation_interval": 0}, "validation_interval must be a positive integer"),
        ({"learning_rate": 0.0}, "learning_rate must be positive"),
        ({"learning_rate": float("nan")}, "learning_rate must be a finite number"),
        ({"weight_decay": -1.0}, "weight_decay must be non-negative"),
        ({"neighbor_dropout": 1.0}, r"neighbor_dropout must be in \[0, 1\)"),
        ({"training_ffid_count": 0}, "training_ffid_count must be a positive integer"),
        ({"training_trace_count": 0}, "training_trace_count must be a positive integer"),
        ({"use_bfloat16": 1}, "use_bfloat16 must be a boolean"),
    ],
)
def test_common_invalid_arguments_are_rejected(
    tmp_path: Path,
    override: dict[str, object],
    match: str,
) -> None:
    recorder = _Recorder([1.0, 2.0])
    with pytest.raises(ValueError, match=match):
        _run(recorder, tmp_path, **override)


def test_generator_type_is_rejected(tmp_path: Path) -> None:
    recorder = _Recorder([1.0, 2.0])
    with pytest.raises(TypeError, match="generator must be a torch.Generator"):
        _run(recorder, tmp_path, generator="not a generator")


def test_shared_batch_validator_rejects_short_tuples(tmp_path: Path) -> None:
    recorder = _Recorder([1.0, 2.0])

    def short_provider(
        batch_size: int,
        *,
        generator: torch.Generator,
        neighbor_dropout: float,
    ) -> tuple[torch.Tensor, ...]:
        return _valid_batch()[:5]

    with pytest.raises(TypeError, match="batch_provider must return six tensors"):
        _run(recorder, tmp_path, provider=short_provider)


def test_result_reports_counts_and_history(tmp_path: Path) -> None:
    recorder = _Recorder([1.0, 2.0, 3.0, 4.0])
    result = _run(
        recorder,
        tmp_path,
        total_steps=3,
        training_ffid_count=7,
        training_trace_count=123,
    )

    assert isinstance(result, WholeShotLoopResult)
    assert result.steps_completed == 3
    assert result.training_ffid_count == 7
    assert result.training_trace_count == 123
    assert len(result.history) == 4
    assert recorder.provider_calls == [(1, 0.0)] * 3
