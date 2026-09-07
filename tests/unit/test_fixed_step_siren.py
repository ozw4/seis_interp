from __future__ import annotations

import copy
import inspect
import math

import numpy as np
import pytest
import torch

from seis_interp.models.siren import Siren
from seis_interp.training.fixed_step_siren import (
    FixedStepSirenResult,
    train_siren_fixed_steps,
)
from seis_interp.training.point_sampler import RandomPointSampler


def _model(*, seed: int = 7, output_features: int = 1) -> Siren:
    torch.manual_seed(seed)
    return Siren(
        input_features=2,
        hidden_width=8,
        hidden_layers=1,
        output_features=output_features,
    )


def _sampler(*, seed: int = 11) -> RandomPointSampler:
    time = np.linspace(-1.0, 1.0, 5, dtype=np.float64)
    spatial = np.array([[-0.5], [0.5]], dtype=np.float64)
    amplitudes = (time[np.newaxis, :] + spatial).astype(np.float32)
    return RandomPointSampler(time, spatial, amplitudes, np.array([0, 1]), random_seed=seed)


def _train(model: Siren, sampler: RandomPointSampler, **overrides: object) -> FixedStepSirenResult:
    arguments = {
        "device": "cpu",
        "learning_rate": 1e-3,
        "batch_size": 4,
        "max_steps": 5,
        "report_interval": 2,
    }
    arguments.update(overrides)
    return train_siren_fixed_steps(model, sampler, **arguments)


def test_exact_number_of_samples_updates_and_zero_grad_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    model.eval()
    sampler = _sampler()
    original_sample = sampler.sample
    original_step = torch.optim.Adam.step
    original_zero_grad = torch.optim.Adam.zero_grad
    sample_sizes = []
    optimizer_calls = []
    zero_grad_arguments = []

    def sample(batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        sample_sizes.append(batch_size)
        return original_sample(batch_size)

    def step(optimizer: torch.optim.Adam) -> None:
        optimizer_calls.append(optimizer)
        original_step(optimizer)

    def zero_grad(optimizer: torch.optim.Adam, *, set_to_none: bool) -> None:
        zero_grad_arguments.append(set_to_none)
        original_zero_grad(optimizer, set_to_none=set_to_none)

    monkeypatch.setattr(sampler, "sample", sample)
    monkeypatch.setattr(torch.optim.Adam, "step", step)
    monkeypatch.setattr(torch.optim.Adam, "zero_grad", zero_grad)

    result = _train(model, sampler, max_steps=5, batch_size=3)

    assert result.steps_completed == 5
    assert sample_sizes == [3] * 5
    assert len(optimizer_calls) == 5
    assert all(optimizer is optimizer_calls[0] for optimizer in optimizer_calls)
    assert zero_grad_arguments == [True] * 5
    assert model.training
    assert all(parameter.device == torch.device("cpu") for parameter in model.parameters())


def test_one_update_matches_independent_adam_step() -> None:
    model = _model()
    expected_model = copy.deepcopy(model)
    coordinates, targets = _sampler().sample(4)
    coordinate_tensor = torch.tensor(coordinates, dtype=torch.float32)
    target_tensor = torch.tensor(targets, dtype=torch.float32).reshape(-1, 1)
    optimizer = torch.optim.Adam(expected_model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    loss = ((expected_model(coordinate_tensor) - target_tensor) ** 2).mean()
    expected_loss = float(loss.detach())
    loss.backward()
    optimizer.step()

    result = _train(model, _sampler(), max_steps=1)

    assert result.final_batch_loss == expected_loss
    assert result.history == ({"step": 1, "train_loss": expected_loss},)
    for name, expected in expected_model.state_dict().items():
        assert torch.equal(model.state_dict()[name], expected), name


@pytest.mark.parametrize(
    ("max_steps", "report_interval", "expected_steps"),
    [(6, 2, [2, 4, 6]), (5, 2, [2, 4, 5]), (3, 10, [3]), (3, 1, [1, 2, 3])],
)
def test_history_records_intervals_and_final_once(
    max_steps: int, report_interval: int, expected_steps: list[int]
) -> None:
    messages = []
    result = _train(
        _model(),
        _sampler(),
        max_steps=max_steps,
        report_interval=report_interval,
        reporter=messages.append,
    )

    assert [record["step"] for record in result.history] == expected_steps
    assert math.isfinite(result.final_batch_loss)
    assert all(math.isfinite(record["train_loss"]) for record in result.history)
    assert len(messages) == len(result.history)
    for message, record in zip(messages, result.history, strict=True):
        assert message == (
            f"siren_volume step {record['step']}/{max_steps}: train_loss={record['train_loss']:.8g}"
        )


def test_history_uses_float64_means_since_previous_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    losses = []
    original_forward = torch.nn.MSELoss.forward

    def forward(
        loss_function: torch.nn.MSELoss, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        loss = original_forward(loss_function, prediction, target)
        losses.append(float(loss.detach()))
        return loss

    monkeypatch.setattr(torch.nn.MSELoss, "forward", forward)
    result = _train(_model(), _sampler(), max_steps=5, report_interval=2)

    assert len(losses) == 5
    assert result.final_batch_loss == losses[-1]
    assert result.history == (
        {"step": 2, "train_loss": float(np.mean(losses[:2], dtype=np.float64))},
        {"step": 4, "train_loss": float(np.mean(losses[2:4], dtype=np.float64))},
        {"step": 5, "train_loss": losses[4]},
    )


def test_without_reporter_is_silent(capsys: pytest.CaptureFixture[str]) -> None:
    _train(_model(), _sampler(), reporter=None)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_same_model_and_sampler_seeds_reproduce_cpu_results_without_reseeding() -> None:
    first_model = _model(seed=23)
    rng_state = torch.random.get_rng_state().clone()
    first_result = _train(first_model, _sampler(seed=29))
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    second_model = _model(seed=23)
    second_result = _train(second_model, _sampler(seed=29))

    assert first_result == second_result
    for name, expected in first_model.state_dict().items():
        assert torch.equal(second_model.state_dict()[name], expected), name


@pytest.mark.parametrize("target_value", [float("nan"), float("inf")])
def test_nonfinite_loss_fails_before_backward_or_update(
    target_value: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _model()
    initial_state = copy.deepcopy(model.state_dict())
    sampler = _sampler()
    coordinates, targets = sampler.sample(4)
    targets[:] = target_value
    sample_calls = []

    def sample(batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        sample_calls.append(batch_size)
        return coordinates, targets

    def unexpected_step(optimizer: torch.optim.Adam) -> None:
        pytest.fail("optimizer must not update on non-finite loss")

    monkeypatch.setattr(sampler, "sample", sample)
    monkeypatch.setattr(torch.optim.Adam, "step", unexpected_step)

    with pytest.raises(RuntimeError, match="non-finite training loss at step 1"):
        _train(model, sampler)

    assert sample_calls == [4]
    assert all(parameter.grad is None for parameter in model.parameters())
    for name, expected in initial_state.items():
        assert torch.equal(model.state_dict()[name], expected), name


def test_model_output_shape_must_match_targets_before_backward() -> None:
    model = _model(output_features=2)

    with pytest.raises(ValueError, match="model output shape must match the target shape"):
        _train(model, _sampler())

    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("invalid", [True, False, 0, -1, 1.5, "2", None])
@pytest.mark.parametrize("name", ["batch_size", "max_steps", "report_interval"])
def test_rejects_invalid_positive_integers(name: str, invalid: object) -> None:
    with pytest.raises(ValueError, match=f"{name} must be a positive integer"):
        _train(_model(), _sampler(), **{name: invalid})


@pytest.mark.parametrize(
    "invalid", [True, False, 0.0, -1.0, float("nan"), float("inf"), "0.01", None]
)
def test_rejects_invalid_learning_rate(invalid: object) -> None:
    with pytest.raises(ValueError, match="learning_rate must be a positive finite number"):
        _train(_model(), _sampler(), learning_rate=invalid)


def test_accepts_numpy_numeric_parameters() -> None:
    result = _train(
        _model(),
        _sampler(),
        learning_rate=np.float64(1e-3),
        batch_size=np.int64(4),
        max_steps=np.int64(2),
        report_interval=np.int64(2),
    )

    assert result.steps_completed == 2


def test_rejects_non_siren_model() -> None:
    with pytest.raises(TypeError, match="model must be a Siren"):
        _train(torch.nn.Linear(2, 1), _sampler())


def test_rejects_non_point_sampler() -> None:
    with pytest.raises(TypeError, match="sampler must be a RandomPointSampler"):
        _train(_model(), object())


def test_api_has_no_validation_checkpoint_or_evaluation_callback_parameters() -> None:
    parameters = inspect.signature(train_siren_fixed_steps).parameters

    assert tuple(parameters) == (
        "model",
        "sampler",
        "device",
        "learning_rate",
        "batch_size",
        "max_steps",
        "report_interval",
        "reporter",
    )
    assert parameters["reporter"].default is None
    assert _train(_model(), _sampler(), max_steps=1).steps_completed == 1
