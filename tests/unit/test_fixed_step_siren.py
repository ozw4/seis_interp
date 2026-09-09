from __future__ import annotations

import copy
import inspect
import math

import numpy as np
import pytest
import torch

from seis_interp.models.siren import Siren
from seis_interp.training import fixed_step_siren as training_module
from seis_interp.training.fixed_step_siren import (
    FixedStepSirenResult,
    train_siren_complete_trace_steps,
    train_siren_fixed_steps,
)
from seis_interp.training.point_sampler import RandomPointSampler
from seis_interp.training.trace_envelope_loss import (
    gaussian_envelope_kernels,
    trace_envelope_mse,
)


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


def _trace_arrays():
    time = np.linspace(-1.0, 1.0, 5, dtype=np.float64)
    spatial = np.array([[-0.75], [0.0], [0.75]], dtype=np.float64)
    amplitudes = (time[None, :] + 2 * spatial).astype(np.float32)
    amplitudes[0] = 0.0
    return time, spatial, amplitudes


def _train_complete(model, arrays=None, **overrides):
    arguments = {
        "device": "cpu",
        "learning_rate": 1e-3,
        "traces_per_step": None,
        "microbatch_size": 4,
        "max_steps": 3,
        "report_interval": 2,
        "random_seed": 29,
    }
    arguments.update(overrides)
    return train_siren_complete_trace_steps(
        model, *(_trace_arrays() if arrays is None else arrays), **arguments
    )


@pytest.mark.parametrize("microbatch_size", [1, 4, 8, 100])
@pytest.mark.parametrize("time_offsets", [None, np.array([-0.3, 0.2, 0.6])])
def test_complete_trace_accumulation_matches_dense_mean_gradients_and_one_adam_update(
    monkeypatch,
    microbatch_size,
    time_offsets,
):
    model = _model()
    expected_model = copy.deepcopy(model)
    time, spatial, amplitudes = _trace_arrays()
    coordinates = np.column_stack(
        [np.tile(time, len(spatial)), np.repeat(spatial, len(time), axis=0)]
    )
    if time_offsets is not None:
        coordinates[:, 0] += np.repeat(time_offsets, len(time))
    coordinate_tensor = torch.tensor(coordinates, dtype=torch.float32)
    target_tensor = torch.tensor(amplitudes.reshape(-1, 1), dtype=torch.float32)
    optimizer = torch.optim.Adam(expected_model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    dense_loss = ((expected_model(coordinate_tensor) - target_tensor) ** 2).mean()
    dense_loss.backward()
    expected_gradients = {
        name: parameter.grad.clone() for name, parameter in expected_model.named_parameters()
    }
    optimizer.step()
    actual_gradients = {}
    original_step = torch.optim.Adam.step

    def record_step(optimizer):
        actual_gradients.update(
            {name: parameter.grad.clone() for name, parameter in model.named_parameters()}
        )
        original_step(optimizer)

    monkeypatch.setattr(torch.optim.Adam, "step", record_step)
    result = _train_complete(
        model,
        microbatch_size=microbatch_size,
        max_steps=1,
        normalized_time_offsets=time_offsets,
    )

    # Float32 forward/reduction rounding varies with matrix batch shape; the
    # weighting contract is checked independently against the dense gradients.
    assert result.final_batch_loss == pytest.approx(float(dense_loss.detach()), rel=5e-7)
    assert result.history == ({"step": 1, "train_loss": result.final_batch_loss},)
    for name, expected in expected_gradients.items():
        torch.testing.assert_close(actual_gradients[name], expected, rtol=2e-5, atol=1e-6)
    for name, expected in expected_model.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("time_offsets", [None, np.array([-0.3, 0.2, 0.6])])
def test_complete_trace_selection_is_seeded_distinct_and_microbatches_cover_every_time(
    monkeypatch, time_offsets
):
    time, spatial, amplitudes = _trace_arrays()
    original_arrays = [array.copy() for array in (time, spatial, amplitudes)]
    original_tensors = training_module.to_model_tensors
    original_step = torch.optim.Adam.step
    steps, chunks = [], []

    def tensors(coordinates, targets, *, device):
        assert len(coordinates) <= 4
        chunks.append((coordinates.copy(), targets.copy()))
        return original_tensors(coordinates, targets, device=device)

    def update(optimizer):
        steps.append(chunks.copy())
        chunks.clear()
        original_step(optimizer)

    monkeypatch.setattr(training_module, "to_model_tensors", tensors)
    monkeypatch.setattr(torch.optim.Adam, "step", update)
    result = _train_complete(
        _model(),
        (time, spatial, amplitudes),
        traces_per_step=2,
        normalized_time_offsets=time_offsets,
    )

    assert result.steps_completed == 3
    assert len(steps) == 3
    rng = np.random.default_rng(29)
    for batches in steps:
        selected = rng.choice(3, size=2, replace=False)
        assert [len(coordinates) for coordinates, _ in batches] == [4, 4, 2]
        coordinates = np.concatenate([coordinates for coordinates, _ in batches])
        targets = np.concatenate([targets for _, targets in batches])
        expected_time = np.tile(time, 2)
        if time_offsets is not None:
            expected_time += np.repeat(time_offsets[selected], len(time))
        np.testing.assert_array_equal(coordinates[:, 0], expected_time)
        np.testing.assert_array_equal(
            coordinates[:, 1:], np.repeat(spatial[selected], len(time), axis=0)
        )
        np.testing.assert_array_equal(targets, amplitudes[selected].reshape(-1))
    for actual, expected in zip((time, spatial, amplitudes), original_arrays, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_full_pool_none_and_explicit_count_use_same_fixed_order_and_do_not_reseed():
    first = _model(seed=23)
    torch_state = torch.random.get_rng_state().clone()
    numpy_state = np.random.get_state()
    first_result = _train_complete(first, traces_per_step=None, random_seed=3)
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    current = np.random.get_state()
    assert current[0] == numpy_state[0]
    np.testing.assert_array_equal(current[1], numpy_state[1])
    assert current[2:] == numpy_state[2:]
    second = _model(seed=23)
    second_result = _train_complete(second, traces_per_step=3, random_seed=97)
    assert first_result == second_result
    for name, expected in first.state_dict().items():
        assert torch.equal(second.state_dict()[name], expected)


def test_complete_trace_repeated_cpu_runs_preserve_local_rng_sequence():
    first, second = _model(seed=23), _model(seed=23)
    first_result = _train_complete(first, traces_per_step=2, random_seed=31)
    second_result = _train_complete(second, traces_per_step=2, random_seed=31)
    assert first_result == second_result
    for name, expected in first.state_dict().items():
        assert torch.equal(second.state_dict()[name], expected)


def test_complete_trace_history_weights_short_microbatch_and_records_partial_intervals(monkeypatch):
    step_losses = []
    original = training_module._complete_trace_update

    def record(*args, **kwargs):
        loss = original(*args, **kwargs)
        step_losses.append(loss)
        return loss

    monkeypatch.setattr(training_module, "_complete_trace_update", record)
    messages = []
    result = _train_complete(_model(), max_steps=5, report_interval=2, reporter=messages.append)
    assert result.history == (
        {"step": 2, "train_loss": float(np.mean(step_losses[:2], dtype=np.float64))},
        {"step": 4, "train_loss": float(np.mean(step_losses[2:4], dtype=np.float64))},
        {"step": 5, "train_loss": step_losses[4]},
    )
    assert result.final_batch_loss == step_losses[-1]
    assert len(messages) == 3


def test_complete_trace_cosine_schedule_updates_once_per_optimizer_step(monkeypatch):
    rates = []
    original = torch.optim.Adam.step

    def record(optimizer):
        rates.append(optimizer.param_groups[0]["lr"])
        original(optimizer)

    monkeypatch.setattr(torch.optim.Adam, "step", record)
    result = _train_complete(
        _model(),
        learning_rate=1e-3,
        learning_rate_schedule="cosine",
        minimum_learning_rate=1e-5,
        max_steps=4,
        report_interval=1,
        microbatch_size=4,
    )
    expected = [1e-5 + (1e-3 - 1e-5) * (1 + np.cos(np.pi * step / 4)) / 2 for step in range(5)]
    assert rates == pytest.approx(expected[:4])
    assert [row["learning_rate"] for row in result.history] == pytest.approx(expected[1:])
    assert result.history[-1]["learning_rate"] == 1e-5


@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_complete_trace_nonfinite_late_microbatch_clears_partial_gradients_without_update(
    monkeypatch, invalid
):
    model = _model()
    initial = copy.deepcopy(model.state_dict())
    time, spatial, amplitudes = _trace_arrays()
    amplitudes[-1, -1] = invalid
    monkeypatch.setattr(torch.optim.Adam, "step", lambda *args: pytest.fail("invalid update"))
    with pytest.raises(RuntimeError, match="non-finite training loss at step 1"):
        _train_complete(model, (time, spatial, amplitudes), max_steps=1, microbatch_size=4)
    assert all(parameter.grad is None for parameter in model.parameters())
    for name, expected in initial.items():
        assert torch.equal(model.state_dict()[name], expected)


def test_complete_trace_invalid_output_shape_blocks_update_and_clears_gradients():
    model = _model(output_features=2)
    with pytest.raises(ValueError, match="model output shape must match the target shape"):
        _train_complete(model)
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize(
    "argument,value,match",
    [
        ("traces_per_step", 0, "traces_per_step"),
        ("traces_per_step", 4, "traces_per_step"),
        ("traces_per_step", True, "traces_per_step"),
        ("microbatch_size", 0, "microbatch_size"),
        ("microbatch_size", 1.5, "microbatch_size"),
        ("random_seed", -1, "random_seed"),
        ("random_seed", True, "random_seed"),
        ("max_steps", 0, "max_steps"),
        ("report_interval", 0, "report_interval"),
        ("learning_rate", np.inf, "learning_rate"),
        ("learning_rate_schedule", "linear", "learning_rate_schedule"),
        ("minimum_learning_rate", 1e-5, "requires"),
    ],
)
def test_complete_trace_configuration_errors_precede_model_mutation(argument, value, match):
    model = _model().eval()
    with pytest.raises(ValueError, match=match):
        _train_complete(model, **{argument: value})
    assert not model.training


@pytest.mark.parametrize("minimum", [None, 0, np.inf, 1e-3, 1e-2])
def test_complete_trace_cosine_requires_lower_positive_finite_minimum(minimum):
    with pytest.raises(ValueError, match="minimum_learning_rate"):
        _train_complete(_model(), learning_rate_schedule="cosine", minimum_learning_rate=minimum)


@pytest.mark.parametrize(
    "invalid,match",
    [
        ("time", "normalized_time"),
        ("spatial", "normalized_spatial"),
        ("targets", "normalized_amplitudes"),
        ("target_dtype", "normalized_amplitudes"),
    ],
)
def test_complete_trace_arrays_require_compact_aligned_normalized_inputs(invalid, match):
    time, spatial, amplitudes = _trace_arrays()
    if invalid == "time":
        time = time.astype(np.float32)
    elif invalid == "spatial":
        spatial = np.ones((3, 2), dtype=np.float64)
    elif invalid == "targets":
        amplitudes = amplitudes[:, :-1]
    else:
        amplitudes = amplitudes.astype(np.int64)
    with pytest.raises(ValueError, match=match):
        _train_complete(_model(), (time, spatial, amplitudes))


def test_complete_trace_api_has_no_heldout_evaluator_or_checkpoint_inputs():
    parameters = inspect.signature(train_siren_complete_trace_steps).parameters
    assert not any(
        "validation" in key or "checkpoint" in key or "evaluation" in key for key in parameters
    )


def test_complete_trace_none_time_offsets_preserve_exact_results_and_state():
    first, second = _model(seed=31), _model(seed=31)
    first_result = _train_complete(first, traces_per_step=2)
    second_result = _train_complete(second, traces_per_step=2, normalized_time_offsets=None)
    assert first_result == second_result
    for name, expected in first.state_dict().items():
        assert torch.equal(second.state_dict()[name], expected)


@pytest.mark.parametrize(
    "invalid",
    [
        [0.0, 0.0, 0.0],
        np.zeros(3, dtype=np.float32),
        np.zeros(3, dtype=np.int64),
        np.zeros(3, dtype=bool),
        np.zeros(2, dtype=np.float64),
        np.zeros((3, 1), dtype=np.float64),
        np.array(0.0, dtype=np.float64),
        np.array([0.0, np.nan, 0.0]),
        np.array([0.0, np.inf, 0.0]),
        np.array([0.0, -np.inf, 0.0]),
    ],
)
def test_complete_trace_invalid_time_offsets_precede_model_mutation(invalid):
    model = _model().eval()
    initial = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="normalized_time_offsets"):
        _train_complete(model, normalized_time_offsets=invalid)
    assert not model.training
    assert all(parameter.grad is None for parameter in model.parameters())
    for name, expected in initial.items():
        assert torch.equal(model.state_dict()[name], expected)


def _envelope_trace_arrays():
    time = np.linspace(-1.0, 1.0, 37, dtype=np.float64)
    spatial = np.array([[-0.75], [0.0], [0.75]], dtype=np.float64)
    amplitudes = np.stack([np.sin(3 * time), np.cos(5 * time), 0.5 + time])
    amplitudes /= np.sqrt(np.mean(amplitudes**2, axis=1))[:, None]
    return time, spatial, amplitudes.astype(np.float32)


@pytest.mark.parametrize("microbatch_size", [37, 80, 500])
@pytest.mark.parametrize("offsets", [None, np.array([-0.3, 0.2, 0.6])])
def test_envelope_accumulation_matches_dense_gradients_and_adam_update(
    monkeypatch, microbatch_size, offsets
):
    model = _model()
    reference = copy.deepcopy(model)
    time, spatial, amplitudes = _envelope_trace_arrays()
    coordinates = np.column_stack((np.tile(time, 3), np.repeat(spatial, len(time), axis=0)))
    if offsets is not None:
        coordinates[:, 0] += np.repeat(offsets, len(time))
    predicted = reference(torch.tensor(coordinates, dtype=torch.float32))
    target = torch.tensor(amplitudes.reshape(-1, 1))
    kernels = gaussian_envelope_kernels([4, 8], device="cpu", dtype=torch.float32)
    waveform = torch.nn.functional.mse_loss(predicted, target)
    envelope = trace_envelope_mse(predicted.reshape(3, 37), target.reshape(3, 37), kernels=kernels)
    expected_loss = waveform + 0.6 * envelope
    optimizer = torch.optim.Adam(reference.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    expected_loss.backward()
    gradients = {name: parameter.grad.clone() for name, parameter in reference.named_parameters()}
    optimizer.step()
    actual_gradients = {}
    original_step = torch.optim.Adam.step

    def record_step(optimizer):
        actual_gradients.update(
            {name: parameter.grad.clone() for name, parameter in model.named_parameters()}
        )
        original_step(optimizer)

    monkeypatch.setattr(torch.optim.Adam, "step", record_step)
    result = _train_complete(
        model,
        (time, spatial, amplitudes),
        microbatch_size=microbatch_size,
        max_steps=1,
        normalized_time_offsets=offsets,
        envelope_loss={"weight": 0.6, "sigma_samples": [4, 8], "decay_steps": 2500},
    )
    assert result.final_batch_loss == pytest.approx(float(expected_loss.detach()), rel=5e-7)
    assert result.history[0]["waveform_mse"] == pytest.approx(float(waveform.detach()), rel=5e-7)
    assert result.history[0]["weighted_envelope_mse"] == pytest.approx(
        float((0.6 * envelope).detach()), rel=5e-7
    )
    assert result.history[0]["envelope_weight"] == 0.6
    for name, expected in gradients.items():
        torch.testing.assert_close(actual_gradients[name], expected, rtol=2e-5, atol=1e-6)
    for name, expected in reference.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], expected, rtol=1e-6, atol=1e-7)


def test_envelope_zero_weight_restores_original_chunks_and_preserves_sample_selection(monkeypatch):
    time, spatial, amplitudes = _envelope_trace_arrays()
    originals = [value.copy() for value in (time, spatial, amplitudes)]
    chunks, steps = [], []
    tensor_builder = training_module.to_model_tensors
    optimizer_step = torch.optim.Adam.step
    envelope_function = training_module.trace_envelope_mse
    envelope_calls = []

    def tensors(coordinates, target, *, device):
        assert len(coordinates) <= 50
        chunks.append((coordinates.copy(), target.copy()))
        return tensor_builder(coordinates, target, device=device)

    def update(optimizer):
        steps.append(chunks.copy())
        chunks.clear()
        optimizer_step(optimizer)

    def envelope(*args, **kwargs):
        assert len(steps) == 0, "envelope must not be evaluated at zero weight"
        envelope_calls.append(args[0].shape)
        return envelope_function(*args, **kwargs)

    monkeypatch.setattr(training_module, "to_model_tensors", tensors)
    monkeypatch.setattr(torch.optim.Adam, "step", update)
    monkeypatch.setattr(training_module, "trace_envelope_mse", envelope)
    model = _model()
    torch_state = torch.random.get_rng_state().clone()
    numpy_state = np.random.get_state()
    result = _train_complete(
        model,
        (time, spatial, amplitudes),
        traces_per_step=2,
        microbatch_size=50,
        max_steps=2,
        report_interval=1,
        envelope_loss={"weight": 1.0, "sigma_samples": [4, 8], "decay_steps": 2},
    )
    assert envelope_calls == [torch.Size([1, 37]), torch.Size([1, 37])]
    assert [[len(points) for points, _ in step] for step in steps] == [[37, 37], [50, 24]]
    rng = np.random.default_rng(29)
    for chunks in steps:
        rows = rng.choice(3, size=2, replace=False)
        points = np.concatenate([points for points, _ in chunks])
        targets = np.concatenate([targets for _, targets in chunks])
        np.testing.assert_array_equal(points[:, 0], np.tile(time, 2))
        np.testing.assert_array_equal(points[:, 1:], np.repeat(spatial[rows], 37, axis=0))
        np.testing.assert_array_equal(targets, amplitudes[rows].reshape(-1))
    assert [record["envelope_weight"] for record in result.history] == [1.0, 0.0]
    assert result.history[1]["weighted_envelope_mse"] == 0.0
    assert result.history[1]["waveform_mse"] == result.history[1]["train_loss"]
    assert not any("raw_envelope" in key for record in result.history for key in record)
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    current_numpy = np.random.get_state()
    assert current_numpy[0] == numpy_state[0]
    np.testing.assert_array_equal(current_numpy[1], numpy_state[1])
    assert current_numpy[2:] == numpy_state[2:]
    for actual, expected in zip((time, spatial, amplitudes), originals, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_short_envelope_run_keeps_declared_decay_and_reports_interval_components():
    result = _train_complete(
        _model(),
        _envelope_trace_arrays(),
        microbatch_size=80,
        max_steps=10,
        report_interval=4,
        envelope_loss={"weight": 1.0, "sigma_samples": [4, 8], "decay_steps": 2500},
    )
    assert [record["step"] for record in result.history] == [4, 8, 10]
    for record in result.history:
        assert record["envelope_weight"] == pytest.approx(1 - (record["step"] - 1) / 2499)
        assert record["weighted_envelope_mse"] > 0
        assert record["train_loss"] == pytest.approx(
            record["waveform_mse"] + record["weighted_envelope_mse"], rel=1e-7
        )


def test_envelope_none_preserves_default_history_weights_and_skips_kernel_construction(monkeypatch):
    monkeypatch.setattr(
        training_module,
        "gaussian_envelope_kernels",
        lambda *args, **kwargs: pytest.fail("default must not build envelope kernels"),
    )
    first, second = _model(seed=31), _model(seed=31)
    first_result = _train_complete(first, traces_per_step=2)
    second_result = _train_complete(second, traces_per_step=2, envelope_loss=None)
    assert first_result == second_result
    assert all(set(record) == {"step", "train_loss"} for record in first_result.history)
    for name, expected in first.state_dict().items():
        assert torch.equal(second.state_dict()[name], expected)


@pytest.mark.parametrize("failure", ["radius", "microbatch", "invalid_options"])
def test_envelope_validation_precedes_model_mutation(failure):
    arrays = _envelope_trace_arrays()
    options = {"weight": 1.0, "sigma_samples": [4, 8], "decay_steps": 2500}
    microbatch = 80
    if failure == "radius":
        arrays = tuple(
            value[:, :32] if index == 2 else value[:32] if index == 0 else value
            for index, value in enumerate(arrays)
        )
    elif failure == "microbatch":
        microbatch = 36
    else:
        options["weight"] = True
    model = _model().eval()
    initial = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="envelope_loss"):
        _train_complete(model, arrays, microbatch_size=microbatch, envelope_loss=options)
    assert not model.training
    assert all(parameter.grad is None for parameter in model.parameters())
    for name, expected in initial.items():
        assert torch.equal(model.state_dict()[name], expected)


def test_nonfinite_late_envelope_microbatch_clears_partial_gradients_before_update(monkeypatch):
    time, spatial, amplitudes = _envelope_trace_arrays()
    amplitudes[-1, -1] = np.nan
    model = _model()
    initial = copy.deepcopy(model.state_dict())
    monkeypatch.setattr(torch.optim.Adam, "step", lambda *args: pytest.fail("invalid update"))
    with pytest.raises(RuntimeError, match="non-finite training loss at step 1"):
        _train_complete(
            model,
            (time, spatial, amplitudes),
            microbatch_size=80,
            max_steps=1,
            envelope_loss={"weight": 1.0, "sigma_samples": [4, 8], "decay_steps": 2500},
        )
    assert all(parameter.grad is None for parameter in model.parameters())
    for name, expected in initial.items():
        assert torch.equal(model.state_dict()[name], expected)
