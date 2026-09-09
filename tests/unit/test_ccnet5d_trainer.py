from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from seis_interp.models.ccnet5d import CCNet5D
from seis_interp.training import ccnet5d_trainer
from seis_interp.training.ccnet5d_trainer import train_ccnet5d

PATCH_SHAPE = (1, 2, 1, 1, 1)
OBSERVED_MASK = np.array([True, False], dtype=np.bool_).reshape(PATCH_SHAPE[1:])


def _model(seed: int = 7) -> CCNet5D:
    torch.manual_seed(seed)
    return CCNet5D(
        hidden_channels=1,
        intermediate_channels=1,
        kernel_size=1,
        output_activation="linear",
    )


def _plan(fit_count: int) -> SimpleNamespace:
    return SimpleNamespace(
        patch_shape=PATCH_SHAPE,
        fit=tuple(range(fit_count)),
        selection=(0,),
    )


def _patch(index: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inputs = np.array([1.0 + index, 0.0], dtype=np.float32).reshape(PATCH_SHAPE)
    labels = np.array([-1.0, 0.5 + index], dtype=np.float32).reshape(PATCH_SHAPE)
    return inputs, labels, OBSERVED_MASK.copy()


def _metrics(relative_error: float = 0.25) -> dict[str, object]:
    return {
        "patch_count": 1,
        "missing_sample_count": 1,
        "reference_energy": 4.0,
        "error_energy": 4.0 * relative_error,
        "relative_squared_error": relative_error,
        "snr_db": float(-10.0 * math.log10(relative_error)) if relative_error else None,
        "snr_status": "finite" if relative_error else "perfect_reconstruction",
        "metric_scope": "selection_patch_instances_missing_only",
    }


def _train(
    model: CCNet5D,
    plan: object,
    **overrides: object,
) -> object:
    arguments = {
        "device": "cpu",
        "random_seed": 13,
        "max_epochs": 1,
        "learning_rate": 1e-3,
        "decay_after_epochs": 2,
        "decay_factor": 0.1,
        "validate_every_steps": 2,
        "report_every_steps": 2,
    }
    arguments.update(overrides)
    return train_ccnet5d(model, SimpleNamespace(amplitude_rms=2.0), plan, **arguments)


def test_one_update_matches_independent_adam_mse_on_complete_signed_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    expected_model = copy.deepcopy(model)
    inputs, labels, mask = _patch()
    monkeypatch.setattr(
        ccnet5d_trainer,
        "load_ccnet_patch",
        lambda *args, **kwargs: (inputs.copy(), labels.copy(), mask.copy()),
    )
    monkeypatch.setattr(
        ccnet5d_trainer, "evaluate_ccnet5d_selection", lambda *args, **kwargs: _metrics()
    )
    optimizer = torch.optim.Adam(expected_model.parameters(), lr=1e-3)
    input_tensor = torch.from_numpy(inputs[None, None])
    target_tensor = torch.from_numpy(labels[None, None])
    optimizer.zero_grad(set_to_none=True)
    expected_loss = torch.nn.functional.mse_loss(expected_model(input_tensor), target_tensor)
    expected_loss.backward()
    optimizer.step()

    result = _train(
        model,
        _plan(1),
        validate_every_steps=10,
        report_every_steps=1,
    )

    assert result.steps_completed == 1
    assert result.training_history == (
        {
            "epoch": 1,
            "step": 1,
            "train_loss": float(expected_loss.detach()),
            "learning_rate": 1e-3,
        },
    )
    for name, expected in expected_model.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], expected, rtol=0, atol=0)


def test_each_epoch_visits_every_descriptor_once_in_reproducible_shuffled_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ccnet5d_trainer, "evaluate_ccnet5d_selection", lambda *args, **kwargs: _metrics()
    )
    plan = _plan(4)
    original_plan = copy.deepcopy(plan)

    def run_once() -> tuple[list[int], object]:
        visited = []

        def load(source: object, plan: object, *, region: str, index: int):
            assert region == "fit"
            visited.append(index)
            return _patch(index)

        monkeypatch.setattr(ccnet5d_trainer, "load_ccnet_patch", load)
        return visited, _train(_model(), plan, random_seed=23, max_epochs=2)

    first_order, first = run_once()
    second_order, second = run_once()
    expected_rng = np.random.default_rng(23)
    expected_order = [
        *expected_rng.permutation(4).tolist(),
        *expected_rng.permutation(4).tolist(),
    ]

    assert first_order == second_order == expected_order
    assert sorted(first_order[:4]) == sorted(first_order[4:]) == [0, 1, 2, 3]
    assert first.steps_completed == second.steps_completed == 8
    assert plan == original_plan
    for name, expected in first.best_state_dict.items():
        torch.testing.assert_close(second.best_state_dict[name], expected, rtol=0, atol=0)


def test_total_optimizer_steps_equal_epochs_times_fit_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    original_step = torch.optim.Adam.step

    def step(optimizer: torch.optim.Adam, closure=None):
        calls.append(optimizer)
        return original_step(optimizer, closure)

    monkeypatch.setattr(torch.optim.Adam, "step", step)
    monkeypatch.setattr(ccnet5d_trainer, "load_ccnet_patch", lambda *args, **kwargs: _patch())
    monkeypatch.setattr(
        ccnet5d_trainer, "evaluate_ccnet5d_selection", lambda *args, **kwargs: _metrics()
    )

    result = _train(_model(), _plan(3), max_epochs=2)

    assert result.epochs_completed == 2
    assert result.steps_completed == 6
    assert len(calls) == 6
    assert all(optimizer is calls[0] for optimizer in calls)


def test_learning_rate_decays_after_boundary_epoch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ccnet5d_trainer, "load_ccnet_patch", lambda *args, **kwargs: _patch())
    monkeypatch.setattr(
        ccnet5d_trainer, "evaluate_ccnet5d_selection", lambda *args, **kwargs: _metrics()
    )

    result = _train(
        _model(),
        _plan(1),
        max_epochs=3,
        learning_rate=2e-3,
        decay_after_epochs=2,
        decay_factor=0.25,
        report_every_steps=1,
    )

    assert [row["learning_rate"] for row in result.training_history] == [
        2e-3,
        2e-3,
        5e-4,
    ]


@pytest.mark.parametrize(
    ("fit_count", "expected_steps"),
    [(5, [2, 4, 5]), (4, [2, 4])],
)
def test_selection_cadence_includes_final_step_without_duplicates(
    monkeypatch: pytest.MonkeyPatch,
    fit_count: int,
    expected_steps: list[int],
) -> None:
    fit_calls = []
    selection_calls = []

    def load(source: object, plan: object, *, region: str, index: int):
        fit_calls.append(index)
        return _patch(index)

    def evaluate(*args: object, **kwargs: object) -> dict[str, object]:
        selection_calls.append(len(fit_calls))
        return _metrics()

    monkeypatch.setattr(ccnet5d_trainer, "load_ccnet_patch", load)
    monkeypatch.setattr(ccnet5d_trainer, "evaluate_ccnet5d_selection", evaluate)

    result = _train(
        _model(),
        _plan(fit_count),
        validate_every_steps=2,
        report_every_steps=10,
    )

    assert selection_calls == expected_steps
    assert [row["step"] for row in result.selection_history] == expected_steps


def test_best_snapshot_is_cloned_at_first_minimum_and_survives_later_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ccnet5d_trainer, "load_ccnet_patch", lambda *args, **kwargs: _patch())
    scores = iter((0.5, 0.1, 0.1))
    states_at_selection = []

    def evaluate(model: CCNet5D, *args: object, **kwargs: object) -> dict[str, object]:
        states_at_selection.append(copy.deepcopy(model.state_dict()))
        return _metrics(next(scores))

    monkeypatch.setattr(ccnet5d_trainer, "evaluate_ccnet5d_selection", evaluate)
    model = _model()

    result = _train(
        model,
        _plan(3),
        validate_every_steps=1,
        report_every_steps=10,
        learning_rate=1e-2,
    )

    assert (result.best_epoch, result.best_step) == (1, 2)
    assert result.best_selection_metrics["relative_squared_error"] == 0.1
    assert result.final_selection_metrics["relative_squared_error"] == 0.1
    for name, expected in states_at_selection[1].items():
        torch.testing.assert_close(result.best_state_dict[name], expected, rtol=0, atol=0)
        assert result.best_state_dict[name].device.type == "cpu"
        assert result.best_state_dict[name].data_ptr() != expected.data_ptr()
    assert any(
        not torch.equal(result.best_state_dict[name], model.state_dict()[name])
        for name in result.best_state_dict
    )


def test_selection_patch_is_never_passed_to_training_loss_or_backward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit_patch = _patch()
    selection_patch = (
        np.zeros(PATCH_SHAPE, dtype=np.float32),
        np.full(PATCH_SHAPE, 99.0, dtype=np.float32),
        OBSERVED_MASK.copy(),
    )
    seen_targets = []
    grad_modes = []

    class RecordingCCNet5D(CCNet5D):
        def forward(self, values: torch.Tensor) -> torch.Tensor:
            grad_modes.append(torch.is_grad_enabled())
            return super().forward(values)

    model = RecordingCCNet5D(
        hidden_channels=1,
        intermediate_channels=1,
        kernel_size=1,
        output_activation="linear",
    )
    original_forward = torch.nn.MSELoss.forward

    def loss_forward(
        loss_function: torch.nn.MSELoss,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        seen_targets.append(target.detach().cpu().clone())
        return original_forward(loss_function, prediction, target)

    def load(source: object, plan: object, *, region: str, index: int):
        return fit_patch if region == "fit" else selection_patch

    monkeypatch.setattr(torch.nn.MSELoss, "forward", loss_forward)
    monkeypatch.setattr(ccnet5d_trainer, "load_ccnet_patch", load)
    from seis_interp.evaluation import ccnet5d_selection

    monkeypatch.setattr(ccnet5d_selection, "load_ccnet_patch", load)

    _train(model, _plan(1), validate_every_steps=1)

    assert len(seen_targets) == 1
    torch.testing.assert_close(
        seen_targets[0], torch.from_numpy(fit_patch[1][None, None]), rtol=0, atol=0
    )
    assert grad_modes == [True, False]


def test_tiny_signed_patch_fit_reduces_complete_patch_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch = _patch()
    monkeypatch.setattr(ccnet5d_trainer, "load_ccnet_patch", lambda *args, **kwargs: patch)
    monkeypatch.setattr(
        ccnet5d_trainer, "evaluate_ccnet5d_selection", lambda *args, **kwargs: _metrics()
    )
    model = _model()
    inputs = torch.from_numpy(patch[0][None, None])
    targets = torch.from_numpy(patch[1][None, None])
    with torch.no_grad():
        initial_loss = float(torch.nn.functional.mse_loss(model(inputs), targets))

    _train(
        model,
        _plan(1),
        max_epochs=20,
        learning_rate=5e-2,
        decay_after_epochs=20,
        validate_every_steps=20,
        report_every_steps=20,
    )
    with torch.no_grad():
        final_loss = float(torch.nn.functional.mse_loss(model(inputs), targets))

    assert final_loss < initial_loss


def test_nonfinite_loss_fails_before_backward_and_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, labels, mask = _patch()
    labels[0, 0, 0, 0, 0] = np.nan
    monkeypatch.setattr(
        ccnet5d_trainer,
        "load_ccnet_patch",
        lambda *args, **kwargs: (inputs, labels, mask),
    )
    monkeypatch.setattr(
        torch.optim.Adam,
        "step",
        lambda *args, **kwargs: pytest.fail("optimizer must not update a non-finite loss"),
    )
    model = _model()

    with pytest.raises(RuntimeError, match="non-finite training loss at step 1"):
        _train(model, _plan(1))

    assert all(parameter.grad is None for parameter in model.parameters())


def test_interval_history_and_reporter_use_float64_means_and_silent_default(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    losses = []
    original_forward = torch.nn.MSELoss.forward

    def loss_forward(
        loss_function: torch.nn.MSELoss,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        loss = original_forward(loss_function, prediction, target)
        losses.append(float(loss.detach()))
        return loss

    monkeypatch.setattr(torch.nn.MSELoss, "forward", loss_forward)
    monkeypatch.setattr(ccnet5d_trainer, "load_ccnet_patch", lambda *args, **kwargs: _patch())
    monkeypatch.setattr(
        ccnet5d_trainer, "evaluate_ccnet5d_selection", lambda *args, **kwargs: _metrics()
    )

    silent_result = _train(_model(), _plan(3), report_every_steps=2)
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
    assert [row["step"] for row in silent_result.training_history] == [2, 3]
    assert silent_result.training_history[0]["train_loss"] == float(
        np.mean(losses[:2], dtype=np.float64)
    )
    assert silent_result.training_history[1]["train_loss"] == losses[2]

    messages = []
    _train(_model(), _plan(1), report_every_steps=1, reporter=messages.append)
    assert len(messages) == 1
    assert messages[0].startswith("ccnet5d epoch 1/1 step 1/1:")


@pytest.mark.parametrize("batch_size", [2, 4, 8])
def test_batches_match_independent_adam_keep_partial_batch_and_weight_history(
    monkeypatch: pytest.MonkeyPatch, batch_size: int
) -> None:
    fit_count = 5
    model = _model()
    reference = copy.deepcopy(model)
    optimizer = torch.optim.Adam(reference.parameters(), lr=1e-3)
    rng = np.random.default_rng(13)
    expected_order = []
    expected_sizes = []
    expected_losses = []
    for _ in range(2):
        order = rng.permutation(fit_count)
        expected_order.extend(order.tolist())
        for start in range(0, fit_count, batch_size):
            indices = order[start : start + batch_size]
            patches = [_patch(int(index)) for index in indices]
            inputs = torch.from_numpy(np.stack([patch[0] for patch in patches])[:, None])
            labels = torch.from_numpy(np.stack([patch[1] for patch in patches])[:, None])
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(reference(inputs), labels)
            expected_losses.append(float(loss.detach()))
            expected_sizes.append(len(indices))
            loss.backward()
            optimizer.step()

    seen_order = []
    seen_sizes = []

    def load(source, plan, *, region, index):
        assert region == "fit"
        seen_order.append(index)
        return _patch(index)

    def record_forward(module, args):
        values = args[0]
        seen_sizes.append(values.shape[0])
        assert torch.count_nonzero(values[:, :, :, ~OBSERVED_MASK]) == 0

    monkeypatch.setattr(ccnet5d_trainer, "load_ccnet_patch", load)
    monkeypatch.setattr(
        ccnet5d_trainer, "evaluate_ccnet5d_selection", lambda *args, **kwargs: _metrics()
    )
    handle = model.register_forward_pre_hook(record_forward)
    result = _train(
        model,
        _plan(fit_count),
        batch_size=batch_size,
        max_epochs=2,
        report_every_steps=100,
        validate_every_steps=2,
    )
    handle.remove()

    assert seen_order == expected_order
    assert seen_sizes == expected_sizes
    expected_steps = 2 * math.ceil(fit_count / batch_size)
    assert result.steps_completed == expected_steps
    assert [row["step"] for row in result.selection_history] == list(
        range(2, expected_steps + 1, 2)
    )
    assert result.training_history == (
        {
            "epoch": 2,
            "step": expected_steps,
            "train_loss": float(np.average(expected_losses, weights=expected_sizes)),
            "learning_rate": 1e-3,
            "sample_count": 2 * fit_count * math.prod(PATCH_SHAPE),
            "loss_aggregation": "sample_weighted_mean",
        },
    )
    for name, tensor in reference.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], tensor, rtol=0, atol=0)


@pytest.mark.parametrize("batch_size", [0, -1, True, False, 1.5, None, "2"])
def test_invalid_batch_size_fails_before_patch_or_optimizer(monkeypatch, batch_size):
    monkeypatch.setattr(
        ccnet5d_trainer, "load_ccnet_patch", lambda *a, **kw: pytest.fail("must not read patches")
    )
    monkeypatch.setattr(
        torch.optim, "Adam", lambda *a, **kw: pytest.fail("must not initialize optimizer")
    )
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        _train(_model(), _plan(1), batch_size=batch_size)


@pytest.mark.parametrize("benchmark", [None, 0, 1, "false", [], np.bool_(False)])
def test_invalid_benchmark_fails_before_patch_or_optimizer(monkeypatch, benchmark):
    monkeypatch.setattr(
        ccnet5d_trainer, "load_ccnet_patch", lambda *a, **kw: pytest.fail("must not read patches")
    )
    monkeypatch.setattr(
        torch.optim, "Adam", lambda *a, **kw: pytest.fail("must not initialize optimizer")
    )
    with pytest.raises(ValueError, match="cudnn_benchmark must be a boolean"):
        _train(_model(), _plan(1), cudnn_benchmark=benchmark)


@pytest.mark.parametrize("ambient", [True, False])
def test_default_batch_and_benchmark_preserve_weights_history_rng_and_ambient(monkeypatch, ambient):
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", ambient)
    monkeypatch.setattr(ccnet5d_trainer, "load_ccnet_patch", lambda *a, **kw: _patch(kw["index"]))
    monkeypatch.setattr(ccnet5d_trainer, "evaluate_ccnet5d_selection", lambda *a, **kw: _metrics())
    records = []
    for options in ({}, {"batch_size": 1, "cudnn_benchmark": True}):
        model = _model()
        before = torch.get_rng_state().clone()
        result = _train(model, _plan(3), max_epochs=2, **options)
        assert torch.equal(before, torch.get_rng_state())
        assert torch.backends.cudnn.benchmark is ambient
        records.append((model.state_dict(), result))
    left, right = records
    assert left[1].training_history == right[1].training_history
    assert left[1].selection_history == right[1].selection_history
    assert left[1].steps_completed == right[1].steps_completed == 6
    for name, tensor in left[0].items():
        assert torch.equal(tensor, right[0][name])


def test_false_benchmark_applies_before_any_forward(monkeypatch):
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)
    monkeypatch.setattr(ccnet5d_trainer, "load_ccnet_patch", lambda *a, **kw: _patch())
    monkeypatch.setattr(ccnet5d_trainer, "evaluate_ccnet5d_selection", lambda *a, **kw: _metrics())
    model = _model()
    modes = []
    handle = model.register_forward_pre_hook(
        lambda module, args: modes.append(torch.backends.cudnn.benchmark)
    )
    _train(model, _plan(3), cudnn_benchmark=False, batch_size=2)
    handle.remove()
    assert modes == [False, False]
