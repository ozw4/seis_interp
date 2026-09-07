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
