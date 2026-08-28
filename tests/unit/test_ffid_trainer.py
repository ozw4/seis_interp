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
from seis_interp.training.correlation_loss import trace_correlation_loss
from seis_interp.training.ffid_batches import FullFfidBatch, FullFfidBatchSampler
from seis_interp.training.ffid_trainer import train_siren_by_ffid


def _normalization() -> NormalizationParameters:
    return NormalizationParameters(
        coordinate_order=MODEL_COORDINATE_ORDER,
        coordinate_min=(-1.0, -1.0, -1.0, -1.0, -1.0, -1.0),
        coordinate_max=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        amplitude_rms=1.0,
    )


def _sampler(
    seed: int = 9,
    *,
    amplitude_scaling: str = "train_global_rms",
) -> FullFfidBatchSampler:
    time = np.array([-1.0, 1.0], dtype=np.float64)
    spatial = np.arange(30, dtype=np.float64).reshape(6, 5) / 30.0
    amplitudes = np.arange(1, 13, dtype=np.float32).reshape(6, 2) / 10.0
    return FullFfidBatchSampler(
        time,
        spatial,
        amplitudes,
        {
            10: np.array([0], dtype=np.int64),
            20: np.array([1, 2], dtype=np.int64),
            30: np.array([3, 4, 5], dtype=np.int64),
        },
        amplitude_rms=1.0,
        random_seed=seed,
        amplitude_scaling=amplitude_scaling,
    )


def _scores(values: list[float]):
    iterator: Iterator[float] = iter(values)

    def evaluate(_model: torch.nn.Module) -> float:
        return next(iterator)

    return evaluate


class _LifetimeCheckingIterator(Iterator[FullFfidBatch]):
    def __init__(self, release_checks: list[bool]) -> None:
        self._release_checks = release_checks
        self._index = 0
        self._previous_references: (
            tuple[
                weakref.ReferenceType[np.ndarray],
                weakref.ReferenceType[np.ndarray],
            ]
            | None
        ) = None

    def __next__(self) -> FullFfidBatch:
        if self._previous_references is not None:
            self._release_checks.append(
                all(reference() is None for reference in self._previous_references)
            )
            self._previous_references = None
        if self._index == 2:
            raise StopIteration

        coordinates = np.full((2, 6), self._index / 10.0, dtype=np.float64)
        targets = np.full(2, (self._index + 1) / 10.0, dtype=np.float32)
        batch = FullFfidBatch(
            ffid=10 + self._index * 10,
            coordinates=coordinates,
            targets=targets,
            trace_count=1,
            point_count=2,
        )
        self._previous_references = (weakref.ref(coordinates), weakref.ref(targets))
        self._index += 1
        return batch


class _LifetimeCheckingSampler:
    ffid_count = 2
    amplitude_scaling = "train_global_rms"

    def __init__(self) -> None:
        self.release_checks: list[bool] = []

    def iter_epoch(self) -> Iterator[FullFfidBatch]:
        return _LifetimeCheckingIterator(self.release_checks)


def test_full_ffid_trainer_counts_variable_batches_and_selects_global_snr(
    tmp_path: Path,
) -> None:
    torch.manual_seed(4)
    model = Siren(hidden_width=8, hidden_layers=1)
    checkpoint = tmp_path / "best.pt"
    messages: list[str] = []

    result = train_siren_by_ffid(
        model,
        _sampler(),
        _scores([1.0, 2.0, 1.5, 1.0]),
        _normalization(),
        device="cpu",
        loss="l2",
        optimizer="adam",
        learning_rate=1.0e-3,
        max_epochs=6,
        early_stopping_patience=2,
        validation_ffid_count=3,
        checkpoint_path=checkpoint,
        reporter=messages.append,
    )

    assert result.best_epoch == 2
    assert result.best_validation_global_snr_db == 2.0
    assert result.epochs_completed == 4
    assert result.global_steps == 12
    assert result.stopped_early
    assert result.training_ffid_count == result.validation_ffid_count == 3
    assert all(
        set(record)
        == {
            "epoch",
            "global_step",
            "mean_ffid_batch_loss",
            "validation_global_snr_db",
        }
        for record in result.history
    )
    assert messages[0].endswith("start")
    assert "validation_global_snr_db=" in messages[-1]
    loaded = load_siren_checkpoint(checkpoint)
    assert loaded.epoch == 2
    assert loaded.global_step == 6
    assert loaded.validation_median_trace_snr_db is None
    assert loaded.validation_global_snr_db == 2.0


def test_full_ffid_trainer_adds_trace_correlation_over_each_variable_size_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import seis_interp.training.ffid_trainer as trainer_module

    observed_shapes: list[tuple[int, int]] = []
    observed_eps: list[float] = []

    def recording_correlation_loss(
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        eps: float,
    ) -> torch.Tensor:
        assert prediction.shape == target.shape
        observed_shapes.append(tuple(prediction.shape))
        observed_eps.append(eps)
        return trace_correlation_loss(prediction, target, eps=eps)

    monkeypatch.setattr(trainer_module, "trace_correlation_loss", recording_correlation_loss)
    messages: list[str] = []
    correlation_weight = 0.25
    correlation_eps = 2.5e-4

    result = train_siren_by_ffid(
        Siren(hidden_width=8, hidden_layers=1),
        _sampler(),
        _scores([1.0]),
        _normalization(),
        device="cpu",
        loss="l2",
        optimizer="adam",
        learning_rate=1.0e-3,
        max_epochs=1,
        early_stopping_patience=1,
        validation_ffid_count=3,
        checkpoint_path=tmp_path / "best.pt",
        reporter=messages.append,
        correlation_weight=correlation_weight,
        correlation_eps=correlation_eps,
    )

    assert sorted(observed_shapes) == [(1, 2), (2, 2), (3, 2)]
    assert observed_eps == [correlation_eps] * 3
    (history_row,) = result.history
    assert set(history_row) == {
        "epoch",
        "global_step",
        "mean_ffid_batch_loss",
        "mean_ffid_batch_mse_loss",
        "mean_ffid_batch_correlation_loss",
        "validation_global_snr_db",
    }
    assert history_row["mean_ffid_batch_loss"] == pytest.approx(
        history_row["mean_ffid_batch_mse_loss"]
        + correlation_weight * history_row["mean_ffid_batch_correlation_loss"]
    )
    assert "correlation_weight=0.25" in messages[0]
    assert "correlation_eps=0.00025" in messages[0]
    assert "mean_ffid_batch_mse_loss=" in messages[-1]
    assert "mean_ffid_batch_correlation_loss=" in messages[-1]


def test_zero_correlation_weight_preserves_the_pure_mse_history_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import seis_interp.training.ffid_trainer as trainer_module

    monkeypatch.setattr(
        trainer_module,
        "trace_correlation_loss",
        lambda *_args, **_kwargs: pytest.fail(
            "zero correlation weight must not evaluate the auxiliary loss"
        ),
    )

    result = train_siren_by_ffid(
        Siren(hidden_width=8, hidden_layers=1),
        _sampler(),
        _scores([1.0]),
        _normalization(),
        device="cpu",
        loss="l2",
        optimizer="adam",
        learning_rate=1.0e-3,
        max_epochs=1,
        early_stopping_patience=1,
        validation_ffid_count=3,
        checkpoint_path=tmp_path / "best.pt",
        reporter=lambda _message: None,
        correlation_weight=0.0,
        correlation_eps=2.5e-4,
    )

    assert set(result.history[0]) == {
        "epoch",
        "global_step",
        "mean_ffid_batch_loss",
        "validation_global_snr_db",
    }


@pytest.mark.parametrize(
    "correlation_weight",
    [True, -0.1, float("nan"), float("inf"), -float("inf"), "0.1"],
)
def test_full_ffid_trainer_rejects_invalid_correlation_weight(
    tmp_path: Path,
    correlation_weight: object,
) -> None:
    with pytest.raises(ValueError, match="correlation_weight must be a non-negative finite number"):
        train_siren_by_ffid(
            Siren(hidden_width=8, hidden_layers=1),
            _sampler(),
            _scores([1.0]),
            _normalization(),
            device="cpu",
            loss="l2",
            optimizer="adam",
            learning_rate=1.0e-3,
            max_epochs=1,
            early_stopping_patience=1,
            validation_ffid_count=3,
            checkpoint_path=tmp_path / "best.pt",
            reporter=lambda _message: None,
            correlation_weight=correlation_weight,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "correlation_eps",
    [True, 0.0, -0.1, float("nan"), float("inf"), -float("inf"), "1e-4"],
)
def test_full_ffid_trainer_rejects_invalid_correlation_epsilon(
    tmp_path: Path,
    correlation_eps: object,
) -> None:
    with pytest.raises(ValueError, match="correlation_eps must be a positive finite number"):
        train_siren_by_ffid(
            Siren(hidden_width=8, hidden_layers=1),
            _sampler(),
            _scores([1.0]),
            _normalization(),
            device="cpu",
            loss="l2",
            optimizer="adam",
            learning_rate=1.0e-3,
            max_epochs=1,
            early_stopping_patience=1,
            validation_ffid_count=3,
            checkpoint_path=tmp_path / "best.pt",
            reporter=lambda _message: None,
            correlation_weight=0.1,
            correlation_eps=correlation_eps,  # type: ignore[arg-type]
        )


def test_full_ffid_trainer_releases_each_batch_before_requesting_the_next(
    tmp_path: Path,
) -> None:
    sampler = _LifetimeCheckingSampler()

    result = train_siren_by_ffid(
        Siren(hidden_width=8, hidden_layers=1),
        sampler,
        _scores([1.0]),
        _normalization(),
        device="cpu",
        loss="l2",
        optimizer="adam",
        learning_rate=1.0e-3,
        max_epochs=1,
        early_stopping_patience=1,
        validation_ffid_count=2,
        checkpoint_path=tmp_path / "best.pt",
        reporter=lambda _message: None,
    )

    assert result.global_steps == 2
    assert sampler.release_checks == [True, True]


@pytest.mark.parametrize("invalid_loss", [float("nan"), float("inf"), -float("inf")])
def test_full_ffid_trainer_rejects_non_finite_loss_before_backward_and_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_loss: float,
) -> None:
    class _SecondBatchNonFiniteLoss(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(
            self,
            prediction: torch.Tensor,
            target: torch.Tensor,
        ) -> torch.Tensor:
            self.calls += 1
            if self.calls == 2:
                return prediction.sum() * 0.0 + invalid_loss
            return torch.nn.functional.mse_loss(prediction, target)

    loss_function = _SecondBatchNonFiniteLoss()
    monkeypatch.setattr(torch.nn, "MSELoss", lambda: loss_function)
    backward_calls = 0
    original_backward = torch.Tensor.backward

    def record_backward(tensor: torch.Tensor, *args: object, **kwargs: object) -> None:
        nonlocal backward_calls
        backward_calls += 1
        original_backward(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "backward", record_backward)
    optimizer_steps = 0
    original_step = torch.optim.Adam.step

    def record_step(
        optimizer: torch.optim.Adam,
        closure: object | None = None,
    ) -> torch.Tensor | None:
        nonlocal optimizer_steps
        optimizer_steps += 1
        return original_step(optimizer, closure=closure)

    monkeypatch.setattr(torch.optim.Adam, "step", record_step)
    time = np.array([-1.0, 1.0], dtype=np.float64)
    spatial = np.arange(5, dtype=np.float64).reshape(1, 5) / 5.0
    amplitudes = np.array([[0.1, 0.2]], dtype=np.float32)
    sampler = FullFfidBatchSampler(
        time,
        spatial,
        amplitudes,
        {2348: np.array([0], dtype=np.int64)},
        amplitude_rms=1.0,
        random_seed=9,
    )
    model = Siren(hidden_width=8, hidden_layers=1)

    with pytest.raises(RuntimeError, match="non-finite training loss") as exc_info:
        train_siren_by_ffid(
            model,
            sampler,
            _scores([1.0]),
            _normalization(),
            device="cpu",
            loss="l2",
            optimizer="adam",
            learning_rate=1.0e-3,
            max_epochs=2,
            early_stopping_patience=2,
            validation_ffid_count=1,
            checkpoint_path=tmp_path / "best.pt",
            reporter=lambda _message: None,
        )

    message = str(exc_info.value)
    assert "FFID 2348" in message
    assert "epoch=2" in message
    assert "global_step=2" in message
    assert f"loss={invalid_loss}" in message
    assert backward_calls == optimizer_steps == 1


@pytest.mark.parametrize("invalid_loss", [float("nan"), float("inf"), -float("inf")])
def test_full_ffid_trainer_rejects_non_finite_correlation_before_backward_and_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_loss: float,
) -> None:
    import seis_interp.training.ffid_trainer as trainer_module

    correlation_calls = 0

    def second_batch_non_finite_correlation(
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        eps: float,
    ) -> torch.Tensor:
        nonlocal correlation_calls
        correlation_calls += 1
        if correlation_calls == 2:
            return prediction.sum() * 0.0 + invalid_loss
        return trace_correlation_loss(prediction, target, eps=eps)

    monkeypatch.setattr(
        trainer_module,
        "trace_correlation_loss",
        second_batch_non_finite_correlation,
    )
    backward_calls = 0
    original_backward = torch.Tensor.backward

    def record_backward(tensor: torch.Tensor, *args: object, **kwargs: object) -> None:
        nonlocal backward_calls
        backward_calls += 1
        original_backward(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "backward", record_backward)
    optimizer_steps = 0
    original_step = torch.optim.Adam.step

    def record_step(
        optimizer: torch.optim.Adam,
        closure: object | None = None,
    ) -> torch.Tensor | None:
        nonlocal optimizer_steps
        optimizer_steps += 1
        return original_step(optimizer, closure=closure)

    monkeypatch.setattr(torch.optim.Adam, "step", record_step)
    time = np.array([-1.0, 1.0], dtype=np.float64)
    spatial = np.arange(5, dtype=np.float64).reshape(1, 5) / 5.0
    amplitudes = np.array([[0.1, 0.2]], dtype=np.float32)
    sampler = FullFfidBatchSampler(
        time,
        spatial,
        amplitudes,
        {2348: np.array([0], dtype=np.int64)},
        amplitude_rms=1.0,
        random_seed=9,
    )

    with pytest.raises(RuntimeError, match="non-finite correlation loss") as exc_info:
        train_siren_by_ffid(
            Siren(hidden_width=8, hidden_layers=1),
            sampler,
            _scores([1.0]),
            _normalization(),
            device="cpu",
            loss="l2",
            optimizer="adam",
            learning_rate=1.0e-3,
            max_epochs=2,
            early_stopping_patience=2,
            validation_ffid_count=1,
            checkpoint_path=tmp_path / "best.pt",
            reporter=lambda _message: None,
            correlation_weight=0.1,
            correlation_eps=1.0e-4,
        )

    message = str(exc_info.value)
    assert "FFID 2348" in message
    assert "epoch=2" in message
    assert "global_step=2" in message
    assert f"correlation_loss={invalid_loss}" in message
    assert backward_calls == optimizer_steps == 1


@pytest.mark.parametrize(
    ("override", "message"),
    [({"loss": "l1"}, "only l2"), ({"optimizer": "sgd"}, "only adam")],
)
def test_full_ffid_trainer_rejects_unsupported_training_contract(
    tmp_path: Path,
    override: dict[str, str],
    message: str,
) -> None:
    arguments = {"loss": "l2", "optimizer": "adam", **override}

    with pytest.raises(ValueError, match=message):
        train_siren_by_ffid(
            Siren(hidden_width=8, hidden_layers=1),
            _sampler(),
            _scores([1.0]),
            _normalization(),
            device="cpu",
            learning_rate=1.0e-3,
            max_epochs=1,
            early_stopping_patience=1,
            validation_ffid_count=3,
            checkpoint_path=tmp_path / "best.pt",
            reporter=lambda _message: None,
            **arguments,
        )


def test_full_ffid_trainer_is_reproducible_for_the_same_seed(tmp_path: Path) -> None:
    histories = []
    states = []
    for run_index in range(2):
        torch.manual_seed(12)
        model = Siren(hidden_width=8, hidden_layers=1)
        result = train_siren_by_ffid(
            model,
            _sampler(seed=5),
            _scores([0.5, 0.75]),
            _normalization(),
            device="cpu",
            loss="l2",
            optimizer="adam",
            learning_rate=1.0e-3,
            max_epochs=2,
            early_stopping_patience=2,
            validation_ffid_count=3,
            checkpoint_path=tmp_path / f"best_{run_index}.pt",
            reporter=lambda _message: None,
        )
        histories.append(result.history)
        states.append({name: value.detach().clone() for name, value in model.state_dict().items()})

    assert histories[0] == histories[1]
    for name in states[0]:
        torch.testing.assert_close(states[0][name], states[1][name], rtol=0.0, atol=0.0)


def test_full_ffid_trainer_accepts_perfect_validation_and_checkpoints_it(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "best.pt"

    result = train_siren_by_ffid(
        Siren(hidden_width=8, hidden_layers=1),
        _sampler(),
        _scores([float("inf"), float("inf")]),
        _normalization(),
        device="cpu",
        loss="l2",
        optimizer="adam",
        learning_rate=1.0e-3,
        max_epochs=3,
        early_stopping_patience=1,
        validation_ffid_count=3,
        checkpoint_path=checkpoint,
        reporter=lambda _message: None,
    )

    assert result.best_epoch == 1
    assert result.best_validation_global_snr_db == float("inf")
    assert result.epochs_completed == 2
    assert result.stopped_early
    assert all(record["validation_global_snr_db"] == float("inf") for record in result.history)
    assert load_siren_checkpoint(checkpoint).validation_global_snr_db == float("inf")


def test_full_ffid_trainer_rejects_scaling_that_does_not_match_sampler(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="must match the FullFfidBatchSampler"):
        train_siren_by_ffid(
            Siren(hidden_width=8, hidden_layers=1),
            _sampler(),
            _scores([1.0]),
            _normalization(),
            device="cpu",
            loss="l2",
            optimizer="adam",
            learning_rate=1.0e-3,
            max_epochs=1,
            early_stopping_patience=1,
            validation_ffid_count=3,
            checkpoint_path=tmp_path / "best.pt",
            reporter=lambda _message: None,
            amplitude_scaling="per_trace_rms",
        )


def test_full_ffid_trainer_labels_per_trace_validation_as_oracle(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "best.pt"
    messages: list[str] = []

    train_siren_by_ffid(
        Siren(hidden_width=8, hidden_layers=1),
        _sampler(amplitude_scaling="per_trace_rms"),
        _scores([1.0]),
        _normalization(),
        device="cpu",
        loss="l2",
        optimizer="adam",
        learning_rate=1.0e-3,
        max_epochs=1,
        early_stopping_patience=1,
        validation_ffid_count=3,
        checkpoint_path=checkpoint,
        reporter=messages.append,
        amplitude_scaling="per_trace_rms",
    )

    assert "validation_metric_domain=oracle_per_trace_unit_rms" in messages[0]
    assert "oracle_per_trace_unit_rms_global_snr_db=" in messages[-1]
    assert load_siren_checkpoint(checkpoint).validation_metric_domain == (
        "oracle_per_trace_unit_rms"
    )


@pytest.mark.parametrize("invalid_score", [float("nan"), -float("inf")])
def test_full_ffid_trainer_rejects_invalid_validation_scores(
    tmp_path: Path,
    invalid_score: float,
) -> None:
    with pytest.raises(ValueError, match="finite or positive infinity"):
        train_siren_by_ffid(
            Siren(hidden_width=8, hidden_layers=1),
            _sampler(),
            _scores([invalid_score]),
            _normalization(),
            device="cpu",
            loss="l2",
            optimizer="adam",
            learning_rate=1.0e-3,
            max_epochs=1,
            early_stopping_patience=1,
            validation_ffid_count=3,
            checkpoint_path=tmp_path / "best.pt",
            reporter=lambda _message: None,
        )
