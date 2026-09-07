from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from seis_interp.evaluation import ccnet5d_selection
from seis_interp.evaluation.ccnet5d_selection import (
    evaluate_ccnet5d_selection,
    validate_ccnet5d_selection_targets,
)

PATCH_SHAPE = (1, 2, 1, 1, 1)
OBSERVED_MASK = np.array([True, False], dtype=np.bool_).reshape(PATCH_SHAPE[1:])


class BroadcastObservedValue(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self.grad_enabled: list[bool] = []

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        self.grad_enabled.append(torch.is_grad_enabled())
        first_trace = values[:, :, :, :1, :, :, :]
        return torch.zeros_like(values) + first_trace + self.anchor * 0.0


def _plan(count: int = 2) -> SimpleNamespace:
    return SimpleNamespace(patch_shape=PATCH_SHAPE, selection=tuple(range(count)))


def _source(rms: float = 2.0) -> SimpleNamespace:
    return SimpleNamespace(amplitude_rms=rms)


def _patch(observed_value: float, missing_label: float) -> tuple[np.ndarray, ...]:
    labels = np.array([observed_value, missing_label], dtype=np.float32).reshape(PATCH_SHAPE)
    inputs = np.where(OBSERVED_MASK[None], labels, 0.0).astype(np.float32)
    return inputs, labels, OBSERVED_MASK.copy()


def test_target_preflight_reads_every_fixed_descriptor_and_keeps_zero_patches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patches = (_patch(5.0, 0.0), _patch(0.0, 3.0))
    requested = []

    def load(source: object, plan: object, *, region: str, index: int) -> tuple[np.ndarray, ...]:
        requested.append((region, index))
        return patches[index]

    monkeypatch.setattr(ccnet5d_selection, "load_ccnet_patch", load)

    assert validate_ccnet5d_selection_targets(_source(), _plan()) is None
    assert requested == [("selection", 0), ("selection", 1)]


def test_target_preflight_rejects_zero_aggregate_missing_reference_energy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested = []

    def load(source: object, plan: object, *, region: str, index: int) -> tuple[np.ndarray, ...]:
        requested.append((region, index))
        return _patch(99.0, 0.0)

    monkeypatch.setattr(ccnet5d_selection, "load_ccnet_patch", load)

    with pytest.raises(ValueError, match="selection target reference energy"):
        validate_ccnet5d_selection_targets(_source(), _plan())

    assert requested == [("selection", 0), ("selection", 1)]


def test_target_preflight_rejects_nonfinite_physical_reference_energy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ccnet5d_selection,
        "load_ccnet_patch",
        lambda *args, **kwargs: _patch(0.0, 1.0),
    )

    with pytest.raises(ValueError, match="selection target reference energy"):
        validate_ccnet5d_selection_targets(_source(rms=1e308), _plan(count=1))


def test_selection_accumulates_physical_energies_over_patch_instances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patches = (_patch(0.0, 1.0), _patch(2.0, 3.0))
    requested = []

    def load(source: object, plan: object, *, region: str, index: int) -> tuple[np.ndarray, ...]:
        requested.append((region, index))
        return patches[index]

    monkeypatch.setattr(ccnet5d_selection, "load_ccnet_patch", load)

    metrics = evaluate_ccnet5d_selection(BroadcastObservedValue(), _source(), _plan(), device="cpu")

    assert requested == [("selection", 0), ("selection", 1)]
    assert metrics == {
        "patch_count": 2,
        "missing_sample_count": 2,
        "reference_energy": 40.0,
        "error_energy": 8.0,
        "relative_squared_error": 0.2,
        "snr_db": pytest.approx(10.0 * math.log10(5.0)),
        "snr_status": "finite",
        "metric_scope": "selection_patch_instances_missing_only",
    }


def test_selection_ignores_errors_on_observed_traces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = np.array([99.0, 3.0], dtype=np.float32).reshape(PATCH_SHAPE)
    inputs = np.array([3.0, 0.0], dtype=np.float32).reshape(PATCH_SHAPE)
    monkeypatch.setattr(
        ccnet5d_selection,
        "load_ccnet_patch",
        lambda *args, **kwargs: (inputs, labels, OBSERVED_MASK.copy()),
    )

    metrics = evaluate_ccnet5d_selection(
        BroadcastObservedValue(), _source(rms=1.0), _plan(count=1), device="cpu"
    )

    assert metrics["error_energy"] == 0.0
    assert metrics["relative_squared_error"] == 0.0
    assert metrics["snr_db"] is None
    assert metrics["snr_status"] == "perfect_reconstruction"


@pytest.mark.parametrize("initial_training", [True, False])
def test_selection_disables_gradients_and_restores_model_mode(
    monkeypatch: pytest.MonkeyPatch, initial_training: bool
) -> None:
    monkeypatch.setattr(
        ccnet5d_selection,
        "load_ccnet_patch",
        lambda *args, **kwargs: _patch(1.0, 2.0),
    )
    model = BroadcastObservedValue()
    model.train(initial_training)

    evaluate_ccnet5d_selection(model, _source(), _plan(count=1), device="cpu")

    assert model.training is initial_training
    assert model.grad_enabled == [False]
    assert model.anchor.grad is None


def test_selection_rejects_zero_reference_and_restores_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ccnet5d_selection,
        "load_ccnet_patch",
        lambda *args, **kwargs: _patch(1.0, 0.0),
    )
    model = BroadcastObservedValue()
    model.train()

    with pytest.raises(ValueError, match="reference energy"):
        evaluate_ccnet5d_selection(model, _source(), _plan(count=1), device="cpu")

    assert model.training


def test_perfect_metrics_are_strict_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ccnet5d_selection,
        "load_ccnet_patch",
        lambda *args, **kwargs: _patch(2.0, 2.0),
    )

    metrics = evaluate_ccnet5d_selection(
        BroadcastObservedValue(), _source(), _plan(count=1), device="cpu"
    )

    assert metrics["snr_db"] is None
    assert metrics["snr_status"] == "perfect_reconstruction"
    json.dumps(metrics, allow_nan=False)
