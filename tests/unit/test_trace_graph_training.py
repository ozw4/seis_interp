"""Unit tests for the trace-graph trainer and checkpoint round trip."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from seis_interp.models.trace_graph_interpolator import (
    SOURCE_RECEIVER_BIPARTITE_GRAPH_MODE,
    TRACE_LATTICE_GRAPH_MODE,
    TraceGraphInterpolator,
)
from seis_interp.training.trace_graph_checkpoints import (
    load_trace_graph_checkpoint,
    save_trace_graph_checkpoint,
)
from seis_interp.training.trace_graph_trainer import train_trace_graph_interpolator

SOURCES = 3
TIME = 20


def _small_model(graph_mode: str = TRACE_LATTICE_GRAPH_MODE) -> TraceGraphInterpolator:
    return TraceGraphInterpolator(
        width=8,
        graph_mode=graph_mode,
        message_passing_rounds=2,
        time_downsample_factor=5,
        stem_kernel_size=3,
        temporal_kernel_size=3,
        temporal_dilations=(1, 2),
        spatial_kernel_size=3,
        attention_width=4,
    )


class _SyntheticProvider:
    """Deterministic gather batches from a fixed synthetic train pool."""

    def __init__(self) -> None:
        pool_generator = torch.Generator().manual_seed(21)
        self.neighbors = torch.randn(4, SOURCES, 8, 68, TIME, generator=pool_generator)
        self.availability = torch.rand(4, SOURCES, 8, 68, generator=pool_generator) > 0.2
        self.source_deltas = torch.randn(4, SOURCES, 2, generator=pool_generator) * 40.0 + 120.0
        self.target_coordinates = torch.rand(4, 2, generator=pool_generator)
        self.targets = torch.randn(4, 8, 68, TIME, generator=pool_generator)
        self.target_availability = torch.rand(4, 8, 68, generator=pool_generator) > 0.2

    def __call__(
        self,
        batch_size: int,
        *,
        generator: torch.Generator,
        neighbor_dropout: float,
    ) -> tuple[torch.Tensor, ...]:
        indices = torch.randint(4, (batch_size,), generator=generator)
        return (
            self.neighbors[indices],
            self.availability[indices],
            self.source_deltas[indices],
            self.target_coordinates[indices],
            self.targets[indices],
            self.target_availability[indices],
        )


class _SyntheticEvaluator:
    """Score the model against one held-out synthetic gather."""

    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(33)
        self.neighbors = torch.randn(1, SOURCES, 8, 68, TIME, generator=generator)
        self.availability = torch.rand(1, SOURCES, 8, 68, generator=generator) > 0.2
        self.source_deltas = torch.randn(1, SOURCES, 2, generator=generator) * 40.0 + 120.0
        self.target_coordinates = torch.rand(1, 2, generator=generator)
        self.targets = torch.randn(1, 8, 68, TIME, generator=generator)
        self.call_count = 0

    def __call__(self, model: TraceGraphInterpolator) -> float:
        self.call_count += 1
        prediction = model(
            self.neighbors,
            self.availability,
            self.source_deltas,
            self.target_coordinates,
        )
        signal = torch.sum(torch.square(self.targets.double()))
        error = torch.sum(torch.square(self.targets.double() - prediction.double()))
        return float(10.0 * torch.log10(signal / error))


def _train(
    model: TraceGraphInterpolator,
    checkpoint_path: Path,
    **overrides: object,
):
    keywords: dict[str, object] = {
        "device": "cpu",
        "generator": torch.Generator().manual_seed(5),
        "checkpoint_path": checkpoint_path,
        "total_steps": 3,
        "batch_size": 2,
        "neighbor_dropout": 0.0,
        "spectrum_weight": 0.0,
        "slope_weight": 0.0,
        "amplitude_weight": 0.0,
        "learning_rate": 1.0e-3,
        "weight_decay": 0.0,
        "validation_interval": 1,
        "use_bfloat16": False,
        "training_ffid_count": 4,
        "training_trace_count": 100,
    }
    keywords.update(overrides)
    return train_trace_graph_interpolator(
        model,
        _SyntheticProvider(),
        _SyntheticEvaluator(),
        **keywords,
    )


def test_trainer_writes_step_zero_baseline_and_history(tmp_path: Path) -> None:
    torch.manual_seed(0)
    checkpoint_path = tmp_path / "artifacts" / "best.pt"
    result = _train(_small_model(), checkpoint_path)
    assert checkpoint_path.exists()
    assert result.steps_completed == 3
    assert result.history[0]["step"] == 0
    assert len(result.history) == 4
    for row in result.history[1:]:
        for key in (
            "loss",
            "mask_mse",
            "spectrum_loss",
            "slope_loss",
            "amplitude_loss",
            "learning_rate",
            "validation_global_snr_db",
        ):
            assert key in row


def test_trainer_records_composite_terms_when_enabled(tmp_path: Path) -> None:
    torch.manual_seed(0)
    result = _train(
        _small_model(),
        tmp_path / "best.pt",
        spectrum_weight=0.1,
        slope_weight=0.1,
        amplitude_weight=0.1,
    )
    row = result.history[-1]
    assert row["spectrum_loss"] > 0.0
    assert row["slope_loss"] > 0.0
    assert row["amplitude_loss"] > 0.0
    assert row["loss"] > row["mask_mse"]


def test_trainer_disabled_terms_stay_zero(tmp_path: Path) -> None:
    torch.manual_seed(0)
    result = _train(_small_model(), tmp_path / "best.pt")
    row = result.history[-1]
    assert row["spectrum_loss"] == 0.0
    assert row["slope_loss"] == 0.0
    assert row["amplitude_loss"] == 0.0
    assert row["loss"] == row["mask_mse"]


def test_trainer_rejects_negative_loss_weights(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="spectrum_weight"):
        _train(_small_model(), tmp_path / "best.pt", spectrum_weight=-0.1)


def test_trainer_rejects_wrong_model_type(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="TraceGraphInterpolator"):
        train_trace_graph_interpolator(
            torch.nn.Linear(2, 2),
            _SyntheticProvider(),
            _SyntheticEvaluator(),
            device="cpu",
            generator=torch.Generator().manual_seed(5),
            checkpoint_path=tmp_path / "best.pt",
            total_steps=1,
            batch_size=1,
            neighbor_dropout=0.0,
            spectrum_weight=0.0,
            slope_weight=0.0,
            amplitude_weight=0.0,
            learning_rate=1.0e-3,
            weight_decay=0.0,
            validation_interval=1,
            use_bfloat16=False,
            training_ffid_count=1,
            training_trace_count=1,
        )


@pytest.mark.parametrize(
    "graph_mode",
    [TRACE_LATTICE_GRAPH_MODE, SOURCE_RECEIVER_BIPARTITE_GRAPH_MODE],
)
def test_checkpoint_round_trip_reproduces_outputs(tmp_path: Path, graph_mode: str) -> None:
    torch.manual_seed(0)
    model = _small_model(graph_mode)
    checkpoint_path = tmp_path / "best.pt"
    _train(model, checkpoint_path, total_steps=2)
    save_trace_graph_checkpoint(
        checkpoint_path,
        model,
        best_step=2,
        best_validation_global_snr_db=1.25,
    )
    loaded = load_trace_graph_checkpoint(checkpoint_path)
    assert loaded.best_step == 2
    assert loaded.best_validation_global_snr_db == 1.25
    assert loaded.graph_mode == graph_mode
    assert loaded.amplitude_scaling == "per_trace_rms"
    assert loaded.validation_metric_domain == "oracle_per_trace_unit_rms"
    evaluator = _SyntheticEvaluator()
    model.eval()
    loaded.model.eval()
    assert evaluator(loaded.model) == pytest.approx(evaluator(model))


def test_checkpoint_rejects_foreign_model_type(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "best.pt"
    model = _small_model()
    save_trace_graph_checkpoint(
        checkpoint_path,
        model,
        best_step=0,
        best_validation_global_snr_db=0.0,
    )
    payload = torch.load(checkpoint_path, weights_only=True)
    payload["model_type"] = "something_else"
    torch.save(payload, checkpoint_path)
    with pytest.raises(ValueError, match="model_type"):
        load_trace_graph_checkpoint(checkpoint_path)
