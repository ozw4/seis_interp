from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from seis_interp.models.relational_trace_graph import RelationalTraceGraphInterpolator
from seis_interp.processing.trace_graph_preprocessing import fit_trace_graph_preprocessing
from seis_interp.processing.trace_graph_settings import TraceGraphSettings
from seis_interp.training import relational_trace_graph_trainer as trainer_module
from seis_interp.training.relational_trace_graph_prediction import predict_relational_trace_graph
from seis_interp.training.relational_trace_graph_trainer import train_relational_trace_graph
from seis_interp.training.trace_graph_episodes import TraceGraphEpisodeGenerator
from tests.fixtures.trace_graph_training import make_trace_graph_training_domains


def _model():
    torch.manual_seed(91)
    return RelationalTraceGraphInterpolator(
        width=8,
        message_passing_rounds=2,
        time_downsample_factor=2,
        temporal_dilations=(1, 2),
        stem_kernel_size=3,
        temporal_kernel_size=3,
        attention_width=5,
        relation_embedding_dim=3,
        relation_fusion="learned_gate",
    )


def _setup():
    train, validation, values = make_trace_graph_training_domains()
    preprocessing = fit_trace_graph_preprocessing(
        train, values, position_scale_m=2.0, offset_scale_m=2.0, azimuth_min_offset_m=0.01
    )
    settings = TraceGraphSettings(((2.0, 2.0),) * 4, neighbors_per_relation=3)
    return train, validation, values, preprocessing, settings


def _train(model, *, kind="random_trace", **overrides):
    train, validation, values, preprocessing, graph_settings = _setup()
    arguments = {
        "graph_settings": graph_settings,
        "episode_kind_probabilities": {kind: 1.0},
        "missing_fractions": [0.5],
        "random_seed": 24,
        "max_steps": 16,
        "query_batch_size": 4,
        "validation_interval": 8,
        "validation_query_batch_size": 2,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "gradient_clip_norm": 1.0,
        "training_amplitudes": values,
        "validation_amplitudes": values,
    }
    arguments.update(overrides)
    return train_relational_trace_graph(model, train, validation, preprocessing, **arguments)


@pytest.mark.parametrize("kind", ["random_trace", "random_whole_ffid"])
def test_analytic_training_completes_learns_and_backpropagates_after_zero_init(kind) -> None:
    model = _model()
    result = _train(model, kind=kind)
    assert result.steps_completed == result.episodes_completed == 16
    assert not result.final_episode_interrupted
    assert result.query_count == 64 and result.sample_count == 64 * 7
    assert result.no_context_query_count == 0
    losses = [row["batch_loss"] for row in result.training_history]
    assert np.mean(losses[-4:]) < losses[0] * 0.4
    assert model.encoder.stem.weight.grad.abs().sum() > 0
    assert model.rounds[0].gamma[0].weight.grad.abs().sum() > 0
    assert model.rounds[0].relation_gate[0].weight.grad.abs().sum() > 0
    assert [row["step"] for row in result.validation_history] == [8, 16]
    assert result.train_loss == pytest.approx(
        sum(row["normalized_error_energy"] for row in result.training_history) / result.sample_count
    )


def test_validation_frequency_preserves_masks_queries_training_states_and_losses(
    monkeypatch,
) -> None:
    actual = trainer_module.read_trace_graph_training_labels
    calls = []

    def record(domain, episode, query_ids, *args, **kwargs):
        calls.append((episode.hidden_trace_ids.copy(), query_ids.copy()))
        return actual(domain, episode, query_ids, *args, **kwargs)

    monkeypatch.setattr(trainer_module, "read_trace_graph_training_labels", record)
    one = _train(_model(), max_steps=6, validation_interval=1, query_batch_size=3)
    first_calls, calls[:] = calls.copy(), []
    two = _train(_model(), max_steps=6, validation_interval=5, query_batch_size=3)
    for left, right in zip(first_calls, calls, strict=True):
        np.testing.assert_array_equal(left[0], right[0])
        np.testing.assert_array_equal(left[1], right[1])
    assert one.training_history == two.training_history
    for name in one.final_state_dict:
        assert torch.equal(one.final_state_dict[name], two.final_state_dict[name])
    assert [row["step"] for row in two.validation_history] == [5, 6]


def test_best_snapshot_is_deep_copied_earliest_tie_and_predicts_independently(monkeypatch) -> None:
    captured = []
    actual = trainer_module.evaluate_trace_graph_prediction
    errors = iter([1.0, 1.0, 2.0])
    model = _model()

    def score(*args, **kwargs):
        captured.append(
            {name: value.detach().clone() for name, value in model.state_dict().items()}
        )
        metrics = actual(*args, **kwargs)
        metrics["evaluation_target"]["error_energy"] = next(errors)
        return metrics

    monkeypatch.setattr(trainer_module, "evaluate_trace_graph_prediction", score)
    result = _train(model, max_steps=3, validation_interval=1)
    assert result.best_step == 1
    assert any(
        not torch.equal(result.best_state_dict[name], result.final_state_dict[name])
        for name in result.best_state_dict
    )
    for name, value in result.best_state_dict.items():
        assert torch.equal(value, captured[0][name])
        assert value.data_ptr() != model.state_dict()[name].data_ptr()
        assert value.data_ptr() != result.final_state_dict[name].data_ptr()
    _, validation, values, preprocessing, settings = _setup()
    best = _model()
    best.load_state_dict(result.best_state_dict)
    recorded = _model()
    recorded.load_state_dict(captured[0])
    expected = predict_relational_trace_graph(
        recorded, validation, preprocessing, graph_settings=settings, amplitudes=values
    )
    restored = predict_relational_trace_graph(
        best, validation, preprocessing, graph_settings=settings, amplitudes=values
    )
    np.testing.assert_array_equal(restored.prediction, expected.prediction)


def test_validation_labels_affect_only_scores_not_training_or_current_prediction() -> None:
    _, validation, values, preprocessing, settings = _setup()
    changed = values.copy()
    changed[validation.array_rows[validation.query_mask]] *= -5
    first_model, second_model = _model(), _model()
    one = _train(first_model, max_steps=3, validation_interval=1)
    two = _train(second_model, max_steps=3, validation_interval=1, validation_amplitudes=changed)
    assert one.training_history == two.training_history
    assert one.final_validation_metrics != two.final_validation_metrics
    for name in one.final_state_dict:
        assert torch.equal(one.final_state_dict[name], two.final_state_dict[name])
    first = predict_relational_trace_graph(
        first_model, validation, preprocessing, graph_settings=settings, amplitudes=values
    )
    second = predict_relational_trace_graph(
        second_model, validation, preprocessing, graph_settings=settings, amplitudes=changed
    )
    np.testing.assert_array_equal(first.prediction, second.prediction)


def test_zero_context_and_final_small_batch_use_exact_unpadded_sample_denominator() -> None:
    train, validation, values, preprocessing, _ = _setup()
    values[:8] *= np.arange(1, 9, dtype=np.float32)[:, None]
    tiny_radius = TraceGraphSettings(((0.001, 0.001),) * 4)
    seed = 19
    episode = TraceGraphEpisodeGenerator(
        train, random_seed=seed, kind_probabilities={"random_trace": 1.0}, missing_fractions=[0.5]
    ).next_episode()
    rows = np.array(
        [
            train.array_rows[np.flatnonzero(train.trace_ids == value)[0]]
            for value in episode.query_trace_ids
        ]
    )
    normalized = (values[rows].astype(np.float64) / preprocessing.amplitude_scale).astype(
        np.float32
    )
    expected_errors = np.square(normalized).astype(np.float64)
    result = train_relational_trace_graph(
        _model(),
        train,
        validation,
        preprocessing,
        graph_settings=tiny_radius,
        episode_kind_probabilities={"random_trace": 1.0},
        missing_fractions=[0.5],
        random_seed=seed,
        max_steps=2,
        query_batch_size=3,
        validation_interval=1,
        learning_rate=0.01,
        weight_decay=0.0,
        training_amplitudes=values,
        validation_amplitudes=values,
    )
    assert [row["query_count"] for row in result.training_history] == [3, 1]
    assert result.query_count == result.no_context_query_count == 4
    assert result.no_context_rate == 1.0
    assert result.sample_count == 4 * 7  # Codec pads to eight; the objective does not.
    assert result.normalized_error_energy == pytest.approx(expected_errors.sum())
    assert result.train_loss == pytest.approx(expected_errors.mean())
    assert result.training_history[-1]["train_loss"] == pytest.approx(expected_errors.mean())
    assert result.best_step == 1  # Identical zero predictions keep the first validation.
    assert result.episodes_completed == 1 and not result.final_episode_interrupted


def test_final_step_validates_and_records_interrupted_episode() -> None:
    messages = []
    result = _train(
        _model(), max_steps=1, validation_interval=10, query_batch_size=3, reporter=messages.append
    )
    assert result.best_step == result.steps_completed == 1
    assert [row["step"] for row in result.validation_history] == [1]
    assert result.episodes_completed == 0 and result.final_episode_interrupted
    assert result.episode_history[0]["query_count"] == 3
    assert result.episode_history[0]["hidden_count"] == 4
    assert result.episode_history[0]["completed"] is False
    assert any("episode 1" in message for message in messages)
    assert any("validation step 1" in message for message in messages)


@pytest.mark.parametrize("partition", ["test", "train"])
def test_trainer_rejects_test_or_training_domain_for_fixed_validation(partition) -> None:
    train, validation, values, preprocessing, settings = _setup()
    validation = replace(validation, inputs_lock={"partition": partition})
    with pytest.raises(ValueError, match="validation partition"):
        train_relational_trace_graph(
            _model(),
            train,
            validation,
            preprocessing,
            graph_settings=settings,
            episode_kind_probabilities={"random_trace": 1.0},
            missing_fractions=[0.5],
            random_seed=1,
            max_steps=1,
            query_batch_size=2,
            validation_interval=1,
            learning_rate=0.01,
            weight_decay=0.0,
            training_amplitudes=values,
            validation_amplitudes=values,
        )
