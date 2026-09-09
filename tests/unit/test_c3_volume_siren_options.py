from __future__ import annotations

import pytest

from seis_interp.training.c3_volume_siren_options import (
    complete_trace_training_options,
    initial_time_weight_scale,
)


def test_random_points_remain_the_default_and_reject_ignored_options():
    assert complete_trace_training_options({"learning_rate": 1e-4}) is None
    for key in (
        "traces_per_step",
        "learning_rate_schedule",
        "minimum_learning_rate",
        "envelope_loss",
    ):
        with pytest.raises(ValueError, match="require random_complete_traces"):
            complete_trace_training_options({key: None})


@pytest.mark.parametrize("count", [None, 4])
def test_complete_traces_keep_explicit_full_or_sampled_pool_and_cosine(count):
    options = complete_trace_training_options(
        {
            "batch_mode": "random_complete_traces",
            "traces_per_step": count,
            "learning_rate": 1e-4,
            "learning_rate_schedule": "cosine",
            "minimum_learning_rate": 1e-6,
        }
    )
    assert options == {
        "traces_per_step": count,
        "learning_rate_schedule": "cosine",
        "minimum_learning_rate": 1e-6,
    }


@pytest.mark.parametrize(
    "change",
    [
        {"batch_mode": "unknown"},
        {"traces_per_step": True},
        {"traces_per_step": 0},
        {"traces_per_step": 1.5},
        {"learning_rate_schedule": "unknown"},
        {"learning_rate_schedule": "cosine"},
        {"learning_rate_schedule": "cosine", "minimum_learning_rate": 1e-4},
        {"learning_rate_schedule": "cosine", "minimum_learning_rate": float("nan")},
        {"minimum_learning_rate": 1e-6},
    ],
)
def test_invalid_complete_trace_training_options_fail(change):
    training = {
        "batch_mode": "random_complete_traces",
        "traces_per_step": None,
        "learning_rate": 1e-4,
        **change,
    }
    with pytest.raises(ValueError):
        complete_trace_training_options(training)


def test_complete_trace_pool_size_must_be_declared():
    with pytest.raises(ValueError, match="requires training.traces_per_step"):
        complete_trace_training_options({"batch_mode": "random_complete_traces"})


def test_time_weight_scale_defaults_to_one_and_accepts_explicit_multiplier():
    assert initial_time_weight_scale({}) == 1.0
    assert initial_time_weight_scale({"initial_time_weight_scale": 3}) == 3.0


@pytest.mark.parametrize("value", [True, 0, -1, "3", None, float("nan"), float("inf")])
def test_time_weight_scale_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="initial_time_weight_scale"):
        initial_time_weight_scale({"initial_time_weight_scale": value})


def test_envelope_options_preserve_declared_decay_in_short_preflight():
    envelope = {"weight": 1.0, "sigma_samples": [4.0, 8.0], "decay_steps": 2500}
    training = {
        "batch_mode": "random_complete_traces",
        "traces_per_step": None,
        "learning_rate": 1e-4,
        "batch_size": 262144,
        "max_steps": 10,
        "envelope_loss": envelope,
    }
    assert complete_trace_training_options(training, time_count=384)["envelope_loss"] == envelope
    with pytest.raises(ValueError):
        complete_trace_training_options(training, time_count=16)
    with pytest.raises(ValueError):
        complete_trace_training_options({**training, "batch_size": 383}, time_count=384)
