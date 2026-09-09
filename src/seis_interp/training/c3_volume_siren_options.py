"""Validate opt-in complete-trace training options for a C3 SIREN run."""

from __future__ import annotations

from collections.abc import Mapping

from seis_interp import config_values
from seis_interp.configuration import ConfigurationError
from seis_interp.training.trace_envelope_loss import validate_trace_envelope_loss_options


def initial_time_weight_scale(training: Mapping) -> float:
    """Validate the optional time-column initialization multiplier."""
    return config_values.positive_float(
        training.get("initial_time_weight_scale", 1.0), "training.initial_time_weight_scale"
    )


def complete_trace_training_options(
    training: Mapping, *, time_count: int | None = None
) -> dict | None:
    """Return complete-trace trainer arguments, preserving the random-point default."""
    mode = training.get("batch_mode", "random_points")
    extra_keys = {
        "traces_per_step",
        "learning_rate_schedule",
        "minimum_learning_rate",
        "envelope_loss",
    }
    if mode == "random_points":
        if extra_keys.intersection(training):
            raise ConfigurationError("complete-trace options require random_complete_traces")
        return None
    if mode != "random_complete_traces":
        raise ConfigurationError(
            "training.batch_mode must be random_points or random_complete_traces"
        )
    if "traces_per_step" not in training:
        raise ConfigurationError("random_complete_traces requires training.traces_per_step")
    count = training["traces_per_step"]
    if count is not None:
        count = config_values.positive_integer(count, "training.traces_per_step")
    schedule = training.get("learning_rate_schedule", "constant")
    if schedule not in ("constant", "cosine"):
        raise ConfigurationError("training.learning_rate_schedule must be constant or cosine")
    minimum = training.get("minimum_learning_rate")
    if schedule == "cosine":
        minimum = config_values.positive_float(minimum, "training.minimum_learning_rate")
        if minimum >= config_values.positive_float(
            training["learning_rate"], "training.learning_rate"
        ):
            raise ConfigurationError("minimum_learning_rate must be below learning_rate")
    elif minimum is not None:
        raise ConfigurationError("minimum_learning_rate requires the cosine schedule")
    options = {
        "traces_per_step": count,
        "learning_rate_schedule": schedule,
        "minimum_learning_rate": minimum,
    }
    envelope = validate_trace_envelope_loss_options(
        training.get("envelope_loss"),
        time_count=time_count,
        microbatch_size=training.get("batch_size") if time_count is not None else None,
    )
    if envelope is not None:
        options["envelope_loss"] = envelope
    return options
