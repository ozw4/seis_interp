"""Fixed observed-only GNN configuration without checkpoint selection."""

from collections.abc import Mapping
from dataclasses import dataclass

from seis_interp import config_values as values
from seis_interp.configuration import ConfigurationError
from seis_interp.evaluation.c3_volume_metrics import validate_c3_volume_evaluation_config
from seis_interp.processing.trace_graph_settings import TraceGraphSettings
from seis_interp.relational_trace_graph_config import (
    validate_trace_graph_graph_config,
    validate_trace_graph_model_config,
)
from seis_interp.training.mixed_precision import validate_mixed_precision
from seis_interp.training.trace_graph_sampling import validate_trace_graph_edge_sampling
from seis_interp.training.trace_relative_loss import POC_TRACE_LOSSES


@dataclass(frozen=True)
class RelationalTraceGraphPocSettings:
    """Constructor and fixed optimizer options for one volume."""

    model: dict[str, object]
    graph: TraceGraphSettings
    geometry: dict[str, float]
    training: dict[str, object]
    prediction_query_batch_size: int


def validate_relational_trace_graph_poc_config(config: Mapping) -> RelationalTraceGraphPocSettings:
    """Reject partition training, validation, and alternative amplitude/loss protocols."""
    allowed = {
        "project",
        "data",
        "benchmark_case",
        "benchmark_volume",
        "interpolation_mask",
        "model",
        "graph",
        "geometry_features",
        "training",
        "prediction",
        "evaluation",
    }
    if set(config) != allowed:
        raise ConfigurationError(f"PoC configuration requires exactly {sorted(allowed)}")
    project = values.exact_section(config, "project", {"random_seed"})
    values.nonnegative_integer(project["random_seed"], "project.random_seed")
    values.require_exact(config, "data.dataset_id", "seg_c3_na")
    model = validate_trace_graph_model_config(config)
    if model.get("amplitude_mode", "train_global_rms") != "train_global_rms":
        raise ConfigurationError("PoC model.amplitude_mode must be train_global_rms")
    if model.get("method_variant", "relational") != "relational":
        raise ConfigurationError("PoC model.method_variant must be relational")
    graph = validate_trace_graph_graph_config(config)
    graph.validate_model_config(model)
    geometry = dict(
        values.exact_section(
            config,
            "geometry_features",
            {"position_scale_m", "offset_scale_m", "azimuth_min_offset_m"},
        )
    )
    for key in ("position_scale_m", "offset_scale_m"):
        geometry[key] = values.positive_float(geometry[key], f"geometry_features.{key}")
    geometry["azimuth_min_offset_m"] = values.nonnegative_float(
        geometry["azimuth_min_offset_m"], "geometry_features.azimuth_min_offset_m"
    )
    raw_training = config.get("training")
    optional_training_keys = {"mixed_precision", "edge_sampling"}
    present_optional = (
        optional_training_keys.intersection(raw_training)
        if isinstance(raw_training, Mapping)
        else set()
    )
    training = dict(
        values.exact_section(
            config,
            "training",
            {
                "device",
                "model_initialization_seed",
                "episode_seed",
                "loss",
                "amplitude_scaling",
                "optimizer",
                "max_steps",
                "query_batch_size",
                "learning_rate",
                "weight_decay",
                "gradient_clip_norm",
                "inner_mask_fraction",
                "report_interval",
            }
            | present_optional,
        )
    )
    try:
        training["mixed_precision"] = validate_mixed_precision(
            training.get("mixed_precision", "off")
        )
    except ValueError as error:
        raise ConfigurationError(str(error)) from error
    if "edge_sampling" in training:
        try:
            training["edge_sampling"] = validate_trace_graph_edge_sampling(
                training["edge_sampling"], graph
            )
        except ValueError as error:
            raise ConfigurationError(str(error)) from error
    if training["loss"] not in POC_TRACE_LOSSES:
        raise ConfigurationError(f"training.loss must be one of {POC_TRACE_LOSSES!r}")
    for key, expected in {
        "amplitude_scaling": "observed_volume_global_rms",
        "optimizer": "adamw",
    }.items():
        values.require_exact(config, f"training.{key}", expected)
        training.pop(key)
    if not isinstance(training["device"], str) or not training["device"].strip():
        raise ConfigurationError("training.device must be a nonempty string")
    for key in ("model_initialization_seed", "episode_seed"):
        training[key] = values.nonnegative_integer(training[key], f"training.{key}")
    for key in ("max_steps", "query_batch_size", "report_interval"):
        training[key] = values.positive_integer(training[key], f"training.{key}")
    for key in ("learning_rate", "inner_mask_fraction"):
        training[key] = values.positive_float(training[key], f"training.{key}")
    if training["inner_mask_fraction"] >= 1:
        raise ConfigurationError("training.inner_mask_fraction must be less than 1")
    training["weight_decay"] = values.nonnegative_float(
        training["weight_decay"], "training.weight_decay"
    )
    if training["gradient_clip_norm"] is not None:
        training["gradient_clip_norm"] = values.positive_float(
            training["gradient_clip_norm"], "training.gradient_clip_norm"
        )
    prediction = values.exact_section(config, "prediction", {"query_batch_size"})
    batch_size = values.positive_integer(
        prediction["query_batch_size"], "prediction.query_batch_size"
    )
    values.exact_section(config, "evaluation", {"primary_metric", "domain"})
    validate_c3_volume_evaluation_config(config)
    return RelationalTraceGraphPocSettings(model, graph, geometry, training, batch_size)
