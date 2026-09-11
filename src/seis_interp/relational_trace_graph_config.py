"""Executable configuration for masked trace training and frozen prediction."""

from __future__ import annotations

from collections.abc import Mapping

from seis_interp import config_values
from seis_interp.configuration import ConfigurationError
from seis_interp.evaluation.c3_volume_metrics import validate_c3_volume_evaluation_config
from seis_interp.evaluation.trace_graph_diagnostic_metrics import TraceGraphDiagnosticBands
from seis_interp.processing.c3_volume_index import validated_index_range
from seis_interp.processing.trace_graph_geometry import RELATION_NAMES
from seis_interp.processing.trace_graph_settings import TraceGraphSettings

METHOD = "relational_trace_graph"
TRAINING_REGIME = "masked_train_partition"


def validate_relational_trace_graph_training_config(
    config: Mapping[str, object],
) -> tuple[dict[str, object], TraceGraphSettings, dict[str, object]]:
    """Validate consumed study settings and return model, graph and trainer options."""
    _root_sections(
        config,
        {
            "project",
            "data",
            "model",
            "graph",
            "geometry_features",
            "training_data",
            "training_mask",
            "training",
            "evaluation",
            "diagnostics",
        },
    )
    _project_and_data(config)
    model = validate_trace_graph_model_config(config)
    graph = validate_trace_graph_graph_config(config)
    graph.validate_model_config(model)
    geometry = config_values.exact_section(
        config, "geometry_features", {"position_scale_m", "offset_scale_m", "azimuth_min_offset_m"}
    )
    for name in ("position_scale_m", "offset_scale_m"):
        config_values.positive_float(geometry[name], f"geometry_features.{name}")
    config_values.nonnegative_float(
        geometry["azimuth_min_offset_m"], "geometry_features.azimuth_min_offset_m"
    )
    data = _section_with_options(
        config, "training_data", {"pool", "time_samples"}, {"max_abs_amplitude"}
    )
    if data["pool"] not in ("all_train_traces", "mask_observed"):
        raise ConfigurationError("training_data.pool must be all_train_traces or mask_observed")
    validated_index_range(data["time_samples"], name="training_data.time_samples")
    if "max_abs_amplitude" in data:
        config_values.positive_float(data["max_abs_amplitude"], "training_data.max_abs_amplitude")
    mask = config_values.exact_section(
        config, "training_mask", {"kinds", "kind_probabilities", "missing_fractions"}
    )
    kinds = mask["kinds"]
    if (
        not isinstance(kinds, list)
        or not kinds
        or any(kind not in ("random_trace", "random_whole_ffid") for kind in kinds)
        or len(set(kinds)) != len(kinds)
    ):
        raise ConfigurationError("training_mask.kinds must contain unique supported kinds")
    probabilities = mask["kind_probabilities"]
    if not isinstance(probabilities, list) or len(probabilities) != len(kinds):
        raise ConfigurationError("training_mask.kind_probabilities must match kinds")
    probabilities = [
        config_values.nonnegative_float(p, "training_mask.kind_probabilities")
        for p in probabilities
    ]
    if abs(sum(probabilities) - 1.0) > 1e-8:
        raise ConfigurationError("training_mask.kind_probabilities must sum to 1")
    fractions = mask["missing_fractions"]
    if not isinstance(fractions, list) or not fractions:
        raise ConfigurationError("training_mask.missing_fractions must be nonempty")
    for value in fractions:
        fraction = config_values.positive_float(value, "training_mask.missing_fractions")
        if fraction >= 1:
            raise ConfigurationError("training_mask.missing_fractions must be in (0, 1)")
    training = dict(
        _section_with_options(
            config,
            "training",
            {
                "device",
                "random_seed",
                "loss",
                "max_steps",
                "query_batch_size",
                "validation_interval",
                "learning_rate",
                "weight_decay",
                "gradient_clip_norm",
            },
            {"cudnn_benchmark"},
        )
    )
    cudnn_benchmark = training.pop("cudnn_benchmark", True)
    if not isinstance(cudnn_benchmark, bool):
        raise ConfigurationError("training.cudnn_benchmark must be a boolean")
    if not cudnn_benchmark:
        training["cudnn_benchmark"] = False
    loss = training.pop("loss")
    if loss not in ("masked_mse", "masked_trace_relative_mse"):
        raise ConfigurationError("training.loss must be masked_mse or masked_trace_relative_mse")
    if loss != "masked_mse":
        training["loss"] = loss
    _device_name(training.pop("device"), "training.device")
    training["random_seed"] = config_values.nonnegative_integer(
        training["random_seed"], "training.random_seed"
    )
    for name in ("max_steps", "query_batch_size", "validation_interval"):
        training[name] = config_values.positive_integer(training[name], f"training.{name}")
    training["learning_rate"] = config_values.positive_float(
        training["learning_rate"], "training.learning_rate"
    )
    training["weight_decay"] = config_values.nonnegative_float(
        training["weight_decay"], "training.weight_decay"
    )
    if training["gradient_clip_norm"] is not None:
        training["gradient_clip_norm"] = config_values.positive_float(
            training["gradient_clip_norm"], "training.gradient_clip_norm"
        )
    evaluation = config_values.exact_section(
        config, "evaluation", {"primary_metric", "domain", "query_batch_size"}
    )
    validate_c3_volume_evaluation_config(config)
    training["validation_query_batch_size"] = config_values.positive_integer(
        evaluation["query_batch_size"], "evaluation.query_batch_size"
    )
    training["episode_kind_probabilities"] = dict(zip(kinds, probabilities, strict=True))
    training["missing_fractions"] = tuple(fractions)
    bands = trace_graph_diagnostic_bands(config)
    if bands is not None:
        training["diagnostic_bands"] = bands
    return model, graph, training


def validate_relational_trace_graph_prediction_config(config: Mapping[str, object]) -> None:
    """Keep model, graph, geometry and amplitude settings exclusively in the checkpoint."""
    _root_sections(
        config,
        {
            "project",
            "data",
            "prediction",
            "evaluation",
            "benchmark_case",
            "interpolation_mask",
            "benchmark_volume",
            "diagnostics",
        },
    )
    _project_and_data(config)
    prediction = config_values.exact_section(config, "prediction", {"device", "query_batch_size"})
    _device_name(prediction["device"], "prediction.device")
    config_values.positive_integer(prediction["query_batch_size"], "prediction.query_batch_size")
    config_values.exact_section(config, "evaluation", {"primary_metric", "domain"})
    validate_c3_volume_evaluation_config(config)
    trace_graph_diagnostic_bands(config)
    case = config_values.exact_section(config, "benchmark_case", {"id"})
    if not isinstance(case["id"], str) or not case["id"]:
        raise ConfigurationError("benchmark_case.id must be a nonempty string")
    if "interpolation_mask" in config:
        config_values.exact_section(
            config, "interpolation_mask", {"partition", "kind", "missing_fraction"}
        )
    if "benchmark_volume" in config:
        config_values.exact_section(config, "benchmark_volume", {"id", "selection"})


def validate_trace_graph_model_config(config: Mapping[str, object]) -> dict[str, object]:
    """Validate model constructor settings independently of experiment wiring."""
    model = dict(
        _section_with_options(
            config,
            "model",
            {
                "name",
                "width",
                "message_passing_rounds",
                "time_downsample_factor",
                "stem_kernel_size",
                "temporal_kernel_size",
                "temporal_dilations",
                "attention_width",
                "relation_embedding_dim",
                "relation_fusion",
            },
            {
                "method_variant",
                "explicit_azimuth_features",
                "amplitude_mode",
                "max_edge_time_shift_samples",
            },
        )
    )
    if model.pop("name") != METHOD:
        raise ConfigurationError(f"model.name must be {METHOD!r}")
    for name in (
        "width",
        "message_passing_rounds",
        "time_downsample_factor",
        "attention_width",
        "relation_embedding_dim",
    ):
        model[name] = config_values.positive_integer(model[name], f"model.{name}")
    if model["width"] % 8:
        raise ConfigurationError("model.width must be divisible by 8")
    for name in ("stem_kernel_size", "temporal_kernel_size"):
        model[name] = config_values.odd_positive_integer(model[name], f"model.{name}")
    model["temporal_dilations"] = config_values.validated_positive_integer_list(
        model["temporal_dilations"], "model.temporal_dilations"
    )
    if len(model["temporal_dilations"]) != model["message_passing_rounds"]:
        raise ConfigurationError("model.temporal_dilations must match message_passing_rounds")
    if model["relation_fusion"] not in ("mean", "learned_gate"):
        raise ConfigurationError("model.relation_fusion must be mean or learned_gate")
    variant = model.get("method_variant", "relational")
    if variant not in ("relational", "plain_gcn_row_normalized", "untyped_edge_conditioned"):
        raise ConfigurationError("unsupported model.method_variant")
    if variant != "relational" and model["relation_fusion"] != "mean":
        raise ConfigurationError("comparison models require relation_fusion=mean")
    if not isinstance(model.get("explicit_azimuth_features", True), bool):
        raise ConfigurationError("model.explicit_azimuth_features must be boolean")
    if model.get("amplitude_mode", "train_global_rms") not in (
        "train_global_rms",
        "observed_trace_rms",
    ):
        raise ConfigurationError("unsupported model.amplitude_mode")
    if "max_edge_time_shift_samples" in model:
        model["max_edge_time_shift_samples"] = config_values.nonnegative_integer(
            model["max_edge_time_shift_samples"], "model.max_edge_time_shift_samples"
        )
    return model


def validate_trace_graph_graph_config(config: Mapping[str, object]) -> TraceGraphSettings:
    """Validate geometry-only neighbor settings for a trace graph."""
    graph = _section_with_options(
        config,
        "graph",
        {"neighbors_per_relation", "max_normalized_distance", "candidate_chunk_size", "relations"},
        {
            "topology",
            "excluded_relation",
            "common_distance_scales_m",
            "single_4d_neighbors",
            "neighbor_search",
        },
    )
    relations = config_values.exact_section(graph, "relations", set(RELATION_NAMES))
    scales = []
    for relation in RELATION_NAMES:
        keys = (
            ("source_scale_m", "receiver_scale_m")
            if relation in ("source", "receiver")
            else ("midpoint_scale_m", "offset_vector_scale_m")
        )
        values = config_values.exact_section(relations, relation, set(keys))
        scales.append(
            tuple(
                config_values.positive_float(values[name], f"graph.relations.{relation}.{name}")
                for name in keys
            )
        )
    return TraceGraphSettings(
        relation_scales_m=tuple(scales),
        neighbors_per_relation=config_values.positive_integer(
            graph["neighbors_per_relation"], "graph.neighbors_per_relation"
        ),
        radius=config_values.positive_float(
            graph["max_normalized_distance"], "graph.max_normalized_distance"
        ),
        candidate_chunk_size=config_values.positive_integer(
            graph["candidate_chunk_size"], "graph.candidate_chunk_size"
        ),
        **{
            name: graph[name]
            for name in (
                "topology",
                "excluded_relation",
                "common_distance_scales_m",
                "single_4d_neighbors",
                "neighbor_search",
            )
            if name in graph
        },
    )


def trace_graph_diagnostic_bands(config: Mapping[str, object]) -> TraceGraphDiagnosticBands | None:
    """Read fixed diagnostic cuts; absent diagnostics leave existing runs unchanged."""
    if "diagnostics" not in config:
        return None
    values = config_values.exact_section(
        config, "diagnostics", {"time_s", "offset_m", "azimuth_deg"}
    )
    return TraceGraphDiagnosticBands(**values)


def trace_graph_method_variant(model_config: Mapping[str, object]) -> str:
    """Name relation fusion for the proposed model and the actual control otherwise."""
    variant = model_config.get("method_variant", "relational")
    return str(model_config["relation_fusion"] if variant == "relational" else variant)


def _section_with_options(config, name, required, optional):
    section = config.get(name)
    if (
        not isinstance(section, Mapping)
        or not required <= set(section)
        or set(section) - (required | optional)
    ):
        raise ConfigurationError(
            f"{name} configuration requires {sorted(required)} with optional {sorted(optional)}"
        )
    return section


def _project_and_data(config: Mapping[str, object]) -> None:
    project = config_values.exact_section(config, "project", {"random_seed"})
    config_values.nonnegative_integer(project["random_seed"], "project.random_seed")
    data = config_values.exact_section(config, "data", {"dataset_id"})
    if not isinstance(data["dataset_id"], str) or not data["dataset_id"]:
        raise ConfigurationError("data.dataset_id must be a nonempty string")


def _device_name(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{name} must be a nonempty string")


def _root_sections(config: Mapping[str, object], allowed: set[str]) -> None:
    if set(config) - allowed:
        raise ConfigurationError(
            f"unsupported relational trace graph config sections: {sorted(set(config) - allowed)}"
        )
