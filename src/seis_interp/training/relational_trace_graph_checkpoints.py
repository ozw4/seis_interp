"""Complete inference snapshots for the grid-free relational trace model."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import torch

from seis_interp.models.relational_trace_graph import RelationalTraceGraphInterpolator
from seis_interp.processing.trace_graph_geometry import (
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    RELATION_NAMES,
)
from seis_interp.processing.trace_graph_preprocessing import TraceGraphPreprocessing
from seis_interp.processing.trace_graph_settings import TraceGraphSettings
from seis_interp.training.trace_relative_loss import POC_TRACE_LOSSES

RELATIONAL_TRACE_GRAPH_MODEL_TYPE = "relational_trace_graph"
BEST_VALIDATION_CHECKPOINT_ROLE = "best_validation"
FINAL_CHECKPOINT_ROLE = "final"
_MODEL_FIELDS = {
    "width",
    "message_passing_rounds",
    "time_downsample_factor",
    "stem_kernel_size",
    "temporal_kernel_size",
    "temporal_dilations",
    "attention_width",
    "relation_embedding_dim",
    "relation_fusion",
}
_MODEL_OPTIONS = {
    "method_variant",
    "explicit_azimuth_features",
    "amplitude_mode",
    "max_edge_time_shift_samples",
}


@dataclass(frozen=True)
class LoadedRelationalTraceGraphCheckpoint:
    """Rebuilt model, fixed inputs, and independent training provenance."""

    model: RelationalTraceGraphInterpolator
    preprocessing: TraceGraphPreprocessing
    graph_settings: TraceGraphSettings
    training_mask: dict[str, object]
    training_provenance: dict[str, object]
    training_random_seed: int
    checkpoint_role: str
    global_step: int
    selection_metrics: dict[str, object]


@dataclass(frozen=True)
class LoadedRelationalTraceGraphPocCheckpoint:
    """Strictly restored per-volume inference state and training metadata."""

    model: RelationalTraceGraphInterpolator
    preprocessing: TraceGraphPreprocessing
    graph_settings: TraceGraphSettings
    metadata: dict[str, object]


def save_relational_trace_graph_poc_checkpoint(
    path: Path,
    *,
    model_config: Mapping[str, object],
    state_dict: Mapping[str, torch.Tensor],
    preprocessing: TraceGraphPreprocessing,
    graph_settings: TraceGraphSettings,
    metadata: Mapping[str, object],
    inputs_lock: Mapping[str, object],
) -> None:
    """Validate and atomically save a CPU final snapshot without advancing RNG."""
    config = _model_config(model_config)
    payload = {
        **_json_mapping(metadata, "metadata"),
        "model_type": RELATIONAL_TRACE_GRAPH_MODEL_TYPE,
        "model_config": config,
        "model_state_dict": _snapshot(state_dict),
        "graph_settings": graph_settings.constructor_config(),
        "relation_names": ["untyped"]
        if graph_settings.topology == "single_4d"
        else list(RELATION_NAMES),
        "node_feature_names": list(NODE_FEATURE_NAMES),
        "edge_feature_names": list(EDGE_FEATURE_NAMES),
        "preprocessing": _json_mapping(asdict(preprocessing), "preprocessing"),
        "time": {
            "time_s": list(preprocessing.time_s),
            "sample_count": len(preprocessing.time_s),
            "time_downsample_factor": config["time_downsample_factor"],
        },
        "inputs_lock": _json_mapping(inputs_lock, "inputs_lock"),
    }
    with torch.random.fork_rng(devices=[]):
        _load_poc_payload(payload, inputs_lock=inputs_lock, device="cpu")
    _atomic_save(path, payload)


def load_relational_trace_graph_poc_checkpoint(
    path: Path,
    *,
    inputs_lock: Mapping[str, object],
    device: torch.device | str = "cpu",
) -> LoadedRelationalTraceGraphPocCheckpoint:
    """Restore inference state only when the entire verified current input lock matches."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    return _load_poc_payload(payload, inputs_lock=inputs_lock, device=device)


def _load_poc_payload(
    payload: object, *, inputs_lock: Mapping[str, object], device: torch.device | str
) -> LoadedRelationalTraceGraphPocCheckpoint:
    if not isinstance(payload, Mapping):
        raise ValueError("PoC checkpoint must contain a mapping")
    expected_lock = _json_mapping(inputs_lock, "inputs_lock")
    if not expected_lock or payload.get("inputs_lock") != expected_lock:
        raise ValueError("checkpoint inputs_lock does not match current inputs")
    for name, expected in {
        "method": RELATIONAL_TRACE_GRAPH_MODEL_TYPE,
        "training_domain": "O_with_inner_pseudo_mask",
        "checkpoint_role": "final",
        "output_amplitude_domain": "physical",
        "case_id": expected_lock.get("case_id"),
        "volume_id": expected_lock.get("volume_id"),
    }.items():
        if expected is None or payload.get(name) != expected:
            raise ValueError(f"checkpoint {name} does not match the PoC contract")
    if payload.get("loss") not in POC_TRACE_LOSSES:
        raise ValueError(f"checkpoint loss must be one of {POC_TRACE_LOSSES!r}")
    for key in ("model_initialization_seed", "episode_seed"):
        _integer(payload.get(key), key, minimum=0)
    _integer(payload.get("steps_completed"), "steps_completed", minimum=1)
    fraction = payload.get("inner_mask_fraction")
    if not _finite_number(fraction) or not 0 < fraction < 1:
        raise ValueError("checkpoint inner_mask_fraction must be in (0, 1)")
    model, preprocessing, graph = _load_inference(
        {**payload, "state_dict": payload.get("model_state_dict")}, device=device
    )
    fit = preprocessing.fit_domain
    if (
        not isinstance(fit, Mapping)
        or set(fit) != {"coordinate_source", "midpoint_bounds_m", "amplitude_source", "inputs_lock"}
        or fit.get("coordinate_source") != "fixed_analysis_domain"
        or fit.get("amplitude_source") != "external_observed_global_rms"
        or fit.get("inputs_lock") != expected_lock
    ):
        raise ValueError("checkpoint preprocessing.fit_domain must match the PoC inputs")
    bounds = np.asarray(fit["midpoint_bounds_m"], dtype=np.float64)
    if (
        bounds.shape != (2, 2)
        or not np.all(np.isfinite(bounds))
        or np.any(bounds[1] < bounds[0])
        or not np.array_equal(
            bounds[0] + (bounds[1] - bounds[0]) / 2, preprocessing.midpoint_origin_m
        )
    ):
        raise ValueError("checkpoint midpoint bounds do not match preprocessing origin")
    if payload.get("normalization") != {
        "type": "global_rms",
        "source": "O_only",
        "scale": preprocessing.amplitude_scale,
    }:
        raise ValueError("checkpoint normalization does not match preprocessing")
    metadata = _json_mapping(
        {key: value for key, value in payload.items() if key != "model_state_dict"}, "metadata"
    )
    model.eval()
    return LoadedRelationalTraceGraphPocCheckpoint(model, preprocessing, graph, metadata)


def save_relational_trace_graph_checkpoint(
    path: Path,
    *,
    model_config: Mapping[str, object],
    state_dict: Mapping[str, torch.Tensor],
    preprocessing: TraceGraphPreprocessing,
    graph_settings: TraceGraphSettings,
    training_mask: Mapping[str, object],
    training_provenance: Mapping[str, object],
    training_random_seed: int,
    checkpoint_role: str,
    global_step: int,
    selection_metrics: Mapping[str, object],
) -> None:
    """Save a detached CPU snapshot; optimizer/resume state is out of scope.

    Accepting a state dictionary separately permits saving a best-validation
    snapshot after training without changing the final model's parameters.
    All metadata is copied into pure JSON values; no module objects are saved.
    A complete temporary file replaces the destination only after saving succeeds.
    """
    if not isinstance(preprocessing, TraceGraphPreprocessing):
        raise TypeError("preprocessing must be TraceGraphPreprocessing")
    if not isinstance(graph_settings, TraceGraphSettings):
        raise TypeError("graph_settings must be TraceGraphSettings")
    config = _model_config(model_config)
    snapshot = _snapshot(state_dict)
    payload = {
        "model_type": RELATIONAL_TRACE_GRAPH_MODEL_TYPE,
        "model_config": config,
        "state_dict": snapshot,
        "relation_names": ["untyped"]
        if graph_settings.topology == "single_4d"
        else list(RELATION_NAMES),
        "node_feature_names": list(NODE_FEATURE_NAMES),
        "edge_feature_names": list(EDGE_FEATURE_NAMES),
        "graph_settings": graph_settings.constructor_config(),
        "preprocessing": _json_mapping(asdict(preprocessing), "preprocessing"),
        "time": {
            "time_s": list(preprocessing.time_s),
            "sample_count": len(preprocessing.time_s),
            "time_downsample_factor": config["time_downsample_factor"],
        },
        "training_mask": _training_mask(training_mask),
        "training_provenance": _json_mapping(training_provenance, "training_provenance"),
        "training_random_seed": _integer(training_random_seed, "training_random_seed", minimum=0),
        "checkpoint_role": _role(checkpoint_role),
        "global_step": _integer(global_step, "global_step", minimum=0),
        "selection_metrics": _json_mapping(selection_metrics, "selection_metrics"),
    }
    # Validate constructor/state and temporal consistency before writing, while
    # keeping a checkpoint save from advancing the caller's model RNG stream.
    with torch.random.fork_rng(devices=[]):
        _load_payload(payload, device="cpu")
    _atomic_save(path, payload)


def _atomic_save(path: Path, payload: Mapping) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.name}-", suffix=".tmp", delete=False
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        torch.save(payload, temporary_path)
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_relational_trace_graph_checkpoint(
    path: Path,
    *,
    device: torch.device | str = "cpu",
) -> LoadedRelationalTraceGraphCheckpoint:
    """Reconstruct from complete configuration and load tensor weights strictly."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    return _load_payload(payload, device=device)


def _load_payload(
    payload: object, *, device: torch.device | str
) -> LoadedRelationalTraceGraphCheckpoint:
    model, preprocessing, graph_settings = _load_inference(payload, device=device)
    _training_fit_domain(preprocessing)
    try:
        mask = _training_mask(payload["training_mask"])
        provenance = _json_mapping(payload["training_provenance"], "training_provenance")
        seed = _integer(payload["training_random_seed"], "training_random_seed", minimum=0)
        role = _role(payload["checkpoint_role"])
        step = _integer(payload["global_step"], "global_step", minimum=0)
        metrics = _json_mapping(payload["selection_metrics"], "selection_metrics")
    except KeyError as error:
        raise ValueError(f"checkpoint is missing required field {error.args[0]!r}") from error
    return LoadedRelationalTraceGraphCheckpoint(
        model, preprocessing, graph_settings, mask, provenance, seed, role, step, metrics
    )


def _load_inference(
    payload: object, *, device: torch.device | str
) -> tuple[RelationalTraceGraphInterpolator, TraceGraphPreprocessing, TraceGraphSettings]:
    if not isinstance(payload, Mapping):
        raise ValueError("relational trace graph checkpoint must contain a mapping")
    if payload.get("model_type") != RELATIONAL_TRACE_GRAPH_MODEL_TYPE:
        raise ValueError("checkpoint model_type must be 'relational_trace_graph'")
    for name, expected in (
        ("node_feature_names", NODE_FEATURE_NAMES),
        ("edge_feature_names", EDGE_FEATURE_NAMES),
    ):
        if payload.get(name) != list(expected):
            raise ValueError(f"checkpoint {name} does not match the required order")
    try:
        config = _model_config(payload["model_config"])
        graph_config = _json_mapping(payload["graph_settings"], "graph_settings")
        graph_required = {
            "relation_scales_m",
            "neighbors_per_relation",
            "radius",
            "candidate_chunk_size",
        }
        graph_optional = {
            "neighbor_search",
            "topology",
            "excluded_relation",
            "common_distance_scales_m",
            "single_4d_neighbors",
        }
        if not graph_required <= set(graph_config) or set(graph_config) - (
            graph_required | graph_optional
        ):
            raise ValueError("checkpoint graph_settings must contain all fixed neighbor settings")
        graph_settings = TraceGraphSettings(**graph_config)
        graph_settings.validate_model_config(config)
        relations = ["untyped"] if graph_settings.topology == "single_4d" else list(RELATION_NAMES)
        if payload.get("relation_names") != relations:
            raise ValueError("checkpoint relation_names does not match the required order")
        preprocessing = _preprocessing(payload["preprocessing"])
        _time_metadata(payload["time"], preprocessing, config)
        state_dict = payload["state_dict"]
    except KeyError as error:
        raise ValueError(f"checkpoint is missing required field {error.args[0]!r}") from error
    if not isinstance(state_dict, Mapping) or not all(
        isinstance(name, str) and isinstance(tensor, torch.Tensor)
        for name, tensor in state_dict.items()
    ):
        raise ValueError("checkpoint state_dict must map names to tensors")
    try:
        model = RelationalTraceGraphInterpolator(**config)
    except (TypeError, ValueError) as error:
        raise ValueError(f"checkpoint model_config is invalid: {error}") from error
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise ValueError("checkpoint state_dict does not match model_config") from error
    model.to(device)
    return model, preprocessing, graph_settings


def _model_config(value: object) -> dict[str, object]:
    config = _json_mapping(value, "model_config")
    if not set(config) >= _MODEL_FIELDS or set(config) - (_MODEL_FIELDS | _MODEL_OPTIONS):
        raise ValueError("checkpoint model_config must contain exactly all constructor fields")
    for name in _MODEL_FIELDS - {"temporal_dilations", "relation_fusion"}:
        _integer(config[name], f"model_config.{name}", minimum=1)
    if "max_edge_time_shift_samples" in config:
        _integer(
            config["max_edge_time_shift_samples"],
            "model_config.max_edge_time_shift_samples",
            minimum=0,
        )
    dilations = config["temporal_dilations"]
    if not isinstance(dilations, list) or len(dilations) != config["message_passing_rounds"]:
        raise ValueError("model_config.temporal_dilations must match message_passing_rounds")
    for dilation in dilations:
        _integer(dilation, "model_config.temporal_dilations", minimum=1)
    return config


def _preprocessing(value: object) -> TraceGraphPreprocessing:
    config = _json_mapping(value, "preprocessing")
    try:
        config["midpoint_origin_m"] = tuple(config["midpoint_origin_m"])
        config["time_s"] = tuple(config["time_s"])
        preprocessing = TraceGraphPreprocessing(**config)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"checkpoint preprocessing is invalid: {error}") from error
    return preprocessing


def _training_fit_domain(preprocessing: TraceGraphPreprocessing) -> None:
    fit = preprocessing.fit_domain
    if not isinstance(fit, Mapping) or fit.get("pool") not in ("all_train_traces", "mask_observed"):
        raise ValueError("checkpoint preprocessing.fit_domain must identify the training pool")
    samples = fit.get("time_samples")
    if not isinstance(samples, list) or len(samples) != 2:
        raise ValueError("checkpoint preprocessing.fit_domain must identify time_samples")
    start = _integer(samples[0], "fit_domain.time_samples start", minimum=0)
    stop = _integer(samples[1], "fit_domain.time_samples stop", minimum=1)
    if stop - start != len(preprocessing.time_s):
        raise ValueError("checkpoint fit_domain.time_samples must match the time grid length")


def _time_metadata(value: object, preprocessing: TraceGraphPreprocessing, config: Mapping) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint time must be a mapping")
    if value.get("time_s") != list(preprocessing.time_s):
        raise ValueError("checkpoint time_s does not match preprocessing time grid")
    count = _integer(value.get("sample_count"), "time.sample_count", minimum=1)
    factor = _integer(value.get("time_downsample_factor"), "time.time_downsample_factor", minimum=1)
    if count != len(preprocessing.time_s):
        raise ValueError("checkpoint sample_count does not match time_s")
    if factor != config["time_downsample_factor"]:
        raise ValueError("checkpoint time_downsample_factor does not match model_config")


def _training_mask(value: object) -> dict[str, object]:
    config = _json_mapping(value, "training_mask")
    if set(config) != {"kinds", "kind_probabilities", "missing_fractions"}:
        raise ValueError("training_mask requires kinds, kind_probabilities and missing_fractions")
    kinds = config["kinds"]
    if (
        not isinstance(kinds, list)
        or not kinds
        or any(kind not in ("random_trace", "random_whole_ffid") for kind in kinds)
        or len(set(kinds)) != len(kinds)
    ):
        raise ValueError("training_mask.kinds must contain unique supported mask kinds")
    probabilities = config["kind_probabilities"]
    if not isinstance(probabilities, list) or len(probabilities) != len(kinds):
        raise ValueError("training_mask.kind_probabilities must match kinds")
    if not all(_finite_number(value) and value >= 0 for value in probabilities) or not np.isclose(
        sum(probabilities), 1.0, rtol=0, atol=1e-8
    ):
        raise ValueError("training_mask.kind_probabilities must be nonnegative and sum to one")
    fractions = config["missing_fractions"]
    if (
        not isinstance(fractions, list)
        or not fractions
        or not all(_finite_number(value) and 0 < value < 1 for value in fractions)
    ):
        raise ValueError("training_mask.missing_fractions must contain fractions in (0, 1)")
    return config


def _snapshot(value: object) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping):
        raise ValueError("state_dict must map names to tensors")
    snapshot = {}
    for name, tensor in value.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("state_dict must map names to tensors")
        snapshot[name] = tensor.detach().cpu().clone()
    return snapshot


def _json_mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    try:
        return json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain strict JSON metadata") from error


def _role(value: object) -> str:
    if value not in (BEST_VALIDATION_CHECKPOINT_ROLE, FINAL_CHECKPOINT_ROLE):
        raise ValueError("checkpoint_role must be 'best_validation' or 'final'")
    return value


def _integer(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _finite_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, Real) and np.isfinite(value)
