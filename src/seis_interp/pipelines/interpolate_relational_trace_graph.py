"""Frozen trace graph benchmark prediction followed by target-only physical scoring."""

from __future__ import annotations

import platform
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from seis_interp import run_records
from seis_interp.configuration import load_resolved_config
from seis_interp.data.file_checksums import file_sha256
from seis_interp.data.relational_trace_graph_run_inputs import (
    trace_graph_run_input_metadata,
    validate_trace_graph_checkpoint_provenance,
)
from seis_interp.data.trace_graph_domain import load_benchmark_trace_graph_domain
from seis_interp.data.trace_graph_prediction_store import save_trace_graph_prediction
from seis_interp.evaluation.trace_graph_diagnostic_metrics import (
    evaluate_trace_graph_baselines,
    evaluate_trace_graph_diagnostic_bands,
)
from seis_interp.evaluation.trace_graph_metrics import evaluate_trace_graph_prediction
from seis_interp.processing.c3_volume_index import validated_index_range
from seis_interp.processing.trace_graph_geometry import (
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    RELATION_NAMES,
)
from seis_interp.relational_trace_graph_config import (
    METHOD,
    TRAINING_REGIME,
    trace_graph_diagnostic_bands,
    trace_graph_method_variant,
    validate_relational_trace_graph_prediction_config,
)
from seis_interp.training.devices import resolve_device
from seis_interp.training.relational_trace_graph_checkpoints import (
    load_relational_trace_graph_checkpoint,
)
from seis_interp.training.relational_trace_graph_prediction import predict_relational_trace_graph


def interpolate_relational_trace_graph_run(
    *,
    config_path: Path,
    checkpoint_path: Path,
    interim_dir: Path,
    processed_dir: Path,
    mask_dir: Path,
    case_dir: Path,
    output_dir: Path,
    volume_dir: Path | None = None,
    device_override: str | None = None,
    progress_reporter: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Use checkpoint time/model/geometry with a verified native or selected dense case."""
    output = Path(output_dir)
    run_records.check_new_output_directory(output)
    config = load_resolved_config(Path(config_path))
    validate_relational_trace_graph_prediction_config(config)
    device = resolve_device(
        config["prediction"]["device"] if device_override is None else device_override,
        config_key="prediction.device",
    )
    config["prediction"]["device"] = str(device)
    started_at = run_records.utc_timestamp()
    git_metadata = run_records.current_git_metadata()
    started = time.perf_counter()
    if progress_reporter:
        progress_reporter("Loading frozen checkpoint and verifying benchmark inputs.")
    checkpoint_hash = file_sha256(Path(checkpoint_path))
    loaded = load_relational_trace_graph_checkpoint(Path(checkpoint_path), device=device)
    selection = validated_index_range(
        loaded.preprocessing.fit_domain["time_samples"], name="checkpoint.time_samples"
    )
    domain = load_benchmark_trace_graph_domain(
        interim_dir=interim_dir,
        processed_dir=processed_dir,
        mask_dir=mask_dir,
        case_dir=case_dir,
        volume_dir=volume_dir,
        time_samples=selection,
    )
    input_metadata = trace_graph_run_input_metadata(
        domain,
        config=config,
        processed_dir=processed_dir,
        case_dir=case_dir,
    )
    validate_trace_graph_checkpoint_provenance(loaded.training_provenance, domain)
    if not np.array_equal(domain.time_s, loaded.preprocessing.time_s):
        raise ValueError("checkpoint time grid does not match the benchmark samples")
    if progress_reporter:
        progress_reporter("Predicting missing queries with the frozen observed domain.")
    predicted = predict_relational_trace_graph(
        loaded.model,
        domain,
        loaded.preprocessing,
        graph_settings=loaded.graph_settings,
        query_batch_size=config["prediction"]["query_batch_size"],
        device=device,
        measure_resources=True,
    )
    if progress_reporter:
        progress_reporter("Evaluating target-only physical amplitudes.")
    metrics = evaluate_trace_graph_prediction(
        predicted.prediction,
        domain,
        query_trace_ids=predicted.query_trace_ids,
        has_observed_context=predicted.has_observed_context,
    )
    bands = trace_graph_diagnostic_bands(config)
    if bands is not None:
        metrics["diagnostic_bands"] = evaluate_trace_graph_diagnostic_bands(
            predicted.prediction,
            domain,
            query_trace_ids=predicted.query_trace_ids,
            bands=bands,
            azimuth_min_offset_m=loaded.preprocessing.azimuth_min_offset_m,
        )
        metrics["baselines"] = evaluate_trace_graph_baselines(
            domain,
            loaded.preprocessing,
            graph_settings=loaded.graph_settings,
            query_trace_ids=predicted.query_trace_ids,
            query_batch_size=config["prediction"]["query_batch_size"],
        )
    identity = {
        "method": METHOD,
        "method_variant": trace_graph_method_variant(loaded.model.constructor_config()),
        "training_regime": TRAINING_REGIME,
    }
    metrics.update({**identity, "case_id": input_metadata["case_id"]})
    output.mkdir(parents=True, exist_ok=False)
    prediction_metadata = save_trace_graph_prediction(
        output / "artifacts",
        predicted.prediction,
        domain,
        query_trace_ids=predicted.query_trace_ids,
        has_observed_context=predicted.has_observed_context,
        volume_dir=volume_dir,
    )
    checkpoint_record = {
        "path": str(checkpoint_path),
        "sha256": checkpoint_hash,
        "role": loaded.checkpoint_role,
        "step": loaded.global_step,
        "training_provenance": loaded.training_provenance,
    }
    inputs_lock = {
        **domain.inputs_lock,
        "checkpoint": checkpoint_record,
        "configuration_sha256": file_sha256(Path(config_path)),
    }
    metadata = {
        **identity,
        **git_metadata,
        "started_at_utc": started_at,
        "finished_at_utc": run_records.utc_timestamp(),
        "status": "success",
        "device": str(device),
        "random_seed": config["project"]["random_seed"],
        "training_random_seed": loaded.training_random_seed,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": str(torch.__version__),
        "input": input_metadata,
        "checkpoint": checkpoint_record,
        "model": loaded.model.constructor_config(),
        "parameter_count": sum(parameter.numel() for parameter in loaded.model.parameters()),
        "graph": {
            **loaded.graph_settings.constructor_config(),
            "relation_names": ["untyped"]
            if loaded.graph_settings.topology == "single_4d"
            else list(RELATION_NAMES),
        },
        "amplitude": {
            "scale_source": "checkpoint_fixed_training_pool",
            "amplitude_scale": loaded.preprocessing.amplitude_scale,
        },
        "geometry_features": {
            "node_feature_names": list(NODE_FEATURE_NAMES),
            "edge_feature_names": list(EDGE_FEATURE_NAMES),
            "position_scale_m": loaded.preprocessing.position_scale_m,
            "offset_scale_m": loaded.preprocessing.offset_scale_m,
            "midpoint_origin_m": list(loaded.preprocessing.midpoint_origin_m),
            "azimuth_min_offset_m": loaded.preprocessing.azimuth_min_offset_m,
        },
        "prediction": {
            **prediction_metadata,
            "query_batch_size": config["prediction"]["query_batch_size"],
            "diagnostics": predicted.diagnostics,
        },
        "resources": {
            "elapsed_seconds": time.perf_counter() - started,
            **run_records.runtime_resource_metadata(device),
        },
    }
    run_records.write_run_outputs(output, config, inputs_lock, metrics, metadata)
    return metrics
