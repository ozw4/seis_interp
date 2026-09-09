"""Verify trace graph artifacts and measure explicit query samples without training."""

from __future__ import annotations

import time
from numbers import Integral
from pathlib import Path

import numpy as np
import torch

from seis_interp.configuration import load_resolved_config
from seis_interp.data.file_checksums import file_sha256
from seis_interp.data.relational_trace_graph_run_inputs import (
    trace_graph_run_input_metadata,
    validate_trace_graph_checkpoint_provenance,
)
from seis_interp.data.trace_graph_domain import (
    load_benchmark_trace_graph_domain,
    load_training_trace_graph_domain,
)
from seis_interp.models.relational_trace_graph import RelationalTraceGraphInterpolator
from seis_interp.processing.trace_graph_preprocessing import fit_trace_graph_preprocessing
from seis_interp.relational_trace_graph_config import (
    trace_graph_diagnostic_bands,
    validate_relational_trace_graph_prediction_config,
    validate_relational_trace_graph_training_config,
)
from seis_interp.training.devices import resolve_device
from seis_interp.training.relational_trace_graph_checkpoints import (
    load_relational_trace_graph_checkpoint,
)
from seis_interp.training.trace_graph_episodes import TraceGraphEpisodeGenerator
from seis_interp.training.trace_graph_preflight import run_trace_graph_preflight


def preflight_relational_trace_graph_run(
    *,
    config_path: Path,
    interim_dir: Path,
    processed_dir: Path,
    mask_dir: Path,
    case_dir: Path,
    checkpoint_path: Path | None = None,
    volume_dir: Path | None = None,
    train_mask_dir: Path | None = None,
    train_case_dir: Path | None = None,
    train_volume_dir: Path | None = None,
    query_count: int = 1,
    query_limit: int = 32,
    device_override: str | None = None,
    evaluate_baselines: bool = False,
    max_graph_seconds: float | None = None,
    max_process_rss_bytes: int | None = None,
    max_cuda_allocated_bytes: int | None = None,
) -> dict[str, object]:
    """Return a strict-JSON report; invalid inputs/resources block without fallback.

    Without a checkpoint, a training config fits its fixed train-pool scales and
    initializes a model but performs no optimizer step. With a checkpoint, use a
    frozen inference config. Exactly ``query_count`` targets in stable ID order
    are measured; this explicit sample is never a full-case performance claim.
    """
    started = time.perf_counter()
    stage = "inputs"
    try:
        for name, value in (("query_count", query_count), ("query_limit", query_limit)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if query_count > query_limit:
            raise ValueError("query_count exceeds the explicit preflight query_limit")
        config = load_resolved_config(Path(config_path))
        source_paths = {
            "interim_dir": interim_dir,
            "processed_dir": processed_dir,
            "mask_dir": mask_dir,
            "case_dir": case_dir,
            "volume_dir": volume_dir,
        }
        if checkpoint_path is None:
            model, domain, fixed, graph, records, device = _training_inputs(
                config,
                **source_paths,
                train_mask_dir=train_mask_dir,
                train_case_dir=train_case_dir,
                train_volume_dir=train_volume_dir,
                device_override=device_override,
            )
            batch_size = config["evaluation"]["query_batch_size"]
        else:
            if any(path is not None for path in (train_mask_dir, train_case_dir, train_volume_dir)):
                raise ValueError("frozen preflight uses checkpoint preprocessing, not train inputs")
            model, domain, fixed, graph, records, device = _frozen_inputs(
                config,
                checkpoint_path=checkpoint_path,
                **source_paths,
                device_override=device_override,
            )
            batch_size = config["prediction"]["query_batch_size"]
        query_ids = np.sort(domain.trace_ids[domain.query_mask])
        if query_count > len(query_ids):
            raise ValueError("query_count exceeds the case's available target count")
        stage = "sample_measurement"
        report = run_trace_graph_preflight(
            model,
            domain,
            fixed,
            graph_settings=graph,
            query_trace_ids=query_ids[:query_count],
            query_limit=query_limit,
            query_batch_size=batch_size,
            device=device,
            evaluate_baselines=evaluate_baselines,
            bands=trace_graph_diagnostic_bands(config),
            max_graph_seconds=max_graph_seconds,
            max_process_rss_bytes=max_process_rss_bytes,
            max_cuda_allocated_bytes=max_cuda_allocated_bytes,
        )
        report.update(
            training_started=False,
            configuration_sha256=file_sha256(Path(config_path)),
            input=records,
            sampled_query_trace_ids=query_ids[:query_count].tolist(),
            total_case_query_count=len(query_ids),
            elapsed_seconds=time.perf_counter() - started,
        )
        return report
    except (OSError, ValueError, RuntimeError, MemoryError) as error:
        return {
            "status": "blocked",
            "stage": stage,
            "blockers": [{"type": type(error).__name__, "message": str(error)}],
            "training_started": False,
            "elapsed_seconds": time.perf_counter() - started,
        }


def _training_inputs(
    config,
    *,
    interim_dir,
    processed_dir,
    mask_dir,
    case_dir,
    volume_dir,
    train_mask_dir,
    train_case_dir,
    train_volume_dir,
    device_override,
):
    model_config, graph, options = validate_relational_trace_graph_training_config(config)
    device = resolve_device(
        config["training"]["device"] if device_override is None else device_override
    )
    selection = tuple(config["training_data"]["time_samples"])
    training = load_training_trace_graph_domain(
        interim_dir=interim_dir,
        processed_dir=processed_dir,
        pool=config["training_data"]["pool"],
        time_samples=selection,
        mask_dir=train_mask_dir,
        case_dir=train_case_dir,
        volume_dir=train_volume_dir,
    )
    domain = load_benchmark_trace_graph_domain(
        interim_dir=interim_dir,
        processed_dir=processed_dir,
        mask_dir=mask_dir,
        case_dir=case_dir,
        volume_dir=volume_dir,
        time_samples=selection,
    )
    training_record = trace_graph_run_input_metadata(
        training,
        config=config,
        processed_dir=processed_dir,
        case_dir=train_case_dir,
        expected_partition="train",
    )
    benchmark_record = trace_graph_run_input_metadata(
        domain,
        config=config,
        processed_dir=processed_dir,
        case_dir=case_dir,
        expected_partition="validation",
    )
    if np.intersect1d(training.trace_ids, domain.trace_ids).size:
        raise ValueError("training and validation trace domains must be disjoint")
    TraceGraphEpisodeGenerator(
        training,
        random_seed=options["random_seed"],
        kind_probabilities=options["episode_kind_probabilities"],
        missing_fractions=options["missing_fractions"],
    )
    fixed = fit_trace_graph_preprocessing(
        training,
        **config["geometry_features"],
        max_abs_amplitude=config["training_data"].get("max_abs_amplitude"),
    )
    if not np.array_equal(domain.time_s, fixed.time_s):
        raise ValueError("validation time_s must match fixed preprocessing")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(options["random_seed"])
        if not options.get("cudnn_benchmark", True):
            torch.backends.cudnn.benchmark = False
        model = RelationalTraceGraphInterpolator(**model_config)
    records = {
        "training": training_record,
        "benchmark": benchmark_record,
        "training_inputs_lock": training.inputs_lock,
        "benchmark_inputs_lock": domain.inputs_lock,
        "model_state": "untrained",
        "training_random_seed": options["random_seed"],
    }
    return model, domain, fixed, graph, records, device


def _frozen_inputs(
    config,
    *,
    checkpoint_path,
    interim_dir,
    processed_dir,
    mask_dir,
    case_dir,
    volume_dir,
    device_override,
):
    validate_relational_trace_graph_prediction_config(config)
    device = resolve_device(
        config["prediction"]["device"] if device_override is None else device_override,
        config_key="prediction.device",
    )
    loaded = load_relational_trace_graph_checkpoint(Path(checkpoint_path), device=device)
    domain = load_benchmark_trace_graph_domain(
        interim_dir=interim_dir,
        processed_dir=processed_dir,
        mask_dir=mask_dir,
        case_dir=case_dir,
        volume_dir=volume_dir,
        time_samples=tuple(loaded.preprocessing.fit_domain["time_samples"]),
    )
    metadata = trace_graph_run_input_metadata(
        domain, config=config, processed_dir=processed_dir, case_dir=case_dir
    )
    validate_trace_graph_checkpoint_provenance(loaded.training_provenance, domain)
    if not np.array_equal(domain.time_s, loaded.preprocessing.time_s):
        raise ValueError("checkpoint time grid does not match benchmark samples")
    records = {
        "benchmark": metadata,
        "benchmark_inputs_lock": domain.inputs_lock,
        "checkpoint_sha256": file_sha256(Path(checkpoint_path)),
        "checkpoint_role": loaded.checkpoint_role,
        "checkpoint_step": loaded.global_step,
    }
    return loaded.model, domain, loaded.preprocessing, loaded.graph_settings, records, device
