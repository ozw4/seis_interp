"""Per-volume observed-only GNN fit, complete prediction, and common evaluation."""

from __future__ import annotations

import platform
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from seis_interp import run_records
from seis_interp.c3_poc_run_records import (
    METADATA_FILE_NAME,
    poc_compute_metadata,
    poc_coverage_metadata,
    poc_run_metadata,
    validate_poc_prediction,
)
from seis_interp.configuration import load_resolved_config
from seis_interp.data.c3_poc_inputs import load_c3_random80_poc_inputs
from seis_interp.data.c3_poc_trace_graph import (
    build_c3_poc_trace_graph_domain,
    build_c3_poc_trace_graph_training_data,
)
from seis_interp.data.c3_trace_graph_prediction import scatter_c3_trace_graph_prediction
from seis_interp.data.file_checksums import file_sha256
from seis_interp.evaluation.c3_volume_metrics import evaluate_c3_volume_prediction
from seis_interp.models.relational_trace_graph import RelationalTraceGraphInterpolator
from seis_interp.processing.c3_benchmark_contract import MAIN_C3_DIMENSIONS, C3BenchmarkDimensions
from seis_interp.relational_trace_graph_poc_config import validate_relational_trace_graph_poc_config
from seis_interp.training.amplitude_scaling import compute_observed_global_rms
from seis_interp.training.devices import resolve_device
from seis_interp.training.randomness import seed_global_model_initialization
from seis_interp.training.relational_trace_graph_checkpoints import (
    save_relational_trace_graph_poc_checkpoint,
)
from seis_interp.training.relational_trace_graph_poc_trainer import train_relational_trace_graph_poc
from seis_interp.training.relational_trace_graph_prediction import predict_relational_trace_graph


def interpolate_relational_trace_graph_run(
    *,
    config_path: Path,
    interim_dir: Path,
    processed_dir: Path,
    mask_dir: Path,
    case_dir: Path,
    volume_dir: Path,
    output_dir: Path,
    device_override: str | None = None,
    progress_reporter: Callable[[str], None] | None = None,
    dimensions: C3BenchmarkDimensions = MAIN_C3_DIMENSIONS,
) -> dict[str, object]:
    """Fit only O and predict every T using the final state, with no external checkpoint."""
    started = time.perf_counter()
    output = Path(output_dir)
    run_records.check_new_output_directory(output)
    config = load_resolved_config(Path(config_path))
    settings = validate_relational_trace_graph_poc_config(config)
    options = dict(settings.training)
    device = resolve_device(
        options.pop("device") if device_override is None else device_override,
        config_key="training.device",
    )
    options.pop("device", None)
    config["training"]["device"] = str(device)
    started_at = run_records.utc_timestamp()
    git_metadata = run_records.current_git_metadata()
    if progress_reporter:
        progress_reporter("Loading observed-only PoC inputs.")
    inputs = load_c3_random80_poc_inputs(
        config=config,
        interim_dir=interim_dir,
        processed_dir=processed_dir,
        mask_dir=mask_dir,
        case_dir=case_dir,
        volume_dir=volume_dir,
        dimensions=dimensions,
    )
    volume = inputs.observed_volume
    scale = compute_observed_global_rms(volume.values, volume.observed_trace_mask)
    training = build_c3_poc_trace_graph_training_data(
        inputs, amplitude_scale=scale, **settings.geometry
    )
    model_seed = options.pop("model_initialization_seed")
    episode_seed = options.pop("episode_seed")
    seed_global_model_initialization(model_seed, device=device)
    model = RelationalTraceGraphInterpolator(**settings.model)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    train_started = time.perf_counter()
    trained = train_relational_trace_graph_poc(
        model,
        training,
        graph_settings=settings.graph,
        device=device,
        reporter=progress_reporter,
        random_seed=episode_seed,
        **options,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - train_started
    identity = {
        "method": "relational_trace_graph",
        "training_domain": "O_with_inner_pseudo_mask",
        "normalization": {"type": "global_rms", "source": "O_only", "scale": scale},
        "loss": options["loss"],
        "inner_mask_fraction": options["inner_mask_fraction"],
        "checkpoint_role": "final",
        "output_amplitude_domain": "physical",
        "case_id": inputs.case["case_id"],
        "volume_id": inputs.volume_metadata["volume_id"],
    }
    output.mkdir(parents=True, exist_ok=False)
    save_relational_trace_graph_poc_checkpoint(
        output / "final.pt",
        model_config=model.constructor_config(),
        state_dict=model.state_dict(),
        preprocessing=training.preprocessing,
        graph_settings=settings.graph,
        metadata={
            **identity,
            "model_initialization_seed": model_seed,
            "episode_seed": episode_seed,
            "steps_completed": trained.steps_completed,
        },
        inputs_lock=inputs.inputs_lock,
    )
    metadata = {
        **identity,
        **git_metadata,
        "started_at_utc": started_at,
        "status": "running",
        "device": str(device),
        "random_seed": config["project"]["random_seed"],
        "model_initialization_seed": model_seed,
        "episode_seed": episode_seed,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": str(torch.__version__),
        "model": model.constructor_config(),
        "graph": settings.graph.constructor_config(),
        "preprocessing": asdict(training.preprocessing),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training": asdict(trained),
        "checkpoint": {
            "artifact": "final.pt",
            "role": "final",
            "sha256": file_sha256(output / "final.pt"),
        },
        "resources": {"training_seconds": training_seconds},
    }

    coverage = np.zeros_like(volume.evaluation_target_trace_mask)

    def common_metadata() -> dict[str, object]:
        return poc_run_metadata(
            inputs.inputs_lock,
            metadata,
            normalization=identity["normalization"],
            objective=identity["loss"],
            operation=metadata["training"],
            coverage=poc_coverage_metadata(
                volume.evaluation_target_trace_mask, coverage, time_sample_count=len(volume.time_s)
            ),
            compute=poc_compute_metadata(
                parameter_count=metadata["parameter_count"],
                optimizer_updates=trained.steps_completed,
                supervised_trace_presentations=trained.query_count,
            ),
        )

    run_records.write_run_outputs(
        output,
        config,
        inputs.inputs_lock,
        {},
        common_metadata(),
        metadata_file_name=METADATA_FILE_NAME,
    )
    try:
        if progress_reporter:
            progress_reporter("Predicting every target with all observed context.")
        prediction_started = time.perf_counter()
        predicted = predict_relational_trace_graph(
            model,
            build_c3_poc_trace_graph_domain(inputs),
            training.preprocessing,
            graph_settings=settings.graph,
            query_batch_size=settings.prediction_query_batch_size,
            observed_waveforms=training.observed_amplitudes,
            device=device,
        )
        dense, coverage = scatter_c3_trace_graph_prediction(
            inputs, predicted.query_trace_ids, predicted.prediction
        )
        metadata["resources"]["prediction_seconds"] = time.perf_counter() - prediction_started
        # Target truth is first materialized by this common evaluation boundary.
        validate_poc_prediction(dense, volume)
        evaluation_started = time.perf_counter()
        evaluation = evaluate_c3_volume_prediction(
            dense,
            volume,
            interim_dir=Path(interim_dir),
            volume_metadata=inputs.volume_metadata,
            target_coverage_mask=coverage,
        )
        metadata["resources"]["evaluation_seconds"] = time.perf_counter() - evaluation_started
        metrics = dict(evaluation)
        metrics.update(
            {
                **identity,
                "optimizer_updates": trained.steps_completed,
                "training": asdict(trained),
                "no_context_query_count": int(np.count_nonzero(~predicted.has_observed_context)),
                "uncovered_trace_count": 0,
                "uncovered_sample_count": 0,
                "warnings": [],
            }
        )
        np.save(output / "prediction.npy", dense, allow_pickle=False)
        artifacts = output / "artifacts"
        artifacts.mkdir()
        np.save(artifacts / "target_coverage.npy", coverage, allow_pickle=False)
        np.save(artifacts / "query_trace_ids.npy", predicted.query_trace_ids, allow_pickle=False)
        metadata.update(
            status="success",
            prediction={
                "artifact": "prediction.npy",
                "sha256": file_sha256(output / "prediction.npy"),
                "shape": list(dense.shape),
                "diagnostics": predicted.diagnostics,
                "query_batch_size": settings.prediction_query_batch_size,
                "no_context_query_count": metrics["no_context_query_count"],
            },
            coverage={
                "target_trace_count": int(np.count_nonzero(volume.evaluation_target_trace_mask)),
                "covered_target_trace_count": int(np.count_nonzero(coverage)),
            },
        )
    except (Exception, KeyboardInterrupt) as error:
        metadata.update(
            status="failed", error={"type": type(error).__name__, "message": str(error)}
        )
        metadata["finished_at_utc"] = run_records.utc_timestamp()
        run_records.write_run_progress(
            output, {}, common_metadata(), metadata_file_name=METADATA_FILE_NAME
        )
        raise
    metadata["finished_at_utc"] = run_records.utc_timestamp()
    metadata["resources"].update(
        end_to_end_seconds=time.perf_counter() - started,
        **run_records.runtime_resource_metadata(device),
    )
    run_records.write_run_progress(
        output, evaluation, common_metadata(), metadata_file_name=METADATA_FILE_NAME
    )
    return metrics
