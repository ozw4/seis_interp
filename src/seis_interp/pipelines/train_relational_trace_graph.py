"""Orchestrate masked trace training, fixed validation and immutable run artifacts."""

from __future__ import annotations

import platform
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path

import numpy as np
import torch

from seis_interp import run_records
from seis_interp.configuration import load_resolved_config
from seis_interp.data.file_checksums import file_sha256
from seis_interp.data.relational_trace_graph_run_inputs import trace_graph_run_input_metadata
from seis_interp.data.trace_graph_domain import (
    load_benchmark_trace_graph_domain,
    load_training_trace_graph_domain,
)
from seis_interp.data.trace_graph_prediction_store import save_trace_graph_prediction
from seis_interp.evaluation.trace_graph_diagnostic_metrics import evaluate_trace_graph_baselines
from seis_interp.models.relational_trace_graph import RelationalTraceGraphInterpolator
from seis_interp.processing.trace_graph_geometry import (
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    RELATION_NAMES,
)
from seis_interp.processing.trace_graph_preprocessing import fit_trace_graph_preprocessing
from seis_interp.relational_trace_graph_config import (
    METHOD,
    TRAINING_REGIME,
    trace_graph_method_variant,
    validate_relational_trace_graph_training_config,
)
from seis_interp.training.devices import resolve_device
from seis_interp.training.randomness import seed_global_model_initialization
from seis_interp.training.relational_trace_graph_checkpoints import (
    load_relational_trace_graph_checkpoint,
    save_relational_trace_graph_checkpoint,
)
from seis_interp.training.relational_trace_graph_prediction import predict_relational_trace_graph
from seis_interp.training.relational_trace_graph_trainer import train_relational_trace_graph
from seis_interp.training.trace_graph_episodes import TraceGraphEpisodeGenerator


def train_relational_trace_graph_run(
    *,
    config_path: Path,
    interim_dir: Path,
    processed_dir: Path,
    validation_mask_dir: Path,
    validation_case_dir: Path,
    output_dir: Path,
    validation_volume_dir: Path | None = None,
    train_mask_dir: Path | None = None,
    train_case_dir: Path | None = None,
    train_volume_dir: Path | None = None,
    device_override: str | None = None,
    progress_reporter: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Train within one verified partition artifact; save best validation predictions."""
    output = Path(output_dir)
    run_records.check_new_output_directory(output)
    config = load_resolved_config(Path(config_path))
    model_config, graph_settings, trainer_options = validate_relational_trace_graph_training_config(
        config
    )
    device = resolve_device(
        config["training"]["device"] if device_override is None else device_override
    )
    config["training"]["device"] = str(device)
    started_at = run_records.utc_timestamp()
    git_metadata = run_records.current_git_metadata()
    started = time.perf_counter()
    if progress_reporter:
        progress_reporter("Verifying train pool and fixed validation case.")
    time_samples = tuple(config["training_data"]["time_samples"])
    training = load_training_trace_graph_domain(
        interim_dir=interim_dir,
        processed_dir=processed_dir,
        pool=config["training_data"]["pool"],
        time_samples=time_samples,
        mask_dir=train_mask_dir,
        case_dir=train_case_dir,
        volume_dir=train_volume_dir,
    )
    validation = load_benchmark_trace_graph_domain(
        interim_dir=interim_dir,
        processed_dir=processed_dir,
        mask_dir=validation_mask_dir,
        case_dir=validation_case_dir,
        volume_dir=validation_volume_dir,
        time_samples=time_samples,
    )
    training_input = trace_graph_run_input_metadata(
        training,
        config=config,
        processed_dir=processed_dir,
        case_dir=train_case_dir,
        expected_partition="train",
    )
    validation_input = trace_graph_run_input_metadata(
        validation,
        config=config,
        processed_dir=processed_dir,
        case_dir=validation_case_dir,
        expected_partition="validation",
    )
    if np.intersect1d(training.trace_ids, validation.trace_ids).size:
        raise ValueError("training and validation trace domains must be disjoint")
    if not np.any(validation.query_mask):
        raise ValueError("fixed validation must have evaluation target queries")
    # Check domain-dependent mask feasibility before creating a run. Construction
    # does not sample an episode or advance the trainer's separate local RNG.
    TraceGraphEpisodeGenerator(
        training,
        random_seed=trainer_options["random_seed"],
        kind_probabilities=trainer_options["episode_kind_probabilities"],
        missing_fractions=trainer_options["missing_fractions"],
    )
    preprocessing = fit_trace_graph_preprocessing(
        training,
        **config["geometry_features"],
        max_abs_amplitude=config["training_data"].get("max_abs_amplitude"),
    )
    if not np.array_equal(validation.time_s, preprocessing.time_s):
        raise ValueError("validation time_s must match the fixed preprocessing time grid")
    seed_global_model_initialization(trainer_options["random_seed"], device=device)
    if not trainer_options.get("cudnn_benchmark", True):
        torch.backends.cudnn.benchmark = False
    model = RelationalTraceGraphInterpolator(**model_config).float()
    provenance = {
        "training_run": git_metadata,
        "configuration_sha256": file_sha256(Path(config_path)),
        "source_inputs_lock": training.inputs_lock,
        "validation_inputs_lock": validation.inputs_lock,
        "training_input": training_input,
        "validation_input": validation_input,
        "training_data": {
            **config["training_data"],
            "additional_train_partition_access": training.pool == "all_train_traces",
        },
        "partition_random_seed": config["project"]["random_seed"],
        "training_random_seed": trainer_options["random_seed"],
    }
    identity = {
        "method": METHOD,
        "method_variant": trace_graph_method_variant(model_config),
        "training_regime": TRAINING_REGIME,
    }
    metadata = {
        **identity,
        **git_metadata,
        "started_at_utc": started_at,
        "finished_at_utc": None,
        "status": "running",
        "phase": "training",
        "device": str(device),
        "random_seed": config["project"]["random_seed"],
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": str(torch.__version__),
        "input": {"training": training_input, "validation": validation_input},
        "training_data": provenance["training_data"],
        "training": {**config["training"], "steps_completed": 0},
        "training_mask": config["training_mask"],
        "graph": {
            **graph_settings.constructor_config(),
            "relation_names": ["untyped"]
            if graph_settings.topology == "single_4d"
            else list(RELATION_NAMES),
        },
        "model": model.constructor_config(),
        "geometry_features": {
            **config["geometry_features"],
            "midpoint_origin_m": list(preprocessing.midpoint_origin_m),
            "node_feature_names": list(NODE_FEATURE_NAMES),
            "edge_feature_names": list(EDGE_FEATURE_NAMES),
        },
        "amplitude": {
            "scale_source": "fixed_training_pool",
            "amplitude_scale": preprocessing.amplitude_scale,
        },
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoints": {},
        "resources": {
            "elapsed_seconds": time.perf_counter() - started,
            **run_records.runtime_resource_metadata(device),
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    artifacts = output / "artifacts"
    artifacts.mkdir()
    metrics = dict(identity)
    run_records.write_run_outputs(output, config, provenance, metrics, metadata)

    def save_checkpoint(state, step, selection, *, role):
        name = "best" if role == "best_validation" else "final"
        save_relational_trace_graph_checkpoint(
            artifacts / f"{name}.pt",
            model_config=model.constructor_config(),
            state_dict=state,
            preprocessing=preprocessing,
            graph_settings=graph_settings,
            training_mask=config["training_mask"],
            training_provenance=provenance,
            training_random_seed=trainer_options["random_seed"],
            checkpoint_role=role,
            global_step=step,
            selection_metrics=selection,
        )
        metadata["checkpoints"][name] = {
            "artifact": f"artifacts/{name}.pt",
            "role": role,
            "step": step,
        }

    def write_progress():
        metadata["resources"] = {
            "elapsed_seconds": time.perf_counter() - started,
            **run_records.runtime_resource_metadata(device),
        }
        run_records.write_run_progress(output, metrics, metadata)

    def save_best(state, step, selection):
        save_checkpoint(state, step, selection, role="best_validation")
        metrics.update(best_step=step, best_validation_metrics=selection)
        metadata["training"]["steps_completed"] = step
        write_progress()

    try:
        trained = train_relational_trace_graph(
            model,
            training,
            validation,
            preprocessing,
            graph_settings=graph_settings,
            device=device,
            reporter=progress_reporter,
            on_best_update=save_best,
            **trainer_options,
        )
        save_checkpoint(
            trained.final_state_dict,
            trained.steps_completed,
            trained.final_validation_metrics,
            role="final",
        )
        metrics = {
            **identity,
            **{
                name: getattr(trained, name)
                for name in (
                    "best_step",
                    "steps_completed",
                    "episodes_completed",
                    "best_validation_metrics",
                    "final_validation_metrics",
                    "training_history",
                    "validation_history",
                    "episode_history",
                    "query_count",
                    "no_context_query_count",
                    "sample_count",
                    "normalized_error_energy",
                    "final_episode_interrupted",
                )
            },
        }
        metrics["no_context_query_fraction"] = trained.no_context_query_count / trained.query_count
        for name in ("training_history", "validation_history", "episode_history"):
            metrics[name] = list(metrics[name])
        metadata["training"]["steps_completed"] = trained.steps_completed
        metadata["phase"] = "validation_prediction"
        write_progress()
        best = load_relational_trace_graph_checkpoint(artifacts / "best.pt", device=device)
        predicted = predict_relational_trace_graph(
            best.model,
            validation,
            best.preprocessing,
            graph_settings=best.graph_settings,
            query_batch_size=trainer_options["validation_query_batch_size"],
            device=device,
            measure_resources=True,
        )
        if "diagnostics" in config:
            metrics["baselines"] = evaluate_trace_graph_baselines(
                validation,
                best.preprocessing,
                graph_settings=best.graph_settings,
                query_trace_ids=predicted.query_trace_ids,
                query_batch_size=trainer_options["validation_query_batch_size"],
            )
        prediction_metadata = save_trace_graph_prediction(
            artifacts,
            predicted.prediction,
            validation,
            query_trace_ids=predicted.query_trace_ids,
            has_observed_context=predicted.has_observed_context,
            volume_dir=validation_volume_dir,
        )
        metadata["prediction"] = {
            **prediction_metadata,
            "partition": "validation",
            "checkpoint_role": "best_validation",
            "diagnostics": predicted.diagnostics,
        }
        metadata.update(
            status="success", phase="complete", finished_at_utc=run_records.utc_timestamp()
        )
        write_progress()
    except (Exception, KeyboardInterrupt) as error:
        metadata.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            finished_at_utc=run_records.utc_timestamp(),
            error={"type": type(error).__name__, "message": str(error)},
        )
        # Preserve the original failure if storage cannot accept another
        # record. The last committed run record and best snapshot remain.
        with suppress(OSError, ValueError):
            write_progress()
        raise
    return metrics
