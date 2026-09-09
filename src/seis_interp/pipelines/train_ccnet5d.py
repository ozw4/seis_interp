"""Orchestrate train-partition CCNet-5D supervision and immutable run records."""

from __future__ import annotations

import json
import platform
import time
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import torch

from seis_interp import config_values, run_records
from seis_interp.configuration import ConfigurationError, load_resolved_config
from seis_interp.data.c3_supervised_source import load_c3_supervised_source
from seis_interp.data.file_checksums import file_sha256
from seis_interp.evaluation.ccnet5d_selection import validate_ccnet5d_selection_targets
from seis_interp.models.ccnet5d import CCNet5D, ccnet5d_method_variant
from seis_interp.training.ccnet5d_checkpoints import save_ccnet5d_checkpoint
from seis_interp.training.ccnet5d_patches import make_ccnet_patch_plan
from seis_interp.training.ccnet5d_trainer import train_ccnet5d
from seis_interp.training.devices import resolve_device
from seis_interp.training.randomness import seed_global_model_initialization

METHOD = "ccnet5d"
TRAINING_REGIME = "supervised_train_partition"


def train_ccnet5d_run(
    *,
    config_path: Path,
    interim_dir: Path,
    processed_dir: Path,
    output_dir: Path,
    device_override: str | None = None,
    progress_reporter: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Fit fixed patches and select a checkpoint without any benchmark labels."""
    output = Path(output_dir)
    run_records.check_new_output_directory(output)
    config = load_resolved_config(Path(config_path))
    model_config, trainer_options = _validate_config(config)
    device = resolve_device(
        config["training"]["device"] if device_override is None else device_override
    )
    config["training"]["device"] = str(device)
    started_at = run_records.utc_timestamp()
    git_metadata = run_records.current_git_metadata()
    timings = {}
    if progress_reporter:
        progress_reporter("Verifying train-partition source and fitting complete-label RMS.")
    started = time.perf_counter()
    source = load_c3_supervised_source(
        interim_dir=Path(interim_dir),
        processed_dir=Path(processed_dir),
        fit_region=config["supervision"]["fit_region"],
        selection_region=config["supervision"]["selection_region"],
    )
    timings["load_and_verification_seconds"] = time.perf_counter() - started
    if config["project"]["random_seed"] != source.inputs_lock["partition_random_seed"]:
        raise ConfigurationError("project.random_seed does not match the prepared partition seed")
    if config["data"]["dataset_id"] != source.inputs_lock["dataset_id"]:
        raise ConfigurationError("data.dataset_id does not match the supervised source")
    patches = config["patches"]
    plan = make_ccnet_patch_plan(
        source,
        patch_shape=patches["shape"],
        fit_count=patches["fit_count"],
        selection_count=patches["selection_count"],
        missing_fraction=patches["missing_fraction"],
        random_seed=patches["random_seed"],
    )
    validate_ccnet5d_selection_targets(source, plan)
    seed_global_model_initialization(trainer_options["random_seed"], device=device)
    if not trainer_options.get("cudnn_benchmark", True):
        torch.backends.cudnn.benchmark = False
    model = CCNet5D(**model_config).float()
    variant = ccnet5d_method_variant(model.output_activation)
    output.mkdir(parents=True, exist_ok=False)
    artifacts = output / "artifacts"
    artifacts.mkdir()
    plan_path = artifacts / "patch_plan.json"
    plan_path.write_text(
        json.dumps(plan.to_dict(), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    provenance = {
        "training_run": dict(git_metadata),
        "source_inputs_lock": source.inputs_lock,
        "patch_plan_sha256": file_sha256(plan_path),
        "patches_random_seed": patches["random_seed"],
        "training_random_seed": trainer_options["random_seed"],
    }
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    trained = train_ccnet5d(
        model, source, plan, device=device, **trainer_options, reporter=progress_reporter
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timings["training_and_selection_seconds"] = time.perf_counter() - started
    for name, state, epoch, step, selection in (
        (
            "best",
            trained.best_state_dict,
            trained.best_epoch,
            trained.best_step,
            trained.best_selection_metrics,
        ),
        (
            "final",
            model.state_dict(),
            trained.epochs_completed,
            trained.steps_completed,
            trained.final_selection_metrics,
        ),
    ):
        save_ccnet5d_checkpoint(
            artifacts / f"{name}.pt",
            model_config=model.constructor_config(),
            state_dict=state,
            amplitude_rms=source.amplitude_rms,
            training_provenance=provenance,
            checkpoint_role="best_selection" if name == "best" else "final",
            epoch=epoch,
            global_step=step,
            selection_metrics=selection,
        )
    finished_at = run_records.utc_timestamp()
    identity = {"method": METHOD, "method_variant": variant, "training_regime": TRAINING_REGIME}
    metrics = {
        **identity,
        "epochs_completed": trained.epochs_completed,
        "steps_completed": trained.steps_completed,
        "best_epoch": trained.best_epoch,
        "best_step": trained.best_step,
        "best_selection_metrics": trained.best_selection_metrics,
        "final_selection_metrics": trained.final_selection_metrics,
        "training_history": list(trained.training_history),
        "selection_history": list(trained.selection_history),
    }
    metadata = {
        **identity,
        **git_metadata,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "status": "success",
        "device": str(device),
        "random_seed": config["project"]["random_seed"],
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": str(torch.__version__),
        "supervision": config["supervision"],
        "amplitude": {
            "scale_source": "fit_region_complete_labels",
            "amplitude_rms": source.amplitude_rms,
        },
        "model": {
            **model.constructor_config(),
            "parameter_dtype": "float32",
            "initialization": "pytorch_conv_default",
            "padding": "same_zero",
            "module_count": 4,
        },
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "patches": {**patches, "patch_plan_sha256": provenance["patch_plan_sha256"]},
        "training": {
            **config["training"],
            "input_dtype": "float32",
            "target_dtype": "float32",
            "steps_completed": trained.steps_completed,
            "epochs_completed": trained.epochs_completed,
        },
        "selection": {
            **config["selection"],
            "metric_scope": "selection_patch_instances_missing_only",
            "best_epoch": trained.best_epoch,
            "best_step": trained.best_step,
        },
        "checkpoints": {
            "best": {"artifact": "artifacts/best.pt", "role": "best_selection"},
            "final": {"artifact": "artifacts/final.pt", "role": "final"},
        },
        "resources": {
            **timings,
            **run_records.runtime_resource_metadata(device),
            "process_max_rss_scope": "whole_process",
            "torch_num_threads": torch.get_num_threads(),
        },
        "warnings": [],
    }
    run_records.write_run_outputs(output, config, provenance, metrics, metadata)
    return metrics


def _validate_config(config: Mapping[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    model = config_values.exact_section(
        config,
        "model",
        {"name", "hidden_channels", "intermediate_channels", "kernel_size", "output_activation"},
    )
    config_values.exact_section(config, "project", {"random_seed"})
    data = config_values.exact_section(config, "data", {"dataset_id"})
    if not isinstance(data["dataset_id"], str) or not data["dataset_id"]:
        raise ConfigurationError("data.dataset_id must be a non-empty string")
    config_values.nonnegative_integer(config["project"]["random_seed"], "project.random_seed")
    config_values.exact_section(
        config,
        "supervision",
        {"partition", "amplitude_normalization", "fit_region", "selection_region"},
    )
    patches = config_values.exact_section(
        config,
        "patches",
        {"shape", "fit_count", "selection_count", "missing_fraction", "mask_kind", "random_seed"},
    )
    training_keys = {
        "optimizer",
        "loss",
        "random_seed",
        "batch_size",
        "max_epochs",
        "learning_rate",
        "decay_after_epochs",
        "decay_factor",
        "validate_every_steps",
        "report_every_steps",
        "device",
    }
    if isinstance(config.get("training"), Mapping) and "cudnn_benchmark" in config["training"]:
        training_keys.add("cudnn_benchmark")
    training = config_values.exact_section(config, "training", training_keys)
    config_values.exact_section(config, "selection", {"metric", "domain"})
    for key, value in {
        "model.name": METHOD,
        "supervision.partition": "train",
        "supervision.amplitude_normalization": "fit_region_global_rms",
        "patches.mask_kind": "random_trace",
        "training.optimizer": "adam",
        "training.loss": "mse_complete_patch",
        "selection.metric": "missing_global_snr_db",
        "selection.domain": "held_out_train_partition_patch_instances",
    }.items():
        config_values.require_exact(config, key, value)
    batch_size = config_values.positive_integer(training["batch_size"], "training.batch_size")
    cudnn_benchmark = training.get("cudnn_benchmark", True)
    if not isinstance(cudnn_benchmark, bool):
        raise ConfigurationError("training.cudnn_benchmark must be a boolean")
    config_values.nonnegative_integer(patches["random_seed"], "patches.random_seed")
    constructor = {
        key: config_values.positive_integer(model[key], f"model.{key}")
        for key in ("hidden_channels", "intermediate_channels")
    }
    constructor["kernel_size"] = config_values.odd_positive_integer(
        model["kernel_size"], "model.kernel_size"
    )
    ccnet5d_method_variant(model["output_activation"])
    constructor["output_activation"] = model["output_activation"]
    options = {
        key: config_values.positive_integer(training[key], f"training.{key}")
        for key in (
            "max_epochs",
            "decay_after_epochs",
            "validate_every_steps",
            "report_every_steps",
        )
    }
    options["random_seed"] = config_values.nonnegative_integer(
        training["random_seed"], "training.random_seed"
    )
    for key in ("learning_rate", "decay_factor"):
        options[key] = config_values.positive_float(training[key], f"training.{key}")
    if options["decay_factor"] > 1:
        raise ConfigurationError("training.decay_factor must be at most 1")
    if batch_size != 1:
        options["batch_size"] = batch_size
    if not cudnn_benchmark:
        options["cudnn_benchmark"] = False
    return constructor, options
