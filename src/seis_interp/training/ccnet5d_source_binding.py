"""Bind CCNet supervision config to either the legacy region or QC training dataset."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from seis_interp.data.c3_benchmark_inputs import (
    load_c3_benchmark_training_dataset_source,
)
from seis_interp.data.c3_supervised_source import C3SupervisedSource, load_c3_supervised_source
from seis_interp.data.file_checksums import file_sha256
from seis_interp.processing.c3_benchmark_contract import (
    MAIN_C3_DIMENSIONS,
    C3BenchmarkDimensions,
)
from seis_interp.training.ccnet5d_checkpoints import load_ccnet5d_checkpoint

QC_TRAINING_DATASET = "qc_training_dataset"
FIXED_CHECKPOINT_RMS = "fixed_checkpoint_fit_region_global_rms"


def load_ccnet5d_supervised_source(
    config: Mapping[str, object],
    *,
    interim_dir: Path,
    processed_dir: Path,
    suite_dir: Path | None = None,
    normalization_checkpoint_path: Path | None = None,
    dimensions: C3BenchmarkDimensions = MAIN_C3_DIMENSIONS,
) -> C3SupervisedSource:
    """Load configured labels and verify any externally fixed RMS provenance."""
    supervision = config["supervision"]
    if supervision.get("sampling_domain") != QC_TRAINING_DATASET:
        return load_c3_supervised_source(
            interim_dir=interim_dir,
            processed_dir=processed_dir,
            fit_region=supervision["fit_region"],
            selection_region=supervision["selection_region"],
        )
    if suite_dir is None:
        raise ValueError("qc_training_dataset supervision requires suite_dir")
    if normalization_checkpoint_path is None:
        raise ValueError("qc_training_dataset supervision requires normalization_checkpoint_path")
    checkpoint_path = Path(normalization_checkpoint_path).resolve()
    expected_hash = supervision["normalization_checkpoint_sha256"]
    checkpoint_hash = file_sha256(checkpoint_path)
    if checkpoint_hash != expected_hash:
        raise ValueError("normalization checkpoint SHA-256 differs from configured value")
    checkpoint = load_ccnet5d_checkpoint(checkpoint_path)
    expected_step = supervision["normalization_checkpoint_global_step"]
    if checkpoint.checkpoint_role != "final" or checkpoint.global_step != expected_step:
        raise ValueError("normalization checkpoint is not the declared final checkpoint")
    model_config = {key: value for key, value in config["model"].items() if key != "name"}
    if checkpoint.model.constructor_config() != model_config:
        raise ValueError("normalization checkpoint model differs from the training model")
    normalization_source = {
        "artifact": str(checkpoint_path),
        "sha256": checkpoint_hash,
        "checkpoint_role": checkpoint.checkpoint_role,
        "global_step": checkpoint.global_step,
        "training_run": checkpoint.training_provenance["training_run"],
        "amplitude_rms": checkpoint.amplitude_rms,
    }
    source = load_c3_benchmark_training_dataset_source(
        Path(suite_dir),
        selection_region=supervision["selection_region"],
        amplitude_rms=checkpoint.amplitude_rms,
        normalization_source=normalization_source,
        dimensions=dimensions,
    )
    checkpoint_source = checkpoint.training_provenance["source_inputs_lock"]
    for group in ("interim", "processed"):
        if checkpoint_source[group] != source.inputs_lock[group]:
            raise ValueError(
                f"normalization checkpoint {group} differs from current training dataset"
            )
    return source
