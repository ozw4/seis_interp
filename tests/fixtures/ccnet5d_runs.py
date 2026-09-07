"""Executable tiny CCNet configurations and correctly bound benchmark artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from seis_interp.data.interpolation_mask_store import load_interpolation_mask
from seis_interp.pipelines.prepare_baseline import prepare_baseline_dataset
from seis_interp.pipelines.prepare_benchmark_case import prepare_benchmark_case
from seis_interp.pipelines.prepare_c3_volume_index import prepare_c3_volume_index
from seis_interp.pipelines.prepare_interpolation_mask import prepare_interpolation_mask
from seis_interp.processing.trace_splits import C3_SOURCE_LINE_BLOCKS_SPLIT_SCOPE
from tests.fixtures.c3_volume_run_artifacts import PreparedC3VolumeRunArtifacts
from tests.fixtures.ccnet5d_artifacts import SOURCE_LINE_RANGES, PreparedCCNet5DArtifacts


def ccnet5d_training_config(artifacts: PreparedCCNet5DArtifacts) -> dict[str, object]:
    return {
        "project": {"random_seed": 42},
        "data": {"dataset_id": "synthetic_c3_ccnet5d"},
        "model": {
            "name": "ccnet5d",
            "hidden_channels": 2,
            "intermediate_channels": 2,
            "kernel_size": 3,
            "output_activation": "linear",
        },
        "supervision": {
            "partition": "train",
            "amplitude_normalization": "fit_region_global_rms",
            "fit_region": artifacts.fit_region,
            "selection_region": artifacts.selection_region,
        },
        "patches": {
            "shape": [2, 1, 1, 2, 2],
            "fit_count": 2,
            "selection_count": 2,
            "missing_fraction": 0.5,
            "mask_kind": "random_trace",
            "random_seed": 42,
        },
        "training": {
            "optimizer": "adam",
            "loss": "mse_complete_patch",
            "random_seed": 7,
            "batch_size": 1,
            "max_epochs": 2,
            "learning_rate": 0.001,
            "decay_after_epochs": 1,
            "decay_factor": 0.1,
            "validate_every_steps": 3,
            "report_every_steps": 1,
            "device": "cpu",
        },
        "selection": {
            "metric": "missing_global_snr_db",
            "domain": "held_out_train_partition_patch_instances",
        },
    }


def write_ccnet5d_config(path: Path, config: dict[str, object]) -> Path:
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def prepare_ccnet5d_benchmark(
    artifacts: PreparedCCNet5DArtifacts,
    *,
    mask_kind: str = "random_trace",
    partition: str = "validation",
    target_offset: float = 0.0,
) -> PreparedC3VolumeRunArtifacts:
    interim, processed = artifacts.interim, artifacts.processed
    mask = processed / "masks" / f"{partition}-{mask_kind}"
    options = {
        "partition": partition,
        "kind": mask_kind,
        "missing_fraction": 0.5,
        "random_seed": 42,
        "config_source": "synthetic_ccnet5d",
    }
    prepare_interpolation_mask(interim, processed, mask, **options)
    if target_offset:
        table, _ = load_interpolation_mask(mask)
        rows = table.loc[table["observation_role"].eq("evaluation_target"), "array_row"].to_numpy(
            dtype=np.int64
        )
        amplitude_path = interim / "amplitudes.npy"
        amplitudes = np.load(amplitude_path, allow_pickle=False)
        amplitudes[rows] += target_offset
        np.save(amplitude_path, amplitudes, allow_pickle=False)
        prepare_baseline_dataset(
            interim,
            processed,
            holdout_fraction=None,
            validation_fraction_of_holdout=None,
            random_seed=42,
            split_scope=C3_SOURCE_LINE_BLOCKS_SPLIT_SCOPE,
            source_line_ranges=SOURCE_LINE_RANGES,
            config_source="studies/synthetic/config.yaml",
            overwrite=True,
        )
        prepare_interpolation_mask(interim, processed, mask, **options, overwrite=True)
    case = processed / "cases" / f"{partition}-{mask_kind}"
    volume = processed / "volumes" / f"{partition}-{mask_kind}"
    prepare_benchmark_case(
        interim,
        processed,
        mask,
        case,
        case_id="synthetic_ccnet5d_case",
        config_source="synthetic_ccnet5d",
    )
    metadata = prepare_c3_volume_index(
        interim,
        processed,
        mask,
        case,
        volume,
        volume_id="synthetic_ccnet5d_volume",
        time_range=(1, 5),
        source_line_range=SOURCE_LINE_RANGES[partition],
        shot_in_line_range=(0, 3),
        relative_receiver_x_range=(0, 2),
        relative_receiver_y_range=(0, 3),
        config_source="synthetic_ccnet5d",
    )
    return PreparedC3VolumeRunArtifacts(interim, processed, mask, case, volume, metadata, mask_kind)


def ccnet5d_inference_config(
    artifacts: PreparedC3VolumeRunArtifacts, *, partition: str = "validation"
) -> dict[str, object]:
    return {
        "project": {"random_seed": 42},
        "data": {"dataset_id": "synthetic_c3_ccnet5d"},
        "interpolation_mask": {
            "partition": partition,
            "kind": artifacts.mask_kind,
            "missing_fraction": 0.5,
        },
        "benchmark_case": {"id": "synthetic_ccnet5d_case"},
        "benchmark_volume": {
            "id": "synthetic_ccnet5d_volume",
            "selection": artifacts.volume_metadata["selection"],
        },
        "prediction": {"device": "cpu", "core_shape": [2, 1, 2, 1, 2]},
        "evaluation": {
            "primary_metric": "physical_amplitude_global_snr_db",
            "domain": "evaluation_target",
        },
    }
