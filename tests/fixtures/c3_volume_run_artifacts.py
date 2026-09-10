"""Synthetic verified C3 artifacts shared by volume-run integration tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from seis_interp.data.interpolation_mask_store import load_interpolation_mask
from seis_interp.pipelines.prepare_baseline import prepare_baseline_dataset
from seis_interp.pipelines.prepare_benchmark_case import prepare_benchmark_case
from seis_interp.pipelines.prepare_c3_volume_index import prepare_c3_volume_index
from seis_interp.pipelines.prepare_interpolation_mask import prepare_interpolation_mask
from seis_interp.processing.interpolation_masks import (
    EVALUATION_TARGET_ROLE,
    OBSERVATION_ROLE_COLUMN,
    RANDOM_TRACE_MASK_KIND,
)
from seis_interp.processing.trace_splits import C3_SOURCE_LINE_BLOCKS_SPLIT_SCOPE, TEST_SPLIT
from tests.fixtures.c3_volume_artifacts import SOURCE_LINE_RANGES, prepare_c3_volume_artifacts


@dataclass(frozen=True)
class PreparedC3VolumeRunArtifacts:
    """Paths and metadata for one synthetic verified benchmark-volume run."""

    interim: Path
    processed: Path
    mask: Path
    case: Path
    volume: Path
    volume_metadata: dict[str, object]
    mask_kind: str


def prepare_c3_volume_run_artifacts(
    tmp_path: Path,
    *,
    mask_kind: str = RANDOM_TRACE_MASK_KIND,
    missing_fraction: float = 0.5,
    target_offset: float = 0.0,
    time_sample_count: int = 4,
    receiver_y_count: int = 3,
) -> PreparedC3VolumeRunArtifacts:
    """Prepare a tiny complete run input, optionally changing only target truth."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    prepared = prepare_c3_volume_artifacts(
        tmp_path,
        mask_kind=mask_kind,
        missing_fraction=missing_fraction,
        random_seed=42,
        time_sample_count=time_sample_count,
    )
    case_dir = prepared.case_dir
    if target_offset:
        mask_table, _ = load_interpolation_mask(prepared.mask_dir)
        target_rows = mask_table.loc[
            mask_table[OBSERVATION_ROLE_COLUMN].eq(EVALUATION_TARGET_ROLE),
            "array_row",
        ].to_numpy(dtype=np.int64)
        amplitude_path = prepared.interim_dir / "amplitudes.npy"
        amplitudes = np.load(amplitude_path, allow_pickle=False)
        changed = np.array(amplitudes, copy=True)
        changed[target_rows] += target_offset
        np.save(amplitude_path, changed, allow_pickle=False)
        prepare_baseline_dataset(
            prepared.interim_dir,
            prepared.processed_dir,
            holdout_fraction=None,
            validation_fraction_of_holdout=None,
            random_seed=42,
            split_scope=C3_SOURCE_LINE_BLOCKS_SPLIT_SCOPE,
            source_line_ranges=SOURCE_LINE_RANGES,
            config_source="studies/synthetic/config.yaml",
            overwrite=True,
        )
        prepare_interpolation_mask(
            prepared.interim_dir,
            prepared.processed_dir,
            prepared.mask_dir,
            partition=TEST_SPLIT,
            kind=mask_kind,
            missing_fraction=missing_fraction,
            random_seed=42,
            config_source="studies/synthetic/config.yaml",
            overwrite=True,
        )
        case_dir = prepared.processed_dir / "cases" / "synthetic-case-adjusted"
        prepare_benchmark_case(
            prepared.interim_dir,
            prepared.processed_dir,
            prepared.mask_dir,
            case_dir,
            case_id="synthetic_case",
            config_source="studies/synthetic/config.yaml",
        )

    volume_dir = prepared.processed_dir / "volumes" / "synthetic-volume"
    volume_metadata = prepare_c3_volume_index(
        prepared.interim_dir,
        prepared.processed_dir,
        prepared.mask_dir,
        case_dir,
        volume_dir,
        volume_id="synthetic_volume",
        time_range=(0, time_sample_count),
        source_line_range=(2, 4),
        shot_in_line_range=(0, 3),
        relative_receiver_x_range=(0, 2),
        relative_receiver_y_range=(0, receiver_y_count),
        config_source="studies/synthetic/config.yaml",
    )
    return PreparedC3VolumeRunArtifacts(
        interim=prepared.interim_dir,
        processed=prepared.processed_dir,
        mask=prepared.mask_dir,
        case=case_dir,
        volume=volume_dir,
        volume_metadata=volume_metadata,
        mask_kind=mask_kind,
    )
