from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from seis_interp.data.c3_masked_gather_source import (
    MaskedC3GatherSource,
    load_masked_c3_gather_source,
)
from seis_interp.data.c3_volume_adapter import (
    ObservedC3Volume,
    load_observed_c3_volume,
    materialize_observed_c3_volume,
)
from seis_interp.data.c3_volume_index_store import load_c3_volume_index
from seis_interp.data.interpolation_mask_store import load_interpolation_mask
from seis_interp.data.masked_gather_inputs import (
    MaskedGatherInputs,
    validate_masked_gather_inputs,
)
from seis_interp.data.trace_store import AMPLITUDES_FILE_NAME, TIME_FILE_NAME
from seis_interp.pipelines.prepare_c3_volume_index import prepare_c3_volume_index
from seis_interp.processing.c3_receiver_grid import RECEIVER_X_COUNT, RECEIVER_Y_COUNT
from seis_interp.processing.interpolation_masks import (
    EVALUATION_TARGET_ROLE,
    OBSERVATION_ROLE_COLUMN,
    RANDOM_WHOLE_FFID_MASK_KIND,
)
from tests.fixtures.c3_volume_artifacts import (
    PreparedC3VolumeArtifacts,
    prepare_c3_volume_artifacts,
)

_CONTEXT_GATHER_COUNT = 2
_RECEIVER_SHAPE = (RECEIVER_X_COUNT, RECEIVER_Y_COUNT)
_TIME_COUNT = 4


def _prepare_volume_index(
    artifacts: PreparedC3VolumeArtifacts,
    *,
    relative_receiver_x_range: tuple[int, int] = (0, _RECEIVER_SHAPE[0]),
    relative_receiver_y_range: tuple[int, int] = (0, _RECEIVER_SHAPE[1]),
) -> Path:
    output = artifacts.processed_dir / "volumes" / "synthetic-volume"
    prepare_c3_volume_index(
        artifacts.interim_dir,
        artifacts.processed_dir,
        artifacts.mask_dir,
        artifacts.case_dir,
        output,
        volume_id="synthetic_volume",
        time_range=(0, _TIME_COUNT),
        source_line_range=artifacts.source_line_range,
        shot_in_line_range=artifacts.shot_in_line_range,
        relative_receiver_x_range=relative_receiver_x_range,
        relative_receiver_y_range=relative_receiver_y_range,
        config_source="studies/synthetic/config.yaml",
    )
    return output


def _load_source_and_volume(
    artifacts: PreparedC3VolumeArtifacts,
    volume_dir: Path,
    *,
    context_gather_count: int = _CONTEXT_GATHER_COUNT,
) -> tuple[MaskedC3GatherSource, ObservedC3Volume, pd.DataFrame, dict[str, object]]:
    source = load_masked_c3_gather_source(
        interim_dir=artifacts.interim_dir,
        processed_dir=artifacts.processed_dir,
        mask_dir=artifacts.mask_dir,
        case_dir=artifacts.case_dir,
        volume_dir=volume_dir,
        context_gather_count=context_gather_count,
        device="cpu",
    )
    volume = load_observed_c3_volume(
        interim_dir=artifacts.interim_dir,
        processed_dir=artifacts.processed_dir,
        mask_dir=artifacts.mask_dir,
        case_dir=artifacts.case_dir,
        volume_dir=volume_dir,
    )
    index_table, metadata = load_c3_volume_index(volume_dir)
    return source, volume, index_table, metadata


def _flat_target_source_indices(source: MaskedC3GatherSource, shot_count: int) -> np.ndarray:
    target_indices = source.target_source_indices
    return target_indices[:, 0] * shot_count + target_indices[:, 1]


def _source_coordinates(index_table: pd.DataFrame) -> np.ndarray:
    source_rows = index_table.drop_duplicates(
        subset=["source_line_index", "shot_in_line_index"]
    ).sort_values(["source_line_index", "shot_in_line_index"])
    return source_rows[["source_x_m", "source_y_m"]].to_numpy(dtype=np.float64)


def _assert_batch_contract(
    inputs: MaskedGatherInputs,
    *,
    batch_size: int,
    context_gather_count: int,
) -> None:
    assert inputs.target_observed.shape == (batch_size, *_RECEIVER_SHAPE, _TIME_COUNT)
    assert inputs.target_observation_mask.shape == (batch_size, *_RECEIVER_SHAPE)
    assert inputs.context_gathers.shape == (
        batch_size,
        context_gather_count,
        *_RECEIVER_SHAPE,
        _TIME_COUNT,
    )
    assert inputs.context_availability.shape == (
        batch_size,
        context_gather_count,
        *_RECEIVER_SHAPE,
    )
    assert inputs.source_deltas_m.shape == (batch_size, context_gather_count, 2)
    assert inputs.target_coordinates.shape == (batch_size, 2)
    assert inputs.target_observed.dtype == torch.float32
    assert inputs.context_gathers.dtype == torch.float32
    assert inputs.source_deltas_m.dtype == torch.float32
    assert inputs.target_coordinates.dtype == torch.float32
    assert inputs.target_observation_mask.dtype == torch.bool
    assert inputs.context_availability.dtype == torch.bool
    assert validate_masked_gather_inputs(inputs) is inputs


def test_random_trace_artifacts_load_partial_target_and_nearest_contexts(
    tmp_path: Path,
) -> None:
    artifacts = prepare_c3_volume_artifacts(tmp_path)
    volume_dir = _prepare_volume_index(artifacts)
    source, volume, index_table, metadata = _load_source_and_volume(artifacts, volume_dir)

    line_count, shot_count = (int(value) for value in metadata["shape"][1:3])  # type: ignore[index]
    source_count = line_count * shot_count
    shot_major_values = volume.values.transpose(1, 2, 3, 4, 0).reshape(
        source_count, *_RECEIVER_SHAPE, _TIME_COUNT
    )
    shot_major_observed = volume.observed_trace_mask.reshape(source_count, *_RECEIVER_SHAPE)
    shot_major_targets = volume.evaluation_target_trace_mask.reshape(source_count, *_RECEIVER_SHAPE)
    expected_target_sources = np.flatnonzero(shot_major_targets.reshape(source_count, -1).any(1))

    assert source.source_count == source_count == 4
    assert source.target_count == len(expected_target_sources) == 4
    assert source.context_gather_count == _CONTEXT_GATHER_COUNT
    assert np.any(
        shot_major_observed[expected_target_sources].reshape(source.target_count, -1).any(1)
        & shot_major_targets[expected_target_sources].reshape(source.target_count, -1).any(1)
    )

    requested = np.arange(source.target_count, dtype=np.int64)
    inputs = source.inputs(requested)
    _assert_batch_contract(
        inputs,
        batch_size=source.target_count,
        context_gather_count=_CONTEXT_GATHER_COUNT,
    )

    target_sources = _flat_target_source_indices(source, shot_count)
    assert np.array_equal(target_sources, expected_target_sources)
    assert np.array_equal(
        source.target_array_rows,
        volume.array_rows.reshape(source_count, *_RECEIVER_SHAPE)[target_sources],
    )
    assert np.array_equal(source.target_evaluation_mask, shot_major_targets[target_sources])

    target_values = inputs.target_observed.cpu().numpy()
    target_observed = inputs.target_observation_mask.cpu().numpy()
    assert np.array_equal(target_values, shot_major_values[target_sources])
    assert np.array_equal(target_observed, shot_major_observed[target_sources])
    assert np.all(target_values[~target_observed] == 0)
    assert np.array_equal(
        target_values[target_observed],
        shot_major_values[target_sources][target_observed],
    )

    context_sources = source.context_source_indices
    assert not np.any(context_sources == target_sources[:, None])
    context_values = inputs.context_gathers.cpu().numpy()
    context_available = inputs.context_availability.cpu().numpy()
    assert np.array_equal(context_values, shot_major_values[context_sources])
    assert np.array_equal(context_available, shot_major_observed[context_sources])
    assert np.any(~context_available)
    assert np.all(context_values[~context_available] == 0)

    coordinates_m = _source_coordinates(index_table)
    coordinate_minimum = coordinates_m.min(axis=0)
    coordinate_span = np.where(
        coordinates_m.max(axis=0) > coordinate_minimum,
        coordinates_m.max(axis=0) - coordinate_minimum,
        1.0,
    )
    normalized = (coordinates_m - coordinate_minimum) / coordinate_span
    assert np.allclose(inputs.target_coordinates.cpu().numpy(), normalized[target_sources])
    assert np.allclose(
        inputs.source_deltas_m.cpu().numpy(),
        coordinates_m[context_sources] - coordinates_m[target_sources, None],
    )


def test_readme_receiver_crop_materializes_dynamic_masked_gathers(tmp_path: Path) -> None:
    artifacts = prepare_c3_volume_artifacts(tmp_path)
    volume_dir = _prepare_volume_index(
        artifacts,
        relative_receiver_x_range=(0, 8),
        relative_receiver_y_range=(18, 50),
    )
    source, volume, _, metadata = _load_source_and_volume(artifacts, volume_dir)

    inputs = source.inputs(np.arange(source.target_count, dtype=np.int64))

    assert metadata["selection"]["relative_receiver_x"] == [0, 8]  # type: ignore[index]
    assert metadata["selection"]["relative_receiver_y"] == [18, 50]  # type: ignore[index]
    assert metadata["shape"] == [_TIME_COUNT, 2, 2, 8, 32]
    assert volume.values.shape == (_TIME_COUNT, 2, 2, 8, 32)
    assert source.target_count == 4
    assert source.target_array_rows.shape == (4, 8, 32)
    assert source.target_evaluation_mask.shape == (4, 8, 32)
    assert inputs.target_observed.shape == (4, 8, 32, _TIME_COUNT)
    assert inputs.target_observation_mask.shape == (4, 8, 32)
    assert inputs.context_gathers.shape == (4, _CONTEXT_GATHER_COUNT, 8, 32, _TIME_COUNT)
    assert inputs.context_availability.shape == (4, _CONTEXT_GATHER_COUNT, 8, 32)
    assert validate_masked_gather_inputs(inputs) is inputs


def test_whole_shot_artifacts_keep_missing_targets_out_of_context(
    tmp_path: Path,
) -> None:
    artifacts = prepare_c3_volume_artifacts(
        tmp_path,
        mask_kind=RANDOM_WHOLE_FFID_MASK_KIND,
    )
    volume_dir = _prepare_volume_index(artifacts)
    source, volume, _, metadata = _load_source_and_volume(artifacts, volume_dir)

    shot_count = int(metadata["shape"][2])  # type: ignore[index]
    target_sources = _flat_target_source_indices(source, shot_count)
    requested = np.arange(source.target_count, dtype=np.int64)
    inputs = source.inputs(requested)
    _assert_batch_contract(
        inputs,
        batch_size=source.target_count,
        context_gather_count=_CONTEXT_GATHER_COUNT,
    )

    assert source.target_count == 2
    assert not inputs.target_observation_mask.any()
    assert not inputs.target_observed.any()
    assert source.target_evaluation_mask.all()
    assert inputs.context_availability.flatten(2).any(dim=2).all()
    assert not np.isin(source.context_source_indices, target_sources).any()

    with pytest.raises(
        ValueError,
        match=r"target source indices .* has only 2 observed context sources; 3 required",
    ):
        load_masked_c3_gather_source(
            interim_dir=artifacts.interim_dir,
            processed_dir=artifacts.processed_dir,
            mask_dir=artifacts.mask_dir,
            case_dir=artifacts.case_dir,
            volume_dir=volume_dir,
            context_gather_count=3,
            device="cpu",
        )

    source_count = source.source_count
    target_mask = volume.evaluation_target_trace_mask.reshape(source_count, *_RECEIVER_SHAPE)[
        target_sources
    ]
    assert target_mask.all()


def test_masked_gather_source_never_recovers_sentinel_target_amplitudes(
    tmp_path: Path,
) -> None:
    artifacts = prepare_c3_volume_artifacts(tmp_path)
    volume_dir = _prepare_volume_index(artifacts)
    index_table, metadata = load_c3_volume_index(volume_dir)
    mask_table, _ = load_interpolation_mask(artifacts.mask_dir)
    amplitudes = np.load(
        artifacts.interim_dir / AMPLITUDES_FILE_NAME,
        allow_pickle=False,
    ).copy()
    time_s = np.load(artifacts.interim_dir / TIME_FILE_NAME, allow_pickle=False)

    sentinel = np.float32(1.2345e20)
    target_rows = mask_table.loc[
        mask_table[OBSERVATION_ROLE_COLUMN].eq(EVALUATION_TARGET_ROLE),
        "array_row",
    ].to_numpy(dtype=np.int64)
    assert len(target_rows) > 0
    amplitudes[target_rows] = sentinel
    observed_volume = materialize_observed_c3_volume(
        amplitudes,
        time_s,
        index_table,
        metadata,
        mask_table,
    )
    source = MaskedC3GatherSource(
        observed_volume,
        index_table,
        metadata,
        context_gather_count=_CONTEXT_GATHER_COUNT,
        device="cpu",
    )
    inputs = source.inputs(np.arange(source.target_count, dtype=np.int64))

    for field in fields(inputs):
        tensor = getattr(inputs, field.name)
        assert not torch.any(tensor == sentinel)
