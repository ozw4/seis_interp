from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from seis_interp.data.c3_volume_adapter import (
    load_observed_c3_volume,
    volume_to_trace_predictions,
)
from seis_interp.data.c3_volume_index_store import load_c3_volume_index
from seis_interp.pipelines.prepare_c3_volume_index import prepare_c3_volume_index
from tests.fixtures.c3_volume_artifacts import (
    PreparedC3VolumeArtifacts,
    prepare_c3_volume_artifacts,
)


def _prepare_volume_index(artifacts: PreparedC3VolumeArtifacts, output: Path) -> None:
    prepare_c3_volume_index(
        artifacts.interim_dir,
        artifacts.processed_dir,
        artifacts.mask_dir,
        artifacts.case_dir,
        output,
        volume_id="synthetic_volume",
        time_range=(0, 4),
        source_line_range=artifacts.source_line_range,
        shot_in_line_range=artifacts.shot_in_line_range,
        relative_receiver_x_range=(0, 8),
        relative_receiver_y_range=(0, 68),
        config_source="studies/synthetic/config.yaml",
    )


def test_case_to_observed_volume_and_inverse_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = prepare_c3_volume_artifacts(tmp_path)
    output = artifacts.processed_dir / "volumes" / "synthetic-volume"
    _prepare_volume_index(artifacts, output)

    volume = load_observed_c3_volume(
        interim_dir=artifacts.interim_dir,
        processed_dir=artifacts.processed_dir,
        mask_dir=artifacts.mask_dir,
        case_dir=artifacts.case_dir,
        volume_dir=output,
    )
    index, _ = load_c3_volume_index(output)
    amplitudes = np.load(artifacts.interim_dir / "amplitudes.npy", allow_pickle=False)
    rows, predictions = volume_to_trace_predictions(volume.values, volume.array_rows)

    observed = volume.observed_trace_mask.reshape(-1)
    targets = volume.evaluation_target_trace_mask.reshape(-1)
    expected_predictions = np.zeros_like(predictions)
    expected_predictions[observed] = amplitudes[rows[observed], :4]

    assert volume.values.shape == (4, 2, 2, 8, 68)
    assert np.any(observed)
    assert np.any(targets)
    assert np.array_equal(
        volume.values[:, volume.observed_trace_mask],
        amplitudes[volume.array_rows[volume.observed_trace_mask], :4].T,
    )
    assert np.all(volume.values[:, volume.evaluation_target_trace_mask] == 0)
    assert np.array_equal(rows, index["array_row"].to_numpy(dtype=np.int64))
    assert np.array_equal(predictions, expected_predictions)

    actual_load = np.load
    injected_amplitude_paths: list[Path] = []

    def load_with_nan_targets(path: Path, *args: object, **kwargs: object) -> np.ndarray:
        values = actual_load(path, *args, **kwargs)
        if Path(path).name == "amplitudes.npy":
            injected_amplitude_paths.append(Path(path))
            values = np.array(values, copy=True)
            values[rows[targets]] = np.nan
        return values

    monkeypatch.setattr("seis_interp.data.c3_volume_adapter.np.load", load_with_nan_targets)
    nan_target_volume = load_observed_c3_volume(
        interim_dir=artifacts.interim_dir,
        processed_dir=artifacts.processed_dir,
        mask_dir=artifacts.mask_dir,
        case_dir=artifacts.case_dir,
        volume_dir=output,
    )

    assert np.array_equal(nan_target_volume.values, volume.values)
    assert injected_amplitude_paths == [artifacts.interim_dir / "amplitudes.npy"]


def test_rejects_a_rank_compressed_missing_source_line(tmp_path: Path) -> None:
    artifacts = prepare_c3_volume_artifacts(
        tmp_path,
        physical_source_line_indices=(0, 1, 2, 4),
    )

    with pytest.raises(
        ValueError,
        match="selected source lines must be contiguous on the 160 m",
    ):
        _prepare_volume_index(
            artifacts,
            artifacts.processed_dir / "volumes" / "missing-source-line",
        )


def test_rejects_a_rank_compressed_missing_shot(tmp_path: Path) -> None:
    artifacts = prepare_c3_volume_artifacts(
        tmp_path,
        omitted_shots=frozenset({(2, 1)}),
    )

    with pytest.raises(
        ValueError,
        match="selected shots in source line 2 must be contiguous on the 80 m",
    ):
        _prepare_volume_index(
            artifacts,
            artifacts.processed_dir / "volumes" / "missing-shot",
        )
