from __future__ import annotations

from pathlib import Path

import numpy as np

from seis_interp.data.c3_volume_adapter import (
    load_observed_c3_volume,
    volume_to_trace_predictions,
)
from seis_interp.pipelines.prepare_c3_volume_index import prepare_c3_volume_index
from tests.fixtures.c3_volume_artifacts import prepare_c3_volume_artifacts


def test_case_to_observed_volume_and_inverse_mapping(tmp_path: Path) -> None:
    artifacts = prepare_c3_volume_artifacts(tmp_path)
    output = artifacts.processed_dir / "volumes" / "synthetic-volume"
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

    volume = load_observed_c3_volume(
        interim_dir=artifacts.interim_dir,
        processed_dir=artifacts.processed_dir,
        mask_dir=artifacts.mask_dir,
        case_dir=artifacts.case_dir,
        volume_dir=output,
    )
    rows, predictions = volume_to_trace_predictions(volume.values, volume.array_rows)

    assert volume.values.shape == (4, 1, 1, 8, 68)
    assert np.all(volume.values[:, volume.evaluation_target_trace_mask] == 0)
    assert np.array_equal(rows, volume.array_rows.reshape(-1))
    assert predictions.shape == (544, 4)
