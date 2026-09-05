from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seis_interp.data.c3_volume_adapter import (
    materialize_observed_c3_volume,
    volume_to_trace_predictions,
)
from seis_interp.data.c3_volume_index_store import write_c3_volume_index
from seis_interp.processing.c3_volume_index import (
    INDEX_CONTRACT,
    VOLUME_AXIS_ORDER,
    build_c3_volume_index,
)
from tests.fixtures.c3_volume_artifacts import make_c3_trace_table


def _volume_inputs(tmp_path: Path) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    table = make_c3_trace_table()
    index = build_c3_volume_index(
        table,
        table["array_row"].to_numpy(),
        source_line_range=(0, 1),
        shot_in_line_range=(0, 1),
        relative_receiver_x_range=(0, 1),
        relative_receiver_y_range=(0, 2),
    )
    base = {
        "volume_id": "synthetic_volume",
        "dataset_id": "synthetic",
        "partition": "test",
        "config_source": None,
        "axis_order": list(VOLUME_AXIS_ORDER),
        "selection": {
            "time": [1, 4],
            "source_line": [0, 1],
            "shot_in_line": [0, 1],
            "relative_receiver_x": [0, 1],
            "relative_receiver_y": [0, 2],
        },
        "shape": [3, 1, 1, 1, 2],
        "trace_count": 2,
        "role_counts": {"observed": 1, "evaluation_target": 1},
        "index_contract": dict(INDEX_CONTRACT),
        "benchmark_case": {
            "case_id": "synthetic_case",
            "file": "benchmark_case.json",
            "sha256": "a" * 64,
        },
    }
    metadata = write_c3_volume_index(tmp_path / "volume", index, base)
    mask = pd.DataFrame(
        {
            "array_row": table["array_row"].to_numpy(dtype=np.int64),
            "observation_role": np.where(
                table["array_row"].eq(index.iloc[0]["array_row"]),
                "observed",
                "evaluation_target",
            ),
        }
    )
    return index, metadata, mask


def test_materializes_observed_only_and_never_validates_target_values(tmp_path: Path) -> None:
    index, metadata, mask = _volume_inputs(tmp_path)
    amplitudes = np.arange(6 * 544, dtype=np.float32).reshape(544, 6)
    target_row = int(index.iloc[1]["array_row"])
    amplitudes[target_row] = np.nan
    original = amplitudes.copy()
    time_s = np.arange(6, dtype=np.float64) * 0.008

    volume = materialize_observed_c3_volume(amplitudes, time_s, index, metadata, mask)

    assert volume.values.shape == (3, 1, 1, 1, 2)
    assert np.array_equal(volume.time_s, time_s[1:4])
    assert np.array_equal(volume.values[..., 0].reshape(-1), amplitudes[0, 1:4])
    assert np.array_equal(volume.values[..., 1], np.zeros((3, 1, 1, 1)))
    assert np.all(volume.observed_trace_mask ^ volume.evaluation_target_trace_mask)
    assert np.array_equal(amplitudes, original, equal_nan=True)


def test_rejects_nonfinite_observed_values(tmp_path: Path) -> None:
    index, metadata, mask = _volume_inputs(tmp_path)
    amplitudes = np.ones((544, 6), dtype=np.float32)
    amplitudes[int(index.iloc[0]["array_row"]), 2] = np.nan

    with pytest.raises(ValueError, match="observed amplitudes"):
        materialize_observed_c3_volume(
            amplitudes,
            np.arange(6, dtype=np.float64),
            index,
            metadata,
            mask,
        )


def test_inverse_mapping_uses_spatial_lexicographic_order() -> None:
    predicted = np.arange(3 * 2 * 2, dtype=np.float32).reshape(3, 1, 1, 2, 2)
    rows = np.array([[[[7, 3], [9, 1]]]], dtype=np.int64)

    output_rows, traces = volume_to_trace_predictions(predicted, rows)

    assert np.array_equal(output_rows, [7, 3, 9, 1])
    assert np.array_equal(traces, predicted.transpose(1, 2, 3, 4, 0).reshape(4, 3))
