from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from seis_interp.data.c3_volume_index_store import (
    VOLUME_INDEX_FILE_NAME,
    VOLUME_METADATA_FILE_NAME,
    load_c3_volume_index,
    validate_c3_volume_index,
    write_c3_volume_index,
)
from seis_interp.processing.c3_volume_index import (
    INDEX_CONTRACT,
    VOLUME_AXIS_ORDER,
    build_c3_volume_index,
)
from tests.fixtures.c3_volume_artifacts import make_c3_trace_table


def _index_table() -> pd.DataFrame:
    table = make_c3_trace_table()
    return build_c3_volume_index(
        table,
        table["array_row"].to_numpy(),
        source_line_range=(0, 1),
        shot_in_line_range=(0, 1),
        relative_receiver_x_range=(0, 1),
        relative_receiver_y_range=(0, 2),
    )


def _metadata() -> dict[str, object]:
    return {
        "volume_id": "synthetic_volume",
        "dataset_id": "synthetic",
        "partition": "test",
        "config_source": "studies/synthetic/config.yaml",
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


def test_write_load_round_trip_is_hash_bound_and_does_not_mutate_inputs(
    tmp_path: Path,
) -> None:
    table = _index_table()
    metadata = _metadata()
    original_table = table.copy(deep=True)
    original_metadata = copy.deepcopy(metadata)

    written = write_c3_volume_index(tmp_path / "volume", table, metadata)
    loaded_table, loaded = load_c3_volume_index(tmp_path / "volume")

    pd.testing.assert_frame_equal(loaded_table, original_table)
    assert metadata == original_metadata
    pd.testing.assert_frame_equal(table, original_table)
    assert loaded == written
    text = (tmp_path / "volume" / VOLUME_METADATA_FILE_NAME).read_text()
    assert text == json.dumps(written, indent=2, sort_keys=True, allow_nan=False) + "\n"
    assert set(path.name for path in (tmp_path / "volume").iterdir()) == {
        VOLUME_INDEX_FILE_NAME,
        VOLUME_METADATA_FILE_NAME,
    }


def test_loader_detects_changed_parquet(tmp_path: Path) -> None:
    output = tmp_path / "volume"
    write_c3_volume_index(output, _index_table(), _metadata())
    index_path = output / VOLUME_INDEX_FILE_NAME
    index_path.write_bytes(index_path.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="SHA-256"):
        load_c3_volume_index(output)


def test_validator_rejects_wrong_order_and_missing_cell(tmp_path: Path) -> None:
    output = tmp_path / "volume"
    metadata = write_c3_volume_index(output, _index_table(), _metadata())
    table = _index_table()

    with pytest.raises(ValueError, match="lexicographic"):
        validate_c3_volume_index(table.iloc[::-1].reset_index(drop=True), metadata)
    with pytest.raises(ValueError, match="cover every cell|row count"):
        validate_c3_volume_index(table.iloc[:-1].copy(), metadata)


def test_overwrite_replaces_only_owned_files(tmp_path: Path) -> None:
    output = tmp_path / "volume"
    write_c3_volume_index(output, _index_table(), _metadata())
    unrelated = output / "notes.txt"
    unrelated.write_text("keep")

    with pytest.raises(FileExistsError):
        write_c3_volume_index(output, _index_table(), _metadata())
    write_c3_volume_index(output, _index_table(), _metadata(), overwrite=True)

    assert unrelated.read_text() == "keep"
