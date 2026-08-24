from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seis_interp.data.trace_schema import PHYSICAL_COORDINATE_ORDER, PHYSICAL_COORDINATE_UNITS
from seis_interp.data.trace_store import OUTPUT_FILE_NAMES, write_interim_trace_dataset

SAMPLE_INTERVAL_S = 0.004


def make_trace_table(trace_count: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trace_index": np.arange(trace_count, dtype=np.int64),
            "ffid": np.full(trace_count, 20, dtype=np.int64),
            "cmp_x_m": np.arange(trace_count, dtype=np.float64) * 10.0,
            "cmp_y_m": np.arange(trace_count, dtype=np.float64) * 20.0,
            "offset_m": np.arange(trace_count, dtype=np.float64) * 30.0,
            "azimuth_deg": np.arange(trace_count, dtype=np.float64),
            "sample_interval_s": np.full(trace_count, SAMPLE_INTERVAL_S, dtype=np.float64),
        }
    )


def make_amplitudes(trace_count: int = 3, sample_count: int = 4) -> np.ndarray:
    return np.arange(trace_count * sample_count, dtype=np.float32).reshape(
        trace_count, sample_count
    )


def make_time_axis(sample_count: int = 4) -> np.ndarray:
    return np.arange(sample_count, dtype=np.float64) * SAMPLE_INTERVAL_S


def make_source_file(tmp_path: Path) -> Path:
    source = tmp_path / "source.sgy"
    source.write_bytes(b"tiny-source-bytes")
    return source


def write_default_dataset(tmp_path: Path, **overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "output_dir": tmp_path / "out",
        "trace_table": make_trace_table(),
        "amplitudes": make_amplitudes(),
        "time_s": make_time_axis(),
        "source_path": make_source_file(tmp_path),
        "dataset_id": "seg_c3_na",
    }
    kwargs.update(overrides)
    return write_interim_trace_dataset(**kwargs)  # type: ignore[arg-type]


def test_writes_the_four_output_files(tmp_path: Path) -> None:
    write_default_dataset(tmp_path)

    for file_name in OUTPUT_FILE_NAMES:
        assert (tmp_path / "out" / file_name).is_file()


def test_array_row_links_parquet_rows_to_amplitude_rows(tmp_path: Path) -> None:
    amplitudes = make_amplitudes()
    write_default_dataset(tmp_path, amplitudes=amplitudes)

    stored_table = pd.read_parquet(tmp_path / "out" / "traces.parquet")
    stored_amplitudes = np.load(tmp_path / "out" / "amplitudes.npy")

    assert stored_table.columns[0] == "array_row"
    assert stored_table["array_row"].tolist() == [0, 1, 2]
    for array_row, trace_index in zip(
        stored_table["array_row"], stored_table["trace_index"], strict=True
    ):
        np.testing.assert_array_equal(stored_amplitudes[array_row], amplitudes[trace_index])


def test_preserves_array_dtypes_and_shapes(tmp_path: Path) -> None:
    write_default_dataset(tmp_path)

    stored_amplitudes = np.load(tmp_path / "out" / "amplitudes.npy")
    stored_time = np.load(tmp_path / "out" / "time_s.npy")

    assert stored_amplitudes.dtype == np.float32
    assert stored_amplitudes.shape == (3, 4)
    assert stored_time.dtype == np.float64
    assert stored_time.shape == (4,)


def test_metadata_records_basename_and_no_absolute_paths(tmp_path: Path) -> None:
    metadata = write_default_dataset(tmp_path)

    assert metadata["source_file"] == "source.sgy"
    raw_metadata = (tmp_path / "out" / "dataset.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in raw_metadata
    assert json.loads(raw_metadata) == metadata


def test_metadata_stores_the_selection_provenance(tmp_path: Path) -> None:
    selection = {"ffid": 20, "expected_trace_count": 3}

    metadata = write_default_dataset(tmp_path, selection=selection)

    stored = json.loads((tmp_path / "out" / "dataset.json").read_text(encoding="utf-8"))
    assert metadata["selection"] == selection
    assert stored["selection"] == selection


def test_metadata_selection_defaults_to_an_empty_mapping(tmp_path: Path) -> None:
    metadata = write_default_dataset(tmp_path)

    assert metadata["selection"] == {}


def test_rejects_a_selection_that_cannot_be_serialised(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="JSON serialisable"):
        write_default_dataset(tmp_path, selection={"ffid": object()})

    assert not (tmp_path / "out").exists()


def test_metadata_records_the_expected_fields(tmp_path: Path) -> None:
    metadata = write_default_dataset(tmp_path)

    assert metadata["dataset_id"] == "seg_c3_na"
    assert metadata["trace_count"] == 3
    assert metadata["sample_count"] == 4
    assert metadata["sample_interval_s"] == SAMPLE_INTERVAL_S
    assert metadata["ffids"] == [20]
    assert metadata["time_origin_s"] == 0.0
    assert metadata["coordinate_order"] == list(PHYSICAL_COORDINATE_ORDER)
    assert metadata["coordinate_units"] == PHYSICAL_COORDINATE_UNITS
    assert metadata["azimuth_convention"] == (
        "degrees(atan2(source_x-receiver_x, source_y-receiver_y)) wrapped to [0, 360)"
    )
    assert metadata["created_at_utc"].endswith("+00:00")


def test_source_sha256_matches_the_source_file(tmp_path: Path) -> None:
    metadata = write_default_dataset(tmp_path)

    assert metadata["source_sha256"] == hashlib.sha256(b"tiny-source-bytes").hexdigest()


def test_row_count_mismatch_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="rows"):
        write_default_dataset(tmp_path, amplitudes=make_amplitudes(trace_count=2))


def test_sample_count_mismatch_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="samples"):
        write_default_dataset(tmp_path, time_s=make_time_axis(sample_count=5))


def test_amplitude_dtype_mismatch_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="amplitudes must be float32"):
        write_default_dataset(tmp_path, amplitudes=make_amplitudes().astype(np.float64))


def test_time_dtype_mismatch_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="time_s must be float64"):
        write_default_dataset(tmp_path, time_s=make_time_axis().astype(np.float32))


def test_duplicate_trace_index_is_an_error(tmp_path: Path) -> None:
    trace_table = make_trace_table()
    trace_table.loc[2, "trace_index"] = 0

    with pytest.raises(ValueError, match="duplicate trace_index"):
        write_default_dataset(tmp_path, trace_table=trace_table)


def test_non_finite_geometry_is_an_error(tmp_path: Path) -> None:
    trace_table = make_trace_table()
    trace_table.loc[1, "offset_m"] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        write_default_dataset(tmp_path, trace_table=trace_table)


def test_non_empty_output_directory_without_overwrite_is_an_error(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "keep.txt").write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_default_dataset(tmp_path)


def test_overwrite_replaces_only_the_generated_files(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    write_default_dataset(tmp_path)
    (output_dir / "keep.txt").write_text("existing", encoding="utf-8")

    metadata = write_default_dataset(
        tmp_path,
        trace_table=make_trace_table(trace_count=2),
        amplitudes=make_amplitudes(trace_count=2),
        overwrite=True,
    )

    assert metadata["trace_count"] == 2
    assert np.load(output_dir / "amplitudes.npy").shape == (2, 4)
    assert (output_dir / "keep.txt").read_text(encoding="utf-8") == "existing"


def test_parent_directories_are_created(tmp_path: Path) -> None:
    write_default_dataset(tmp_path, output_dir=tmp_path / "nested" / "deeper" / "out")

    assert (tmp_path / "nested" / "deeper" / "out" / "dataset.json").is_file()
