from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seis_interp.data.interim_trace_dataset import load_interim_trace_dataset
from seis_interp.data.trace_store import write_interim_trace_dataset

TRACE_COUNT = 4
SAMPLE_COUNT = 3


def _write_interim_dataset(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    source_path = tmp_path / "source.sgy"
    source_path.write_bytes(b"synthetic SEG-Y placeholder")
    output_dir = tmp_path / "interim"
    metadata = write_interim_trace_dataset(
        output_dir=output_dir,
        trace_table=_trace_table(),
        amplitudes=_amplitudes(),
        time_s=_time_axis(),
        source_path=source_path,
        dataset_id="synthetic",
        selection={"ffid": 20, "expected_trace_count": TRACE_COUNT},
    )
    return output_dir, metadata


def _trace_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trace_index": np.arange(10, 10 + TRACE_COUNT, dtype=np.int64),
            "ffid": np.full(TRACE_COUNT, 20, dtype=np.int64),
            "cmp_x_m": np.array([100.0, 110.0, 120.0, 130.0]),
            "cmp_y_m": np.array([200.0, 205.0, 210.0, 215.0]),
            "offset_m": np.array([500.0, 550.0, 600.0, 650.0]),
            "azimuth_deg": np.array([10.0, 20.0, 30.0, 40.0]),
            "sample_interval_s": np.full(TRACE_COUNT, 0.008),
        }
    )


def _amplitudes() -> np.ndarray:
    return np.arange(1, TRACE_COUNT * SAMPLE_COUNT + 1, dtype=np.float32).reshape(
        TRACE_COUNT, SAMPLE_COUNT
    )


def _time_axis() -> np.ndarray:
    return np.arange(SAMPLE_COUNT, dtype=np.float64) * 0.008


def _read_metadata(directory: Path) -> dict[str, object]:
    return json.loads((directory / "dataset.json").read_text(encoding="utf-8"))


def _write_metadata(directory: Path, metadata: object) -> None:
    (directory / "dataset.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_loads_the_table_arrays_and_metadata(tmp_path: Path) -> None:
    directory, expected_metadata = _write_interim_dataset(tmp_path)

    dataset = load_interim_trace_dataset(directory)

    pd.testing.assert_frame_equal(
        dataset.trace_table,
        pd.read_parquet(directory / "traces.parquet"),
    )
    np.testing.assert_array_equal(dataset.amplitudes, _amplitudes())
    np.testing.assert_array_equal(dataset.time_s, _time_axis())
    assert dataset.metadata == expected_metadata


def test_accepts_array_rows_in_a_different_table_order(tmp_path: Path) -> None:
    directory, _ = _write_interim_dataset(tmp_path)
    trace_table = pd.read_parquet(directory / "traces.parquet").iloc[[3, 0, 2, 1]]
    trace_table.to_parquet(directory / "traces.parquet", index=False)

    dataset = load_interim_trace_dataset(directory)

    assert dataset.trace_table["array_row"].tolist() == [3, 0, 2, 1]


@pytest.mark.parametrize(
    "file_name",
    ["traces.parquet", "amplitudes.npy", "time_s.npy", "dataset.json"],
)
def test_rejects_a_missing_required_file(tmp_path: Path, file_name: str) -> None:
    directory, _ = _write_interim_dataset(tmp_path)
    (directory / file_name).unlink()

    with pytest.raises(FileNotFoundError, match=file_name):
        load_interim_trace_dataset(directory)


def test_rejects_metadata_that_is_not_a_json_object(tmp_path: Path) -> None:
    directory, _ = _write_interim_dataset(tmp_path)
    _write_metadata(directory, ["not", "an", "object"])

    with pytest.raises(ValueError, match="JSON object"):
        load_interim_trace_dataset(directory)


def test_rejects_duplicate_array_rows(tmp_path: Path) -> None:
    directory, _ = _write_interim_dataset(tmp_path)
    trace_table = pd.read_parquet(directory / "traces.parquet")
    trace_table["array_row"] = [0, 1, 1, 3]
    trace_table.to_parquet(directory / "traces.parquet", index=False)

    with pytest.raises(ValueError, match="duplicate"):
        load_interim_trace_dataset(directory)


@pytest.mark.parametrize("array_rows", [[0, 1, 2, 4], [-1, 0, 1, 2]])
def test_rejects_gapped_or_out_of_range_array_rows(
    tmp_path: Path,
    array_rows: list[int],
) -> None:
    directory, _ = _write_interim_dataset(tmp_path)
    trace_table = pd.read_parquet(directory / "traces.parquet")
    trace_table["array_row"] = array_rows
    trace_table.to_parquet(directory / "traces.parquet", index=False)

    with pytest.raises(ValueError, match="every integer"):
        load_interim_trace_dataset(directory)


def test_rejects_an_amplitude_shape_mismatch(tmp_path: Path) -> None:
    directory, _ = _write_interim_dataset(tmp_path)
    np.save(directory / "amplitudes.npy", _amplitudes()[:-1])

    with pytest.raises(ValueError, match="rows"):
        load_interim_trace_dataset(directory)


@pytest.mark.parametrize(
    ("file_name", "array", "expected_message"),
    [
        ("amplitudes.npy", _amplitudes().astype(np.float64), "float32"),
        ("time_s.npy", _time_axis().astype(np.float32), "float64"),
    ],
)
def test_rejects_an_array_dtype_mismatch(
    tmp_path: Path,
    file_name: str,
    array: np.ndarray,
    expected_message: str,
) -> None:
    directory, _ = _write_interim_dataset(tmp_path)
    np.save(directory / file_name, array)

    with pytest.raises(ValueError, match=expected_message):
        load_interim_trace_dataset(directory)


def test_rejects_a_metadata_count_mismatch(tmp_path: Path) -> None:
    directory, _ = _write_interim_dataset(tmp_path)
    metadata = _read_metadata(directory)
    metadata["trace_count"] = TRACE_COUNT + 1
    _write_metadata(directory, metadata)

    with pytest.raises(ValueError, match="trace_count"):
        load_interim_trace_dataset(directory)


def test_rejects_a_metadata_file_shape_mismatch(tmp_path: Path) -> None:
    directory, _ = _write_interim_dataset(tmp_path)
    metadata = _read_metadata(directory)
    files = metadata["files"]
    assert isinstance(files, dict)
    amplitude_record = files["amplitudes.npy"]
    assert isinstance(amplitude_record, dict)
    amplitude_record["shape"] = [TRACE_COUNT, SAMPLE_COUNT + 1]
    _write_metadata(directory, metadata)

    with pytest.raises(ValueError, match="shape"):
        load_interim_trace_dataset(directory)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("coordinate_order", ["time_s", "cmp_y_m", "cmp_x_m", "offset_m", "azimuth_deg"]),
        (
            "coordinate_units",
            {
                "time_s": "ms",
                "cmp_x_m": "m",
                "cmp_y_m": "m",
                "offset_m": "m",
                "azimuth_deg": "deg",
            },
        ),
    ],
)
def test_rejects_a_coordinate_schema_mismatch(
    tmp_path: Path,
    key: str,
    value: object,
) -> None:
    directory, _ = _write_interim_dataset(tmp_path)
    metadata = _read_metadata(directory)
    metadata[key] = value
    _write_metadata(directory, metadata)

    with pytest.raises(ValueError, match=key):
        load_interim_trace_dataset(directory)


@pytest.mark.parametrize("target", ["amplitudes", "time", "geometry"])
def test_rejects_non_finite_values(tmp_path: Path, target: str) -> None:
    directory, _ = _write_interim_dataset(tmp_path)
    if target == "amplitudes":
        amplitudes = _amplitudes()
        amplitudes[0, 0] = np.nan
        np.save(directory / "amplitudes.npy", amplitudes)
    elif target == "time":
        time_s = _time_axis()
        time_s[-1] = np.inf
        np.save(directory / "time_s.npy", time_s)
    else:
        trace_table = pd.read_parquet(directory / "traces.parquet")
        trace_table.loc[0, "cmp_x_m"] = np.nan
        trace_table.to_parquet(directory / "traces.parquet", index=False)

    with pytest.raises(ValueError, match="non-finite"):
        load_interim_trace_dataset(directory)
