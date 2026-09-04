from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import seis_interp.pipelines.prepare_interpolation_mask as interpolation_mask_pipeline
from seis_interp.data.file_checksums import file_sha256
from seis_interp.data.interpolation_mask_store import (
    MASK_METADATA_FILE_NAME,
    MASK_TABLE_FILE_NAME,
    load_interpolation_mask,
)
from seis_interp.pipelines.prepare_interpolation_mask import prepare_interpolation_mask
from seis_interp.processing.interpolation_masks import (
    EVALUATION_TARGET_ROLE,
    OBSERVATION_ROLE_COLUMN,
    OBSERVED_ROLE,
    RANDOM_TRACE_MASK_KIND,
    RANDOM_WHOLE_FFID_MASK_KIND,
)
from seis_interp.processing.trace_canonicalization import PHYSICAL_COORDINATE_COLUMNS

DATASET_ID = "synthetic"
CONFIG_SOURCE = "studies/study_001/config.yaml"


def _trace_table() -> pd.DataFrame:
    array_rows = np.arange(16, dtype=np.int64)
    return pd.DataFrame(
        {
            "array_row": array_rows,
            "ffid": np.repeat(np.arange(100, 108, dtype=np.int64), 2),
            "source_x_m": array_rows.astype(np.float64),
            "source_y_m": np.zeros(16, dtype=np.float64),
            "receiver_x_m": array_rows.astype(np.float64) + 100.0,
            "receiver_y_m": np.full(16, 200.0, dtype=np.float64),
        }
    )


def _split_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "array_row": np.arange(16, dtype=np.int64),
            "split": ["train"] * 6 + ["validation"] * 4 + ["test"] * 6,
        }
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_inputs(
    tmp_path: Path,
    *,
    name: str = "inputs",
    trace_order: np.ndarray | None = None,
    split_scope: str = "whole_ffid",
) -> tuple[Path, Path]:
    root = tmp_path / name
    interim = root / "interim"
    processed = root / "processed"
    interim.mkdir(parents=True)
    processed.mkdir(parents=True)

    trace_table = _trace_table()
    if trace_order is not None:
        trace_table = trace_table.iloc[trace_order].reset_index(drop=True)
    trace_table.to_parquet(interim / "traces.parquet", index=False)
    _write_json(interim / "dataset.json", {"dataset_id": DATASET_ID})

    _split_table().to_parquet(processed / "trace_split.parquet", index=False)
    preparation = {
        "dataset_id": DATASET_ID,
        "trace_count": len(trace_table),
        "split_scope": split_scope,
        "split_counts": {"train": 6, "validation": 4, "test": 6},
        "input_files": {
            "traces.parquet": {
                "sha256": file_sha256(interim / "traces.parquet"),
            }
        },
    }
    if split_scope == "whole_ffid":
        preparation["ffid_split_counts"] = {"train": 3, "validation": 2, "test": 3}
    _write_json(processed / "preparation.json", preparation)
    return interim, processed


def _prepare(
    interim: Path,
    processed: Path,
    output: Path,
    **overrides: object,
) -> dict[str, object]:
    arguments: dict[str, object] = {
        "interim_dir": interim,
        "processed_dir": processed,
        "output_dir": output,
        "partition": "test",
        "kind": RANDOM_TRACE_MASK_KIND,
        "missing_fraction": 0.5,
        "random_seed": 42,
        "config_source": CONFIG_SOURCE,
    }
    arguments.update(overrides)
    return prepare_interpolation_mask(**arguments)  # type: ignore[arg-type]


def _role_by_array_row(mask_table: pd.DataFrame) -> dict[int, str]:
    return dict(
        zip(
            mask_table["array_row"].tolist(),
            mask_table[OBSERVATION_ROLE_COLUMN].tolist(),
            strict=True,
        )
    )


def test_random_trace_writes_a_mask_artifact_without_amplitude_inputs(tmp_path: Path) -> None:
    interim, processed = _write_inputs(tmp_path)
    output = tmp_path / "mask"

    summary = _prepare(interim, processed, output)
    mask_table, metadata = load_interpolation_mask(output)

    assert set(path.name for path in output.iterdir()) == {
        MASK_TABLE_FILE_NAME,
        MASK_METADATA_FILE_NAME,
    }
    assert summary == metadata
    assert mask_table.columns.tolist() == ["array_row", OBSERVATION_ROLE_COLUMN]
    assert mask_table[OBSERVATION_ROLE_COLUMN].value_counts().to_dict() == {
        EVALUATION_TARGET_ROLE: 3,
        OBSERVED_ROLE: 3,
    }
    assert not (interim / "amplitudes.npy").exists()
    assert not (processed / "normalization.json").exists()


def test_mask_contains_only_the_requested_partition_rows(tmp_path: Path) -> None:
    interim, processed = _write_inputs(tmp_path)

    _prepare(interim, processed, tmp_path / "mask")
    mask_table, _ = load_interpolation_mask(tmp_path / "mask")

    split_table = pd.read_parquet(processed / "trace_split.parquet")
    expected = set(split_table.loc[split_table["split"].eq("test"), "array_row"])
    assert set(mask_table["array_row"]) == expected
    assert not {0, 6}.intersection(mask_table["array_row"])


def test_trace_row_order_does_not_change_role_mapping(tmp_path: Path) -> None:
    first_interim, first_processed = _write_inputs(tmp_path, name="first")
    second_interim, second_processed = _write_inputs(
        tmp_path,
        name="second",
        trace_order=np.arange(15, -1, -1),
    )

    _prepare(first_interim, first_processed, tmp_path / "first_mask")
    _prepare(second_interim, second_processed, tmp_path / "second_mask")
    first_mask, _ = load_interpolation_mask(tmp_path / "first_mask")
    second_mask, _ = load_interpolation_mask(tmp_path / "second_mask")

    assert _role_by_array_row(first_mask) == _role_by_array_row(second_mask)


def test_pipeline_materializes_only_mask_columns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interim, processed = _write_inputs(tmp_path)
    canonical_input_columns: list[str] = []
    candidate_columns: list[str] = []
    canonicalize = interpolation_mask_pipeline.canonicalize_eligible_physical_coordinates
    make_mask = interpolation_mask_pipeline.make_random_trace_mask

    def recording_canonicalize(
        joined_table: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        canonical_input_columns.extend(joined_table.columns)
        return canonicalize(joined_table)

    def recording_make_mask(
        candidate_table: pd.DataFrame,
        *,
        missing_fraction: float,
        random_seed: int,
    ) -> pd.DataFrame:
        candidate_columns.extend(candidate_table.columns)
        return make_mask(
            candidate_table,
            missing_fraction=missing_fraction,
            random_seed=random_seed,
        )

    monkeypatch.setattr(
        interpolation_mask_pipeline,
        "canonicalize_eligible_physical_coordinates",
        recording_canonicalize,
    )
    monkeypatch.setattr(
        interpolation_mask_pipeline,
        "make_random_trace_mask",
        recording_make_mask,
    )

    _prepare(interim, processed, tmp_path / "mask")

    assert canonical_input_columns == [
        "array_row",
        "ffid",
        *PHYSICAL_COORDINATE_COLUMNS,
        "split",
    ]
    assert candidate_columns == ["array_row", "ffid"]


def test_pipeline_hashes_each_input_file_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interim, processed = _write_inputs(tmp_path)
    hashed_paths: list[Path] = []

    def recording_file_sha256(path: Path) -> str:
        hashed_paths.append(path)
        return file_sha256(path)

    monkeypatch.setattr(
        interpolation_mask_pipeline,
        "file_sha256",
        recording_file_sha256,
    )

    _prepare(interim, processed, tmp_path / "mask")

    expected_paths = [
        interim / "traces.parquet",
        interim / "dataset.json",
        processed / "trace_split.parquet",
        processed / "preparation.json",
    ]
    assert all(hashed_paths.count(path) == 1 for path in expected_paths)
    assert len(hashed_paths) == len(expected_paths)


def test_duplicate_physical_cells_are_canonicalized_before_random_masking(
    tmp_path: Path,
) -> None:
    interim, processed = _write_inputs(tmp_path)
    traces_path = interim / "traces.parquet"
    trace_table = pd.read_parquet(traces_path)
    physical_key = ["source_x_m", "source_y_m", "receiver_x_m", "receiver_y_m"]
    trace_table.loc[15, physical_key] = trace_table.loc[12, physical_key].to_numpy()
    trace_table.to_parquet(traces_path, index=False)
    preparation_path = processed / "preparation.json"
    preparation = _read_json(preparation_path)
    preparation["input_files"]["traces.parquet"]["sha256"] = file_sha256(traces_path)
    _write_json(preparation_path, preparation)

    metadata = _prepare(interim, processed, tmp_path / "mask")
    mask_table, _ = load_interpolation_mask(tmp_path / "mask")
    masked_traces = mask_table.merge(trace_table, on="array_row", validate="one_to_one")

    assert set(mask_table["array_row"]) == {10, 11, 12, 13, 14}
    assert 12 in set(mask_table["array_row"])
    assert 15 not in set(mask_table["array_row"])
    assert not masked_traces.duplicated(physical_key, keep=False).any()
    assert masked_traces.groupby(physical_key)[OBSERVATION_ROLE_COLUMN].nunique().eq(1).all()
    assert metadata["candidate_trace_count"] == 5
    assert metadata["duplicate_physical_coordinates"] == {
        "policy": "keep_lowest_array_row",
        "removed_trace_count": 1,
    }


def test_duplicate_input_order_does_not_change_canonical_random_mask(tmp_path: Path) -> None:
    first_interim, first_processed = _write_inputs(tmp_path, name="first")
    first_traces_path = first_interim / "traces.parquet"
    trace_table = pd.read_parquet(first_traces_path)
    physical_key = ["source_x_m", "source_y_m", "receiver_x_m", "receiver_y_m"]
    trace_table.loc[15, physical_key] = trace_table.loc[12, physical_key].to_numpy()
    trace_table.to_parquet(first_traces_path, index=False)

    second_interim, second_processed = _write_inputs(tmp_path, name="second")
    second_traces_path = second_interim / "traces.parquet"
    trace_table.iloc[::-1].reset_index(drop=True).to_parquet(second_traces_path, index=False)

    for interim, processed in (
        (first_interim, first_processed),
        (second_interim, second_processed),
    ):
        preparation_path = processed / "preparation.json"
        preparation = _read_json(preparation_path)
        preparation["input_files"]["traces.parquet"]["sha256"] = file_sha256(
            interim / "traces.parquet"
        )
        _write_json(preparation_path, preparation)

    first_metadata = _prepare(first_interim, first_processed, tmp_path / "first_mask")
    second_metadata = _prepare(second_interim, second_processed, tmp_path / "second_mask")
    first_mask, _ = load_interpolation_mask(tmp_path / "first_mask")
    second_mask, _ = load_interpolation_mask(tmp_path / "second_mask")

    assert _role_by_array_row(first_mask) == _role_by_array_row(second_mask)
    assert first_metadata["candidate_trace_count"] == second_metadata["candidate_trace_count"]
    assert (
        first_metadata["duplicate_physical_coordinates"]
        == second_metadata["duplicate_physical_coordinates"]
    )


def test_random_whole_ffid_assigns_one_role_to_each_ffid(tmp_path: Path) -> None:
    interim, processed = _write_inputs(tmp_path)

    _prepare(
        interim,
        processed,
        tmp_path / "mask",
        kind=RANDOM_WHOLE_FFID_MASK_KIND,
    )
    mask_table, _ = load_interpolation_mask(tmp_path / "mask")
    trace_table = pd.read_parquet(interim / "traces.parquet")
    joined = mask_table.merge(trace_table, on="array_row", validate="one_to_one")

    assert joined.groupby("ffid")[OBSERVATION_ROLE_COLUMN].nunique().eq(1).all()
    assert set(joined[OBSERVATION_ROLE_COLUMN]) == {
        OBSERVED_ROLE,
        EVALUATION_TARGET_ROLE,
    }


def test_random_whole_ffid_rejects_a_non_whole_ffid_split_scope(tmp_path: Path) -> None:
    interim, processed = _write_inputs(tmp_path, split_scope="global")

    with pytest.raises(ValueError, match="split_scope.*whole_ffid"):
        _prepare(
            interim,
            processed,
            tmp_path / "mask",
            kind=RANDOM_WHOLE_FFID_MASK_KIND,
        )


def test_whole_ffid_scope_rejects_crossing_ffid_outside_requested_partition(
    tmp_path: Path,
) -> None:
    interim, processed = _write_inputs(tmp_path)
    split_path = processed / "trace_split.parquet"
    split_table = pd.read_parquet(split_path)
    split_table.loc[split_table["array_row"].eq(0), "split"] = "validation"
    split_table.to_parquet(split_path, index=False)
    preparation_path = processed / "preparation.json"
    preparation = _read_json(preparation_path)
    preparation["split_counts"] = {"train": 5, "validation": 5, "test": 6}
    preparation["ffid_split_counts"] = {"train": 3, "validation": 3, "test": 3}
    _write_json(preparation_path, preparation)

    with pytest.raises(ValueError, match="disjoint whole-FFID"):
        _prepare(
            interim,
            processed,
            tmp_path / "mask",
            kind=RANDOM_TRACE_MASK_KIND,
        )


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("split_counts", "split_counts do not match"),
        ("ffid_split_counts", "ffid_split_counts do not match"),
    ],
)
def test_rejects_split_counts_inconsistent_with_partition_artifact(
    tmp_path: Path,
    field: str,
    message: str,
) -> None:
    interim, processed = _write_inputs(tmp_path)
    preparation_path = processed / "preparation.json"
    preparation = _read_json(preparation_path)
    preparation[field]["test"] -= 1
    _write_json(preparation_path, preparation)

    with pytest.raises(ValueError, match=message):
        _prepare(interim, processed, tmp_path / "mask")


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing", "exactly match"),
        ("duplicate", "duplicate array_row"),
        ("unknown", "unknown split"),
    ],
)
def test_rejects_invalid_split_tables(tmp_path: Path, case: str, message: str) -> None:
    interim, processed = _write_inputs(tmp_path)
    split_path = processed / "trace_split.parquet"
    split_table = pd.read_parquet(split_path)
    if case == "missing":
        split_table = split_table.iloc[:-1]
    elif case == "duplicate":
        split_table.loc[15, "array_row"] = 14
    else:
        split_table.loc[0, "split"] = "holdout"
    split_table.to_parquet(split_path, index=False)

    with pytest.raises(ValueError, match=message):
        _prepare(interim, processed, tmp_path / "mask")


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("dataset_id", "dataset_id mismatch"),
        ("trace_count", "trace_count"),
        ("trace_hash", "input hash"),
    ],
)
def test_rejects_inconsistent_preparation_metadata(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    interim, processed = _write_inputs(tmp_path)
    preparation_path = processed / "preparation.json"
    preparation = _read_json(preparation_path)
    if case == "dataset_id":
        preparation["dataset_id"] = "other"
    elif case == "trace_count":
        preparation["trace_count"] = 17
    else:
        preparation["input_files"]["traces.parquet"]["sha256"] = "0" * 64
    _write_json(preparation_path, preparation)

    with pytest.raises(ValueError, match=message):
        _prepare(interim, processed, tmp_path / "mask")


def test_pipeline_does_not_modify_partition_artifact(tmp_path: Path) -> None:
    interim, processed = _write_inputs(tmp_path)
    partition_paths = [processed / "trace_split.parquet", processed / "preparation.json"]
    before = {path: path.read_bytes() for path in partition_paths}

    _prepare(interim, processed, tmp_path / "mask")

    assert {path: path.read_bytes() for path in partition_paths} == before


def test_metadata_records_input_hashes_and_candidate_counts(tmp_path: Path) -> None:
    interim, processed = _write_inputs(tmp_path)

    metadata = _prepare(interim, processed, tmp_path / "mask")

    assert metadata["dataset_id"] == DATASET_ID
    assert metadata["partition"] == "test"
    assert metadata["kind"] == RANDOM_TRACE_MASK_KIND
    assert metadata["missing_fraction"] == 0.5
    assert metadata["random_seed"] == 42
    assert metadata["config_source"] == CONFIG_SOURCE
    assert metadata["candidate_trace_count"] == 6
    assert metadata["candidate_ffid_count"] == 3
    assert metadata["duplicate_physical_coordinates"] == {
        "policy": "keep_lowest_array_row",
        "removed_trace_count": 0,
    }
    assert metadata["input_files"] == {
        "interim": {
            "traces.parquet": {"sha256": file_sha256(interim / "traces.parquet")},
            "dataset.json": {"sha256": file_sha256(interim / "dataset.json")},
        },
        "processed": {
            "trace_split.parquet": {"sha256": file_sha256(processed / "trace_split.parquet")},
            "preparation.json": {"sha256": file_sha256(processed / "preparation.json")},
        },
    }
    assert "created_at" not in metadata
    assert "schema_version" not in metadata


def test_overwrite_is_forwarded_to_the_mask_store(tmp_path: Path) -> None:
    interim, processed = _write_inputs(tmp_path)
    output = tmp_path / "mask"
    _prepare(interim, processed, output)
    marker = output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        _prepare(interim, processed, output, random_seed=7)

    metadata = _prepare(interim, processed, output, random_seed=7, overwrite=True)

    assert metadata["random_seed"] == 7
    assert marker.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize(
    ("group", "file_name"),
    [
        ("interim", "traces.parquet"),
        ("interim", "dataset.json"),
        ("processed", "trace_split.parquet"),
        ("processed", "preparation.json"),
    ],
)
def test_rejects_missing_required_files(tmp_path: Path, group: str, file_name: str) -> None:
    interim, processed = _write_inputs(tmp_path)
    directory = interim if group == "interim" else processed
    (directory / file_name).unlink()

    with pytest.raises(FileNotFoundError, match=file_name):
        _prepare(interim, processed, tmp_path / "mask")


def test_rejects_an_empty_requested_partition(tmp_path: Path) -> None:
    interim, processed = _write_inputs(tmp_path)
    split_path = processed / "trace_split.parquet"
    split_table = pd.read_parquet(split_path)
    split_table.loc[split_table["split"].eq("test"), "split"] = "excluded"
    split_table.to_parquet(split_path, index=False)
    preparation_path = processed / "preparation.json"
    preparation = _read_json(preparation_path)
    preparation["split_counts"]["test"] = 0
    preparation["ffid_split_counts"]["test"] = 0
    _write_json(preparation_path, preparation)

    with pytest.raises(ValueError, match="partition 'test' is empty"):
        _prepare(interim, processed, tmp_path / "mask")


@pytest.mark.parametrize("partition", ["excluded", "holdout", ""])
def test_rejects_unsupported_partitions(tmp_path: Path, partition: str) -> None:
    interim, processed = _write_inputs(tmp_path)

    with pytest.raises(ValueError, match="partition"):
        _prepare(interim, processed, tmp_path / "mask", partition=partition)


def test_rejects_an_unknown_mask_kind(tmp_path: Path) -> None:
    interim, processed = _write_inputs(tmp_path)

    with pytest.raises(ValueError, match="kind"):
        _prepare(interim, processed, tmp_path / "mask", kind="native_missing")


def test_rejects_a_validated_but_unhandled_mask_kind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interim, processed = _write_inputs(tmp_path)
    future_kind = "contiguous_shot"
    monkeypatch.setattr(
        interpolation_mask_pipeline,
        "MASK_KINDS",
        (*interpolation_mask_pipeline.MASK_KINDS, future_kind),
    )

    with pytest.raises(AssertionError, match=f"unhandled mask kind: {future_kind}"):
        _prepare(interim, processed, tmp_path / "mask", kind=future_kind)


@pytest.mark.parametrize("config_source", ["", " ", "/tmp/config.yaml", "../config.yaml", "a\\b"])
def test_rejects_non_portable_config_sources(tmp_path: Path, config_source: str) -> None:
    interim, processed = _write_inputs(tmp_path)

    with pytest.raises(ValueError, match="config_source"):
        _prepare(interim, processed, tmp_path / "mask", config_source=config_source)
