from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import seis_interp.data.benchmark_case_inputs as benchmark_inputs
import seis_interp.pipelines.prepare_benchmark_case as benchmark_pipeline
from seis_interp.data.benchmark_case_inputs import (
    collect_benchmark_input_hashes,
    verify_benchmark_case_inputs,
)
from seis_interp.data.benchmark_case_store import load_benchmark_case
from seis_interp.data.file_checksums import file_sha256
from seis_interp.data.interpolation_mask_store import (
    MASK_METADATA_FILE_NAME,
    MASK_TABLE_FILE_NAME,
    load_interpolation_mask,
)
from seis_interp.data.interpolation_mask_store import (
    OUTPUT_FILE_NAMES as MASK_FILE_NAMES,
)
from seis_interp.data.prepared_partition import (
    NORMALIZATION_FILE_NAME,
    PREPARATION_FILE_NAME,
)
from seis_interp.data.prepared_partition import (
    OUTPUT_FILE_NAMES as PREPARED_FILE_NAMES,
)
from seis_interp.data.trace_store import (
    AMPLITUDES_FILE_NAME,
    METADATA_FILE_NAME,
)
from seis_interp.data.trace_store import (
    OUTPUT_FILE_NAMES as INTERIM_FILE_NAMES,
)
from seis_interp.pipelines.prepare_benchmark_case import prepare_benchmark_case
from tests.fixtures.benchmark_case_artifacts import (
    CONFIG_SOURCE,
    DATASET_ID,
    prepare_benchmark_case_artifacts,
)

CASE_ID = "synthetic_test_random_trace_seed42"


def _prepare(
    interim_dir: Path,
    processed_dir: Path,
    mask_dir: Path,
    output_dir: Path,
    **overrides: object,
) -> dict[str, object]:
    arguments: dict[str, object] = {
        "interim_dir": interim_dir,
        "processed_dir": processed_dir,
        "mask_dir": mask_dir,
        "output_dir": output_dir,
        "case_id": CASE_ID,
        "config_source": CONFIG_SOURCE,
    }
    arguments.update(overrides)
    return prepare_benchmark_case(**arguments)  # type: ignore[arg-type]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_prepares_case_with_mask_semantics_and_all_input_hashes(tmp_path: Path) -> None:
    interim, processed, mask = prepare_benchmark_case_artifacts(tmp_path)
    output = processed / "cases" / CASE_ID
    _, mask_metadata = load_interpolation_mask(mask)

    written = _prepare(interim, processed, mask, output)
    loaded = load_benchmark_case(output)
    verify_benchmark_case_inputs(
        loaded,
        interim_dir=interim,
        processed_dir=processed,
        mask_dir=mask,
    )

    assert written == loaded
    assert loaded["case_id"] == CASE_ID
    assert loaded["dataset_id"] == DATASET_ID
    assert loaded["partition"] == mask_metadata["partition"]
    assert loaded["config_source"] == CONFIG_SOURCE
    case_mask = loaded["mask"]
    assert isinstance(case_mask, dict)
    for key in (
        "kind",
        "missing_fraction",
        "random_seed",
        "candidate_trace_count",
        "candidate_ffid_count",
        "counts",
        "duplicate_physical_coordinates",
    ):
        assert case_mask[key] == mask_metadata[key]
    assert loaded["input_files"] == collect_benchmark_input_hashes(interim, processed, mask)


def test_hashes_each_input_file_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    interim, processed, mask = prepare_benchmark_case_artifacts(tmp_path)
    calls: list[Path] = []

    def recording_hash(path: Path) -> str:
        calls.append(Path(path))
        return file_sha256(path)

    monkeypatch.setattr(benchmark_inputs, "file_sha256", recording_hash)

    _prepare(interim, processed, mask, processed / "cases" / CASE_ID)

    expected_paths = {
        *(interim / file_name for file_name in INTERIM_FILE_NAMES),
        *(processed / file_name for file_name in PREPARED_FILE_NAMES),
        *(mask / file_name for file_name in MASK_FILE_NAMES),
    }
    assert set(calls) == expected_paths
    assert set(Counter(calls).values()) == {1}


def test_does_not_modify_prepared_or_mask_artifacts(tmp_path: Path) -> None:
    interim, processed, mask = prepare_benchmark_case_artifacts(tmp_path)
    input_paths = [
        *(processed / file_name for file_name in PREPARED_FILE_NAMES),
        *(mask / file_name for file_name in MASK_FILE_NAMES),
    ]
    before = {path: path.read_bytes() for path in input_paths}

    _prepare(interim, processed, mask, processed / "cases" / CASE_ID)

    assert {path: path.read_bytes() for path in input_paths} == before


def test_rejects_dataset_id_mismatch(tmp_path: Path) -> None:
    interim, processed, mask = prepare_benchmark_case_artifacts(tmp_path)
    preparation_path = processed / PREPARATION_FILE_NAME
    preparation = _read_json(preparation_path)
    preparation["dataset_id"] = "other_dataset"
    _write_json(preparation_path, preparation)

    with pytest.raises(ValueError, match="dataset_id mismatch"):
        _prepare(interim, processed, mask, processed / "case")


def test_rejects_mask_dataset_id_mismatch(tmp_path: Path) -> None:
    interim, processed, mask = prepare_benchmark_case_artifacts(tmp_path)
    metadata_path = mask / MASK_METADATA_FILE_NAME
    metadata = _read_json(metadata_path)
    metadata["dataset_id"] = "other_dataset"
    _write_json(metadata_path, metadata)

    with pytest.raises(ValueError, match="dataset_id mismatch"):
        _prepare(interim, processed, mask, processed / "case")


@pytest.mark.parametrize("field", ["trace_count", "sample_count"])
def test_rejects_preparation_shape_count_mismatch(tmp_path: Path, field: str) -> None:
    interim, processed, mask = prepare_benchmark_case_artifacts(tmp_path)
    preparation_path = processed / PREPARATION_FILE_NAME
    preparation = _read_json(preparation_path)
    preparation[field] = int(preparation[field]) + 1
    _write_json(preparation_path, preparation)

    with pytest.raises(ValueError, match=field):
        _prepare(interim, processed, mask, processed / "case")


def test_rejects_preparation_input_hash_mismatch(tmp_path: Path) -> None:
    interim, processed, mask = prepare_benchmark_case_artifacts(tmp_path)
    preparation_path = processed / PREPARATION_FILE_NAME
    preparation = _read_json(preparation_path)
    preparation["input_files"][AMPLITUDES_FILE_NAME]["sha256"] = "0" * 64
    _write_json(preparation_path, preparation)

    with pytest.raises(ValueError, match="preparation.json input_files"):
        _prepare(interim, processed, mask, processed / "case")


def test_rejects_incorrect_preparation_file_contract(tmp_path: Path) -> None:
    interim, processed, mask = prepare_benchmark_case_artifacts(tmp_path)
    preparation_path = processed / PREPARATION_FILE_NAME
    preparation = _read_json(preparation_path)
    preparation["files"]["trace_split"] = "other.parquet"
    _write_json(preparation_path, preparation)

    with pytest.raises(ValueError, match="preparation.json files"):
        _prepare(interim, processed, mask, processed / "case")


def test_rejects_mask_bound_to_other_partition_artifact(tmp_path: Path) -> None:
    interim, processed, mask = prepare_benchmark_case_artifacts(tmp_path)
    metadata_path = mask / MASK_METADATA_FILE_NAME
    metadata = _read_json(metadata_path)
    metadata["input_files"]["processed"][PREPARATION_FILE_NAME]["sha256"] = "0" * 64
    _write_json(metadata_path, metadata)

    with pytest.raises(ValueError, match="interpolation_mask.json input_files"):
        _prepare(interim, processed, mask, processed / "case")


def test_rejects_mask_candidate_count_mismatch(tmp_path: Path) -> None:
    interim, processed, mask = prepare_benchmark_case_artifacts(tmp_path)
    metadata_path = mask / MASK_METADATA_FILE_NAME
    metadata = _read_json(metadata_path)
    metadata["candidate_trace_count"] = int(metadata["candidate_trace_count"]) + 1
    _write_json(metadata_path, metadata)

    with pytest.raises(ValueError, match="candidate_trace_count"):
        _prepare(interim, processed, mask, processed / "case")


def test_rejects_mask_array_row_outside_interim_range(tmp_path: Path) -> None:
    interim, processed, mask = prepare_benchmark_case_artifacts(tmp_path)
    mask_path = mask / MASK_TABLE_FILE_NAME
    mask_table = pd.read_parquet(mask_path)
    dataset_metadata = _read_json(interim / METADATA_FILE_NAME)
    mask_table.loc[0, "array_row"] = dataset_metadata["trace_count"]
    mask_table.to_parquet(mask_path, index=False)

    with pytest.raises(ValueError, match="outside the interim range"):
        _prepare(interim, processed, mask, processed / "case")


def test_rejects_invalid_normalization_json(tmp_path: Path) -> None:
    interim, processed, mask = prepare_benchmark_case_artifacts(tmp_path)
    (processed / NORMALIZATION_FILE_NAME).write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="normalization parameters"):
        _prepare(interim, processed, mask, processed / "case")


@pytest.mark.parametrize(
    ("group", "file_name"),
    [
        ("interim", AMPLITUDES_FILE_NAME),
        ("processed", NORMALIZATION_FILE_NAME),
        ("mask", MASK_TABLE_FILE_NAME),
    ],
)
def test_verification_rejects_changed_file(
    tmp_path: Path,
    group: str,
    file_name: str,
) -> None:
    interim, processed, mask = prepare_benchmark_case_artifacts(tmp_path)
    case = _prepare(interim, processed, mask, processed / "case")
    directories = {"interim": interim, "processed": processed, "mask": mask}
    changed_path = directories[group] / file_name
    changed_path.write_bytes(changed_path.read_bytes() + b"changed")

    with pytest.raises(ValueError, match="input_files"):
        verify_benchmark_case_inputs(
            case,
            interim_dir=interim,
            processed_dir=processed,
            mask_dir=mask,
        )


def test_verification_rejects_invalid_role_contract(tmp_path: Path) -> None:
    interim, processed, mask = prepare_benchmark_case_artifacts(tmp_path)
    case = _prepare(interim, processed, mask, processed / "case")
    role_contract = case["role_contract"]
    assert isinstance(role_contract, dict)
    role_contract["evaluation_target_amplitude_use"] = "training"

    with pytest.raises(ValueError, match="role_contract"):
        verify_benchmark_case_inputs(
            case,
            interim_dir=interim,
            processed_dir=processed,
            mask_dir=mask,
        )


def test_verification_rejects_incomplete_case_before_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interim, processed, mask = prepare_benchmark_case_artifacts(tmp_path)
    case = _prepare(interim, processed, mask, processed / "case")
    del case["mask"]

    def unexpected_hash_collection(*args: object, **kwargs: object) -> object:
        pytest.fail("input hashes were collected before validating the benchmark case")

    monkeypatch.setattr(
        benchmark_inputs,
        "collect_benchmark_input_hashes",
        unexpected_hash_collection,
    )

    with pytest.raises(ValueError, match="benchmark case"):
        verify_benchmark_case_inputs(
            case,
            interim_dir=interim,
            processed_dir=processed,
            mask_dir=mask,
        )


def test_requires_every_input_file_before_hashing(tmp_path: Path) -> None:
    interim, processed, mask = prepare_benchmark_case_artifacts(tmp_path)
    (mask / MASK_TABLE_FILE_NAME).unlink()

    with pytest.raises(FileNotFoundError, match=MASK_TABLE_FILE_NAME):
        collect_benchmark_input_hashes(interim, processed, mask)


def test_loads_amplitudes_as_memmap_without_value_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interim, processed, mask = prepare_benchmark_case_artifacts(tmp_path)
    actual_load = benchmark_pipeline.load_interim_trace_dataset
    recorded: dict[str, object] = {}

    def recording_load(directory: Path, **kwargs: object) -> object:
        recorded.update(kwargs)
        dataset = actual_load(directory, **kwargs)  # type: ignore[arg-type]
        assert isinstance(dataset.amplitudes, np.memmap)
        return dataset

    monkeypatch.setattr(benchmark_pipeline, "load_interim_trace_dataset", recording_load)

    _prepare(interim, processed, mask, processed / "case")

    assert recorded["memory_map_amplitudes"] is True
    np.testing.assert_array_equal(
        recorded["amplitude_validation_rows"],
        np.empty(0, dtype=np.int64),
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"case_id": "bad/id"}, "case_id"),
        ({"config_source": "/absolute/config.yaml"}, "absolute"),
        ({"config_source": "../config.yaml"}, "escape"),
        ({"overwrite": 1}, "overwrite"),
    ],
)
def test_rejects_invalid_arguments(
    tmp_path: Path,
    overrides: dict[str, object],
    message: str,
) -> None:
    interim, processed, mask = prepare_benchmark_case_artifacts(tmp_path)

    with pytest.raises(ValueError, match=message):
        _prepare(interim, processed, mask, processed / "case", **overrides)
