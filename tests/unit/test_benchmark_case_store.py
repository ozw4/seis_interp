from __future__ import annotations

import copy
import json
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from seis_interp.data.benchmark_case_store import (
    BENCHMARK_CASE_FILE_NAME,
    EVALUATION_TARGET_AMPLITUDE_USE,
    MASK_DOMAIN,
    OUTPUT_FILE_NAMES,
    load_benchmark_case,
    validate_benchmark_case,
    validated_case_id,
    validated_config_source,
    write_benchmark_case,
)
from seis_interp.data.interpolation_mask_store import OUTPUT_FILE_NAMES as MASK_FILE_NAMES
from seis_interp.data.prepared_partition import OUTPUT_FILE_NAMES as PREPARED_FILE_NAMES
from seis_interp.data.trace_store import OUTPUT_FILE_NAMES as INTERIM_FILE_NAMES
from seis_interp.processing.interpolation_masks import (
    EVALUATION_TARGET_ROLE,
    OBSERVED_ROLE,
    RANDOM_TRACE_MASK_KIND,
    RANDOM_WHOLE_FFID_MASK_KIND,
)


def _case() -> dict[str, object]:
    return {
        "case_id": "synthetic_test_seed42",
        "dataset_id": "synthetic",
        "partition": "test",
        "config_source": "studies/synthetic/config.yaml",
        "role_contract": {
            "domain": MASK_DOMAIN,
            "observed_role": OBSERVED_ROLE,
            "evaluation_target_role": EVALUATION_TARGET_ROLE,
            "evaluation_target_amplitude_use": EVALUATION_TARGET_AMPLITUDE_USE,
        },
        "mask": {
            "kind": RANDOM_TRACE_MASK_KIND,
            "missing_fraction": 0.8,
            "random_seed": 42,
            "candidate_trace_count": 10,
            "candidate_ffid_count": 2,
            "counts": {
                "total": 10,
                OBSERVED_ROLE: 2,
                EVALUATION_TARGET_ROLE: 8,
            },
            "duplicate_physical_coordinates": {
                "policy": "keep_lowest_array_row",
                "removed_trace_count": 0,
            },
        },
        "input_files": {
            "interim": {file_name: {"sha256": "a" * 64} for file_name in INTERIM_FILE_NAMES},
            "processed": {file_name: {"sha256": "b" * 64} for file_name in PREPARED_FILE_NAMES},
            "mask": {file_name: {"sha256": "c" * 64} for file_name in MASK_FILE_NAMES},
        },
    }


def _replace(case: dict[str, object], path: tuple[str, ...], value: object) -> None:
    target = case
    for key in path[:-1]:
        target = target[key]  # type: ignore[assignment]
    target[path[-1]] = value


def test_write_load_round_trip_returns_saved_detached_snapshot(tmp_path: Path) -> None:
    case = _case()
    original = copy.deepcopy(case)

    written = write_benchmark_case(tmp_path / "case", case)
    assert case == original
    case["case_id"] = "changed_after_write"
    _replace(case, ("mask", "counts", EVALUATION_TARGET_ROLE), 7)
    loaded = load_benchmark_case(tmp_path / "case")
    saved_text = (tmp_path / "case" / BENCHMARK_CASE_FILE_NAME).read_text(encoding="utf-8")

    assert original == written == loaded == json.loads(saved_text)
    assert saved_text == json.dumps(written, indent=2, sort_keys=True, allow_nan=False) + "\n"
    assert OUTPUT_FILE_NAMES == (BENCHMARK_CASE_FILE_NAME,)


def test_writer_accepts_a_read_only_mapping(tmp_path: Path) -> None:
    case = _case()

    written = write_benchmark_case(tmp_path / "case", MappingProxyType(case))

    assert written == case


@pytest.mark.parametrize("change", ["missing", "unknown", "schema_version"])
def test_rejects_inexact_top_level_keys(change: str) -> None:
    case = _case()
    if change == "missing":
        del case["dataset_id"]
    else:
        case[change] = 1

    with pytest.raises(ValueError, match="benchmark case must contain exactly"):
        validate_benchmark_case(case)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("case_id",), "", "case_id"),
        (("case_id",), "a/b", "path separators"),
        (("case_id",), "a\\b", "path separators"),
        (("dataset_id",), " synthetic", "dataset_id"),
        (("partition",), "excluded", "partition"),
        (("config_source",), "", "config_source"),
        (("mask", "kind"), "periodic_shot", "mask.kind"),
        (("mask", "missing_fraction"), 0.0, "missing_fraction"),
        (("mask", "missing_fraction"), 1.0, "missing_fraction"),
        (("mask", "missing_fraction"), float("inf"), "missing_fraction"),
        (("mask", "missing_fraction"), float("nan"), "missing_fraction"),
        (("mask", "missing_fraction"), Fraction(1, 2), "missing_fraction"),
        (("mask", "random_seed"), -1, "random_seed"),
        (("mask", "random_seed"), True, "random_seed"),
        (("mask", "random_seed"), np.int64(42), "random_seed"),
        (("mask", "candidate_trace_count"), 0, "candidate_trace_count"),
        (("mask", "candidate_ffid_count"), False, "candidate_ffid_count"),
        (("mask", "counts", "observed"), 0, "counts.observed"),
        (("mask", "counts", "evaluation_target"), True, "counts.evaluation_target"),
        (
            ("mask", "duplicate_physical_coordinates", "policy"),
            " ",
            "policy",
        ),
        (
            ("mask", "duplicate_physical_coordinates", "removed_trace_count"),
            -1,
            "removed_trace_count",
        ),
    ],
)
def test_rejects_invalid_scalar_fields(
    path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    case = _case()
    _replace(case, path, value)

    with pytest.raises(ValueError, match=message):
        validate_benchmark_case(case)


def test_accepts_none_config_source() -> None:
    case = _case()
    case["config_source"] = None

    validate_benchmark_case(case)


@pytest.mark.parametrize(
    "config_source",
    [
        "/tmp/config.yaml",
        "C:\\work\\config.yaml",
        "../config.yaml",
        "study\\config.yaml",
    ],
)
def test_rejects_nonportable_config_source(config_source: str) -> None:
    case = _case()
    case["config_source"] = config_source

    with pytest.raises(ValueError, match="config_source"):
        validate_benchmark_case(case)


def test_validated_config_source_returns_portable_value() -> None:
    assert validated_config_source("studies/synthetic/config.yaml") == (
        "studies/synthetic/config.yaml"
    )
    assert validated_config_source(None) is None


def test_whole_ffid_trace_counts_need_not_match_requested_fraction() -> None:
    case = _case()
    _replace(case, ("mask", "kind"), RANDOM_WHOLE_FFID_MASK_KIND)
    _replace(case, ("mask", "counts", OBSERVED_ROLE), 9)
    _replace(case, ("mask", "counts", EVALUATION_TARGET_ROLE), 1)

    validate_benchmark_case(case)


def test_validated_case_id_returns_valid_value() -> None:
    assert validated_case_id("synthetic_test_seed42") == "synthetic_test_seed42"


def test_rejects_changed_role_contract() -> None:
    case = _case()
    _replace(case, ("role_contract", "evaluation_target_amplitude_use"), "training")

    with pytest.raises(ValueError, match="role_contract"):
        validate_benchmark_case(case)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("mask", "counts", "total"), 9, "observed plus evaluation_target"),
        (("mask", "candidate_trace_count"), 11, "candidate_trace_count"),
    ],
)
def test_rejects_inconsistent_counts(
    path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    case = _case()
    _replace(case, path, value)

    with pytest.raises(ValueError, match=message):
        validate_benchmark_case(case)


@pytest.mark.parametrize("section", ["mask", "counts", "duplicate_physical_coordinates"])
def test_rejects_inexact_nested_mask_keys(section: str) -> None:
    case = _case()
    mask = case["mask"]
    assert isinstance(mask, dict)
    target = mask if section == "mask" else mask[section]
    assert isinstance(target, dict)
    target["unknown"] = 1

    with pytest.raises(ValueError, match=section):
        validate_benchmark_case(case)


@pytest.mark.parametrize("change", ["group", "file", "record"])
def test_rejects_inexact_input_file_structure(change: str) -> None:
    case = _case()
    input_files = case["input_files"]
    assert isinstance(input_files, dict)
    if change == "group":
        input_files["extra"] = {}
    elif change == "file":
        interim = input_files["interim"]
        assert isinstance(interim, dict)
        del interim[INTERIM_FILE_NAMES[0]]
    else:
        mask_files = input_files["mask"]
        assert isinstance(mask_files, dict)
        record = mask_files[MASK_FILE_NAMES[0]]
        assert isinstance(record, dict)
        record["size"] = 100

    with pytest.raises(ValueError, match="input_files"):
        validate_benchmark_case(case)


@pytest.mark.parametrize("sha256", ["A" * 64, "a" * 63, "g" * 64])
def test_rejects_invalid_sha256(sha256: str) -> None:
    case = _case()
    _replace(case, ("input_files", "interim", INTERIM_FILE_NAMES[0], "sha256"), sha256)

    with pytest.raises(ValueError, match="lowercase hexadecimal"):
        validate_benchmark_case(case)


def test_writer_rejects_nan(tmp_path: Path) -> None:
    case = _case()
    _replace(case, ("mask", "missing_fraction"), float("nan"))

    with pytest.raises(ValueError, match="missing_fraction"):
        write_benchmark_case(tmp_path / "case", case)


@pytest.mark.parametrize("text", ["[]\n", "NaN\n", "{not-json}\n"])
def test_loader_rejects_invalid_or_non_object_json(tmp_path: Path, text: str) -> None:
    output = tmp_path / "case"
    output.mkdir()
    (output / BENCHMARK_CASE_FILE_NAME).write_text(text, encoding="utf-8")

    with pytest.raises(ValueError):
        load_benchmark_case(output)


def test_loader_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=BENCHMARK_CASE_FILE_NAME):
        load_benchmark_case(tmp_path)


def test_overwrite_false_rejects_nonempty_directory(tmp_path: Path) -> None:
    output = tmp_path / "case"
    output.mkdir()
    (output / "unrelated.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        write_benchmark_case(output, _case())


def test_overwrite_true_replaces_case_and_preserves_unrelated_file(tmp_path: Path) -> None:
    output = tmp_path / "case"
    write_benchmark_case(output, _case())
    unrelated = output / "unrelated.txt"
    unrelated.write_text("keep", encoding="utf-8")
    replacement = _case()
    replacement["case_id"] = "replacement"

    written = write_benchmark_case(output, replacement, overwrite=True)

    assert load_benchmark_case(output) == written
    assert unrelated.read_text(encoding="utf-8") == "keep"


def test_writer_rejects_output_path_that_is_a_file(tmp_path: Path) -> None:
    output = tmp_path / "case"
    output.write_text("not a directory", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not a directory"):
        write_benchmark_case(output, _case())


@pytest.mark.parametrize("owned_path_type", ["directory", "symlink"])
def test_overwrite_rejects_non_regular_owned_path(
    tmp_path: Path,
    owned_path_type: str,
) -> None:
    output = tmp_path / "case"
    output.mkdir()
    case_path = output / BENCHMARK_CASE_FILE_NAME
    if owned_path_type == "directory":
        case_path.mkdir()
    else:
        unrelated = tmp_path / "unrelated.json"
        unrelated.write_text("do not replace", encoding="utf-8")
        case_path.symlink_to(unrelated)

    with pytest.raises(FileExistsError, match=BENCHMARK_CASE_FILE_NAME):
        write_benchmark_case(output, _case(), overwrite=True)
