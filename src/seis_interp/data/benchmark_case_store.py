"""Validate, store, and load model-independent benchmark cases."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath

from seis_interp.data.interpolation_mask_store import OUTPUT_FILE_NAMES as MASK_FILE_NAMES
from seis_interp.data.prepared_partition import OUTPUT_FILE_NAMES as PREPARED_FILE_NAMES
from seis_interp.data.trace_store import OUTPUT_FILE_NAMES as INTERIM_FILE_NAMES
from seis_interp.processing.interpolation_masks import (
    EVALUATION_TARGET_ROLE,
    MASK_KINDS,
    OBSERVED_ROLE,
)
from seis_interp.processing.trace_splits import TEST_SPLIT, TRAIN_SPLIT, VALIDATION_SPLIT

BENCHMARK_CASE_FILE_NAME = "benchmark_case.json"
OUTPUT_FILE_NAMES = (BENCHMARK_CASE_FILE_NAME,)

MASK_DOMAIN = "canonical_present_traces"
EVALUATION_TARGET_AMPLITUDE_USE = "scoring_only"

_TOP_LEVEL_KEYS = frozenset(
    (
        "case_id",
        "dataset_id",
        "partition",
        "config_source",
        "role_contract",
        "mask",
        "input_files",
    )
)
_MASK_KEYS = frozenset(
    (
        "kind",
        "missing_fraction",
        "random_seed",
        "candidate_trace_count",
        "candidate_ffid_count",
        "counts",
        "duplicate_physical_coordinates",
    )
)
_COUNT_KEYS = frozenset(("total", OBSERVED_ROLE, EVALUATION_TARGET_ROLE))
_DUPLICATE_KEYS = frozenset(("policy", "removed_trace_count"))
_PARTITIONS = frozenset((TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT))
_ROLE_CONTRACT = {
    "domain": MASK_DOMAIN,
    "observed_role": OBSERVED_ROLE,
    "evaluation_target_role": EVALUATION_TARGET_ROLE,
    "evaluation_target_amplitude_use": EVALUATION_TARGET_AMPLITUDE_USE,
}
_INPUT_FILE_NAMES = {
    "interim": INTERIM_FILE_NAMES,
    "processed": PREPARED_FILE_NAMES,
    "mask": MASK_FILE_NAMES,
}
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


def validate_benchmark_case(case: Mapping[str, object]) -> None:
    """Validate the complete benchmark-case artifact contract."""
    case_object = _exact_mapping(case, _TOP_LEVEL_KEYS, "benchmark case")

    validated_case_id(case_object["case_id"])
    _trimmed_text(case_object["dataset_id"], "dataset_id")
    partition = case_object["partition"]
    if not isinstance(partition, str) or partition not in _PARTITIONS:
        raise ValueError(f"partition must be one of {sorted(_PARTITIONS)}, got {partition!r}")
    validated_config_source(case_object["config_source"])

    role_contract = case_object["role_contract"]
    if not isinstance(role_contract, Mapping) or dict(role_contract) != _ROLE_CONTRACT:
        raise ValueError(f"role_contract must be exactly {_ROLE_CONTRACT!r}")

    _validate_mask_summary(case_object["mask"])
    _validate_input_files(case_object["input_files"])


def write_benchmark_case(
    output_dir: Path,
    case: Mapping[str, object],
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Validate and write one benchmark-case JSON artifact."""
    validate_benchmark_case(case)
    case_json = _case_json(dict(case))
    stored_case = _decode_case(case_json)

    directory = Path(output_dir)
    _check_output_directory(directory, overwrite=overwrite)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / BENCHMARK_CASE_FILE_NAME).write_text(case_json, encoding="utf-8")
    return stored_case


def load_benchmark_case(directory: Path) -> dict[str, object]:
    """Load and validate one benchmark-case JSON artifact."""
    case_path = Path(directory) / BENCHMARK_CASE_FILE_NAME
    if not case_path.is_file():
        raise FileNotFoundError(f"benchmark case is missing required file: {case_path}")
    return _decode_case(case_path.read_text(encoding="utf-8"))


def validated_case_id(value: object) -> str:
    """Return a validated benchmark-case identifier."""
    case_id = _trimmed_text(value, "case_id")
    if "/" in case_id or "\\" in case_id:
        raise ValueError("case_id must not contain path separators")
    return case_id


def validated_config_source(value: object) -> str | None:
    """Return a portable repository-relative configuration source."""
    if value is None:
        return None
    config_source = _trimmed_text(value, "config_source")
    if "\\" in config_source:
        raise ValueError("config_source must use POSIX path separators")

    posix_path = PurePosixPath(config_source)
    windows_path = PureWindowsPath(config_source)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or windows_path.root
    ):
        raise ValueError("config_source must not be an absolute path")
    if not posix_path.parts:
        raise ValueError("config_source must identify a configuration file")
    if ".." in posix_path.parts or ".." in windows_path.parts:
        raise ValueError("config_source must not escape the repository")
    return posix_path.as_posix()


def _validate_mask_summary(value: object) -> None:
    summary = _exact_mapping(value, _MASK_KEYS, "mask")
    kind = summary["kind"]
    if not isinstance(kind, str) or kind not in MASK_KINDS:
        raise ValueError(f"mask.kind must be one of {list(MASK_KINDS)}")

    fraction = summary["missing_fraction"]
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, float)
        or not math.isfinite(fraction)
        or not 0.0 < fraction < 1.0
    ):
        raise ValueError("mask.missing_fraction must be finite and strictly between 0 and 1")

    _nonnegative_integer(summary["random_seed"], "mask.random_seed")
    candidate_trace_count = _positive_integer(
        summary["candidate_trace_count"], "mask.candidate_trace_count"
    )
    _positive_integer(summary["candidate_ffid_count"], "mask.candidate_ffid_count")

    counts = _exact_mapping(summary["counts"], _COUNT_KEYS, "mask.counts")
    total = _positive_integer(counts["total"], "mask.counts.total")
    observed = _positive_integer(counts[OBSERVED_ROLE], f"mask.counts.{OBSERVED_ROLE}")
    target = _positive_integer(
        counts[EVALUATION_TARGET_ROLE],
        f"mask.counts.{EVALUATION_TARGET_ROLE}",
    )
    if total != observed + target:
        raise ValueError("mask.counts.total must equal observed plus evaluation_target")
    if total != candidate_trace_count:
        raise ValueError("mask.counts.total must equal mask.candidate_trace_count")

    duplicate_summary = _exact_mapping(
        summary["duplicate_physical_coordinates"],
        _DUPLICATE_KEYS,
        "mask.duplicate_physical_coordinates",
    )
    _trimmed_text(
        duplicate_summary["policy"],
        "mask.duplicate_physical_coordinates.policy",
    )
    _nonnegative_integer(
        duplicate_summary["removed_trace_count"],
        "mask.duplicate_physical_coordinates.removed_trace_count",
    )


def _validate_input_files(value: object) -> None:
    input_files = _exact_mapping(value, frozenset(_INPUT_FILE_NAMES), "input_files")
    for group, file_names in _INPUT_FILE_NAMES.items():
        records = _exact_mapping(input_files[group], frozenset(file_names), f"input_files.{group}")
        for file_name in file_names:
            record = _exact_mapping(
                records[file_name],
                frozenset(("sha256",)),
                f"input_files.{group}.{file_name}",
            )
            sha256 = record["sha256"]
            if not isinstance(sha256, str) or _SHA256_PATTERN.fullmatch(sha256) is None:
                raise ValueError(
                    f"input_files.{group}.{file_name}.sha256 must be 64-character "
                    "lowercase hexadecimal"
                )


def _exact_mapping(
    value: object,
    expected_keys: frozenset[str],
    description: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be an object")
    actual_keys = set(value)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys, key=repr)
        raise ValueError(
            f"{description} must contain exactly {sorted(expected_keys)}; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return value


def _trimmed_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a trimmed non-empty string")
    return value


def _positive_integer(value: object, name: str) -> int:
    result = _nonnegative_integer(value, name)
    if result == 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _case_json(case: Mapping[str, object]) -> str:
    try:
        return json.dumps(case, indent=2, sort_keys=True, allow_nan=False) + "\n"
    except (TypeError, ValueError) as error:
        raise ValueError(f"benchmark case is not JSON serializable: {error}") from error


def _decode_case(text: str) -> dict[str, object]:
    try:
        payload = json.loads(text, parse_constant=_reject_nonfinite_json_constant)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{BENCHMARK_CASE_FILE_NAME} contains invalid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{BENCHMARK_CASE_FILE_NAME} must contain a JSON object")
    validate_benchmark_case(payload)
    return payload


def _reject_nonfinite_json_constant(value: str) -> object:
    raise ValueError(f"non-finite numeric value {value!r}")


def _check_output_directory(directory: Path, *, overwrite: bool) -> None:
    if directory.exists() and not directory.is_dir():
        raise FileExistsError(f"output path is not a directory: {directory}")
    if directory.exists() and not overwrite and any(directory.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {directory}; pass overwrite=True to replace "
            f"{BENCHMARK_CASE_FILE_NAME}"
        )
    case_path = directory / BENCHMARK_CASE_FILE_NAME
    if (
        overwrite
        and directory.exists()
        and (case_path.is_symlink() or (case_path.exists() and not case_path.is_file()))
    ):
        raise FileExistsError(f"generated output path is not a file: {case_path}")
