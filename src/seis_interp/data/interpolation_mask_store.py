"""Store and load model-independent interpolation-mask artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from seis_interp.processing.interpolation_masks import (
    EVALUATION_TARGET_ROLE,
    OBSERVATION_ROLE_COLUMN,
    OBSERVED_ROLE,
    validate_interpolation_mask,
)

MASK_TABLE_FILE_NAME = "observation_mask.parquet"
MASK_METADATA_FILE_NAME = "interpolation_mask.json"

OUTPUT_FILE_NAMES = (
    MASK_TABLE_FILE_NAME,
    MASK_METADATA_FILE_NAME,
)


def write_interpolation_mask(
    output_dir: Path,
    mask_table: pd.DataFrame,
    metadata: Mapping[str, object],
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Validate and write one interpolation-mask artifact."""
    validate_interpolation_mask(mask_table)
    stored_metadata = _stored_metadata(mask_table, metadata)
    metadata_json = _metadata_json(stored_metadata)
    stored_metadata = _decode_metadata(metadata_json)

    directory = Path(output_dir)
    _check_output_directory(directory, overwrite=overwrite)
    directory.mkdir(parents=True, exist_ok=True)
    mask_table.to_parquet(directory / MASK_TABLE_FILE_NAME, index=False)
    (directory / MASK_METADATA_FILE_NAME).write_text(metadata_json, encoding="utf-8")
    return stored_metadata


def load_interpolation_mask(
    directory: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load and validate one interpolation-mask artifact."""
    mask_directory = Path(directory)
    paths = {file_name: mask_directory / file_name for file_name in OUTPUT_FILE_NAMES}
    missing = [file_name for file_name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"interpolation mask is missing required files in {mask_directory}: {missing}"
        )

    metadata = _read_metadata(paths[MASK_METADATA_FILE_NAME])
    mask_table = pd.read_parquet(paths[MASK_TABLE_FILE_NAME])
    validate_interpolation_mask(mask_table)
    _validate_stored_metadata(mask_table, metadata)
    return mask_table, metadata


def _stored_metadata(
    mask_table: pd.DataFrame,
    metadata: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(metadata, Mapping):
        raise TypeError(f"metadata must be a mapping, got {type(metadata).__name__}")
    if "schema_version" in metadata:
        raise ValueError("interpolation-mask metadata must not contain schema_version")
    reserved_keys = [key for key in ("counts", "files") if key in metadata]
    if reserved_keys:
        raise ValueError(f"interpolation-mask metadata contains reserved keys: {reserved_keys}")

    stored = dict(metadata)
    stored["counts"] = _mask_counts(mask_table)
    stored["files"] = {"observation_mask": MASK_TABLE_FILE_NAME}
    return stored


def _metadata_json(metadata: Mapping[str, object]) -> str:
    try:
        return json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n"
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"interpolation-mask metadata is not JSON serializable: {error}"
        ) from error


def _read_metadata(path: Path) -> dict[str, object]:
    return _decode_metadata(path.read_text(encoding="utf-8"))


def _decode_metadata(text: str) -> dict[str, object]:
    try:
        payload = json.loads(text, parse_constant=_reject_nonfinite_json_constant)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{MASK_METADATA_FILE_NAME} contains invalid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{MASK_METADATA_FILE_NAME} must contain a JSON object")
    if "schema_version" in payload:
        raise ValueError("interpolation-mask metadata must not contain schema_version")
    return payload


def _reject_nonfinite_json_constant(value: str) -> object:
    raise ValueError(f"non-finite numeric value {value!r}")


def _validate_stored_metadata(
    mask_table: pd.DataFrame,
    metadata: Mapping[str, object],
) -> None:
    expected_files = {"observation_mask": MASK_TABLE_FILE_NAME}
    if metadata.get("files") != expected_files:
        raise ValueError(f"metadata files must be {expected_files}, got {metadata.get('files')!r}")

    expected_counts = _mask_counts(mask_table)
    counts = metadata.get("counts")
    valid_counts = (
        isinstance(counts, Mapping)
        and set(counts) == set(expected_counts)
        and all(
            not isinstance(counts[key], bool)
            and isinstance(counts[key], int)
            and counts[key] == expected
            for key, expected in expected_counts.items()
        )
    )
    if not valid_counts:
        raise ValueError(f"metadata counts are {counts!r} but the table contains {expected_counts}")


def _mask_counts(mask_table: pd.DataFrame) -> dict[str, int]:
    roles = mask_table[OBSERVATION_ROLE_COLUMN]
    return {
        "total": int(len(mask_table)),
        OBSERVED_ROLE: int(roles.eq(OBSERVED_ROLE).sum()),
        EVALUATION_TARGET_ROLE: int(roles.eq(EVALUATION_TARGET_ROLE).sum()),
    }


def _check_output_directory(directory: Path, *, overwrite: bool) -> None:
    if directory.exists() and not directory.is_dir():
        raise FileExistsError(f"output path is not a directory: {directory}")
    if directory.exists() and not overwrite and any(directory.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {directory}; pass overwrite=True to replace "
            "the generated files"
        )
    if overwrite and directory.exists():
        invalid_targets = [
            file_name
            for file_name in OUTPUT_FILE_NAMES
            if (directory / file_name).is_symlink()
            or ((directory / file_name).exists() and not (directory / file_name).is_file())
        ]
        if invalid_targets:
            raise FileExistsError(
                f"generated output paths are not files in {directory}: {invalid_targets}"
            )
