"""Store and load dense SEG C3 NA volume-index artifacts."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_complex_dtype,
    is_integer_dtype,
    is_numeric_dtype,
)

from seis_interp.data.benchmark_case_store import (
    BENCHMARK_CASE_FILE_NAME,
    validated_case_id,
    validated_config_source,
)
from seis_interp.data.file_checksums import file_sha256
from seis_interp.processing.c3_volume_index import (
    INDEX_CONTRACT,
    VOLUME_AXIS_ORDER,
    VOLUME_INDEX_COLUMNS,
    validated_index_range,
)
from seis_interp.processing.interpolation_masks import EVALUATION_TARGET_ROLE, OBSERVED_ROLE
from seis_interp.processing.trace_splits import TEST_SPLIT, TRAIN_SPLIT, VALIDATION_SPLIT

VOLUME_INDEX_FILE_NAME = "volume_index.parquet"
VOLUME_METADATA_FILE_NAME = "volume.json"

OUTPUT_FILE_NAMES = (
    VOLUME_INDEX_FILE_NAME,
    VOLUME_METADATA_FILE_NAME,
)

_SELECTION_KEYS = frozenset(VOLUME_AXIS_ORDER)
_TOP_LEVEL_KEYS = frozenset(
    (
        "volume_id",
        "dataset_id",
        "partition",
        "config_source",
        "axis_order",
        "selection",
        "shape",
        "trace_count",
        "role_counts",
        "index_contract",
        "benchmark_case",
        "files",
    )
)
_BASE_METADATA_KEYS = _TOP_LEVEL_KEYS - {"files"}
_INDEX_COLUMNS = VOLUME_INDEX_COLUMNS[:6]
_LOCAL_INDEX_COLUMNS = VOLUME_INDEX_COLUMNS[2:6]
_COORDINATE_COLUMNS = VOLUME_INDEX_COLUMNS[6:]
_PARTITIONS = frozenset((TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT))
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


def validated_volume_id(value: object) -> str:
    """Return a validated portable volume identifier."""
    volume_id = _trimmed_text(value, "volume_id")
    if "/" in volume_id or "\\" in volume_id:
        raise ValueError("volume_id must not contain path separators")
    return volume_id


def validate_c3_volume_metadata(metadata: Mapping[str, object]) -> None:
    """Validate the complete volume metadata contract."""
    volume = _exact_mapping(metadata, _TOP_LEVEL_KEYS, "volume metadata")
    validated_volume_id(volume["volume_id"])
    _trimmed_text(volume["dataset_id"], "dataset_id")
    if not isinstance(volume["partition"], str) or volume["partition"] not in _PARTITIONS:
        raise ValueError(f"partition must be one of {sorted(_PARTITIONS)}")
    validated_config_source(volume["config_source"])
    if volume["axis_order"] != list(VOLUME_AXIS_ORDER):
        raise ValueError(f"axis_order must be exactly {list(VOLUME_AXIS_ORDER)!r}")

    selection = _exact_mapping(volume["selection"], _SELECTION_KEYS, "selection")
    ranges = [
        validated_index_range(selection[axis], name=f"selection.{axis}")
        for axis in VOLUME_AXIS_ORDER
    ]
    shape = _positive_integer_list(volume["shape"], length=5, name="shape")
    expected_shape = [stop - start for start, stop in ranges]
    if shape != expected_shape:
        raise ValueError(f"shape must match selection lengths {expected_shape}, got {shape}")

    trace_count = _positive_integer(volume["trace_count"], "trace_count")
    expected_trace_count = int(np.prod(shape[1:], dtype=np.int64))
    if trace_count != expected_trace_count:
        raise ValueError(f"trace_count must equal the spatial shape product {expected_trace_count}")

    role_counts = _exact_mapping(
        volume["role_counts"],
        frozenset((OBSERVED_ROLE, EVALUATION_TARGET_ROLE)),
        "role_counts",
    )
    observed = _positive_integer(role_counts[OBSERVED_ROLE], f"role_counts.{OBSERVED_ROLE}")
    target = _positive_integer(
        role_counts[EVALUATION_TARGET_ROLE], f"role_counts.{EVALUATION_TARGET_ROLE}"
    )
    if observed + target != trace_count:
        raise ValueError("role_counts must sum to trace_count")

    if not isinstance(volume["index_contract"], Mapping) or dict(volume["index_contract"]) != dict(
        INDEX_CONTRACT
    ):
        raise ValueError(f"index_contract must be exactly {INDEX_CONTRACT!r}")

    benchmark_case = _exact_mapping(
        volume["benchmark_case"],
        frozenset(("case_id", "file", "sha256")),
        "benchmark_case",
    )
    validated_case_id(benchmark_case["case_id"])
    if benchmark_case["file"] != BENCHMARK_CASE_FILE_NAME:
        raise ValueError(f"benchmark_case.file must be {BENCHMARK_CASE_FILE_NAME!r}")
    _validated_sha256(benchmark_case["sha256"], "benchmark_case.sha256")

    files = _exact_mapping(volume["files"], frozenset((VOLUME_INDEX_FILE_NAME,)), "files")
    index_record = _exact_mapping(
        files[VOLUME_INDEX_FILE_NAME],
        frozenset(("sha256", "row_count")),
        f"files.{VOLUME_INDEX_FILE_NAME}",
    )
    _validated_sha256(index_record["sha256"], f"files.{VOLUME_INDEX_FILE_NAME}.sha256")
    if (
        _positive_integer(index_record["row_count"], f"files.{VOLUME_INDEX_FILE_NAME}.row_count")
        != trace_count
    ):
        raise ValueError(f"files.{VOLUME_INDEX_FILE_NAME}.row_count must equal trace_count")


def validate_c3_volume_index(
    index_table: pd.DataFrame,
    metadata: Mapping[str, object],
) -> None:
    """Validate a volume index against its complete metadata."""
    validate_c3_volume_metadata(metadata)
    if not isinstance(index_table, pd.DataFrame):
        raise TypeError(f"index_table must be a pandas DataFrame, got {type(index_table).__name__}")
    if index_table.columns.tolist() != list(VOLUME_INDEX_COLUMNS):
        raise ValueError(
            f"volume index columns must be exactly {list(VOLUME_INDEX_COLUMNS)!r}, "
            f"got {index_table.columns.tolist()!r}"
        )
    if index_table.empty:
        raise ValueError("volume index must not be empty")

    for column in _INDEX_COLUMNS:
        values = index_table[column]
        if values.isna().any() or is_bool_dtype(values.dtype) or not is_integer_dtype(values.dtype):
            raise ValueError(f"{column} must have a non-missing integer dtype")
        if int(values.min()) < np.iinfo(np.int64).min or int(values.max()) > np.iinfo(np.int64).max:
            raise ValueError(f"{column} values must fit in int64")
    for column in _COORDINATE_COLUMNS:
        values = index_table[column]
        if (
            not is_numeric_dtype(values.dtype)
            or is_bool_dtype(values.dtype)
            or is_complex_dtype(values.dtype)
        ):
            raise ValueError(f"{column} must contain real numeric values")
        try:
            numeric = values.to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{column} must contain real numeric values") from error
        if not np.all(np.isfinite(numeric)):
            raise ValueError(f"{column} must contain finite values")

    integers = index_table[list(_INDEX_COLUMNS)].to_numpy(dtype=np.int64)
    if np.any(integers[:, 0] < 0):
        raise ValueError("array_row values must be nonnegative")
    if len(np.unique(integers[:, 0])) != len(index_table):
        raise ValueError("volume index contains duplicate array_row values")
    if np.any(integers[:, 2:] < 0):
        raise ValueError("local spatial indices must be nonnegative")

    local = integers[:, 2:]
    order = np.lexsort(tuple(local[:, index] for index in range(3, -1, -1)))
    if not np.array_equal(order, np.arange(len(index_table))):
        raise ValueError("volume index rows must be in lexicographic spatial order")

    shape = tuple(int(value) for value in metadata["shape"][1:])  # type: ignore[index]
    if np.any(local >= np.asarray(shape)):
        raise ValueError("local spatial indices are outside metadata shape")
    flat = np.ravel_multi_index(tuple(local[:, index] for index in range(4)), shape)
    expected = np.arange(int(metadata["trace_count"]), dtype=np.int64)
    if not np.array_equal(flat, expected):
        raise ValueError("local spatial cells must cover every cell exactly once")
    if len(index_table) != int(metadata["trace_count"]):
        raise ValueError("volume index row count must equal trace_count")

    source_groups = index_table.groupby(["source_line_index", "shot_in_line_index"], sort=False)
    for column in ("ffid", "source_x_m", "source_y_m"):
        if bool((source_groups[column].nunique(dropna=False) != 1).any()):
            raise ValueError(f"{column} must be constant within each source cell")
    _require_coordinate_rank(index_table, "source_line_index", "source_x_m")
    _require_line_local_shot_ranks(index_table)
    _require_coordinate_rank(index_table, "relative_receiver_x_index", "relative_receiver_x_m")
    _require_coordinate_rank(index_table, "relative_receiver_y_index", "relative_receiver_y_m")


def write_c3_volume_index(
    output_dir: Path,
    index_table: pd.DataFrame,
    metadata: Mapping[str, object],
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Write a validated Parquet mapping and hash-bound JSON metadata."""
    if not isinstance(metadata, Mapping):
        raise TypeError(f"metadata must be a mapping, got {type(metadata).__name__}")
    base = _exact_mapping(metadata, _BASE_METADATA_KEYS, "base volume metadata")
    if not isinstance(overwrite, bool):
        raise ValueError("overwrite must be a boolean")
    canonical = _canonical_index_table(index_table)

    placeholder = dict(base)
    placeholder["files"] = {
        VOLUME_INDEX_FILE_NAME: {"sha256": "0" * 64, "row_count": len(canonical)}
    }
    validate_c3_volume_index(canonical, placeholder)

    directory = Path(output_dir)
    _check_output_directory(directory, overwrite=overwrite)
    directory.mkdir(parents=True, exist_ok=True)
    index_path = directory / VOLUME_INDEX_FILE_NAME
    canonical.to_parquet(index_path, index=False)

    stored = dict(base)
    stored["files"] = {
        VOLUME_INDEX_FILE_NAME: {
            "sha256": file_sha256(index_path),
            "row_count": int(len(canonical)),
        }
    }
    metadata_json = _metadata_json(stored)
    detached = _decode_metadata(metadata_json)
    (directory / VOLUME_METADATA_FILE_NAME).write_text(metadata_json, encoding="utf-8")
    return detached


def load_c3_volume_index(directory: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load a volume artifact and verify its table hash and contracts."""
    volume_directory = Path(directory)
    paths = {name: volume_directory / name for name in OUTPUT_FILE_NAMES}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"C3 volume index is missing required files in {volume_directory}: {missing}"
        )
    metadata = _decode_metadata(paths[VOLUME_METADATA_FILE_NAME].read_text(encoding="utf-8"))
    expected_hash = metadata["files"][VOLUME_INDEX_FILE_NAME]["sha256"]  # type: ignore[index]
    if file_sha256(paths[VOLUME_INDEX_FILE_NAME]) != expected_hash:
        raise ValueError(f"{VOLUME_INDEX_FILE_NAME} SHA-256 does not match volume metadata")
    index_table = pd.read_parquet(paths[VOLUME_INDEX_FILE_NAME])
    validate_c3_volume_index(index_table, metadata)
    return index_table, metadata


def _canonical_index_table(index_table: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(index_table, pd.DataFrame):
        raise TypeError(f"index_table must be a pandas DataFrame, got {type(index_table).__name__}")
    if index_table.columns.tolist() != list(VOLUME_INDEX_COLUMNS):
        raise ValueError(
            f"volume index columns must be exactly {list(VOLUME_INDEX_COLUMNS)!r}, "
            f"got {index_table.columns.tolist()!r}"
        )
    result = index_table.copy()
    for column in _INDEX_COLUMNS:
        values = result[column]
        if values.isna().any() or is_bool_dtype(values.dtype) or not is_integer_dtype(values.dtype):
            raise ValueError(f"{column} must have a non-missing integer dtype")
        if len(values) and (
            int(values.min()) < np.iinfo(np.int64).min or int(values.max()) > np.iinfo(np.int64).max
        ):
            raise ValueError(f"{column} values must fit in int64")
        result[column] = values.astype(np.int64)
    for column in _COORDINATE_COLUMNS:
        result[column] = result[column].astype(np.float64)
    return result


def _require_coordinate_rank(table: pd.DataFrame, index_column: str, value_column: str) -> None:
    grouped = table.groupby(index_column, sort=True)[value_column]
    if bool((grouped.nunique(dropna=False) != 1).any()):
        raise ValueError(f"{value_column} must have exactly one value per {index_column}")
    values = grouped.first().to_numpy(dtype=np.float64)
    if len(values) > 1 and not np.all(np.diff(values) > 0):
        raise ValueError(f"{value_column} must increase with {index_column}")


def _require_line_local_shot_ranks(table: pd.DataFrame) -> None:
    sources = table[["source_line_index", "shot_in_line_index", "source_y_m"]].drop_duplicates()
    if sources.duplicated(["source_line_index", "shot_in_line_index"]).any():
        raise ValueError("source_y_m must have exactly one value per line-local shot index")
    for _, line in sources.groupby("source_line_index", sort=False):
        line = line.sort_values("shot_in_line_index")
        if len(line) > 1 and not np.all(np.diff(line["source_y_m"].to_numpy()) > 0):
            raise ValueError("source_y_m must increase with shot_in_line_index within each line")


def _metadata_json(metadata: Mapping[str, object]) -> str:
    try:
        return json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n"
    except (TypeError, ValueError) as error:
        raise ValueError(f"volume metadata is not JSON serializable: {error}") from error


def _decode_metadata(text: str) -> dict[str, object]:
    try:
        payload = json.loads(text, parse_constant=_reject_nonfinite_json_constant)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{VOLUME_METADATA_FILE_NAME} contains invalid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{VOLUME_METADATA_FILE_NAME} must contain a JSON object")
    validate_c3_volume_metadata(payload)
    return payload


def _reject_nonfinite_json_constant(value: str) -> object:
    raise ValueError(f"non-finite numeric value {value!r}")


def _exact_mapping(
    value: object, expected_keys: frozenset[str], description: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be an object")
    actual = set(value)
    if actual != expected_keys:
        raise ValueError(
            f"{description} must contain exactly {sorted(expected_keys)}; "
            f"missing={sorted(expected_keys - actual)}, "
            f"unexpected={sorted(actual - expected_keys, key=repr)}"
        )
    return value


def _trimmed_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a trimmed non-empty string")
    return value


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive built-in integer")
    return value


def _positive_integer_list(value: object, *, length: int, name: str) -> list[int]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must be a list of {length} positive integers")
    return [_positive_integer(item, f"{name}[{index}]") for index, item in enumerate(value)]


def _validated_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be 64-character lowercase hexadecimal")
    return value


def _check_output_directory(directory: Path, *, overwrite: bool) -> None:
    if directory.exists() and not directory.is_dir():
        raise FileExistsError(f"output path is not a directory: {directory}")
    if directory.exists() and not overwrite and any(directory.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {directory}; pass overwrite=True to replace "
            "the generated files"
        )
    if overwrite and directory.exists():
        invalid = [
            name
            for name in OUTPUT_FILE_NAMES
            if (directory / name).is_symlink()
            or ((directory / name).exists() and not (directory / name).is_file())
        ]
        if invalid:
            raise FileExistsError(f"generated output paths are not files in {directory}: {invalid}")
