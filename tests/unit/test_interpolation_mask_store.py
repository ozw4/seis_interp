from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seis_interp.data.interpolation_mask_store import (
    MASK_METADATA_FILE_NAME,
    MASK_TABLE_FILE_NAME,
    OUTPUT_FILE_NAMES,
    load_interpolation_mask,
    write_interpolation_mask,
)
from seis_interp.processing.interpolation_masks import (
    EVALUATION_TARGET_ROLE,
    OBSERVATION_ROLE_COLUMN,
    OBSERVED_ROLE,
)


def _mask_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "array_row": np.asarray([9, 2, 5], dtype=np.int64),
            OBSERVATION_ROLE_COLUMN: [
                OBSERVED_ROLE,
                EVALUATION_TARGET_ROLE,
                OBSERVED_ROLE,
            ],
        }
    )


def _metadata() -> dict[str, object]:
    return {
        "dataset_id": "synthetic",
        "partition": "test",
        "kind": "random_trace",
        "missing_fraction": 1.0 / 3.0,
        "random_seed": 42,
    }


def _write_valid_artifact(directory: Path) -> dict[str, object]:
    return write_interpolation_mask(directory, _mask_table(), _metadata())


def test_write_load_round_trip_preserves_table_order_and_metadata(tmp_path: Path) -> None:
    output = tmp_path / "mask"

    written = _write_valid_artifact(output)
    loaded_table, loaded_metadata = load_interpolation_mask(output)

    pd.testing.assert_frame_equal(loaded_table, _mask_table())
    assert loaded_metadata == written
    assert tuple(path.name for path in sorted(output.iterdir())) == tuple(sorted(OUTPUT_FILE_NAMES))


def test_returned_metadata_exactly_matches_saved_json(tmp_path: Path) -> None:
    output = tmp_path / "mask"

    written = _write_valid_artifact(output)
    metadata_text = (output / MASK_METADATA_FILE_NAME).read_text(encoding="utf-8")

    assert json.loads(metadata_text) == written
    assert metadata_text == json.dumps(written, indent=2, sort_keys=True, allow_nan=False) + "\n"


def test_writer_returns_a_detached_json_canonical_metadata_snapshot(tmp_path: Path) -> None:
    metadata = {
        **_metadata(),
        "nested": {
            "sequence": ("first", "second"),
            "integer_key": {1: "one"},
        },
    }

    written = write_interpolation_mask(tmp_path / "mask", _mask_table(), metadata)
    metadata["nested"]["sequence"] = ("changed",)  # type: ignore[index]

    saved = json.loads((tmp_path / "mask" / MASK_METADATA_FILE_NAME).read_text(encoding="utf-8"))
    assert written == saved
    assert written["nested"] == {
        "sequence": ["first", "second"],
        "integer_key": {"1": "one"},
    }


def test_writer_computes_counts_from_the_table(tmp_path: Path) -> None:
    written = _write_valid_artifact(tmp_path / "mask")

    assert written["counts"] == {
        "total": 3,
        OBSERVED_ROLE: 2,
        EVALUATION_TARGET_ROLE: 1,
    }
    assert written["files"] == {"observation_mask": MASK_TABLE_FILE_NAME}


def test_writer_does_not_modify_caller_metadata(tmp_path: Path) -> None:
    metadata = _metadata()
    original = copy.deepcopy(metadata)

    _write_valid_artifact(tmp_path / "first")
    write_interpolation_mask(tmp_path / "second", _mask_table(), metadata)

    assert metadata == original


def test_overwrite_false_rejects_a_nonempty_output_directory(tmp_path: Path) -> None:
    output = tmp_path / "mask"
    output.mkdir()
    (output / "unrelated.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        _write_valid_artifact(output)


def test_overwrite_true_replaces_owned_files_and_preserves_unrelated_files(
    tmp_path: Path,
) -> None:
    output = tmp_path / "mask"
    _write_valid_artifact(output)
    unrelated = output / "unrelated.txt"
    unrelated.write_text("keep", encoding="utf-8")
    replacement = pd.DataFrame(
        {
            "array_row": np.asarray([10, 11, 12, 13], dtype=np.int64),
            OBSERVATION_ROLE_COLUMN: [
                OBSERVED_ROLE,
                EVALUATION_TARGET_ROLE,
                EVALUATION_TARGET_ROLE,
                OBSERVED_ROLE,
            ],
        }
    )

    write_interpolation_mask(output, replacement, _metadata(), overwrite=True)
    loaded_table, loaded_metadata = load_interpolation_mask(output)

    pd.testing.assert_frame_equal(loaded_table, replacement)
    assert loaded_metadata["counts"] == {
        "total": 4,
        OBSERVED_ROLE: 2,
        EVALUATION_TARGET_ROLE: 2,
    }
    assert unrelated.read_text(encoding="utf-8") == "keep"


def test_loader_detects_modified_counts(tmp_path: Path) -> None:
    output = tmp_path / "mask"
    metadata = _write_valid_artifact(output)
    metadata["counts"]["total"] = 999  # type: ignore[index]
    (output / MASK_METADATA_FILE_NAME).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="counts"):
        load_interpolation_mask(output)


@pytest.mark.parametrize("corruption", ["unknown_role", "duplicate_row"])
def test_loader_detects_invalid_stored_table(tmp_path: Path, corruption: str) -> None:
    output = tmp_path / "mask"
    _write_valid_artifact(output)
    table = pd.read_parquet(output / MASK_TABLE_FILE_NAME)
    if corruption == "unknown_role":
        table.loc[0, OBSERVATION_ROLE_COLUMN] = "unknown"
    else:
        table.loc[0, "array_row"] = table.loc[1, "array_row"]
    table.to_parquet(output / MASK_TABLE_FILE_NAME, index=False)

    with pytest.raises(ValueError):
        load_interpolation_mask(output)


def test_writer_rejects_schema_version(tmp_path: Path) -> None:
    metadata = {**_metadata(), "schema_version": 1}

    with pytest.raises(ValueError, match="schema_version"):
        write_interpolation_mask(tmp_path / "mask", _mask_table(), metadata)


def test_loader_rejects_schema_version(tmp_path: Path) -> None:
    output = tmp_path / "mask"
    metadata = _write_valid_artifact(output)
    metadata["schema_version"] = 1
    (output / MASK_METADATA_FILE_NAME).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema_version"):
        load_interpolation_mask(output)


@pytest.mark.parametrize("missing_file", OUTPUT_FILE_NAMES)
def test_loader_rejects_missing_file(tmp_path: Path, missing_file: str) -> None:
    output = tmp_path / "mask"
    _write_valid_artifact(output)
    (output / missing_file).unlink()

    with pytest.raises(FileNotFoundError, match=missing_file):
        load_interpolation_mask(output)


@pytest.mark.parametrize("reserved_key", ["counts", "files"])
def test_writer_rejects_reserved_metadata_keys(tmp_path: Path, reserved_key: str) -> None:
    metadata = {**_metadata(), reserved_key: {}}

    with pytest.raises(ValueError, match="reserved"):
        write_interpolation_mask(tmp_path / "mask", _mask_table(), metadata)


@pytest.mark.parametrize("invalid_value", [{1, 2}, float("nan")])
def test_writer_rejects_non_json_metadata(tmp_path: Path, invalid_value: object) -> None:
    metadata = {**_metadata(), "invalid": invalid_value}

    with pytest.raises(ValueError, match="JSON serializable"):
        write_interpolation_mask(tmp_path / "mask", _mask_table(), metadata)


def test_writer_rejects_output_path_that_is_a_file(tmp_path: Path) -> None:
    output = tmp_path / "mask"
    output.write_text("not a directory", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not a directory"):
        _write_valid_artifact(output)


def test_overwrite_rejects_owned_path_that_is_a_directory(tmp_path: Path) -> None:
    output = tmp_path / "mask"
    (output / MASK_TABLE_FILE_NAME).mkdir(parents=True)

    with pytest.raises(FileExistsError, match=MASK_TABLE_FILE_NAME):
        write_interpolation_mask(output, _mask_table(), _metadata(), overwrite=True)


def test_overwrite_rejects_owned_path_that_is_a_symlink(tmp_path: Path) -> None:
    output = tmp_path / "mask"
    output.mkdir()
    unrelated = tmp_path / "unrelated.parquet"
    unrelated.write_bytes(b"do not replace")
    (output / MASK_TABLE_FILE_NAME).symlink_to(unrelated)

    with pytest.raises(FileExistsError, match=MASK_TABLE_FILE_NAME):
        write_interpolation_mask(output, _mask_table(), _metadata(), overwrite=True)

    assert unrelated.read_bytes() == b"do not replace"


def test_loader_rejects_non_object_metadata(tmp_path: Path) -> None:
    output = tmp_path / "mask"
    _write_valid_artifact(output)
    (output / MASK_METADATA_FILE_NAME).write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        load_interpolation_mask(output)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_loader_rejects_nonfinite_json_numbers(tmp_path: Path, constant: str) -> None:
    output = tmp_path / "mask"
    metadata = _write_valid_artifact(output)
    text = json.dumps({**metadata, "invalid": None}, indent=2, sort_keys=True)
    (output / MASK_METADATA_FILE_NAME).write_text(
        text.replace('"invalid": null', f'"invalid": {constant}') + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid JSON"):
        load_interpolation_mask(output)
