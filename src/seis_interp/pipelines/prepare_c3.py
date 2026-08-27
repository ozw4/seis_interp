"""Prepare SEG C3 NA traces as row-oriented interim datasets."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from seis_interp.data.file_checksums import file_sha256
from seis_interp.data.segy_index import scan_segy_headers
from seis_interp.data.segy_reader import build_time_axis, read_trace_amplitudes
from seis_interp.data.trace_store import (
    AMPLITUDES_FILE_NAME,
    METADATA_FILE_NAME,
    OUTPUT_FILE_NAMES,
    TIME_FILE_NAME,
    TRACES_FILE_NAME,
    build_interim_dataset_metadata,
    validate_trace_identity,
    write_interim_trace_dataset,
)
from seis_interp.processing.ffid_selection import annotate_ffid_quality, select_ffid

# Traces in a complete SEG C3 Narrow-Azimuth shot. Shots at the sail-line ends
# have fewer. Other surveys have their own count, so this stays out of
# seis_interp.processing.ffid_selection.
C3_COMPLETE_SHOT_TRACE_COUNT = 544
C3_SURVEY_FFID_RANGE = (2, 4781)

_SURVEY_AMPLITUDE_CHUNK_ROWS = 4096


def prepare_c3_complete_shot(
    input_path: Path,
    output_dir: Path,
    ffid: int | None = None,
    expected_trace_count: int = C3_COMPLETE_SHOT_TRACE_COUNT,
    dataset_id: str = "seg_c3_na",
    overwrite: bool = False,
) -> dict[str, object]:
    """Scan one SEG-Y file, select one FFID and write the interim dataset.

    With ``ffid=None`` the numerically smallest complete FFID is selected.
    Returns exactly the metadata that was written to ``dataset.json``.
    """
    input_path = Path(input_path)

    trace_table = scan_segy_headers(input_path)
    selected_traces = select_ffid(
        trace_table,
        ffid=ffid,
        expected_trace_count=expected_trace_count,
    )

    amplitudes = read_trace_amplitudes(input_path, selected_traces["trace_index"].tolist())
    time_s = build_time_axis(
        int(selected_traces["sample_count"].iloc[0]),
        float(selected_traces["sample_interval_s"].iloc[0]),
    )

    return write_interim_trace_dataset(
        output_dir=Path(output_dir),
        trace_table=selected_traces,
        amplitudes=amplitudes,
        time_s=time_s,
        source_path=input_path,
        dataset_id=dataset_id,
        selection={
            "ffid": int(selected_traces["ffid"].iloc[0]),
            "expected_trace_count": int(expected_trace_count),
        },
        overwrite=overwrite,
    )


def prepare_c3_survey(
    input_paths: Sequence[Path],
    output_dir: Path,
    *,
    dataset_id: str = "seg_c3_na",
    expected_complete_trace_count: int = C3_COMPLETE_SHOT_TRACE_COUNT,
    expected_ffid_ranges: Sequence[tuple[int, int] | None] | None = None,
    expected_survey_ffid_range: tuple[int, int] | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Combine ordered SEG-Y sources into one chunk-written interim dataset.

    ``expected_ffid_ranges`` follows ``input_paths`` order. The official C3
    caller supplies manifest ranges and ``C3_SURVEY_FFID_RANGE``; tests and
    other controlled subsets may omit the survey-wide range check.
    """
    paths = _validated_input_paths(input_paths)
    ranges = _validated_expected_ranges(expected_ffid_ranges, len(paths))
    survey_range = (
        None
        if expected_survey_ffid_range is None
        else _validated_ffid_range(expected_survey_ffid_range, "expected_survey_ffid_range")
    )
    if (
        isinstance(expected_complete_trace_count, bool)
        or not isinstance(expected_complete_trace_count, int)
        or expected_complete_trace_count < 1
    ):
        raise ValueError(
            f"expected_complete_trace_count must be at least 1, got {expected_complete_trace_count}"
        )

    directory = Path(output_dir)
    _check_survey_output_directory(directory, overwrite=overwrite)

    source_tables = _scan_survey_sources(
        paths,
        expected_ffid_ranges=ranges,
        expected_complete_trace_count=expected_complete_trace_count,
    )
    observed_ffids = {int(ffid) for table in source_tables for ffid in table["ffid"].unique()}
    if survey_range is not None:
        _validate_survey_ffid_coverage(observed_ffids, survey_range)

    stored_table = pd.concat(source_tables, ignore_index=True)
    stored_table.insert(0, "array_row", np.arange(len(stored_table), dtype=np.int64))
    validate_trace_identity(stored_table)

    sample_count = int(stored_table["sample_count"].iloc[0])
    sample_interval_s = float(stored_table["sample_interval_s"].iloc[0])
    time_s = build_time_axis(sample_count, sample_interval_s)
    source_metadata = {
        "source_files": [{"name": path.name, "sha256": file_sha256(path)} for path in paths]
    }

    return _write_c3_survey_dataset(
        directory,
        paths=paths,
        source_tables=source_tables,
        stored_table=stored_table,
        time_s=time_s,
        dataset_id=dataset_id,
        source_metadata=source_metadata,
        expected_complete_trace_count=expected_complete_trace_count,
    )


def _scan_survey_sources(
    paths: tuple[Path, ...],
    *,
    expected_ffid_ranges: tuple[tuple[int, int] | None, ...],
    expected_complete_trace_count: int,
) -> tuple[pd.DataFrame, ...]:
    tables: list[pd.DataFrame] = []
    ffid_sources: dict[int, str] = {}
    expected_sample_count: int | None = None
    expected_sample_interval_s: float | None = None

    for path, expected_range in zip(paths, expected_ffid_ranges, strict=True):
        table = (
            scan_segy_headers(path).sort_values("trace_index", kind="stable").reset_index(drop=True)
        )
        observed_range = (int(table["ffid"].min()), int(table["ffid"].max()))
        if expected_range is not None and observed_range != expected_range:
            raise ValueError(
                f"{path.name}: manifest FFID range is {expected_range[0]}-{expected_range[1]} "
                f"but headers contain {observed_range[0]}-{observed_range[1]}"
            )

        ffids = {int(value) for value in table["ffid"].unique()}
        overlaps = sorted(ffid for ffid in ffids if ffid in ffid_sources)
        if overlaps:
            overlap = overlaps[0]
            raise ValueError(
                f"FFID {overlap} occurs in both {ffid_sources[overlap]} and {path.name}"
            )
        ffid_sources.update({ffid: path.name for ffid in ffids})

        sample_counts = table["sample_count"].unique()
        if len(sample_counts) != 1:
            raise ValueError(f"{path.name}: headers contain multiple sample counts")
        sample_count = int(sample_counts[0])
        if expected_sample_count is None:
            expected_sample_count = sample_count
        elif sample_count != expected_sample_count:
            raise ValueError(
                f"{path.name}: sample count is {sample_count}, expected {expected_sample_count}"
            )

        sample_intervals = table["sample_interval_s"].unique()
        if len(sample_intervals) != 1:
            raise ValueError(f"{path.name}: headers contain multiple sample intervals")
        sample_interval_s = float(sample_intervals[0])
        if expected_sample_interval_s is None:
            expected_sample_interval_s = sample_interval_s
        elif sample_interval_s != expected_sample_interval_s:
            raise ValueError(
                f"{path.name}: sample interval is {sample_interval_s}, "
                f"expected {expected_sample_interval_s}"
            )

        tables.append(
            annotate_ffid_quality(
                table,
                expected_trace_count=expected_complete_trace_count,
            )
        )

    return tuple(tables)


def _write_c3_survey_dataset(
    directory: Path,
    *,
    paths: tuple[Path, ...],
    source_tables: tuple[pd.DataFrame, ...],
    stored_table: pd.DataFrame,
    time_s: np.ndarray,
    dataset_id: str,
    source_metadata: dict[str, object],
    expected_complete_trace_count: int,
) -> dict[str, object]:
    directory.mkdir(parents=True, exist_ok=True)
    temporary_paths: dict[str, Path] = {}
    amplitude_memmap: np.memmap | None = None
    try:
        for file_name, prefix, suffix in (
            (TRACES_FILE_NAME, "traces", ".parquet"),
            (AMPLITUDES_FILE_NAME, "amplitudes", ".npy"),
            (TIME_FILE_NAME, "time", ".npy"),
            (METADATA_FILE_NAME, "dataset", ".json"),
        ):
            temporary_paths[file_name] = _temporary_output_path(directory, prefix, suffix)
        sample_count = len(time_s)
        amplitude_memmap = np.lib.format.open_memmap(
            temporary_paths[AMPLITUDES_FILE_NAME],
            mode="w+",
            dtype=np.float32,
            shape=(len(stored_table), sample_count),
        )
        next_array_row = 0
        for path, table in zip(paths, source_tables, strict=True):
            trace_indices = table["trace_index"].to_numpy(dtype=np.int64)
            for start in range(0, len(trace_indices), _SURVEY_AMPLITUDE_CHUNK_ROWS):
                stop = min(start + _SURVEY_AMPLITUDE_CHUNK_ROWS, len(trace_indices))
                chunk = read_trace_amplitudes(path, trace_indices[start:stop])
                _validate_survey_amplitude_chunk(
                    chunk,
                    source_name=path.name,
                    expected_rows=stop - start,
                    expected_sample_count=sample_count,
                )
                amplitude_memmap[next_array_row : next_array_row + len(chunk)] = chunk
                next_array_row += len(chunk)
        if next_array_row != len(stored_table):
            raise RuntimeError(
                f"wrote {next_array_row} amplitude rows for {len(stored_table)} trace rows"
            )
        amplitude_memmap.flush()

        metadata = build_interim_dataset_metadata(
            stored_table,
            amplitude_memmap,
            time_s,
            dataset_id=dataset_id,
            source_metadata=source_metadata,
            selection={
                "ffid_scope": "all",
                "include_incomplete_ffids": True,
                "expected_complete_trace_count": int(expected_complete_trace_count),
            },
        )
        ffid_quality = stored_table.groupby("ffid", sort=True)["is_complete_ffid"].first()
        complete_ffid_count = int(ffid_quality.sum())
        metadata.update(
            {
                "source_file_count": len(paths),
                "ffid_count": int(len(ffid_quality)),
                "complete_ffid_count": complete_ffid_count,
                "incomplete_ffid_count": int(len(ffid_quality) - complete_ffid_count),
            }
        )
        amplitude_memmap = None

        stored_table.to_parquet(temporary_paths[TRACES_FILE_NAME], index=False)
        with temporary_paths[TIME_FILE_NAME].open("wb") as stream:
            np.save(stream, time_s)
        temporary_paths[METADATA_FILE_NAME].write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        _commit_survey_outputs(directory, temporary_paths)
        return metadata
    finally:
        amplitude_memmap = None
        for temporary_path in temporary_paths.values():
            temporary_path.unlink(missing_ok=True)


def _validate_survey_amplitude_chunk(
    chunk: np.ndarray,
    *,
    source_name: str,
    expected_rows: int,
    expected_sample_count: int,
) -> None:
    expected_shape = (expected_rows, expected_sample_count)
    if chunk.shape != expected_shape:
        raise ValueError(
            f"{source_name}: amplitude chunk has shape {chunk.shape}, expected {expected_shape}"
        )
    if chunk.dtype != np.float32:
        raise ValueError(f"{source_name}: amplitude chunk must be float32, got {chunk.dtype}")
    if not np.all(np.isfinite(chunk)):
        raise ValueError(f"{source_name}: amplitude chunk contains non-finite values")


def _commit_survey_outputs(directory: Path, temporary_paths: dict[str, Path]) -> None:
    (directory / METADATA_FILE_NAME).unlink(missing_ok=True)
    for file_name in (TRACES_FILE_NAME, AMPLITUDES_FILE_NAME, TIME_FILE_NAME):
        temporary_paths[file_name].replace(directory / file_name)
    temporary_paths[METADATA_FILE_NAME].replace(directory / METADATA_FILE_NAME)


def _temporary_output_path(directory: Path, prefix: str, suffix: str) -> Path:
    with tempfile.NamedTemporaryFile(
        dir=directory,
        prefix=f".{prefix}-",
        suffix=suffix,
        delete=False,
    ) as temporary_file:
        return Path(temporary_file.name)


def _validated_input_paths(input_paths: Sequence[Path]) -> tuple[Path, ...]:
    if isinstance(input_paths, (str, bytes)):
        raise ValueError("input_paths must be a non-empty sequence of paths")
    paths = tuple(Path(path) for path in input_paths)
    if not paths:
        raise ValueError("input_paths must not be empty")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"SEG-Y source files not found: {missing}")
    names = [path.name for path in paths]
    if len(names) != len(set(names)):
        raise ValueError("input source files must have unique basenames")
    return paths


def _validated_expected_ranges(
    expected_ranges: Sequence[tuple[int, int] | None] | None,
    source_count: int,
) -> tuple[tuple[int, int] | None, ...]:
    if expected_ranges is None:
        return (None,) * source_count
    if isinstance(expected_ranges, (str, bytes)):
        raise ValueError("expected_ffid_ranges must follow input_paths order")
    ranges = tuple(expected_ranges)
    if len(ranges) != source_count:
        raise ValueError(
            f"expected_ffid_ranges has {len(ranges)} entries for {source_count} input files"
        )
    return tuple(
        None if value is None else _validated_ffid_range(value, f"expected_ffid_ranges[{index}]")
        for index, value in enumerate(ranges)
    )


def _validated_ffid_range(value: object, name: str) -> tuple[int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{name} must contain an inclusive minimum and maximum FFID")
    minimum, maximum = value
    if (
        isinstance(minimum, bool)
        or not isinstance(minimum, int)
        or isinstance(maximum, bool)
        or not isinstance(maximum, int)
        or minimum < 0
        or maximum < minimum
    ):
        raise ValueError(f"{name} must contain increasing non-negative integer FFIDs")
    return int(minimum), int(maximum)


def _validate_survey_ffid_coverage(
    observed_ffids: set[int],
    expected_range: tuple[int, int],
) -> None:
    expected_ffids = set(range(expected_range[0], expected_range[1] + 1))
    missing = sorted(expected_ffids - observed_ffids)
    unexpected = sorted(observed_ffids - expected_ffids)
    if missing or unexpected:
        raise ValueError(
            "survey FFID union does not match expected inclusive range "
            f"{expected_range[0]}-{expected_range[1]}: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )


def _check_survey_output_directory(directory: Path, *, overwrite: bool) -> None:
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
            if (directory / file_name).exists() and not (directory / file_name).is_file()
        ]
        if invalid_targets:
            raise FileExistsError(
                f"generated output paths are not files in {directory}: {invalid_targets}"
            )
