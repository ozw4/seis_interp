"""End-to-end acceptance test for the Step 1 CLI path.

SEG-Y -> header scan -> first complete FFID -> amplitudes -> time axis ->
traces.parquet / amplitudes.npy / time_s.npy / dataset.json.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seis_interp.cli import main
from tests.fixtures.tiny_segy import TinySegyFile, write_tiny_segy

# FFID 10 holds three traces and comes first; FFID 20 holds four and is the
# first complete FFID for this expected count.
EXPECTED_TRACE_COUNT = 4
COMPLETE_FFID = 20
FIRST_TRACE_OF_COMPLETE_FFID = 3


@pytest.fixture
def tiny_segy(tmp_path: Path) -> TinySegyFile:
    return write_tiny_segy(tmp_path / "tiny.sgy")


@pytest.fixture
def output_dir(tiny_segy: TinySegyFile, tmp_path: Path) -> Path:
    directory = tmp_path / "interim" / "complete_shot"
    exit_code = main(
        [
            "data",
            "prepare-c3-shot",
            "--input",
            str(tiny_segy.path),
            "--output",
            str(directory),
            "--expected-traces",
            str(EXPECTED_TRACE_COUNT),
        ]
    )
    assert exit_code == 0
    return directory


def test_cli_selects_the_complete_ffid_after_an_incomplete_one(output_dir: Path) -> None:
    stored_table = pd.read_parquet(output_dir / "traces.parquet")

    assert stored_table["ffid"].unique().tolist() == [COMPLETE_FFID]
    assert len(stored_table) == EXPECTED_TRACE_COUNT
    assert stored_table["array_row"].tolist() == list(range(EXPECTED_TRACE_COUNT))


def test_amplitudes_match_the_fixture_traces(output_dir: Path, tiny_segy: TinySegyFile) -> None:
    stored_table = pd.read_parquet(output_dir / "traces.parquet")
    stored_amplitudes = np.load(output_dir / "amplitudes.npy")

    for array_row, trace_index in zip(
        stored_table["array_row"], stored_table["trace_index"], strict=True
    ):
        np.testing.assert_array_equal(
            stored_amplitudes[array_row], tiny_segy.amplitudes[trace_index]
        )
    assert stored_table["trace_index"].tolist() == [
        FIRST_TRACE_OF_COMPLETE_FFID + offset for offset in range(EXPECTED_TRACE_COUNT)
    ]


def test_time_axis_matches_the_sample_interval(output_dir: Path, tiny_segy: TinySegyFile) -> None:
    stored_time = np.load(output_dir / "time_s.npy")

    np.testing.assert_allclose(
        stored_time,
        np.arange(tiny_segy.sample_count) * tiny_segy.sample_interval_s,
    )


def test_metadata_describes_the_written_files(output_dir: Path, tiny_segy: TinySegyFile) -> None:
    metadata = json.loads((output_dir / "dataset.json").read_text(encoding="utf-8"))

    assert metadata["source_file"] == "tiny.sgy"
    assert metadata["source_sha256"] == hashlib.sha256(tiny_segy.path.read_bytes()).hexdigest()
    assert metadata["ffids"] == [COMPLETE_FFID]
    assert metadata["selection"] == {
        "ffid": COMPLETE_FFID,
        "expected_trace_count": EXPECTED_TRACE_COUNT,
    }
    assert metadata["trace_count"] == EXPECTED_TRACE_COUNT
    assert metadata["sample_count"] == tiny_segy.sample_count
    assert metadata["files"]["amplitudes.npy"] == {
        "dtype": "float32",
        "shape": [EXPECTED_TRACE_COUNT, tiny_segy.sample_count],
    }
    assert metadata["files"]["time_s.npy"] == {
        "dtype": "float64",
        "shape": [tiny_segy.sample_count],
    }


def test_metadata_contains_no_host_absolute_path(output_dir: Path, tiny_segy: TinySegyFile) -> None:
    raw_metadata = (output_dir / "dataset.json").read_text(encoding="utf-8")

    assert str(tiny_segy.path.parent) not in raw_metadata
    assert str(output_dir) not in raw_metadata


def test_cli_returns_non_zero_for_an_unknown_ffid(tiny_segy: TinySegyFile, tmp_path: Path) -> None:
    exit_code = main(
        [
            "data",
            "prepare-c3-shot",
            "--input",
            str(tiny_segy.path),
            "--output",
            str(tmp_path / "unknown"),
            "--expected-traces",
            str(EXPECTED_TRACE_COUNT),
            "--ffid",
            "999",
        ]
    )

    assert exit_code == 1
