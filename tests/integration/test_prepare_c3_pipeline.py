from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seis_interp.cli import main
from seis_interp.pipelines.prepare_c3 import prepare_c3_complete_shot
from tests.fixtures.tiny_segy import TinySegyFile, write_tiny_segy

COMPLETE_FFID = 20
COMPLETE_TRACE_COUNT = 4
INCOMPLETE_FFID = 10


@pytest.fixture
def tiny_segy(tmp_path: Path) -> TinySegyFile:
    return write_tiny_segy(tmp_path / "tiny.sgy")


def test_selects_the_first_complete_ffid(tiny_segy: TinySegyFile, tmp_path: Path) -> None:
    summary = prepare_c3_complete_shot(
        input_path=tiny_segy.path,
        output_dir=tmp_path / "out",
        expected_trace_count=COMPLETE_TRACE_COUNT,
    )

    assert summary["selected_ffid"] == COMPLETE_FFID
    assert summary["ffids"] == [COMPLETE_FFID]
    assert summary["trace_count"] == COMPLETE_TRACE_COUNT


def test_explicit_ffid_is_used(tiny_segy: TinySegyFile, tmp_path: Path) -> None:
    summary = prepare_c3_complete_shot(
        input_path=tiny_segy.path,
        output_dir=tmp_path / "out",
        ffid=COMPLETE_FFID,
        expected_trace_count=COMPLETE_TRACE_COUNT,
    )

    assert summary["selected_ffid"] == COMPLETE_FFID


def test_incomplete_ffid_is_rejected(tiny_segy: TinySegyFile, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="incomplete"):
        prepare_c3_complete_shot(
            input_path=tiny_segy.path,
            output_dir=tmp_path / "out",
            ffid=INCOMPLETE_FFID,
            expected_trace_count=COMPLETE_TRACE_COUNT,
        )


def test_writes_the_four_output_files(tiny_segy: TinySegyFile, tmp_path: Path) -> None:
    output_dir = tmp_path / "out"

    prepare_c3_complete_shot(
        input_path=tiny_segy.path,
        output_dir=output_dir,
        expected_trace_count=COMPLETE_TRACE_COUNT,
    )

    for file_name in ("traces.parquet", "amplitudes.npy", "time_s.npy", "dataset.json"):
        assert (output_dir / file_name).is_file()


def test_trace_table_and_amplitudes_stay_aligned(tiny_segy: TinySegyFile, tmp_path: Path) -> None:
    output_dir = tmp_path / "out"

    prepare_c3_complete_shot(
        input_path=tiny_segy.path,
        output_dir=output_dir,
        expected_trace_count=COMPLETE_TRACE_COUNT,
    )

    stored_table = pd.read_parquet(output_dir / "traces.parquet")
    stored_amplitudes = np.load(output_dir / "amplitudes.npy")

    assert len(stored_table) == stored_amplitudes.shape[0] == COMPLETE_TRACE_COUNT
    assert stored_amplitudes.shape[1] == tiny_segy.sample_count
    for array_row, trace_index in zip(
        stored_table["array_row"], stored_table["trace_index"], strict=True
    ):
        np.testing.assert_array_equal(
            stored_amplitudes[array_row], tiny_segy.amplitudes[trace_index]
        )


def test_cli_returns_zero_and_prints_a_summary(
    tiny_segy: TinySegyFile, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(
        [
            "prepare-c3-shot",
            "--input",
            str(tiny_segy.path),
            "--output",
            str(tmp_path / "out"),
            "--expected-traces",
            str(COMPLETE_TRACE_COUNT),
        ]
    )

    assert exit_code == 0
    assert f"Selected FFID: {COMPLETE_FFID}" in capsys.readouterr().out


def test_cli_json_output_is_parsable(
    tiny_segy: TinySegyFile, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(
        [
            "prepare-c3-shot",
            "--input",
            str(tiny_segy.path),
            "--output",
            str(tmp_path / "out"),
            "--expected-traces",
            str(COMPLETE_TRACE_COUNT),
            "--json",
        ]
    )

    summary = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert summary["selected_ffid"] == COMPLETE_FFID
    assert summary["source_file"] == "tiny.sgy"


def test_cli_reports_errors_and_returns_non_zero(
    tiny_segy: TinySegyFile, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(
        [
            "prepare-c3-shot",
            "--input",
            str(tiny_segy.path),
            "--output",
            str(tmp_path / "out"),
            "--expected-traces",
            str(COMPLETE_TRACE_COUNT),
            "--ffid",
            str(INCOMPLETE_FFID),
        ]
    )

    assert exit_code == 1
    assert "prepare-c3-shot failed" in capsys.readouterr().err


def test_cli_overwrite_replaces_the_generated_files(
    tiny_segy: TinySegyFile, tmp_path: Path
) -> None:
    arguments = [
        "prepare-c3-shot",
        "--input",
        str(tiny_segy.path),
        "--output",
        str(tmp_path / "out"),
        "--expected-traces",
        str(COMPLETE_TRACE_COUNT),
    ]

    assert main(arguments) == 0
    assert main(arguments) == 1
    assert main([*arguments, "--overwrite"]) == 0
