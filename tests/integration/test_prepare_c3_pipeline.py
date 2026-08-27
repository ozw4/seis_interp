from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seis_interp.cli import main
from seis_interp.data import segy_reader as segy_reader_module
from seis_interp.pipelines import prepare_c3 as prepare_c3_module
from seis_interp.pipelines.prepare_c3 import prepare_c3_complete_shot, prepare_c3_survey
from tests.fixtures.tiny_segy import TinySegyFile, TinyTraceHeader, write_tiny_segy

COMPLETE_FFID = 20
COMPLETE_TRACE_COUNT = 4
INCOMPLETE_FFID = 10


@pytest.fixture
def tiny_segy(tmp_path: Path) -> TinySegyFile:
    return write_tiny_segy(tmp_path / "tiny.sgy")


def _survey_headers(ffids: list[int]) -> tuple[TinyTraceHeader, ...]:
    return tuple(
        TinyTraceHeader(
            ffid=ffid,
            coordinate_scalar=1,
            source_x=1000 + index * 10,
            source_y=2000,
            receiver_x=1100 + index * 20,
            receiver_y=2100,
        )
        for index, ffid in enumerate(ffids)
    )


def _write_survey_sources(
    tmp_path: Path,
    *,
    second_ffids: list[int] | None = None,
    second_sample_count: int = 8,
    second_sample_interval_s: float = 0.004,
) -> tuple[TinySegyFile, TinySegyFile]:
    first = write_tiny_segy(
        tmp_path / "first.sgy",
        headers=_survey_headers([10, 10, 11]),
    )
    second = write_tiny_segy(
        tmp_path / "second.sgy",
        headers=_survey_headers(second_ffids or [20, 20]),
        sample_count=second_sample_count,
        sample_interval_s=second_sample_interval_s,
    )
    return first, second


def test_selects_the_first_complete_ffid(tiny_segy: TinySegyFile, tmp_path: Path) -> None:
    summary = prepare_c3_complete_shot(
        input_path=tiny_segy.path,
        output_dir=tmp_path / "out",
        expected_trace_count=COMPLETE_TRACE_COUNT,
    )

    assert summary["selection"]["ffid"] == COMPLETE_FFID
    assert summary["ffids"] == [COMPLETE_FFID]
    assert summary["trace_count"] == COMPLETE_TRACE_COUNT


def test_returned_summary_equals_the_written_metadata(
    tiny_segy: TinySegyFile, tmp_path: Path
) -> None:
    output_dir = tmp_path / "out"

    summary = prepare_c3_complete_shot(
        input_path=tiny_segy.path,
        output_dir=output_dir,
        expected_trace_count=COMPLETE_TRACE_COUNT,
    )

    stored = json.loads((output_dir / "dataset.json").read_text(encoding="utf-8"))

    assert stored == summary


def test_metadata_records_how_the_traces_were_selected(
    tiny_segy: TinySegyFile, tmp_path: Path
) -> None:
    output_dir = tmp_path / "out"

    prepare_c3_complete_shot(
        input_path=tiny_segy.path,
        output_dir=output_dir,
        expected_trace_count=COMPLETE_TRACE_COUNT,
    )

    stored = json.loads((output_dir / "dataset.json").read_text(encoding="utf-8"))

    assert stored["selection"] == {
        "ffid": COMPLETE_FFID,
        "expected_trace_count": COMPLETE_TRACE_COUNT,
    }


def test_explicit_ffid_is_used(tiny_segy: TinySegyFile, tmp_path: Path) -> None:
    summary = prepare_c3_complete_shot(
        input_path=tiny_segy.path,
        output_dir=tmp_path / "out",
        ffid=COMPLETE_FFID,
        expected_trace_count=COMPLETE_TRACE_COUNT,
    )

    assert summary["selection"]["ffid"] == COMPLETE_FFID


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
            "data",
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
            "data",
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
    assert summary["selection"]["ffid"] == COMPLETE_FFID
    assert summary["source_file"] == "tiny.sgy"


def test_cli_reports_errors_and_returns_non_zero(
    tiny_segy: TinySegyFile, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(
        [
            "data",
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
    assert "data prepare-c3-shot failed" in capsys.readouterr().err


def test_cli_overwrite_replaces_the_generated_files(
    tiny_segy: TinySegyFile, tmp_path: Path
) -> None:
    arguments = [
        "data",
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


def test_survey_combines_sources_in_declared_local_trace_order(tmp_path: Path) -> None:
    first, second = _write_survey_sources(tmp_path)
    output_dir = tmp_path / "survey"

    summary = prepare_c3_survey(
        [first.path, second.path],
        output_dir,
        expected_complete_trace_count=2,
        expected_ffid_ranges=[(10, 11), (20, 20)],
    )

    trace_table = pd.read_parquet(output_dir / "traces.parquet")
    amplitudes = np.load(output_dir / "amplitudes.npy", allow_pickle=False)
    assert trace_table["source_file"].tolist() == [
        "first.sgy",
        "first.sgy",
        "first.sgy",
        "second.sgy",
        "second.sgy",
    ]
    assert trace_table["trace_index"].tolist() == [0, 1, 2, 0, 1]
    assert trace_table["array_row"].tolist() == list(range(5))
    np.testing.assert_array_equal(amplitudes, np.vstack([first.amplitudes, second.amplitudes]))
    assert summary["source_files"] == [
        {
            "name": "first.sgy",
            "sha256": hashlib.sha256(first.path.read_bytes()).hexdigest(),
        },
        {
            "name": "second.sgy",
            "sha256": hashlib.sha256(second.path.read_bytes()).hexdigest(),
        },
    ]
    assert "source_file" not in summary
    assert summary["ffids"] == [10, 11, 20]
    assert summary["ffid_count"] == 3
    assert summary["complete_ffid_count"] == 2
    assert summary["incomplete_ffid_count"] == 1
    assert summary["selection"] == {
        "ffid_scope": "all",
        "include_incomplete_ffids": True,
        "expected_complete_trace_count": 2,
    }
    incomplete_rows = trace_table.loc[trace_table["ffid"] == 11]
    assert incomplete_rows["trace_count_in_ffid"].tolist() == [1]
    assert incomplete_rows["is_complete_ffid"].tolist() == [False]


def test_survey_reuses_preverified_source_sha256_without_rehashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, second = _write_survey_sources(tmp_path)
    expected = ("a" * 64, "b" * 64)
    monkeypatch.setattr(
        prepare_c3_module,
        "file_sha256",
        lambda _path: pytest.fail("preverified sources must not be hashed again"),
    )

    summary = prepare_c3_survey(
        [first.path, second.path],
        tmp_path / "survey",
        expected_complete_trace_count=2,
        source_sha256=expected,
    )

    assert summary["source_files"] == [
        {"name": "first.sgy", "sha256": expected[0]},
        {"name": "second.sgy", "sha256": expected[1]},
    ]


@pytest.mark.parametrize(
    ("source_sha256", "message"),
    [
        (("a" * 64,), "entries for 2 input paths"),
        (("a" * 64, "not-a-digest"), r"source_sha256\[1\]"),
    ],
)
def test_survey_rejects_invalid_preverified_source_sha256(
    tmp_path: Path,
    source_sha256: tuple[str, ...],
    message: str,
) -> None:
    first, second = _write_survey_sources(tmp_path)

    with pytest.raises(ValueError, match=message):
        prepare_c3_survey(
            [first.path, second.path],
            tmp_path / "survey",
            expected_complete_trace_count=2,
            source_sha256=source_sha256,
        )


def test_survey_reads_amplitudes_in_bounded_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = _write_survey_sources(tmp_path)
    calls: list[tuple[str, list[int]]] = []
    original_reader = prepare_c3_module.iter_trace_amplitude_chunks

    def recording_reader(
        path: Path,
        trace_indices: Sequence[int],
        *,
        chunk_size: int,
    ) -> Iterator[np.ndarray]:
        indices = [int(value) for value in trace_indices]
        start = 0
        for chunk in original_reader(path, indices, chunk_size=chunk_size):
            calls.append((Path(path).name, indices[start : start + len(chunk)]))
            start += len(chunk)
            yield chunk

    monkeypatch.setattr(prepare_c3_module, "_SURVEY_AMPLITUDE_CHUNK_ROWS", 2)
    monkeypatch.setattr(prepare_c3_module, "iter_trace_amplitude_chunks", recording_reader)

    prepare_c3_survey(
        [first.path, second.path],
        tmp_path / "survey",
        expected_complete_trace_count=2,
    )

    assert calls == [
        ("first.sgy", [0, 1]),
        ("first.sgy", [2]),
        ("second.sgy", [0, 1]),
    ]
    assert max(len(indices) for _, indices in calls) == 2


def test_survey_opens_each_source_once_for_all_amplitude_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = _write_survey_sources(tmp_path)
    scanned_tables = {
        path: prepare_c3_module.scan_segy_headers(path) for path in (first.path, second.path)
    }
    open_counts: Counter[str] = Counter()
    original_open = segy_reader_module.segyio.open

    def stored_scan(path: Path) -> pd.DataFrame:
        return scanned_tables[Path(path)].copy()

    def recording_open(*args: object, **kwargs: object):
        open_counts[Path(str(args[0])).name] += 1
        return original_open(*args, **kwargs)

    monkeypatch.setattr(prepare_c3_module, "scan_segy_headers", stored_scan)
    monkeypatch.setattr(prepare_c3_module, "_SURVEY_AMPLITUDE_CHUNK_ROWS", 2)
    monkeypatch.setattr(segy_reader_module.segyio, "open", recording_open)

    prepare_c3_survey(
        [first.path, second.path],
        tmp_path / "survey",
        expected_complete_trace_count=2,
    )

    assert open_counts == {"first.sgy": 1, "second.sgy": 1}


@pytest.mark.parametrize(
    ("second_ffids", "second_sample_count", "second_sample_interval_s", "message"),
    [
        ([11, 11], 8, 0.004, "occurs in both"),
        ([20, 20], 7, 0.004, "sample count"),
        ([20, 20], 8, 0.002, "sample interval"),
    ],
)
def test_survey_rejects_incompatible_sources(
    tmp_path: Path,
    second_ffids: list[int],
    second_sample_count: int,
    second_sample_interval_s: float,
    message: str,
) -> None:
    first, second = _write_survey_sources(
        tmp_path,
        second_ffids=second_ffids,
        second_sample_count=second_sample_count,
        second_sample_interval_s=second_sample_interval_s,
    )

    with pytest.raises(ValueError, match=message):
        prepare_c3_survey(
            [first.path, second.path],
            tmp_path / "survey",
            expected_complete_trace_count=2,
        )


def test_survey_rejects_a_manifest_ffid_range_mismatch(tmp_path: Path) -> None:
    first, second = _write_survey_sources(tmp_path)

    with pytest.raises(ValueError, match="manifest FFID range"):
        prepare_c3_survey(
            [first.path, second.path],
            tmp_path / "survey",
            expected_complete_trace_count=2,
            expected_ffid_ranges=[(10, 12), (20, 20)],
        )


def test_survey_rejects_incomplete_expected_survey_coverage(tmp_path: Path) -> None:
    first, second = _write_survey_sources(tmp_path)

    with pytest.raises(ValueError, match="FFID union"):
        prepare_c3_survey(
            [first.path, second.path],
            tmp_path / "survey",
            expected_complete_trace_count=2,
            expected_survey_ffid_range=(10, 20),
        )


def test_survey_failure_does_not_leave_dataset_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = _write_survey_sources(tmp_path)
    output_dir = tmp_path / "survey"

    def fail_reader(
        path: Path,
        trace_indices: object,
        *,
        chunk_size: int,
    ) -> Iterator[np.ndarray]:
        raise ValueError("synthetic amplitude read failure")

    monkeypatch.setattr(prepare_c3_module, "iter_trace_amplitude_chunks", fail_reader)

    with pytest.raises(ValueError, match="synthetic amplitude"):
        prepare_c3_survey(
            [first.path, second.path],
            output_dir,
            expected_complete_trace_count=2,
        )

    assert not (output_dir / "dataset.json").exists()
    assert not any(path.name.startswith(".") for path in output_dir.iterdir())


def test_survey_requires_overwrite_for_a_non_empty_output(tmp_path: Path) -> None:
    first, second = _write_survey_sources(tmp_path)
    output_dir = tmp_path / "survey"
    arguments = {
        "input_paths": [first.path, second.path],
        "output_dir": output_dir,
        "expected_complete_trace_count": 2,
    }

    prepare_c3_survey(**arguments)
    marker = output_dir / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        prepare_c3_survey(**arguments)
    summary = prepare_c3_survey(**arguments, overwrite=True)

    assert summary["trace_count"] == 5
    assert marker.read_text(encoding="utf-8") == "keep"
