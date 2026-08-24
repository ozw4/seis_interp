from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from seis_interp.cli import main

SUMMARY: dict[str, object] = {
    "schema_version": 1,
    "dataset_id": "synthetic",
    "source_file": "synthetic.sgy",
    "source_sha256": "a" * 64,
    "input_dataset_metadata_sha256": "b" * 64,
    "trace_count": 20,
    "sample_count": 4,
    "random_seed": 42,
    "holdout_fraction": 0.2,
    "validation_fraction_of_holdout": 0.25,
    "split_counts": {"train": 16, "validation": 1, "test": 3},
    "files": {
        "trace_split": "trace_split.parquet",
        "normalization": "normalization.json",
    },
}


def _arguments(tmp_path: Path) -> list[str]:
    return [
        "data",
        "prepare-baseline",
        "--input",
        str(tmp_path / "interim"),
        "--output",
        str(tmp_path / "processed"),
        "--holdout-fraction",
        "0.20",
        "--validation-fraction-of-holdout",
        "0.25",
        "--random-seed",
        "42",
    ]


def test_cli_passes_all_arguments_to_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    received: dict[str, Any] = {}

    def fake_prepare_baseline_dataset(**kwargs: Any) -> dict[str, object]:
        received.update(kwargs)
        return SUMMARY

    monkeypatch.setattr(
        "seis_interp.pipelines.prepare_baseline.prepare_baseline_dataset",
        fake_prepare_baseline_dataset,
    )

    exit_code = main([*_arguments(tmp_path), "--overwrite"])

    assert exit_code == 0
    assert received == {
        "interim_dir": tmp_path / "interim",
        "output_dir": tmp_path / "processed",
        "holdout_fraction": 0.2,
        "validation_fraction_of_holdout": 0.25,
        "random_seed": 42,
        "overwrite": True,
    }


def test_cli_json_output_is_parsable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "seis_interp.pipelines.prepare_baseline.prepare_baseline_dataset",
        lambda **kwargs: SUMMARY,
    )

    exit_code = main([*_arguments(tmp_path), "--json"])

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == SUMMARY


def test_cli_human_output_contains_split_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "seis_interp.pipelines.prepare_baseline.prepare_baseline_dataset",
        lambda **kwargs: SUMMARY,
    )

    exit_code = main(_arguments(tmp_path))
    output = capsys.readouterr().out

    assert exit_code == 0
    assert f"Input dataset: {tmp_path / 'interim'}" in output
    assert f"Output directory: {tmp_path / 'processed'}" in output
    assert "Traces: 20" in output
    assert "Train traces: 16" in output
    assert "Validation traces: 1" in output
    assert "Test traces: 3" in output


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("dataset.json is missing"),
        FileExistsError("output is not empty"),
        ValueError("invalid split fraction"),
        json.JSONDecodeError("invalid metadata", "{", 1),
    ],
)
def test_cli_reports_input_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: Exception,
) -> None:
    def fail(**kwargs: Any) -> dict[str, object]:
        raise error

    monkeypatch.setattr(
        "seis_interp.pipelines.prepare_baseline.prepare_baseline_dataset",
        fail,
    )

    exit_code = main(_arguments(tmp_path))

    assert exit_code == 1
    assert capsys.readouterr().err == f"data prepare-baseline failed: {error}\n"


def test_cli_does_not_hide_unexpected_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(**kwargs: Any) -> dict[str, object]:
        raise RuntimeError("unexpected failure")

    monkeypatch.setattr(
        "seis_interp.pipelines.prepare_baseline.prepare_baseline_dataset",
        fail,
    )

    with pytest.raises(RuntimeError, match="unexpected failure"):
        main(_arguments(tmp_path))
