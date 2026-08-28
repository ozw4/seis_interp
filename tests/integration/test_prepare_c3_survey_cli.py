from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from seis_interp.cli import main
from seis_interp.data.seg_c3_na import (
    DatasetManifest,
    FileSpec,
    VerificationResult,
)

FILE_SPECS = (
    FileSpec("part_a.sgy", "https://example.test/a", None, None, 2, 3),
    FileSpec("part_b.sgy", "https://example.test/b", None, None, 4, 5),
)
SOURCE_SHA256 = ("a" * 64, "b" * 64)
SUMMARY: dict[str, object] = {
    "source_file_count": 2,
    "ffids": [2, 3, 4, 5],
    "ffid_count": 4,
    "complete_ffid_count": 3,
    "incomplete_ffid_count": 1,
    "trace_count": 20,
    "sample_count": 8,
}


def _arguments(tmp_path: Path) -> list[str]:
    return [
        "data",
        "prepare-c3-survey",
        "--manifest",
        str(tmp_path / "manifest.yaml"),
        "--data-root",
        str(tmp_path / "data"),
        "--output",
        str(tmp_path / "out"),
    ]


def _source_paths(tmp_path: Path) -> tuple[Path, ...]:
    source_directory = tmp_path / "data" / "external" / "seg_c3_na"
    return tuple(source_directory / spec.name for spec in FILE_SPECS)


def _verification_results(tmp_path: Path) -> tuple[VerificationResult, ...]:
    return tuple(
        VerificationResult(
            spec.name,
            True,
            "ok",
            "verified",
            path=path,
            sha256=sha256,
        )
        for spec, path, sha256 in zip(
            FILE_SPECS,
            _source_paths(tmp_path),
            SOURCE_SHA256,
            strict=True,
        )
    )


def _mock_manifest_and_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = DatasetManifest(tmp_path / "manifest.yaml", FILE_SPECS)
    monkeypatch.setattr("seis_interp.cli.load_manifest", lambda _path: manifest)
    monkeypatch.setattr(
        "seis_interp.cli.verify_seg_c3_na",
        lambda _manifest, _root: _verification_results(tmp_path),
    )


def test_prepare_c3_survey_cli_verifies_then_preserves_manifest_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_manifest_and_verification(tmp_path, monkeypatch)
    received: dict[str, Any] = {}

    def fake_prepare_c3_survey(**kwargs: Any) -> dict[str, object]:
        received.update(kwargs)
        return SUMMARY

    monkeypatch.setattr(
        "seis_interp.pipelines.prepare_c3.prepare_c3_survey",
        fake_prepare_c3_survey,
    )

    assert main([*_arguments(tmp_path), "--overwrite"]) == 0
    assert received == {
        "input_paths": [
            tmp_path / "data" / "external" / "seg_c3_na" / "part_a.sgy",
            tmp_path / "data" / "external" / "seg_c3_na" / "part_b.sgy",
        ],
        "output_dir": tmp_path / "out",
        "dataset_id": "seg_c3_na",
        "expected_complete_trace_count": 544,
        "expected_ffid_ranges": [(2, 3), (4, 5)],
        "expected_survey_ffid_range": (2, 4782),
        "source_sha256": SOURCE_SHA256,
        "overwrite": True,
    }


def test_prepare_c3_survey_cli_stops_before_pipeline_on_verification_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = DatasetManifest(tmp_path / "manifest.yaml", FILE_SPECS)
    monkeypatch.setattr("seis_interp.cli.load_manifest", lambda _path: manifest)
    monkeypatch.setattr(
        "seis_interp.cli.verify_seg_c3_na",
        lambda _manifest, _root: (
            VerificationResult("part_a.sgy", False, "checksum_mismatch", "bad checksum"),
        ),
    )
    monkeypatch.setattr(
        "seis_interp.pipelines.prepare_c3.prepare_c3_survey",
        lambda **_kwargs: pytest.fail("pipeline must not run after failed verification"),
    )

    assert main(_arguments(tmp_path)) == 1
    assert "source verification failed" in capsys.readouterr().err


def test_prepare_c3_survey_cli_requires_verification_for_every_manifest_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = DatasetManifest(tmp_path / "manifest.yaml", FILE_SPECS)
    monkeypatch.setattr("seis_interp.cli.load_manifest", lambda _path: manifest)
    monkeypatch.setattr(
        "seis_interp.cli.verify_seg_c3_na",
        lambda _manifest, _root: (VerificationResult("part_a.sgy", True, "ok", "verified"),),
    )
    monkeypatch.setattr(
        "seis_interp.pipelines.prepare_c3.prepare_c3_survey",
        lambda **_kwargs: pytest.fail("pipeline must not run after incomplete verification"),
    )

    assert main(_arguments(tmp_path)) == 1
    assert "must contain every manifest file in order" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("order", "do not match manifest order"),
        ("path", "refers to"),
    ],
)
def test_prepare_c3_survey_cli_rejects_misaligned_verification_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    case: str,
    message: str,
) -> None:
    manifest = DatasetManifest(tmp_path / "manifest.yaml", FILE_SPECS)
    results = list(_verification_results(tmp_path))
    if case == "order":
        results.reverse()
    else:
        results[0] = VerificationResult(
            results[0].name,
            True,
            "ok",
            "verified",
            path=tmp_path / "wrong" / results[0].name,
            sha256=results[0].sha256,
        )
    monkeypatch.setattr("seis_interp.cli.load_manifest", lambda _path: manifest)
    monkeypatch.setattr(
        "seis_interp.cli.verify_seg_c3_na",
        lambda _manifest, _root: tuple(results),
    )
    monkeypatch.setattr(
        "seis_interp.pipelines.prepare_c3.prepare_c3_survey",
        lambda **_kwargs: pytest.fail("pipeline must not run after misaligned verification"),
    )

    assert main(_arguments(tmp_path)) == 1
    assert message in capsys.readouterr().err


def test_prepare_c3_survey_cli_prints_json_and_human_summaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _mock_manifest_and_verification(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "seis_interp.pipelines.prepare_c3.prepare_c3_survey",
        lambda **_kwargs: SUMMARY,
    )

    assert main([*_arguments(tmp_path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == SUMMARY
    assert main(_arguments(tmp_path)) == 0
    output = capsys.readouterr().out
    assert "Source files: 2" in output
    assert "FFID range: 2-5" in output
    assert "Incomplete FFIDs: 1" in output
