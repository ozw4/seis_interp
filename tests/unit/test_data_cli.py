from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from seis_interp.cli import build_parser, main
from seis_interp.commands import data as data_commands


def test_download_cli_forwards_paths_and_options(tmp_path: Path, monkeypatch, capsys) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    data_root = tmp_path / "data-root"
    expected_lock = data_root / "external" / "seg_c3_na" / "download.lock.yaml"
    captured: dict[str, object] = {}

    def fake_download(manifest, root, *, force, resume, timeout_s):
        captured.update(
            manifest=manifest,
            root=root,
            force=force,
            resume=resume,
            timeout_s=timeout_s,
        )
        return expected_lock

    monkeypatch.setattr(data_commands, "download_seg_c3_na", fake_download)

    exit_code = main(
        [
            "data",
            "download",
            "seg_c3_na",
            "--manifest",
            str(manifest_path),
            "--data-root",
            str(data_root),
            "--force",
            "--no-resume",
            "--timeout-s",
            "12.5",
        ]
    )

    assert exit_code == 0
    assert captured == {
        "manifest": manifest_path,
        "root": data_root,
        "force": True,
        "resume": False,
        "timeout_s": 12.5,
    }
    assert str(expected_lock) in capsys.readouterr().out


def test_download_cli_reports_failures_with_the_existing_prefix(monkeypatch, capsys) -> None:
    def failing_download(manifest, root, *, force, resume, timeout_s):
        raise ValueError("manifest is unreadable")

    monkeypatch.setattr(data_commands, "download_seg_c3_na", failing_download)

    exit_code = main(["data", "download", "seg_c3_na"])

    assert exit_code == 1
    assert "Download failed: manifest is unreadable" in capsys.readouterr().err


def test_data_parser_exposes_the_six_subcommands(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["data", "--help"])

    assert excinfo.value.code == 0
    help_text = capsys.readouterr().out
    for name in (
        "download",
        "verify",
        "inspect",
        "prepare-c3-shot",
        "prepare-c3-survey",
        "prepare-baseline",
    ):
        assert name in help_text


def test_importing_data_commands_defers_preparation_pipelines() -> None:
    probe = (
        "import sys\n"
        "import seis_interp.commands.data\n"
        "eager = [\n"
        "    name\n"
        "    for name in (\n"
        "        'seis_interp.pipelines.prepare_c3',\n"
        "        'seis_interp.pipelines.prepare_baseline',\n"
        "        'seis_interp.processing.trace_amplitude_filter',\n"
        "        'torch',\n"
        "    )\n"
        "    if name in sys.modules\n"
        "]\n"
        "assert not eager, f'eagerly imported: {eager}'\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
