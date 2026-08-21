from __future__ import annotations

from pathlib import Path

from seis_interp import cli


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

    monkeypatch.setattr(cli, "download_seg_c3_na", fake_download)

    exit_code = cli.main(
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
