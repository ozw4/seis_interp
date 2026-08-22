from __future__ import annotations

from pathlib import Path

from seis_interp import cli


def test_inspect_cli_forwards_paths_and_sample_count(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    data_root = tmp_path / "data-root"
    report = object()
    captured: dict[str, object] = {}

    def fake_inspect(manifest, root, *, sample_trace_count):
        captured.update(
            manifest=manifest,
            root=root,
            sample_trace_count=sample_trace_count,
        )
        return report

    monkeypatch.setattr(cli, "inspect_seg_c3_na", fake_inspect)
    monkeypatch.setattr(cli, "format_seg_c3_na_inspection", lambda value: "inspection report")
    monkeypatch.setattr(cli, "seg_c3_na_inspection_ok", lambda value: value is report)

    exit_code = cli.main(
        [
            "data",
            "inspect",
            "seg_c3_na",
            "--manifest",
            str(manifest_path),
            "--data-root",
            str(data_root),
            "--sample-traces",
            "16",
        ]
    )

    assert exit_code == 0
    assert captured == {
        "manifest": manifest_path,
        "root": data_root,
        "sample_trace_count": 16,
    }
    assert capsys.readouterr().out.strip() == "inspection report"
