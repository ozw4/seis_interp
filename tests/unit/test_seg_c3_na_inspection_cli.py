from __future__ import annotations

import json
from pathlib import Path

from seis_interp.cli import main
from seis_interp.commands import data as data_commands


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

    monkeypatch.setattr(data_commands, "inspect_seg_c3_na", fake_inspect)
    monkeypatch.setattr(
        data_commands, "format_seg_c3_na_inspection", lambda value: "inspection report"
    )
    monkeypatch.setattr(data_commands, "seg_c3_na_inspection_ok", lambda value: value is report)

    exit_code = main(
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


def test_inspect_cli_json_flag_prints_sorted_indented_json(monkeypatch, capsys) -> None:
    report = object()
    monkeypatch.setattr(data_commands, "inspect_seg_c3_na", lambda *args, **kwargs: report)
    monkeypatch.setattr(
        data_commands,
        "seg_c3_na_inspection_to_dict",
        lambda value: {"b": 2, "a": 1} if value is report else {},
    )
    monkeypatch.setattr(data_commands, "seg_c3_na_inspection_ok", lambda value: value is report)

    exit_code = main(["data", "inspect", "seg_c3_na", "--json"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert output == json.dumps({"a": 1, "b": 2}, indent=2, sort_keys=True) + "\n"
