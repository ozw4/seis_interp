from __future__ import annotations

from seis_interp.cli import collect_environment, main


def test_collect_environment_has_expected_sections() -> None:
    report = collect_environment()

    assert set(report) == {"python", "packages", "torch", "commands", "data_root"}
    assert {"codex", "claude", "gh"} <= set(report["commands"])


def test_doctor_json_exits_successfully(capsys) -> None:
    exit_code = main(["doctor", "--json"])

    assert exit_code == 0
    assert '"python"' in capsys.readouterr().out
