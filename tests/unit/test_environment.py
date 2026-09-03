from __future__ import annotations

import builtins
from typing import Any

import pytest

from seis_interp.cli import collect_environment, main
from seis_interp.commands import doctor as doctor_command


def _report(*, commands_available: bool, data_root_readable: bool) -> dict[str, Any]:
    command = {"available": commands_available, "path": None, "version": "1.0"}
    return {
        "python": {"version": "3.10.0", "executable": "/usr/bin/python", "platform": "linux"},
        "packages": {"numpy": "2.0.0", "PyYAML": None},
        "torch": {
            "available": False,
            "version": None,
            "cuda_available": False,
            "cuda_version": None,
            "device_count": 0,
            "devices": [],
        },
        "commands": {"codex": dict(command), "claude": dict(command), "gh": dict(command)},
        "data_root": {"path": "/data", "exists": True, "readable": data_root_readable},
    }


def test_collect_environment_has_expected_sections() -> None:
    report = collect_environment()

    assert set(report) == {"python", "packages", "torch", "commands", "data_root"}
    assert {"codex", "claude", "gh"} <= set(report["commands"])


def test_doctor_json_exits_successfully(capsys) -> None:
    exit_code = main(["doctor", "--json"])

    assert exit_code == 0
    assert '"python"' in capsys.readouterr().out


@pytest.mark.parametrize(
    ("commands_available", "data_root_readable", "expected_exit_code"),
    [(True, True, 0), (False, True, 1), (True, False, 1)],
)
def test_doctor_strict_requires_ai_clis_and_readable_data_root(
    monkeypatch: pytest.MonkeyPatch,
    commands_available: bool,
    data_root_readable: bool,
    expected_exit_code: int,
) -> None:
    monkeypatch.setattr(
        doctor_command,
        "collect_environment",
        lambda: _report(
            commands_available=commands_available,
            data_root_readable=data_root_readable,
        ),
    )

    assert main(["doctor", "--strict"]) == expected_exit_code


def test_doctor_human_readable_output_keeps_main_labels(
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        doctor_command,
        "collect_environment",
        lambda: _report(commands_available=True, data_root_readable=True),
    )

    assert main(["doctor"]) == 0
    output = capsys.readouterr().out
    assert "Python: 3.10.0 (/usr/bin/python)" in output
    assert "PyTorch: not installed" in output
    assert "Packages:" in output
    assert "  PyYAML: not installed" in output
    assert "Commands:" in output
    assert "Data root: /data | exists=True | readable=True" in output


def test_torch_environment_reports_missing_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def _no_torch(name: str, *args: object, **kwargs: object):
        if name == "torch":
            raise ImportError("No module named 'torch'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_torch)

    assert doctor_command._torch_environment() == {
        "available": False,
        "version": None,
        "cuda_available": False,
        "cuda_version": None,
        "device_count": 0,
        "devices": [],
    }
