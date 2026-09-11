from __future__ import annotations

import json
import subprocess
import sys

import pytest

from seis_interp.cli import build_parser, main


def _arguments(tmp_path):
    names = (
        "interim",
        "processed",
        "mask",
        "case",
        "volume",
        "pocs-config",
        "drr-config",
        "nersi-config",
        "ccnet5d-config",
        "gnn-config",
        "output",
    )
    arguments = ["poc", "run-all"]
    for name in names:
        arguments.extend([f"--{name}", str(tmp_path / name)])
    return arguments


def _summary(tmp_path, *, failed=False):
    methods = ("pocs", "drr", "nersi", "ccnet5d", "relational_trace_graph")
    return {
        "status": "failed" if failed else "success",
        "comparison_valid": not failed,
        "runs": [
            {
                "method": method,
                "status": "failed" if failed and method == "nersi" else "success",
                "output_directory": str(tmp_path / "output" / method),
                "error_type": "ProcessError" if failed and method == "nersi" else None,
                "error_message": "training failed" if failed and method == "nersi" else None,
            }
            for method in methods
        ],
    }


def test_run_all_requires_all_data_config_and_output_paths(tmp_path):
    arguments = _arguments(tmp_path)
    for index in range(2, len(arguments), 2):
        with pytest.raises(SystemExit) as error:
            build_parser().parse_args(arguments[:index] + arguments[index + 2 :])
        assert error.value.code == 2


@pytest.mark.parametrize("failed", [False, True])
def test_run_all_json_dispatch_and_exit_status(tmp_path, monkeypatch, capsys, failed):
    from seis_interp.pipelines import run_c3_random80_poc

    received = {}
    summary = _summary(tmp_path, failed=failed)

    def run(**kwargs):
        received.update(kwargs)
        return summary

    monkeypatch.setattr(run_c3_random80_poc, "run_c3_random80_poc", run)
    assert main([*_arguments(tmp_path), "--json"]) == int(failed)
    captured = capsys.readouterr()
    assert json.loads(captured.out) == summary
    assert captured.err == ""
    assert received == {
        **{
            f"{name}_dir": tmp_path / name
            for name in ("interim", "processed", "mask", "case", "volume", "output")
        },
        **{
            f"{name}_config": tmp_path / f"{name}-config"
            for name in ("pocs", "drr", "nersi", "ccnet5d", "gnn")
        },
    }


def test_run_all_text_reports_method_failures_and_summary_paths(tmp_path, monkeypatch, capsys):
    from seis_interp.pipelines import run_c3_random80_poc

    summary = _summary(tmp_path, failed=True)
    summary["error_type"] = "ValueError"
    summary["error_message"] = "PoC methods used different verified input locks"
    monkeypatch.setattr(run_c3_random80_poc, "run_c3_random80_poc", lambda **kwargs: summary)
    assert main(_arguments(tmp_path)) == 1
    captured = capsys.readouterr()
    assert "Comparison status: failed" in captured.out
    assert "Comparison valid: False" in captured.out
    for run in summary["runs"]:
        assert f"{run['method']}: {run['status']} ({run['output_directory']})" in captured.out
    assert "ProcessError: training failed" in captured.out
    for name in ("summary.json", "summary.csv"):
        assert str(tmp_path / "output" / name) in captured.out
    assert summary["error_message"] in captured.err


@pytest.mark.parametrize("error_type", [OSError, RuntimeError, ValueError])
def test_run_all_handler_reports_errors(tmp_path, monkeypatch, capsys, error_type):
    from seis_interp.pipelines import run_c3_random80_poc

    def fail(**kwargs):
        raise error_type("cannot create comparison")

    monkeypatch.setattr(run_c3_random80_poc, "run_c3_random80_poc", fail)
    assert main(_arguments(tmp_path)) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "poc run-all failed: cannot create comparison" in captured.err


def test_run_all_help_is_lightweight():
    probe = (
        "import sys\n"
        "from seis_interp.cli import main\n"
        "try:\n"
        "    main(['poc', 'run-all', '--help'])\n"
        "except SystemExit as error:\n"
        "    assert error.code == 0\n"
        "else:\n"
        "    raise AssertionError('help must exit')\n"
        "assert 'torch' not in sys.modules\n"
        "assert not any(name.startswith('seis_interp.pipelines.') for name in sys.modules)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe], check=False, capture_output=True, text=True, timeout=60
    )
    assert completed.returncode == 0, completed.stderr
    for option in (
        "--pocs-config",
        "--drr-config",
        "--nersi-config",
        "--ccnet5d-config",
        "--gnn-config",
    ):
        assert option in completed.stdout
    for option in ("--overwrite", "--retry", "--seed", "--device"):
        assert option not in completed.stdout
