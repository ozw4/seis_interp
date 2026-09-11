"""Independent process orchestration and artifact-only PoC comparisons."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from seis_interp.pipelines import run_c3_random80_poc as pipeline
from tests.fixtures.c3_poc_runs import poc_input_lock, write_poc_run_artifacts

METHODS = ("pocs", "drr", "nersi", "ccnet5d", "relational_trace_graph")


def _arguments(tmp_path):
    return {
        **{
            f"{name}_dir": tmp_path / name
            for name in ("interim", "processed", "mask", "case", "volume", "output")
        },
        **{
            f"{name}_config": tmp_path / f"{name}.yaml"
            for name in ("pocs", "drr", "nersi", "ccnet5d", "gnn")
        },
    }


def _fake_processes(monkeypatch, *, failed=None, mutate=None, spawn_error=False):
    calls = []

    def run(arguments, *, stdout_path, stderr_path):
        calls.append(arguments)
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        if arguments[:2] == ["poc", "check"]:
            lock = poc_input_lock()
            checked = {
                key: lock[key]
                for key in (
                    "benchmark_id",
                    "dataset_id",
                    "case_id",
                    "volume_id",
                    "selection",
                    "shape",
                    "observed_trace_count",
                    "target_trace_count",
                )
            }
            checked["observed_global_rms"] = 1.5
            stdout_path.write_text(json.dumps(checked), encoding="utf-8")
            return
        method = arguments[1].replace("-", "_")
        directory = Path(arguments[arguments.index("--output") + 1])
        assert not directory.exists()
        if method == failed:
            if spawn_error:
                raise OSError("process could not start")
            directory.mkdir()
            (directory / "partial.txt").write_text("keep failed run", encoding="utf-8")
            stderr_path.write_text("training failed with test error", encoding="utf-8")
            raise subprocess.CalledProcessError(2, arguments)
        write_poc_run_artifacts(directory, method)
        if mutate is not None:
            mutate(method, directory)

    monkeypatch.setattr(pipeline, "run_poc_process", run)
    return calls


def _edit_json(path, mutate):
    record = json.loads(path.read_text(encoding="utf-8"))
    mutate(record)
    path.write_text(json.dumps(record), encoding="utf-8")


def test_all_methods_run_once_after_preflight_then_validate_and_write_summary(
    tmp_path, monkeypatch
):
    calls = _fake_processes(monkeypatch)
    validate = pipeline.validate_poc_run_artifacts

    def validate_after_processes(*args):
        assert len(calls) == 6
        return validate(*args)

    monkeypatch.setattr(pipeline, "validate_poc_run_artifacts", validate_after_processes)
    arguments = _arguments(tmp_path)
    summary = pipeline.run_c3_random80_poc(**arguments)
    assert summary["status"] == "success"
    assert summary["comparison_valid"] is True
    assert summary["inputs_lock"] == poc_input_lock()
    assert summary["preflight"]["status"] == "success"
    assert [call[:2] for call in calls] == [
        ["poc", "check"],
        *[["interpolate", method.replace("_", "-")] for method in METHODS],
    ]
    for call in calls:
        for name in ("interim", "processed", "mask", "case", "volume"):
            assert call[call.index(f"--{name}") + 1] == str(arguments[f"{name}_dir"])
    for call, config_name in zip(
        calls[1:], ("pocs", "drr", "nersi", "ccnet5d", "gnn"), strict=True
    ):
        assert call[call.index("--config") + 1] == str(arguments[f"{config_name}_config"])
    assert [row["method"] for row in summary["runs"]] == list(METHODS)
    assert all(row["status"] == "success" for row in summary["runs"])
    assert all(row["coverage"]["complete"] for row in summary["runs"])
    output = arguments["output_dir"]
    assert json.loads((output / "summary.json").read_text()) == summary
    with (output / "summary.csv").open(newline="", encoding="utf-8") as stream:
        csv_rows = list(csv.DictReader(stream))
    assert [row["method"] for row in csv_rows] == list(METHODS)
    assert all(row["prediction_sha256"] for row in csv_rows)
    assert csv_rows[0]["checkpoint_path"] == ""
    assert csv_rows[0]["prediction_seconds"] == ""


@pytest.mark.parametrize("spawn_error", [False, True])
def test_failed_method_is_retained_and_remaining_methods_still_run(
    tmp_path, monkeypatch, spawn_error
):
    calls = _fake_processes(monkeypatch, failed="nersi", spawn_error=spawn_error)
    summary = pipeline.run_c3_random80_poc(**_arguments(tmp_path))
    assert len(calls) == 6
    assert summary["status"] == "failed"
    assert summary["comparison_valid"] is False
    assert summary["inputs_lock"] == poc_input_lock()
    assert [row["status"] for row in summary["runs"]] == [
        "success",
        "success",
        "failed",
        "success",
        "success",
    ]
    failed = summary["runs"][2]
    assert failed["error_type"] == ("OSError" if spawn_error else "CalledProcessError")
    assert (
        "process could not start" if spawn_error else "training failed with test error"
    ) in failed["error_message"]
    assert failed["snr_db"] is None
    for method in METHODS:
        directory = tmp_path / "output" / method
        assert directory.is_dir()
        if method != "nersi":
            assert (directory / "prediction.npy").is_file()
        elif not spawn_error:
            assert (directory / "partial.txt").read_text() == "keep failed run"


def test_nested_lock_mismatch_invalidates_comparison_without_altering_runs(tmp_path, monkeypatch):
    def mutate(method, directory):
        if method == "nersi":
            _edit_json(
                directory / "inputs.lock.json",
                lambda lock: lock["benchmark_volume"]["files"].update(
                    {"another-file": {"sha256": "e" * 64}}
                ),
            )

    calls = _fake_processes(monkeypatch, mutate=mutate)
    summary = pipeline.run_c3_random80_poc(**_arguments(tmp_path))
    assert len(calls) == 6
    assert summary["status"] == "failed"
    assert summary["comparison_valid"] is False
    assert summary["inputs_lock"] is None
    assert summary["error_type"] == "ValueError"
    assert summary["error_message"] == "PoC methods used different verified input locks"
    assert all(row["status"] == "success" for row in summary["runs"])
    assert all((tmp_path / "output" / method / "prediction.npy").is_file() for method in METHODS)


@pytest.mark.parametrize("corruption", ["prediction_hash", "coverage", "metrics", "preflight"])
def test_invalid_artifacts_are_not_reported_as_success(tmp_path, monkeypatch, corruption):
    def mutate(method, directory):
        if method != "pocs":
            return
        if corruption == "prediction_hash":
            (directory / "prediction.npy").write_bytes(b"wrong prediction bytes")
        elif corruption == "coverage":
            _edit_json(
                directory / "metadata.json",
                lambda record: record["coverage"].update(complete=False),
            )
        elif corruption == "metrics":
            _edit_json(
                directory / "metrics.json", lambda record: record["evaluation_target"].pop("rmse")
            )
        else:
            _edit_json(
                directory / "inputs.lock.json", lambda record: record.update(dataset_id="different")
            )

    calls = _fake_processes(monkeypatch, mutate=mutate)
    summary = pipeline.run_c3_random80_poc(**_arguments(tmp_path))
    assert len(calls) == 6
    assert summary["status"] == "failed"
    assert summary["comparison_valid"] is False
    assert summary["runs"][0]["status"] == "failed"
    assert summary["runs"][0]["error_type"] == "ValueError"
    assert all(row["status"] == "success" for row in summary["runs"][1:])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_run_metrics_fail_but_summary_remains_strict_json(tmp_path, monkeypatch, value):
    def mutate(method, directory):
        if method == "drr":
            _edit_json(
                directory / "metrics.json",
                lambda record: record["evaluation_target"].update(rmse=value),
            )

    _fake_processes(monkeypatch, mutate=mutate)
    summary = pipeline.run_c3_random80_poc(**_arguments(tmp_path))
    assert summary["runs"][1]["status"] == "failed"
    json.dumps(summary, allow_nan=False)

    def invalid_constant(token):
        raise AssertionError(f"non-JSON numeric token: {token}")

    stored = json.loads(
        (tmp_path / "output" / "summary.json").read_text(), parse_constant=invalid_constant
    )
    assert stored == summary


@pytest.mark.parametrize("nonempty", [False, True])
def test_existing_output_root_is_rejected_without_execution_or_changes(
    tmp_path, monkeypatch, nonempty
):
    calls = _fake_processes(monkeypatch)
    output = tmp_path / "output"
    output.mkdir()
    if nonempty:
        (output / "summary.json").write_text("preserve existing results", encoding="utf-8")
    before = {file.name: file.read_bytes() for file in output.iterdir()}
    with pytest.raises(FileExistsError):
        pipeline.run_c3_random80_poc(**_arguments(tmp_path))
    assert not calls
    assert {file.name: file.read_bytes() for file in output.iterdir()} == before


def test_preflight_failure_prevents_all_method_processes_and_writes_summary(tmp_path, monkeypatch):
    calls = []

    def fail(arguments, *, stdout_path, stderr_path):
        calls.append(arguments)
        stderr_path.write_text("invalid fixed selection", encoding="utf-8")
        raise subprocess.CalledProcessError(1, arguments)

    monkeypatch.setattr(pipeline, "run_poc_process", fail)
    summary = pipeline.run_c3_random80_poc(**_arguments(tmp_path))
    assert len(calls) == 1
    assert calls[0][:2] == ["poc", "check"]
    assert summary["preflight"]["status"] == "failed"
    assert summary["comparison_valid"] is False
    assert "invalid fixed selection" in summary["error_message"]
    assert all(row["status"] == "not_run" for row in summary["runs"])
    assert (tmp_path / "output" / "summary.json").is_file()
    assert (tmp_path / "output" / "summary.csv").is_file()
    assert not any((tmp_path / "output" / method).exists() for method in METHODS)


def test_orchestrator_does_not_open_arrays_or_input_data(tmp_path, monkeypatch):
    import numpy as np

    _fake_processes(monkeypatch)

    def forbidden(*args, **kwargs):
        raise AssertionError("run-all must not deserialize amplitude or prediction arrays")

    monkeypatch.setattr(np, "load", forbidden)
    arguments = _arguments(tmp_path)
    assert not any(
        arguments[f"{name}_dir"].exists()
        for name in ("interim", "processed", "mask", "case", "volume")
    )
    assert pipeline.run_c3_random80_poc(**arguments)["status"] == "success"


def test_process_helper_uses_current_python_module_and_streams_logs(tmp_path, monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        kwargs["stdout"].write("child stdout\n")
        kwargs["stderr"].write("child stderr\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(pipeline.subprocess, "run", run)
    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"
    pipeline.run_poc_process(
        ["interpolate", "pocs", "--config", "a config.yaml"], stdout_path=stdout, stderr_path=stderr
    )
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command == [
        sys.executable,
        "-m",
        "seis_interp.cli",
        "interpolate",
        "pocs",
        "--config",
        "a config.yaml",
    ]
    assert kwargs["check"] is True
    assert set(kwargs) == {"stdout", "stderr", "check"}
    assert kwargs["stdout"].closed and kwargs["stderr"].closed
    assert stdout.read_text() == "child stdout\n"
    assert stderr.read_text() == "child stderr\n"


@pytest.mark.parametrize("method", METHODS)
def test_each_real_method_cli_can_start_in_its_own_process(tmp_path, method):
    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"
    pipeline.run_poc_process(
        ["interpolate", method.replace("_", "-"), "--help"],
        stdout_path=stdout,
        stderr_path=stderr,
    )
    help_text = stdout.read_text()
    for name in ("config", "interim", "processed", "mask", "case", "volume", "output"):
        assert f"--{name}" in help_text
    assert stderr.read_text() == ""
