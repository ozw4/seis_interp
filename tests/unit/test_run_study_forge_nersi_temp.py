import ast
from pathlib import Path

import pytest

from scripts import run_study_forge_nersi_temp as runner


@pytest.fixture
def calls(monkeypatch, tmp_path):
    recorded = []
    monkeypatch.setattr(runner, "OUTPUT_DIRECTORY", tmp_path / "runs")

    def run(repo, config, inputs, output, *, device_override, reporter):
        recorded.append(("run", config, inputs, output, device_override))
        return output / "run"

    def check(repo, config, inputs, *, device_override):
        recorded.append(("check", config, inputs, device_override))
        return {"report": True}

    def summary(output):
        recorded.append(("summary", output))
        return output / "summary.csv"

    monkeypatch.setattr(runner, "run_forge_nersi_experiment", run)
    monkeypatch.setattr(runner, "check_forge_nersi_experiment", check)
    monkeypatch.setattr(runner, "format_check_report", lambda report: "VALID")
    monkeypatch.setattr(runner, "write_experiment_summary", summary)
    monkeypatch.setattr(runner, "format_experiment_summary", lambda path: f"table {path.name}")
    return recorded


def test_default_run_uses_the_workspace_files(calls, tmp_path):
    assert runner.main([]) == 0
    assert calls == [
        ("run", runner.DEFAULT_CONFIG_PATH, runner.DEFAULT_INPUTS_PATH, tmp_path / "runs", None)
    ]
    assert runner.main(["--device", "cuda:1"]) == 0
    assert calls[-1][-1] == "cuda:1"


def test_check_and_summarize_do_not_train(calls, tmp_path, capsys):
    assert runner.main(["--check"]) == 0
    assert calls == [("check", runner.DEFAULT_CONFIG_PATH, runner.DEFAULT_INPUTS_PATH, None)]
    assert capsys.readouterr().out.strip() == "VALID"
    assert runner.main(["--summarize"]) == 0
    assert calls[-1] == ("summary", tmp_path / "runs")
    assert capsys.readouterr().out.strip() == "table summary.csv"
    calls.clear()
    with pytest.raises(SystemExit) as caught:
        runner.main(["--check", "--summarize"])
    assert caught.value.code == 2
    assert calls == []


def test_config_and_inputs_paths_are_forwarded(calls, tmp_path):
    config, inputs = tmp_path / "a.yaml", tmp_path / "i.yaml"
    arguments = ["--config", str(config), "--inputs", str(inputs), "--device", "cuda:1"]
    assert runner.main([*arguments, "--check"]) == 0
    assert calls[-1] == ("check", config, inputs, "cuda:1")
    assert runner.main(arguments) == 0
    assert calls[-1] == ("run", config, inputs, tmp_path / "runs", "cuda:1")


def test_pipeline_error_returns_one(calls, monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise ValueError("training.batch_size: too large")

    monkeypatch.setattr(runner, "run_forge_nersi_experiment", fail)
    assert runner.main([]) == 1
    assert capsys.readouterr().err.strip() == "ERROR: training.batch_size: too large"


def test_cublas_workspace_is_configured_before_any_torch_import():
    tree = ast.parse(Path(runner.__file__).read_text())
    statements = tree.body
    setdefault = next(
        index
        for index, node in enumerate(statements)
        if isinstance(node, ast.Expr)
        and ast.unparse(node) == "os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')"
    )
    imports = [
        index
        for index, node in enumerate(statements)
        if isinstance(node, (ast.Import, ast.ImportFrom)) and ast.unparse(node) != "import os"
    ]
    assert imports and setdefault < min(imports)
