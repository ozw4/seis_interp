from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from seis_interp.cli import main


def _arguments(tmp_path: Path) -> list[str]:
    return [
        "train",
        "ccnet5d",
        "--config",
        str(tmp_path / "config.yaml"),
        "--interim",
        str(tmp_path / "interim"),
        "--processed",
        str(tmp_path / "processed"),
        "--output",
        str(tmp_path / "run"),
    ]


def _install_pipeline_stub(monkeypatch: pytest.MonkeyPatch, function: object) -> None:
    name = "seis_interp.pipelines.train_ccnet5d"
    module = ModuleType(name)
    module.train_ccnet5d_run = function  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, name, module)


@pytest.mark.parametrize("json_output", [False, True])
@pytest.mark.parametrize("device_override", [None, "cpu"])
def test_ccnet5d_training_dispatches_paths_and_keeps_progress_on_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    json_output: bool,
    device_override: str | None,
) -> None:
    received = {}
    expected = {
        "method": "ccnet5d",
        "training_regime": "supervised_train_partition",
        "steps_completed": 4,
        "best_step": 2,
        "best_selection_metrics": {"snr_db": 12.5, "snr_status": "finite"},
        "warnings": ["check selection coverage"],
    }

    def train_ccnet5d_run(**kwargs):
        received.update(kwargs)
        kwargs["progress_reporter"]("Training CCNet5D step 4/4")
        return expected

    _install_pipeline_stub(monkeypatch, train_ccnet5d_run)
    arguments = _arguments(tmp_path)
    if device_override is not None:
        arguments.extend(["--device", device_override])
    if json_output:
        arguments.append("--json")

    assert main(arguments) == 0

    captured = capsys.readouterr()
    if json_output:
        assert (
            json.loads(
                captured.out,
                parse_constant=lambda value: pytest.fail(f"non-finite JSON: {value}"),
            )
            == expected
        )
    else:
        assert captured.out == (
            f"Output directory: {tmp_path / 'run'}\n"
            "Completed steps: 4\n"
            "Best step: 2\n"
            "Best internal selection global S/N: 12.5 dB\n"
        )
        assert "Target" not in captured.out and "Benchmark" not in captured.out
    assert captured.err == "Training CCNet5D step 4/4\nWarning: check selection coverage\n"
    assert received == {
        "config_path": tmp_path / "config.yaml",
        "interim_dir": tmp_path / "interim",
        "processed_dir": tmp_path / "processed",
        "output_dir": tmp_path / "run",
        "device_override": device_override,
        "progress_reporter": received["progress_reporter"],
    }


def test_ccnet5d_training_labels_perfect_selection_without_a_numeric_infinity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install_pipeline_stub(
        monkeypatch,
        lambda **kwargs: {
            "steps_completed": 4,
            "best_step": 2,
            "best_selection_metrics": {"snr_db": None, "snr_status": "perfect_reconstruction"},
        },
    )

    assert main(_arguments(tmp_path)) == 0

    assert "Best internal selection global S/N: perfect reconstruction" in capsys.readouterr().out


@pytest.mark.parametrize(
    "error", [FileNotFoundError("missing labels"), ValueError("invalid patch")]
)
def test_ccnet5d_training_reports_expected_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: Exception,
) -> None:
    def train_ccnet5d_run(**kwargs):
        raise error

    _install_pipeline_stub(monkeypatch, train_ccnet5d_run)

    assert main(_arguments(tmp_path)) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"train ccnet5d failed: {error}\n"


def test_ccnet5d_training_requires_all_four_paths(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(["train", "ccnet5d"])

    assert error.value.code == 2
    stderr = capsys.readouterr().err
    for option in ("--config", "--interim", "--processed", "--output"):
        assert option in stderr
