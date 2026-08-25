from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from seis_interp.cli import main
from seis_interp.training.checkpoints import load_siren_checkpoint
from tests.integration.test_train_siren_pipeline import _build_training_fixture


def test_cli_runs_training_with_device_override_and_json_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config, interim, processed = _build_training_fixture(tmp_path, configured_device="cuda")
    output = tmp_path / "run"

    exit_code = main(
        [
            "train",
            "siren",
            "--config",
            str(config),
            "--interim",
            str(interim),
            "--processed",
            str(processed),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--json",
        ]
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary == json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    saved_config = yaml.safe_load((output / "config.resolved.yaml").read_text(encoding="utf-8"))
    assert saved_config["training"]["device"] == "cpu"
    assert load_siren_checkpoint(output / "artifacts" / "best.pt").model.input_features == 6


def test_cli_reports_nonempty_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config, interim, processed = _build_training_fixture(tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    (output / "marker").write_text("keep", encoding="utf-8")

    exit_code = main(
        [
            "train",
            "siren",
            "--config",
            str(config),
            "--interim",
            str(interim),
            "--processed",
            str(processed),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 1
    assert "train siren failed" in capsys.readouterr().err
