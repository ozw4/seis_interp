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
    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert summary == json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert "random_points 1/2 start" in captured.err
    assert "random_points 1/2 end" in captured.err
    saved_config = yaml.safe_load((output / "config.resolved.yaml").read_text(encoding="utf-8"))
    assert saved_config["training"]["device"] == "cpu"
    assert load_siren_checkpoint(output / "artifacts" / "best.pt").model.input_features == 6


def test_cli_labels_both_validation_metrics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config, interim, processed = _build_training_fixture(tmp_path)
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
        ]
    )

    assert exit_code == 0
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    printed = capsys.readouterr().out
    assert (
        f"Best validation median trace S/N: {metrics['best_validation_median_trace_snr_db']} dB"
        in printed
    )
    assert (
        f"Global validation S/N at best epoch: {metrics['best_validation_global_snr_db']} dB"
        in printed
    )


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


def test_cli_rejects_unknown_training_amplitude_scaling_before_data_loading(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config, _, _ = _build_training_fixture(tmp_path)
    config_payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    config_payload["training"]["amplitude_scaling"] = "global_rms"
    config.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")
    output = tmp_path / "run"

    exit_code = main(
        [
            "train",
            "siren",
            "--config",
            str(config),
            "--interim",
            str(tmp_path / "missing-interim"),
            "--processed",
            str(tmp_path / "missing-processed"),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 1
    error = capsys.readouterr().err
    assert "train siren failed" in error
    assert "training.amplitude_scaling must be one of" in error
    assert not output.exists()


def test_cli_does_not_offer_training_overwrite(tmp_path: Path) -> None:
    config, interim, processed = _build_training_fixture(tmp_path)

    with pytest.raises(SystemExit, match="2"):
        main(
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
                str(tmp_path / "run"),
                "--overwrite",
            ]
        )
