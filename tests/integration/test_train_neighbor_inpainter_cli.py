from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from seis_interp.cli import main
from seis_interp.training.neighbor_inpainter_checkpoints import (
    load_neighbor_inpainter_checkpoint,
)
from tests.fixtures.neighbor_training import (
    prepare_neighbor_training_fixture,
)


def test_cli_runs_neighbor_inpainter_with_device_override_and_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config, interim, processed = prepare_neighbor_training_fixture(
        tmp_path,
        configured_device="cuda:0",
    )
    output = tmp_path / "run"

    exit_code = main(
        [
            "train",
            "neighbor-inpainter",
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
    assert "neighbor_trace_inpainter 1/2" in captured.err
    assert "oracle_per_trace_unit_rms_global_snr_db=" in captured.err
    resolved = yaml.safe_load((output / "config.resolved.yaml").read_text(encoding="utf-8"))
    assert resolved["training"]["device"] == "cpu"
    checkpoint = load_neighbor_inpainter_checkpoint(output / "artifacts/best.pt")
    assert (
        checkpoint.best_validation_global_snr_db
        == summary["oracle_per_trace_unit_rms_global_snr_db"]
    )


def test_cli_does_not_offer_neighbor_training_overwrite(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "train",
                "neighbor-inpainter",
                "--config",
                str(tmp_path / "config.yaml"),
                "--interim",
                str(tmp_path / "interim"),
                "--processed",
                str(tmp_path / "processed"),
                "--output",
                str(tmp_path / "run"),
                "--overwrite",
            ]
        )
