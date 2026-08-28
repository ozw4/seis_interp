from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts import run_study_all_ffid_temp as runner


def _set_scratch_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    output = tmp_path / "runs" / "study_all_ffid_temp" / "current"
    monkeypatch.setattr(runner, "OUTPUT_DIRECTORY", output)
    return output


def test_successful_run_replaces_the_previous_scratch_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _set_scratch_output(tmp_path, monkeypatch)
    output.mkdir(parents=True)
    (output / "old-result").write_text("old", encoding="utf-8")
    received: dict[str, Any] = {}

    def fake_train_siren_run(**kwargs: Any) -> dict[str, object]:
        received.update(kwargs)
        assert (output / "old-result").is_file()
        staging = kwargs["output_dir"]
        staging.mkdir(parents=True)
        (staging / "new-result").write_text("new", encoding="utf-8")
        return {"best_epoch": 1}

    monkeypatch.setattr(runner, "train_siren_run", fake_train_siren_run)
    config = tmp_path / "config.yaml"

    assert runner.run(config_path=config, device_override="cpu") == {"best_epoch": 1}
    assert received["config_path"] == config
    assert received["device_override"] == "cpu"
    assert received["output_dir"] != output
    assert not (output / "old-result").exists()
    assert (output / "new-result").read_text(encoding="utf-8") == "new"
    assert list(output.parent.iterdir()) == [output]


def test_failed_run_preserves_the_previous_scratch_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _set_scratch_output(tmp_path, monkeypatch)
    output.mkdir(parents=True)
    (output / "old-result").write_text("old", encoding="utf-8")

    def failing_train_siren_run(**kwargs: Any) -> dict[str, object]:
        staging = kwargs["output_dir"]
        staging.mkdir(parents=True)
        (staging / "partial-result").write_text("partial", encoding="utf-8")
        raise RuntimeError("training failed")

    monkeypatch.setattr(runner, "train_siren_run", failing_train_siren_run)

    with pytest.raises(RuntimeError, match="training failed"):
        runner.run(config_path=tmp_path / "config.yaml")

    assert (output / "old-result").read_text(encoding="utf-8") == "old"
    assert list(output.parent.iterdir()) == [output]


def test_run_refuses_to_replace_a_non_directory_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _set_scratch_output(tmp_path, monkeypatch)
    output.parent.mkdir(parents=True)
    output.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(
        runner,
        "train_siren_run",
        lambda **_kwargs: pytest.fail("training must not start for an unsafe output path"),
    )

    with pytest.raises(FileExistsError, match="file or symlink"):
        runner.run(config_path=tmp_path / "config.yaml")

    assert output.read_text(encoding="utf-8") == "keep"


def test_main_labels_per_trace_validation_as_oracle(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        runner,
        "run",
        lambda **_kwargs: {
            "best_epoch": 2,
            "best_validation_global_snr_db": 3.5,
            "global_steps": 8,
            "amplitude_scaling": "per_trace_rms",
            "validation_metric_domain": "oracle_per_trace_unit_rms",
        },
    )

    assert runner.main([]) == 0
    output = capsys.readouterr().out
    assert "Amplitude scaling: per_trace_rms" in output
    assert "Validation metric domain: oracle per-trace unit RMS" in output
    assert "Best oracle-normalized validation global S/N: 3.5 dB" in output
