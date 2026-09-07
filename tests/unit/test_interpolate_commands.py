from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from seis_interp.cli import build_parser, main

_PIPELINE_MODULE = "seis_interp.pipelines.interpolate_pocs"


def _arguments(tmp_path: Path, command: str = "pocs") -> list[str]:
    return [
        "interpolate",
        command,
        "--config",
        str(tmp_path / "config.yaml"),
        "--interim",
        str(tmp_path / "interim"),
        "--processed",
        str(tmp_path / "processed"),
        "--mask",
        str(tmp_path / "mask"),
        "--case",
        str(tmp_path / "case"),
        "--volume",
        str(tmp_path / "volume"),
        "--output",
        str(tmp_path / "run"),
    ]


def _summary(
    *,
    snr_db: float | None = 12.5,
    snr_status: str = "finite",
    warnings: list[str] | None = None,
) -> dict[str, object]:
    return {
        "method": "pocs_fourier_5d",
        "case_id": "synthetic_case",
        "volume_id": "synthetic_volume",
        "evaluation_domain": "evaluation_target",
        "amplitude_domain": "physical",
        "evaluation_target": {
            "trace_count": 2,
            "sample_count": 8,
            "reference_energy": 10.0,
            "error_energy": 1.0,
            "snr_db": snr_db,
            "snr_status": snr_status,
            "rmse": 0.25,
            "relative_l2": 0.1,
        },
        "zero_fill": {
            "snr_db": 0.0,
            "snr_status": "finite",
            "rmse": 1.0,
        },
        "observed_max_abs_error": 0.0,
        "uncovered_sample_count": 0,
        "warnings": [] if warnings is None else warnings,
    }


def _install_pipeline_stub(monkeypatch: pytest.MonkeyPatch, function: object) -> None:
    module = ModuleType(_PIPELINE_MODULE)
    module.interpolate_pocs_run = function  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, _PIPELINE_MODULE, module)


def _install_drr_pipeline_stub(monkeypatch: pytest.MonkeyPatch, function: object) -> None:
    module_name = "seis_interp.pipelines.interpolate_drr"
    module = ModuleType(module_name)
    module.interpolate_drr_run = function  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module_name, module)


def _drr_summary(**kwargs: object) -> dict[str, object]:
    return _summary(**kwargs) | {  # type: ignore[arg-type]
        "method": "damped_rank_reduction_5d",
        "uncovered_trace_count": 0,
    }


def _help_text(argv: list[str], capsys: pytest.CaptureFixture[str]) -> str:
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args([*argv, "--help"])

    assert excinfo.value.code == 0
    return capsys.readouterr().out


def test_interpolate_help_is_available_at_every_parser_level(capsys) -> None:
    assert "interpolate" in _help_text([], capsys)
    assert "pocs" in _help_text(["interpolate"], capsys)

    pocs_help = _help_text(["interpolate", "pocs"], capsys)
    for option in (
        "--config",
        "--interim",
        "--processed",
        "--mask",
        "--case",
        "--volume",
        "--output",
        "--json",
    ):
        assert option in pocs_help
    for unsupported in ("--device", "--overwrite", "--seed", "--iterations"):
        assert unsupported not in pocs_help


def test_pocs_dispatches_paths_and_keeps_json_stdout_clean(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    received: dict[str, object] = {}
    expected = _summary(warnings=["3 samples were not covered"])

    def interpolate_pocs_run(**kwargs: object) -> dict[str, object]:
        received.update(kwargs)
        reporter = kwargs["progress_reporter"]
        assert callable(reporter)
        reporter("loading verified inputs")
        return expected

    _install_pipeline_stub(monkeypatch, interpolate_pocs_run)

    assert main([*_arguments(tmp_path), "--json"]) == 0

    captured = capsys.readouterr()
    assert json.loads(captured.out) == expected
    assert "loading verified inputs" not in captured.out
    assert "loading verified inputs" in captured.err
    assert captured.err.count("Warning: 3 samples were not covered") == 1
    assert received == {
        "config_path": tmp_path / "config.yaml",
        "interim_dir": tmp_path / "interim",
        "processed_dir": tmp_path / "processed",
        "mask_dir": tmp_path / "mask",
        "case_dir": tmp_path / "case",
        "volume_dir": tmp_path / "volume",
        "output_dir": tmp_path / "run",
        "progress_reporter": received["progress_reporter"],
    }


@pytest.mark.parametrize(
    ("snr_db", "snr_status", "expected_text"),
    [
        (12.5, "finite", "Target global S/N: 12.5 dB"),
        (None, "finite", "Target global S/N: unavailable (invalid finite S/N)"),
        (None, "perfect_reconstruction", "Target global S/N: perfect reconstruction"),
        (
            None,
            "undefined_zero_reference",
            "Target global S/N: undefined (zero reference energy)",
        ),
    ],
)
def test_pocs_human_output_handles_every_snr_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    snr_db: float | None,
    snr_status: str,
    expected_text: str,
) -> None:
    def interpolate_pocs_run(**kwargs: object) -> dict[str, object]:
        assert kwargs["progress_reporter"] is None
        return _summary(snr_db=snr_db, snr_status=snr_status)

    _install_pipeline_stub(monkeypatch, interpolate_pocs_run)

    assert main(_arguments(tmp_path)) == 0

    captured = capsys.readouterr()
    assert expected_text in captured.out
    assert "Target RMSE: 0.25" in captured.out
    assert "Observed maximum absolute error: 0.0" in captured.out
    assert "Uncovered samples: 0" in captured.out
    assert captured.err == ""


def test_pocs_human_summary_preserves_exact_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install_pipeline_stub(monkeypatch, lambda **kwargs: _summary())

    assert main(_arguments(tmp_path)) == 0

    captured = capsys.readouterr()
    assert captured.out == (
        f"Output directory: {tmp_path / 'run'}\n"
        "Method: pocs_fourier_5d\n"
        "Benchmark case: synthetic_case\n"
        "Benchmark volume: synthetic_volume\n"
        "Target global S/N: 12.5 dB\n"
        "Target RMSE: 0.25\n"
        "Observed maximum absolute error: 0.0\n"
        "Uncovered samples: 0\n"
    )
    assert captured.err == ""


@pytest.mark.parametrize(
    "error",
    [
        FileExistsError("run output path already exists"),
        ValueError("benchmark case does not match"),
    ],
)
def test_pocs_reports_expected_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: Exception,
) -> None:
    def interpolate_pocs_run(**kwargs: object) -> dict[str, object]:
        raise error

    _install_pipeline_stub(monkeypatch, interpolate_pocs_run)

    assert main(_arguments(tmp_path)) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"interpolate pocs failed: {error}" in captured.err


def test_pocs_requires_all_seven_paths(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["interpolate", "pocs"])

    assert excinfo.value.code == 2
    error = capsys.readouterr().err
    for option in (
        "--config",
        "--interim",
        "--processed",
        "--mask",
        "--case",
        "--volume",
        "--output",
    ):
        assert option in error


def test_drr_help_exposes_only_the_shared_path_and_output_options(capsys) -> None:
    assert "drr" in _help_text(["interpolate"], capsys)
    help_text = _help_text(["interpolate", "drr"], capsys)
    for option in (
        "--config",
        "--interim",
        "--processed",
        "--mask",
        "--case",
        "--volume",
        "--output",
        "--json",
    ):
        assert option in help_text
    for unsupported in ("--rank", "--damping-power", "--iterations", "--device", "--overwrite"):
        assert unsupported not in help_text


def test_drr_dispatches_paths_and_keeps_strict_json_stdout_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    received: dict[str, object] = {}
    expected = _drr_summary(warnings=["2 traces are uncovered and evaluated as zero"])

    def interpolate_drr_run(**kwargs: object) -> dict[str, object]:
        received.update(kwargs)
        reporter = kwargs["progress_reporter"]
        assert callable(reporter)
        reporter("loading verified DRR inputs")
        return expected

    _install_drr_pipeline_stub(monkeypatch, interpolate_drr_run)

    assert main([*_arguments(tmp_path, "drr"), "--json"]) == 0

    captured = capsys.readouterr()
    assert (
        json.loads(
            captured.out,
            parse_constant=lambda constant: pytest.fail(f"non-finite JSON constant: {constant}"),
        )
        == expected
    )
    assert "loading verified DRR inputs" not in captured.out
    assert "loading verified DRR inputs" in captured.err
    assert captured.err.count("Warning: 2 traces are uncovered and evaluated as zero") == 1
    assert received == {
        "config_path": tmp_path / "config.yaml",
        "interim_dir": tmp_path / "interim",
        "processed_dir": tmp_path / "processed",
        "mask_dir": tmp_path / "mask",
        "case_dir": tmp_path / "case",
        "volume_dir": tmp_path / "volume",
        "output_dir": tmp_path / "run",
        "progress_reporter": received["progress_reporter"],
    }


@pytest.mark.parametrize(
    ("snr_db", "snr_status", "snr_text"),
    [
        (12.5, "finite", "12.5 dB"),
        (None, "perfect_reconstruction", "perfect reconstruction"),
        (None, "undefined_zero_reference", "undefined (zero reference energy)"),
    ],
)
def test_drr_human_output_includes_primary_metrics_and_uncovered_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    snr_db: float | None,
    snr_status: str,
    snr_text: str,
) -> None:
    def interpolate_drr_run(**kwargs: object) -> dict[str, object]:
        assert kwargs["progress_reporter"] is None
        return _drr_summary(snr_db=snr_db, snr_status=snr_status, warnings=["check coverage"])

    _install_drr_pipeline_stub(monkeypatch, interpolate_drr_run)

    assert main(_arguments(tmp_path, "drr")) == 0

    captured = capsys.readouterr()
    assert f"Output directory: {tmp_path / 'run'}" in captured.out
    assert "Method: damped_rank_reduction_5d" in captured.out
    assert "Benchmark case: synthetic_case" in captured.out
    assert "Benchmark volume: synthetic_volume" in captured.out
    assert f"Target global S/N: {snr_text}" in captured.out
    assert "Target RMSE: 0.25" in captured.out
    assert "Observed maximum absolute error: 0.0" in captured.out
    assert "Uncovered traces: 0" in captured.out
    assert "Uncovered samples: 0" in captured.out
    assert captured.err == "Warning: check coverage\n"


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("missing input"),
        FileExistsError("run already exists"),
        OSError("unreadable input"),
        RuntimeError("could not determine Git state"),
        ValueError("invalid rank"),
    ],
)
def test_drr_reports_expected_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: Exception,
) -> None:
    def interpolate_drr_run(**kwargs: object) -> dict[str, object]:
        raise error

    _install_drr_pipeline_stub(monkeypatch, interpolate_drr_run)

    assert main(_arguments(tmp_path, "drr")) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"interpolate drr failed: {error}\n"


def test_drr_requires_all_seven_paths(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["interpolate", "drr"])

    assert excinfo.value.code == 2
    error = capsys.readouterr().err
    for option in (
        "--config",
        "--interim",
        "--processed",
        "--mask",
        "--case",
        "--volume",
        "--output",
    ):
        assert option in error


def test_importing_interpolate_commands_defers_pipeline_and_heavy_dependencies() -> None:
    probe = (
        "import sys\n"
        "import seis_interp.commands.interpolate\n"
        "eager = [\n"
        "    name\n"
        "    for name in (\n"
        "        'seis_interp.pipelines.interpolate_pocs',\n"
        "        'seis_interp.pipelines.interpolate_drr',\n"
        "        'numpy',\n"
        "        'pandas',\n"
        "        'torch',\n"
        "    )\n"
        "    if name in sys.modules\n"
        "]\n"
        "assert not eager, f'eagerly imported: {eager}'\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
