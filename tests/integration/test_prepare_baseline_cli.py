from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from seis_interp.cli import main
from seis_interp.processing.trace_amplitude_filter import TraceAmplitudeFilterConfig

COORDINATE_NORMALIZATION_METHOD = "train_minmax_linear_plus_azimuth_sin_cos"

SUMMARY: dict[str, object] = {
    "dataset_id": "synthetic",
    "source_file": "synthetic.sgy",
    "source_sha256": "a" * 64,
    "input_files": {
        "traces.parquet": {"sha256": "b" * 64},
        "amplitudes.npy": {"sha256": "c" * 64},
        "time_s.npy": {"sha256": "d" * 64},
        "dataset.json": {"sha256": "e" * 64},
    },
    "trace_count": 20,
    "sample_count": 4,
    "config_source": "studies/study/config.yaml",
    "normalization": {
        "coordinates": COORDINATE_NORMALIZATION_METHOD,
        "amplitude": "train_global_rms",
    },
    "random_seed": 42,
    "holdout_fraction": 0.2,
    "validation_fraction_of_holdout": 0.25,
    "split_counts": {"train": 16, "validation": 1, "test": 3},
    "files": {
        "trace_split": "trace_split.parquet",
        "normalization": "normalization.json",
    },
}


def _write_study_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    base_seed: int = 7,
    study_seed: int = 42,
    holdout_fraction: float = 0.2,
    validation_fraction: float = 0.25,
    split_scope: str | None = None,
    trace_amplitude_filter: dict[str, object] | None = None,
    legacy_study_seed: bool = False,
) -> Path:
    repository = tmp_path / "repository"
    monkeypatch.setattr("seis_interp.commands.data.REPOSITORY_ROOT", repository)
    default_path = repository / "configs" / "default.yaml"
    study_path = repository / "studies" / "study" / "config.yaml"
    default_path.parent.mkdir(parents=True)
    study_path.parent.mkdir(parents=True)
    (repository / "pyproject.toml").write_text("[project]\nname = 'test'\n", encoding="utf-8")
    default_path.write_text(
        yaml.safe_dump(
            {
                "project": {"random_seed": base_seed},
                "normalization": {
                    "coordinates": COORDINATE_NORMALIZATION_METHOD,
                    "amplitude": "train_global_rms",
                },
            }
        ),
        encoding="utf-8",
    )
    study: dict[str, object] = {"status": "draft"}
    if legacy_study_seed:
        study["random_seed"] = study_seed
    sampling: dict[str, object] = {"validation_fraction_of_holdout": validation_fraction}
    holdout_key = (
        "random_ffid_holdout_fraction"
        if split_scope == "whole_ffid"
        else "random_trace_holdout_fraction"
    )
    sampling[holdout_key] = holdout_fraction
    if split_scope is not None:
        sampling["split_scope"] = split_scope
    if trace_amplitude_filter is not None:
        sampling["trace_amplitude_filter"] = trace_amplitude_filter
    study_path.write_text(
        yaml.safe_dump(
            {
                "extends": "../../configs/default.yaml",
                "project": {"random_seed": study_seed},
                "study": study,
                "sampling": sampling,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return study_path


def _arguments(tmp_path: Path, config_path: Path) -> list[str]:
    return [
        "data",
        "prepare-baseline",
        "--config",
        str(config_path),
        "--input",
        str(tmp_path / "interim"),
        "--output",
        str(tmp_path / "processed"),
    ]


def test_cli_passes_all_arguments_to_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_study_config(tmp_path, monkeypatch)
    received: dict[str, Any] = {}

    def fake_prepare_baseline_dataset(**kwargs: Any) -> dict[str, object]:
        received.update(kwargs)
        return SUMMARY

    monkeypatch.setattr(
        "seis_interp.pipelines.prepare_baseline.prepare_baseline_dataset",
        fake_prepare_baseline_dataset,
    )

    exit_code = main([*_arguments(tmp_path, config_path), "--overwrite"])

    assert exit_code == 0
    assert received == {
        "interim_dir": tmp_path / "interim",
        "output_dir": tmp_path / "processed",
        "holdout_fraction": 0.2,
        "validation_fraction_of_holdout": 0.25,
        "random_seed": 42,
        "split_scope": "global",
        "coordinate_normalization": COORDINATE_NORMALIZATION_METHOD,
        "amplitude_normalization": "train_global_rms",
        "trace_amplitude_filter": None,
        "config_source": "studies/study/config.yaml",
        "overwrite": True,
    }


def test_cli_overrides_config_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = _write_study_config(tmp_path, monkeypatch)
    received: dict[str, Any] = {}

    def fake_prepare_baseline_dataset(**kwargs: Any) -> dict[str, object]:
        received.update(kwargs)
        return SUMMARY

    monkeypatch.setattr(
        "seis_interp.pipelines.prepare_baseline.prepare_baseline_dataset",
        fake_prepare_baseline_dataset,
    )

    exit_code = main(
        [
            *_arguments(tmp_path, config_path),
            "--holdout-fraction",
            "0.30",
            "--validation-fraction-of-holdout",
            "0.50",
            "--random-seed",
            "0",
        ]
    )

    assert exit_code == 0
    assert received["holdout_fraction"] == 0.3
    assert received["validation_fraction_of_holdout"] == 0.5
    assert received["random_seed"] == 0


def test_cli_resolves_split_scope_from_config_and_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_study_config(tmp_path, monkeypatch, split_scope="per_ffid")
    received: list[str] = []

    def fake_prepare_baseline_dataset(**kwargs: Any) -> dict[str, object]:
        received.append(kwargs["split_scope"])
        return SUMMARY

    monkeypatch.setattr(
        "seis_interp.pipelines.prepare_baseline.prepare_baseline_dataset",
        fake_prepare_baseline_dataset,
    )

    assert main(_arguments(tmp_path, config_path)) == 0
    assert (
        main(
            [
                *_arguments(tmp_path, config_path),
                "--split-scope",
                "global",
            ]
        )
        == 0
    )
    assert received == ["per_ffid", "global"]


def test_cli_accepts_whole_ffid_split_scope_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_study_config(tmp_path, monkeypatch, split_scope="whole_ffid")
    received: list[str] = []

    def fake_prepare_baseline_dataset(**kwargs: Any) -> dict[str, object]:
        received.append(kwargs["split_scope"])
        return SUMMARY

    monkeypatch.setattr(
        "seis_interp.pipelines.prepare_baseline.prepare_baseline_dataset",
        fake_prepare_baseline_dataset,
    )

    assert main(_arguments(tmp_path, config_path)) == 0
    assert received == ["whole_ffid"]


def test_cli_requires_holdout_override_when_split_unit_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_study_config(tmp_path, monkeypatch, split_scope="per_ffid")

    exit_code = main([*_arguments(tmp_path, config_path), "--split-scope", "whole_ffid"])

    assert exit_code == 1
    assert "--holdout-fraction is required" in capsys.readouterr().err


def test_cli_accepts_split_unit_change_with_explicit_holdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_study_config(tmp_path, monkeypatch, split_scope="per_ffid")
    received: dict[str, object] = {}

    def fake_prepare_baseline_dataset(**kwargs: Any) -> dict[str, object]:
        received.update(kwargs)
        return SUMMARY

    monkeypatch.setattr(
        "seis_interp.pipelines.prepare_baseline.prepare_baseline_dataset",
        fake_prepare_baseline_dataset,
    )

    assert (
        main(
            [
                *_arguments(tmp_path, config_path),
                "--split-scope",
                "whole_ffid",
                "--holdout-fraction",
                "0.75",
            ]
        )
        == 0
    )
    assert received["split_scope"] == "whole_ffid"
    assert received["holdout_fraction"] == 0.75


def test_cli_passes_the_configured_trace_amplitude_filter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_study_config(
        tmp_path,
        monkeypatch,
        trace_amplitude_filter={
            "exclude_all_zero": True,
            "max_abs_amplitude": 10_000.0,
        },
    )
    received: dict[str, object] = {}

    def fake_prepare_baseline_dataset(**kwargs: Any) -> dict[str, object]:
        received.update(kwargs)
        return SUMMARY

    monkeypatch.setattr(
        "seis_interp.pipelines.prepare_baseline.prepare_baseline_dataset",
        fake_prepare_baseline_dataset,
    )

    assert main(_arguments(tmp_path, config_path)) == 0
    assert received["trace_amplitude_filter"] == TraceAmplitudeFilterConfig(
        exclude_all_zero=True,
        max_abs_amplitude=10_000.0,
    )


def test_cli_requires_study_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "seis_interp.pipelines.prepare_baseline.prepare_baseline_dataset",
        lambda **kwargs: pytest.fail("pipeline must not run without --config"),
    )

    with pytest.raises(SystemExit) as error:
        main(
            [
                "data",
                "prepare-baseline",
                "--input",
                str(tmp_path / "interim"),
                "--output",
                str(tmp_path / "processed"),
            ]
        )

    assert error.value.code == 2
    assert "the following arguments are required: --config" in capsys.readouterr().err


def test_cli_json_output_is_parsable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_study_config(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "seis_interp.pipelines.prepare_baseline.prepare_baseline_dataset",
        lambda **kwargs: SUMMARY,
    )

    exit_code = main([*_arguments(tmp_path, config_path), "--json"])

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == SUMMARY


def test_cli_human_output_contains_split_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_study_config(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "seis_interp.pipelines.prepare_baseline.prepare_baseline_dataset",
        lambda **kwargs: SUMMARY,
    )

    exit_code = main(_arguments(tmp_path, config_path))
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Configuration: studies/study/config.yaml" in output
    assert f"Input dataset: {tmp_path / 'interim'}" in output
    assert f"Output directory: {tmp_path / 'processed'}" in output
    assert "Traces: 20" in output
    assert "Train traces: 16" in output
    assert "Validation traces: 1" in output
    assert "Test traces: 3" in output


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("dataset.json is missing"),
        FileExistsError("output is not empty"),
        ValueError("invalid split fraction"),
        json.JSONDecodeError("invalid metadata", "{", 1),
    ],
)
def test_cli_reports_input_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: Exception,
) -> None:
    config_path = _write_study_config(tmp_path, monkeypatch)

    def fail(**kwargs: Any) -> dict[str, object]:
        raise error

    monkeypatch.setattr(
        "seis_interp.pipelines.prepare_baseline.prepare_baseline_dataset",
        fail,
    )

    exit_code = main(_arguments(tmp_path, config_path))

    assert exit_code == 1
    assert capsys.readouterr().err == f"data prepare-baseline failed: {error}\n"


def test_cli_does_not_hide_unexpected_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_study_config(tmp_path, monkeypatch)

    def fail(**kwargs: Any) -> dict[str, object]:
        raise RuntimeError("unexpected failure")

    monkeypatch.setattr(
        "seis_interp.pipelines.prepare_baseline.prepare_baseline_dataset",
        fail,
    )

    with pytest.raises(RuntimeError, match="unexpected failure"):
        main(_arguments(tmp_path, config_path))


def test_cli_does_not_hide_unexpected_pipeline_io_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_study_config(tmp_path, monkeypatch)

    def fail(**kwargs: Any) -> dict[str, object]:
        raise OSError("unexpected output failure")

    monkeypatch.setattr(
        "seis_interp.pipelines.prepare_baseline.prepare_baseline_dataset",
        fail,
    )

    with pytest.raises(OSError, match="unexpected output failure"):
        main(_arguments(tmp_path, config_path))


def test_cli_reports_missing_config_value_without_calling_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_study_config(tmp_path, monkeypatch)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    del config["sampling"]["validation_fraction_of_holdout"]
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    def unexpected_call(**kwargs: Any) -> dict[str, object]:
        raise AssertionError("pipeline must not run with unresolved configuration")

    monkeypatch.setattr(
        "seis_interp.pipelines.prepare_baseline.prepare_baseline_dataset",
        unexpected_call,
    )

    exit_code = main(_arguments(tmp_path, config_path))

    assert exit_code == 1
    assert "sampling.validation_fraction_of_holdout" in capsys.readouterr().err


def test_cli_rejects_legacy_study_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_study_config(
        tmp_path,
        monkeypatch,
        legacy_study_seed=True,
    )

    monkeypatch.setattr(
        "seis_interp.pipelines.prepare_baseline.prepare_baseline_dataset",
        lambda **kwargs: pytest.fail("pipeline must not run with study.random_seed"),
    )

    exit_code = main(_arguments(tmp_path, config_path))

    assert exit_code == 1
    assert "use project.random_seed" in capsys.readouterr().err


def test_cli_rejects_unsupported_normalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_study_config(tmp_path, monkeypatch)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["normalization"] = {"amplitude": "per_trace_peak"}
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    monkeypatch.setattr(
        "seis_interp.pipelines.prepare_baseline.prepare_baseline_dataset",
        lambda **kwargs: pytest.fail("pipeline must not run with unsupported normalization"),
    )

    exit_code = main(_arguments(tmp_path, config_path))

    assert exit_code == 1
    assert "normalization.amplitude" in capsys.readouterr().err
