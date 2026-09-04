from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from seis_interp.cli import build_parser, main
from seis_interp.commands import data as data_commands


def test_download_cli_forwards_paths_and_options(tmp_path: Path, monkeypatch, capsys) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    data_root = tmp_path / "data-root"
    expected_lock = data_root / "external" / "seg_c3_na" / "download.lock.yaml"
    captured: dict[str, object] = {}

    def fake_download(manifest, root, *, force, resume, timeout_s):
        captured.update(
            manifest=manifest,
            root=root,
            force=force,
            resume=resume,
            timeout_s=timeout_s,
        )
        return expected_lock

    monkeypatch.setattr(data_commands, "download_seg_c3_na", fake_download)

    exit_code = main(
        [
            "data",
            "download",
            "seg_c3_na",
            "--manifest",
            str(manifest_path),
            "--data-root",
            str(data_root),
            "--force",
            "--no-resume",
            "--timeout-s",
            "12.5",
        ]
    )

    assert exit_code == 0
    assert captured == {
        "manifest": manifest_path,
        "root": data_root,
        "force": True,
        "resume": False,
        "timeout_s": 12.5,
    }
    assert str(expected_lock) in capsys.readouterr().out


def test_download_cli_reports_failures_with_the_existing_prefix(monkeypatch, capsys) -> None:
    def failing_download(manifest, root, *, force, resume, timeout_s):
        raise ValueError("manifest is unreadable")

    monkeypatch.setattr(data_commands, "download_seg_c3_na", failing_download)

    exit_code = main(["data", "download", "seg_c3_na"])

    assert exit_code == 1
    assert "Download failed: manifest is unreadable" in capsys.readouterr().err


def test_data_parser_exposes_the_eight_subcommands(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["data", "--help"])

    assert excinfo.value.code == 0
    help_text = capsys.readouterr().out
    for name in (
        "download",
        "verify",
        "inspect",
        "prepare-c3-shot",
        "prepare-c3-survey",
        "prepare-baseline",
        "prepare-mask",
        "prepare-benchmark-case",
    ):
        assert name in help_text


def test_importing_data_commands_defers_preparation_pipelines() -> None:
    probe = (
        "import sys\n"
        "import seis_interp.commands.data\n"
        "eager = [\n"
        "    name\n"
        "    for name in (\n"
        "        'seis_interp.pipelines.prepare_c3',\n"
        "        'seis_interp.pipelines.prepare_baseline',\n"
        "        'seis_interp.pipelines.prepare_interpolation_mask',\n"
        "        'seis_interp.pipelines.prepare_benchmark_case',\n"
        "        'seis_interp.processing.trace_amplitude_filter',\n"
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


MASK_SUMMARY: dict[str, object] = {
    "dataset_id": "synthetic",
    "partition": "test",
    "kind": "random_trace",
    "missing_fraction": 0.8,
    "random_seed": 42,
    "config_source": "studies/study/config.yaml",
    "candidate_trace_count": 10,
    "candidate_ffid_count": 2,
    "input_files": {
        "interim": {
            "traces.parquet": {"sha256": "a" * 64},
            "dataset.json": {"sha256": "b" * 64},
        },
        "processed": {
            "trace_split.parquet": {"sha256": "c" * 64},
            "preparation.json": {"sha256": "d" * 64},
        },
    },
    "counts": {"total": 10, "observed": 2, "evaluation_target": 8},
    "files": {"observation_mask": "observation_mask.parquet"},
}


def _write_mask_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    seed: object = 42,
    partition: object = "test",
    kind: object = "random_trace",
    missing_fraction: object = 0.8,
    study_seed: object | None = None,
) -> Path:
    repository = tmp_path / "repository"
    config_path = repository / "studies" / "study" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    (repository / "pyproject.toml").write_text("[project]\nname = 'test'\n", encoding="utf-8")
    payload: dict[str, object] = {
        "project": {"random_seed": seed},
        "interpolation_mask": {
            "partition": partition,
            "kind": kind,
            "missing_fraction": missing_fraction,
        },
    }
    if study_seed is not None:
        payload["study"] = {"random_seed": study_seed}
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    monkeypatch.setattr(data_commands, "REPOSITORY_ROOT", repository)
    return config_path


def _mask_arguments(tmp_path: Path, config_path: Path) -> list[str]:
    return [
        "data",
        "prepare-mask",
        "--config",
        str(config_path),
        "--input",
        str(tmp_path / "interim"),
        "--processed",
        str(tmp_path / "processed"),
        "--output",
        str(tmp_path / "mask"),
    ]


def test_prepare_mask_requires_all_path_arguments(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["data", "prepare-mask"])

    assert excinfo.value.code == 2
    error = capsys.readouterr().err
    for option in ("--config", "--input", "--processed", "--output"):
        assert option in error


def test_prepare_mask_passes_config_values_to_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_mask_config(tmp_path, monkeypatch)
    received: dict[str, object] = {}

    def fake_prepare_interpolation_mask(**kwargs: Any) -> dict[str, object]:
        received.update(kwargs)
        return MASK_SUMMARY

    monkeypatch.setattr(
        "seis_interp.pipelines.prepare_interpolation_mask.prepare_interpolation_mask",
        fake_prepare_interpolation_mask,
    )

    exit_code = main([*_mask_arguments(tmp_path, config_path), "--overwrite"])

    assert exit_code == 0
    assert received == {
        "interim_dir": tmp_path / "interim",
        "processed_dir": tmp_path / "processed",
        "output_dir": tmp_path / "mask",
        "partition": "test",
        "kind": "random_trace",
        "missing_fraction": 0.8,
        "random_seed": 42,
        "config_source": "studies/study/config.yaml",
        "overwrite": True,
    }


def test_prepare_mask_rejects_study_random_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_mask_config(tmp_path, monkeypatch, study_seed=7)

    exit_code = main(_mask_arguments(tmp_path, config_path))

    assert exit_code == 1
    assert "study.random_seed is not supported" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("overrides", "expected_message"),
    [
        ({"kind": "unknown"}, "interpolation_mask.kind"),
        ({"partition": "excluded"}, "interpolation_mask.partition"),
        ({"missing_fraction": 0.0}, "interpolation_mask.missing_fraction"),
        ({"missing_fraction": 1.0}, "interpolation_mask.missing_fraction"),
        ({"missing_fraction": float("nan")}, "interpolation_mask.missing_fraction"),
        ({"seed": -1}, "project.random_seed"),
    ],
)
def test_prepare_mask_rejects_invalid_config_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    overrides: dict[str, object],
    expected_message: str,
) -> None:
    config_path = _write_mask_config(tmp_path, monkeypatch, **overrides)

    exit_code = main(_mask_arguments(tmp_path, config_path))

    assert exit_code == 1
    assert expected_message in capsys.readouterr().err


def test_prepare_mask_json_output_contains_only_the_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_mask_config(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "seis_interp.pipelines.prepare_interpolation_mask.prepare_interpolation_mask",
        lambda **kwargs: MASK_SUMMARY,
    )

    exit_code = main([*_mask_arguments(tmp_path, config_path), "--json"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert json.loads(captured.out) == MASK_SUMMARY


def test_prepare_mask_human_output_contains_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_mask_config(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "seis_interp.pipelines.prepare_interpolation_mask.prepare_interpolation_mask",
        lambda **kwargs: MASK_SUMMARY,
    )

    exit_code = main(_mask_arguments(tmp_path, config_path))
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Configuration: studies/study/config.yaml" in output
    assert f"Input dataset: {tmp_path / 'interim'}" in output
    assert f"Dataset partition: {tmp_path / 'processed'}" in output
    assert f"Output directory: {tmp_path / 'mask'}" in output
    assert "Mask kind: random_trace" in output
    assert "Partition: test" in output
    assert "Candidate traces: 10" in output
    assert "Observed traces: 2" in output
    assert "Evaluation target traces: 8" in output


def test_prepare_mask_reports_pipeline_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_mask_config(tmp_path, monkeypatch)

    def fail(**kwargs: Any) -> dict[str, object]:
        raise ValueError("invalid partition artifact")

    monkeypatch.setattr(
        "seis_interp.pipelines.prepare_interpolation_mask.prepare_interpolation_mask",
        fail,
    )

    exit_code = main(_mask_arguments(tmp_path, config_path))

    assert exit_code == 1
    assert capsys.readouterr().err == ("data prepare-mask failed: invalid partition artifact\n")


CASE_SUMMARY: dict[str, object] = {
    "case_id": "synthetic_test_seed42",
    "dataset_id": "synthetic",
    "partition": "test",
    "config_source": "studies/study/config.yaml",
    "role_contract": {
        "domain": "canonical_present_traces",
        "observed_role": "observed",
        "evaluation_target_role": "evaluation_target",
        "evaluation_target_amplitude_use": "scoring_only",
    },
    "mask": {
        "kind": "random_trace",
        "missing_fraction": 0.8,
        "random_seed": 42,
        "candidate_trace_count": 10,
        "candidate_ffid_count": 2,
        "counts": {"total": 10, "observed": 2, "evaluation_target": 8},
        "duplicate_physical_coordinates": {
            "policy": "keep_lowest_array_row",
            "removed_trace_count": 0,
        },
    },
    "input_files": {},
}


def _write_benchmark_case_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    case_id: object = "synthetic_test_seed42",
    include_case_id: bool = True,
) -> Path:
    repository = tmp_path / "repository"
    config_path = repository / "studies" / "study" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    (repository / "pyproject.toml").write_text("[project]\nname = 'test'\n", encoding="utf-8")
    benchmark_case = {"id": case_id} if include_case_id else {}
    config_path.write_text(
        yaml.safe_dump({"benchmark_case": benchmark_case}),
        encoding="utf-8",
    )
    monkeypatch.setattr(data_commands, "REPOSITORY_ROOT", repository)
    return config_path


def _benchmark_case_arguments(tmp_path: Path, config_path: Path) -> list[str]:
    return [
        "data",
        "prepare-benchmark-case",
        "--config",
        str(config_path),
        "--input",
        str(tmp_path / "interim"),
        "--processed",
        str(tmp_path / "processed"),
        "--mask",
        str(tmp_path / "mask"),
        "--output",
        str(tmp_path / "case"),
    ]


def test_prepare_benchmark_case_requires_all_path_arguments(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["data", "prepare-benchmark-case"])

    assert excinfo.value.code == 2
    error = capsys.readouterr().err
    for option in ("--config", "--input", "--processed", "--mask", "--output"):
        assert option in error


def test_prepare_benchmark_case_passes_config_and_paths_to_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_benchmark_case_config(tmp_path, monkeypatch)
    received: dict[str, object] = {}

    def fake_prepare_benchmark_case(**kwargs: Any) -> dict[str, object]:
        received.update(kwargs)
        return CASE_SUMMARY

    monkeypatch.setattr(
        "seis_interp.pipelines.prepare_benchmark_case.prepare_benchmark_case",
        fake_prepare_benchmark_case,
    )

    exit_code = main([*_benchmark_case_arguments(tmp_path, config_path), "--overwrite"])

    assert exit_code == 0
    assert received == {
        "interim_dir": tmp_path / "interim",
        "processed_dir": tmp_path / "processed",
        "mask_dir": tmp_path / "mask",
        "output_dir": tmp_path / "case",
        "case_id": "synthetic_test_seed42",
        "config_source": "studies/study/config.yaml",
        "overwrite": True,
    }


@pytest.mark.parametrize(
    ("case_id", "include_case_id", "message"),
    [
        ("", True, "case_id"),
        ("bad/id", True, "case_id"),
        (None, False, "benchmark_case.id"),
    ],
)
def test_prepare_benchmark_case_rejects_invalid_or_missing_case_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    case_id: object,
    include_case_id: bool,
    message: str,
) -> None:
    config_path = _write_benchmark_case_config(
        tmp_path,
        monkeypatch,
        case_id=case_id,
        include_case_id=include_case_id,
    )

    exit_code = main(_benchmark_case_arguments(tmp_path, config_path))

    assert exit_code == 1
    assert message in capsys.readouterr().err


def test_prepare_benchmark_case_reports_pipeline_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_benchmark_case_config(tmp_path, monkeypatch)

    def fail(**kwargs: Any) -> dict[str, object]:
        raise ValueError("mask does not match partition")

    monkeypatch.setattr(
        "seis_interp.pipelines.prepare_benchmark_case.prepare_benchmark_case",
        fail,
    )

    exit_code = main(_benchmark_case_arguments(tmp_path, config_path))

    assert exit_code == 1
    assert capsys.readouterr().err == (
        "data prepare-benchmark-case failed: mask does not match partition\n"
    )


def test_prepare_benchmark_case_json_output_contains_only_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_benchmark_case_config(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "seis_interp.pipelines.prepare_benchmark_case.prepare_benchmark_case",
        lambda **kwargs: CASE_SUMMARY,
    )

    exit_code = main([*_benchmark_case_arguments(tmp_path, config_path), "--json"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert json.loads(captured.out) == CASE_SUMMARY


def test_prepare_benchmark_case_human_output_contains_contract_and_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_benchmark_case_config(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "seis_interp.pipelines.prepare_benchmark_case.prepare_benchmark_case",
        lambda **kwargs: CASE_SUMMARY,
    )

    exit_code = main(_benchmark_case_arguments(tmp_path, config_path))
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Configuration: studies/study/config.yaml" in output
    assert "Case ID: synthetic_test_seed42" in output
    assert "Dataset ID: synthetic" in output
    assert "Partition: test" in output
    assert "Mask kind: random_trace" in output
    assert "Observed traces: 2" in output
    assert "Evaluation target traces: 8" in output
    assert f"Input dataset: {tmp_path / 'interim'}" in output
    assert f"Prepared partition: {tmp_path / 'processed'}" in output
    assert f"Mask artifact: {tmp_path / 'mask'}" in output
    assert f"Output directory: {tmp_path / 'case'}" in output


@pytest.mark.parametrize(
    "option",
    ["--case-id", "--partition", "--kind", "--missing-fraction", "--random-seed"],
)
def test_prepare_benchmark_case_has_no_mask_condition_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    option: str,
) -> None:
    config_path = _write_benchmark_case_config(tmp_path, monkeypatch)

    with pytest.raises(SystemExit) as excinfo:
        main([*_benchmark_case_arguments(tmp_path, config_path), option, "value"])

    assert excinfo.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


def test_prepare_benchmark_case_help_describes_exact_hash_binding(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["data", "prepare-benchmark-case", "--help"])

    assert excinfo.value.code == 0
    assert "exact file hashes" in capsys.readouterr().out
