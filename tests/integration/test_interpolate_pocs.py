from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import yaml

from seis_interp import run_records
from seis_interp.cli import main
from seis_interp.data.c3_volume_adapter import load_observed_c3_volume
from seis_interp.data.c3_volume_index_store import load_c3_volume_index
from seis_interp.data.interpolation_mask_store import load_interpolation_mask
from seis_interp.pipelines.interpolate_pocs import (
    METHOD,
    PREDICTION_RELATIVE_PATH,
    interpolate_pocs_run,
)
from seis_interp.pipelines.prepare_baseline import prepare_baseline_dataset
from seis_interp.pipelines.prepare_benchmark_case import prepare_benchmark_case
from seis_interp.pipelines.prepare_c3_volume_index import prepare_c3_volume_index
from seis_interp.pipelines.prepare_interpolation_mask import prepare_interpolation_mask
from seis_interp.processing.interpolation_masks import (
    EVALUATION_TARGET_ROLE,
    OBSERVATION_ROLE_COLUMN,
    RANDOM_TRACE_MASK_KIND,
    RANDOM_WHOLE_FFID_MASK_KIND,
)
from seis_interp.processing.trace_splits import C3_SOURCE_LINE_BLOCKS_SPLIT_SCOPE, TEST_SPLIT
from tests.fixtures.c3_volume_artifacts import SOURCE_LINE_RANGES, prepare_c3_volume_artifacts


@dataclass(frozen=True)
class _PocsArtifacts:
    interim: Path
    processed: Path
    mask: Path
    case: Path
    volume: Path
    volume_metadata: dict[str, object]
    mask_kind: str


def _prepare_pocs_artifacts(
    tmp_path: Path,
    *,
    mask_kind: str = RANDOM_TRACE_MASK_KIND,
    target_offset: float = 0.0,
    time_sample_count: int = 4,
) -> _PocsArtifacts:
    tmp_path.mkdir(parents=True, exist_ok=True)
    prepared = prepare_c3_volume_artifacts(
        tmp_path,
        mask_kind=mask_kind,
        missing_fraction=0.5,
        random_seed=42,
    )
    case_dir = prepared.case_dir
    if target_offset:
        mask_table, _ = load_interpolation_mask(prepared.mask_dir)
        target_rows = mask_table.loc[
            mask_table[OBSERVATION_ROLE_COLUMN].eq(EVALUATION_TARGET_ROLE),
            "array_row",
        ].to_numpy(dtype=np.int64)
        amplitude_path = prepared.interim_dir / "amplitudes.npy"
        amplitudes = np.load(amplitude_path, allow_pickle=False)
        changed = np.array(amplitudes, copy=True)
        changed[target_rows] += target_offset
        np.save(amplitude_path, changed, allow_pickle=False)
        prepare_baseline_dataset(
            prepared.interim_dir,
            prepared.processed_dir,
            holdout_fraction=None,
            validation_fraction_of_holdout=None,
            random_seed=42,
            split_scope=C3_SOURCE_LINE_BLOCKS_SPLIT_SCOPE,
            source_line_ranges=SOURCE_LINE_RANGES,
            config_source="studies/synthetic/config.yaml",
            overwrite=True,
        )
        prepare_interpolation_mask(
            prepared.interim_dir,
            prepared.processed_dir,
            prepared.mask_dir,
            partition=TEST_SPLIT,
            kind=mask_kind,
            missing_fraction=0.5,
            random_seed=42,
            config_source="studies/synthetic/config.yaml",
            overwrite=True,
        )
        case_dir = prepared.processed_dir / "cases" / "synthetic-case-adjusted"
        prepare_benchmark_case(
            prepared.interim_dir,
            prepared.processed_dir,
            prepared.mask_dir,
            case_dir,
            case_id="synthetic_case",
            config_source="studies/synthetic/config.yaml",
        )

    volume_dir = prepared.processed_dir / "volumes" / "synthetic-volume"
    volume_metadata = prepare_c3_volume_index(
        prepared.interim_dir,
        prepared.processed_dir,
        prepared.mask_dir,
        case_dir,
        volume_dir,
        volume_id="synthetic_volume",
        time_range=(0, time_sample_count),
        source_line_range=(2, 4),
        shot_in_line_range=(0, 3),
        relative_receiver_x_range=(0, 2),
        relative_receiver_y_range=(0, 3),
        config_source="studies/synthetic/config.yaml",
    )
    return _PocsArtifacts(
        interim=prepared.interim_dir,
        processed=prepared.processed_dir,
        mask=prepared.mask_dir,
        case=case_dir,
        volume=volume_dir,
        volume_metadata=volume_metadata,
        mask_kind=mask_kind,
    )


def _write_config(
    path: Path,
    artifacts: _PocsArtifacts,
    *,
    windowed: bool = False,
    changes: dict[str, object] | None = None,
) -> Path:
    pocs: dict[str, object] = {
        "n_iterations": 3,
        "threshold_start": 1.0,
        "threshold_end": 0.1,
        "window_shape": [3, 1, 2, 2, 2] if windowed else None,
        "overlap": [1, 0, 1, 1, 1] if windowed else None,
    }
    if changes:
        pocs.update(changes)
    case_mask = json.loads((artifacts.case / "benchmark_case.json").read_text())["mask"]
    config = {
        "project": {"random_seed": 42},
        "interpolation_mask": {
            "partition": "test",
            "kind": artifacts.mask_kind,
            "missing_fraction": case_mask["missing_fraction"],
        },
        "benchmark_case": {"id": "synthetic_case"},
        "benchmark_volume": {
            "id": "synthetic_volume",
            "selection": artifacts.volume_metadata["selection"],
        },
        "pocs": pocs,
        "evaluation": {
            "primary_metric": "physical_amplitude_global_snr_db",
            "domain": "evaluation_target",
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _run(artifacts: _PocsArtifacts, config: Path, output: Path) -> dict[str, object]:
    return interpolate_pocs_run(
        config_path=config,
        interim_dir=artifacts.interim,
        processed_dir=artifacts.processed,
        mask_dir=artifacts.mask,
        case_dir=artifacts.case,
        volume_dir=artifacts.volume,
        output_dir=output,
    )


@pytest.mark.parametrize("windowed", [False, True])
@pytest.mark.parametrize("time_sample_count", [3, 4])
def test_run_writes_prediction_metrics_and_complete_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    windowed: bool,
    time_sample_count: int,
) -> None:
    git_metadata = {"git_commit": "a" * 40, "git_worktree_dirty": windowed}
    monkeypatch.setattr(run_records, "current_git_metadata", lambda: dict(git_metadata))
    artifacts = _prepare_pocs_artifacts(tmp_path, time_sample_count=time_sample_count)
    config = _write_config(tmp_path / "config.yaml", artifacts, windowed=windowed)
    output = tmp_path / "run"
    progress: list[str] = []

    metrics = interpolate_pocs_run(
        config_path=config,
        interim_dir=artifacts.interim,
        processed_dir=artifacts.processed,
        mask_dir=artifacts.mask,
        case_dir=artifacts.case,
        volume_dir=artifacts.volume,
        output_dir=output,
        progress_reporter=progress.append,
    )

    output_files = sorted(
        path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()
    )
    assert output_files == [
        "artifacts/prediction.npy",
        "config.resolved.yaml",
        "inputs.lock.json",
        "metrics.json",
        "run.json",
    ]
    stored_metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    inputs_lock = json.loads((output / "inputs.lock.json").read_text(encoding="utf-8"))
    prediction = np.load(output / PREDICTION_RELATIVE_PATH, allow_pickle=False)
    observed = load_observed_c3_volume(
        interim_dir=artifacts.interim,
        processed_dir=artifacts.processed,
        mask_dir=artifacts.mask,
        case_dir=artifacts.case,
        volume_dir=artifacts.volume,
    )

    assert metrics == stored_metrics
    assert metrics["method"] == METHOD
    assert metrics["warnings"] == []
    assert prediction.shape == observed.values.shape
    assert prediction.dtype == observed.values.dtype
    assert np.all(np.isfinite(prediction))
    assert np.array_equal(
        prediction[:, observed.observed_trace_mask],
        observed.values[:, observed.observed_trace_mask],
    )
    assert run["device"] == "cpu"
    assert run["python_version"]
    assert run["numpy_version"] == np.__version__
    assert run["random_seed"] == 42
    assert run["git_commit"] == git_metadata["git_commit"]
    assert run["git_worktree_dirty"] is git_metadata["git_worktree_dirty"]
    assert run["pocs"]["frequency_bins"] == "all_rfft_bins"
    assert run["prediction"]["axis_order"] == list(artifacts.volume_metadata["axis_order"])
    assert run["window"]["block_count"] >= 1
    assert run["window"]["requested_shape"] == ([3, 1, 2, 2, 2] if windowed else None)
    assert inputs_lock["benchmark_case"]["input_files"]
    assert set(inputs_lock["benchmark_volume"]["files"]) == {
        "volume_index.parquet",
        "volume.json",
    }
    assert len(progress) == 4


def test_random_whole_ffid_mask_runs_and_preserves_observations(tmp_path: Path) -> None:
    artifacts = _prepare_pocs_artifacts(tmp_path, mask_kind=RANDOM_WHOLE_FFID_MASK_KIND)
    config = _write_config(tmp_path / "config.yaml", artifacts)

    metrics = _run(artifacts, config, tmp_path / "run")

    assert metrics["evaluation_target"]["trace_count"] > 0
    prediction = np.load(tmp_path / "run" / PREDICTION_RELATIVE_PATH, allow_pickle=False)
    observed = load_observed_c3_volume(
        interim_dir=artifacts.interim,
        processed_dir=artifacts.processed,
        mask_dir=artifacts.mask,
        case_dir=artifacts.case,
        volume_dir=artifacts.volume,
    )
    assert np.array_equal(
        prediction[:, observed.observed_trace_mask],
        observed.values[:, observed.observed_trace_mask],
    )


def test_empty_windows_remain_zero_are_evaluated_and_report_a_warning(tmp_path: Path) -> None:
    artifacts = _prepare_pocs_artifacts(tmp_path, mask_kind=RANDOM_WHOLE_FFID_MASK_KIND)
    config = _write_config(
        tmp_path / "config.yaml",
        artifacts,
        changes={
            "window_shape": [4, 1, 1, 2, 3],
            "overlap": [0, 0, 0, 0, 0],
        },
    )
    output = tmp_path / "run"

    metrics = _run(artifacts, config, output)

    prediction = np.load(output / PREDICTION_RELATIVE_PATH, allow_pickle=False)
    observed = load_observed_c3_volume(
        interim_dir=artifacts.interim,
        processed_dir=artifacts.processed,
        mask_dir=artifacts.mask,
        case_dir=artifacts.case,
        volume_dir=artifacts.volume,
    )
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    expected_uncovered = int(
        observed.values.shape[0] * np.count_nonzero(observed.evaluation_target_trace_mask)
    )
    target = metrics["evaluation_target"]
    zero_fill = metrics["zero_fill"]
    assert np.all(prediction[:, observed.evaluation_target_trace_mask] == 0.0)
    assert target["error_energy"] == target["reference_energy"]
    assert target["relative_l2"] == 1.0
    assert target["snr_db"] == zero_fill["snr_db"]
    assert target["rmse"] == zero_fill["rmse"]
    assert metrics["uncovered_sample_count"] == expected_uncovered
    assert len(metrics["warnings"]) == 1
    assert run["window"]["empty_block_count"] > 0
    assert run["window"]["uncovered_sample_count"] == expected_uncovered


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"extra": 1}, "unexpected"),
        ({"n_iterations": 1}, "at least 2"),
        ({"threshold_start": 1.1}, "at most 1"),
        ({"threshold_end": 0.0}, "positive"),
        ({"overlap": [3, 0, 1, 1, 1], "window_shape": [3, 1, 2, 2, 2]}, "overlap"),
    ],
)
def test_invalid_pocs_configuration_creates_no_output(
    tmp_path: Path,
    changes: dict[str, object],
    match: str,
) -> None:
    artifacts = _prepare_pocs_artifacts(tmp_path)
    config = _write_config(tmp_path / "config.yaml", artifacts, changes=changes)
    output = tmp_path / "run"

    with pytest.raises(ValueError, match=match):
        _run(artifacts, config, output)

    assert not output.exists()


@pytest.mark.parametrize(
    ("key", "bad_value"),
    [("domain", "all_traces"), ("primary_metric", "correlation")],
)
def test_invalid_evaluation_contract_creates_no_output(
    tmp_path: Path,
    key: str,
    bad_value: str,
) -> None:
    artifacts = _prepare_pocs_artifacts(tmp_path)
    config_path = _write_config(tmp_path / "config.yaml", artifacts)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["evaluation"][key] = bad_value
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    output = tmp_path / "run"

    with pytest.raises(ValueError, match=f"evaluation.{key}"):
        _run(artifacts, config_path, output)

    assert not output.exists()


def test_declared_volume_mismatch_and_broken_binding_create_no_output(tmp_path: Path) -> None:
    artifacts = _prepare_pocs_artifacts(tmp_path)
    config = _write_config(tmp_path / "config.yaml", artifacts)
    text = config.read_text(encoding="utf-8").replace("synthetic_volume", "wrong_volume", 1)
    config.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="benchmark_volume.id"):
        _run(artifacts, config, tmp_path / "mismatch-run")
    assert not (tmp_path / "mismatch-run").exists()

    valid_config = _write_config(tmp_path / "valid.yaml", artifacts)
    amplitudes = np.load(artifacts.interim / "amplitudes.npy", allow_pickle=False)
    changed = np.array(amplitudes, copy=True)
    changed[0, 0] += 1.0
    np.save(artifacts.interim / "amplitudes.npy", changed, allow_pickle=False)
    with pytest.raises(ValueError, match="input_files"):
        _run(artifacts, valid_config, tmp_path / "binding-run")
    assert not (tmp_path / "binding-run").exists()


@pytest.mark.parametrize(
    ("keys", "bad_value", "match"),
    [
        (("benchmark_case", "id"), "wrong_case", "benchmark_case.id"),
        (("benchmark_volume", "id"), "wrong_volume", "benchmark_volume.id"),
        (
            ("benchmark_volume", "selection", "time"),
            [1, 5],
            "benchmark_volume.selection",
        ),
        (("interpolation_mask", "partition"), "validation", "interpolation_mask.partition"),
        (("interpolation_mask", "kind"), "random_whole_ffid", "interpolation_mask.kind"),
        (("interpolation_mask", "missing_fraction"), 0.25, "missing_fraction"),
        (("project", "random_seed"), 7, "project.random_seed"),
    ],
)
def test_declared_input_contradictions_create_no_output(
    tmp_path: Path,
    keys: tuple[str, ...],
    bad_value: object,
    match: str,
) -> None:
    artifacts = _prepare_pocs_artifacts(tmp_path)
    base_config = _write_config(tmp_path / "base.yaml", artifacts)
    config = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    changed = deepcopy(config)
    parent = changed
    for key in keys[:-1]:
        parent = parent[key]
    parent[keys[-1]] = bad_value
    variant = tmp_path / "variant.yaml"
    variant.write_text(yaml.safe_dump(changed, sort_keys=False), encoding="utf-8")
    output = tmp_path / "run"

    with pytest.raises(ValueError, match=match):
        _run(artifacts, variant, output)

    assert not output.exists()


def test_existing_output_is_immutable(tmp_path: Path) -> None:
    artifacts = _prepare_pocs_artifacts(tmp_path)
    config = _write_config(tmp_path / "config.yaml", artifacts)
    output = tmp_path / "run"
    output.mkdir()
    marker = output / "marker.txt"
    marker.write_text("unchanged", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        _run(artifacts, config, output)

    assert marker.read_text(encoding="utf-8") == "unchanged"


def test_prediction_is_reproducible_and_target_amplitudes_do_not_leak(tmp_path: Path) -> None:
    first = _prepare_pocs_artifacts(tmp_path / "first")
    changed_truth = _prepare_pocs_artifacts(tmp_path / "changed", target_offset=5000.0)
    first_config = _write_config(tmp_path / "first.yaml", first)
    changed_config = _write_config(tmp_path / "changed.yaml", changed_truth)

    first_metrics = _run(first, first_config, tmp_path / "first-run")
    repeated_metrics = _run(first, first_config, tmp_path / "repeated-run")
    changed_metrics = _run(changed_truth, changed_config, tmp_path / "changed-run")
    first_prediction = np.load(tmp_path / "first-run" / PREDICTION_RELATIVE_PATH)
    repeated_prediction = np.load(tmp_path / "repeated-run" / PREDICTION_RELATIVE_PATH)
    changed_prediction = np.load(tmp_path / "changed-run" / PREDICTION_RELATIVE_PATH)

    assert np.array_equal(first_prediction, repeated_prediction)
    assert np.array_equal(first_prediction, changed_prediction)
    assert first_metrics["evaluation_target"] == repeated_metrics["evaluation_target"]
    assert first_metrics["evaluation_target"] != changed_metrics["evaluation_target"]


def test_importing_cpu_pipeline_does_not_import_torch() -> None:
    code = """
import builtins
import sys
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise RuntimeError('torch import attempted')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import seis_interp.pipelines.interpolate_pocs
assert 'torch' not in sys.modules
"""

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_real_cli_runs_real_pipeline_end_to_end(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifacts = _prepare_pocs_artifacts(tmp_path)
    config = _write_config(tmp_path / "config.yaml", artifacts)
    output = tmp_path / "cli-run"

    exit_code = main(
        [
            "interpolate",
            "pocs",
            "--config",
            str(config),
            "--interim",
            str(artifacts.interim),
            "--processed",
            str(artifacts.processed),
            "--mask",
            str(artifacts.mask),
            "--case",
            str(artifacts.case),
            "--volume",
            str(artifacts.volume),
            "--output",
            str(output),
            "--json",
        ]
    )

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert exit_code == 0
    assert summary == json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert (output / PREDICTION_RELATIVE_PATH).is_file()
    assert "Loading and verifying C3 inputs" in captured.err


def test_saved_volume_metadata_still_matches_loaded_artifact(tmp_path: Path) -> None:
    artifacts = _prepare_pocs_artifacts(tmp_path)

    _, loaded = load_c3_volume_index(artifacts.volume)

    assert loaded == artifacts.volume_metadata
