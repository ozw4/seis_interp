from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

import seis_interp.pipelines.interpolate_pocs as pocs_pipeline
import seis_interp.training.amplitude_scaling as amplitude_scaling
from seis_interp import run_records
from seis_interp.cli import main
from seis_interp.data.c3_poc_inputs import (
    C3_RANDOM80_POC_BENCHMARK_ID,
    C3_RANDOM80_POC_DATASET_ID,
    load_c3_random80_poc_inputs,
)
from seis_interp.data.c3_volume_adapter import load_observed_c3_volume
from seis_interp.data.c3_volume_index_store import load_c3_volume_index
from seis_interp.pipelines.interpolate_pocs import (
    METHOD,
    PREDICTION_RELATIVE_PATH,
    interpolate_pocs_run,
)
from seis_interp.processing.c3_benchmark_contract import C3BenchmarkDimensions
from seis_interp.processing.interpolation_masks import (
    RANDOM_TRACE_MASK_KIND,
    RANDOM_WHOLE_FFID_MASK_KIND,
)
from seis_interp.processing.pocs_windows import WindowedPocsResult
from tests.fixtures.c3_volume_run_artifacts import (
    PreparedC3VolumeRunArtifacts,
    prepare_c3_volume_run_artifacts,
)


@pytest.fixture(autouse=True)
def _use_synthetic_poc_dimensions(monkeypatch: pytest.MonkeyPatch) -> None:
    def load_synthetic_poc_inputs(**kwargs: object) -> object:
        volume_dir = kwargs["volume_dir"]
        assert isinstance(volume_dir, Path)
        _, metadata = load_c3_volume_index(volume_dir)
        selection = metadata["selection"]
        assert isinstance(selection, dict)
        source_start, source_stop = selection["source_line"]
        dimensions = C3BenchmarkDimensions(
            time_range=tuple(selection["time"]),
            sail_line_numbers=(source_start, source_stop - 1),
            shape=tuple(metadata["shape"]),
        )
        return load_c3_random80_poc_inputs(**kwargs, dimensions=dimensions)

    monkeypatch.setattr(
        pocs_pipeline,
        "load_c3_random80_poc_inputs",
        load_synthetic_poc_inputs,
    )


def _prepare_poc_artifacts(
    tmp_path: Path,
    *,
    mask_kind: str = RANDOM_TRACE_MASK_KIND,
    target_offset: float = 0.0,
    time_sample_count: int = 4,
) -> PreparedC3VolumeRunArtifacts:
    return prepare_c3_volume_run_artifacts(
        tmp_path,
        dataset_id=C3_RANDOM80_POC_DATASET_ID,
        mask_kind=mask_kind,
        missing_fraction=0.8,
        target_offset=target_offset,
        time_sample_count=time_sample_count,
    )


def _write_config(
    path: Path,
    artifacts: PreparedC3VolumeRunArtifacts,
    *,
    windowed: bool = False,
    changes: dict[str, object] | None = None,
) -> Path:
    pocs: dict[str, object] = {
        "n_iterations": 3,
        "threshold_start": 1.0,
        "threshold_end": 0.1,
        "window_shape": [3, 2, 3, 2, 3] if windowed else None,
        "overlap": [1, 0, 0, 0, 0] if windowed else None,
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


def _run(
    artifacts: PreparedC3VolumeRunArtifacts,
    config: Path,
    output: Path,
) -> dict[str, object]:
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
    artifacts = _prepare_poc_artifacts(tmp_path, time_sample_count=time_sample_count)
    config = _write_config(tmp_path / "config.yaml", artifacts, windowed=windowed)
    output = tmp_path / "run"
    progress: list[str] = []
    timestamps: list[str] = []

    def recorded_timestamp() -> str:
        if not timestamps:
            assert not output.exists()
            value = "2026-09-07T10:00:00Z"
        else:
            saved_prediction = np.load(output / PREDICTION_RELATIVE_PATH, allow_pickle=False)
            assert np.all(np.isfinite(saved_prediction))
            assert not (output / "metadata.json").exists()
            value = "2026-09-07T10:00:10Z"
        timestamps.append(value)
        return value

    monkeypatch.setattr(run_records, "utc_timestamp", recorded_timestamp)

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
    assert output_files == sorted(
        [
            "prediction.npy",
            "config.resolved.yaml",
            "inputs.lock.json",
            "metrics.json",
            "metadata.json",
        ]
    )
    stored_metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    run = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    inputs_lock = json.loads((output / "inputs.lock.json").read_text(encoding="utf-8"))
    prediction = np.load(output / PREDICTION_RELATIVE_PATH, allow_pickle=False)
    observed = load_observed_c3_volume(
        interim_dir=artifacts.interim,
        processed_dir=artifacts.processed,
        mask_dir=artifacts.mask,
        case_dir=artifacts.case,
        volume_dir=artifacts.volume,
    )

    assert stored_metrics == {key: metrics[key] for key in stored_metrics}
    assert set(stored_metrics) == {
        "evaluation_domain",
        "amplitude_domain",
        "evaluation_target",
        "zero_fill",
        "observed_max_abs_error",
    }
    assert metrics["method"] == METHOD
    assert metrics["warnings"] == []
    assert {
        "snr_db",
        "rmse",
        "relative_l2",
        "mean_trace_relative_mse",
    } <= metrics["evaluation_target"].keys()
    assert (
        metrics["evaluation_target"]["covered_target_trace_count"]
        == metrics["evaluation_target"]["target_trace_count"]
    )
    assert prediction.shape == observed.values.shape
    assert prediction.dtype == observed.values.dtype
    assert np.all(np.isfinite(prediction))
    assert np.array_equal(
        prediction[:, observed.observed_trace_mask],
        observed.values[:, observed.observed_trace_mask],
    )
    assert run["method_details"]["device"] == "cpu"
    assert run["method_details"]["python_version"]
    assert run["method_details"]["numpy_version"] == np.__version__
    assert run["method_details"]["random_seed"] == 42
    assert run["method_details"]["git_commit"] == git_metadata["git_commit"]
    assert run["method_details"]["git_worktree_dirty"] is git_metadata["git_worktree_dirty"]
    assert run["method"] == "pocs"
    assert run["input_amplitude_domain"] == "physical"
    assert run["output_amplitude_domain"] == "physical"
    assert run["method_details"]["benchmark_global_rms_normalization"] is False
    assert run["method_details"]["observed_data_constraint"] is True
    assert run["method_details"]["native_iterative_updates"] is True
    assert run["status"] == "success"
    assert timestamps == ["2026-09-07T10:00:00Z", "2026-09-07T10:00:10Z"]
    assert run["method_details"]["started_at_utc"] == timestamps[0]
    assert run["method_details"]["finished_at_utc"] == timestamps[1]
    assert run["method_details"]["pocs"]["frequency_bins"] == "all_rfft_bins"
    assert run["method_details"]["pocs"]["n_iterations"] == 3
    assert run["method_details"]["prediction"]["axis_order"] == list(
        artifacts.volume_metadata["axis_order"]
    )
    assert run["method_details"]["window"]["block_count"] >= 1
    assert run["method_details"]["window"]["requested_shape"] == (
        [3, 2, 3, 2, 3] if windowed else None
    )
    assert run["coverage"] == {
        "target_trace_count": int(observed.evaluation_target_trace_mask.sum()),
        "covered_target_trace_count": int(observed.evaluation_target_trace_mask.sum()),
        "target_coverage_fraction": 1.0,
        "uncovered_sample_count": 0,
    }
    assert run["timing"]["reconstruction_seconds"] >= 0.0
    assert run["timing"]["end_to_end_seconds"] >= run["timing"]["reconstruction_seconds"]
    case_path = artifacts.case / "benchmark_case.json"
    case = json.loads(case_path.read_text(encoding="utf-8"))
    assert inputs_lock == {
        "benchmark_case": {
            "case_id": "synthetic_case",
            "file": "benchmark_case.json",
            "sha256": hashlib.sha256(case_path.read_bytes()).hexdigest(),
            "input_files": json.loads(case_path.read_text(encoding="utf-8"))["input_files"],
        },
        "benchmark_volume": {
            "volume_id": "synthetic_volume",
            "files": {
                name: {"sha256": hashlib.sha256((artifacts.volume / name).read_bytes()).hexdigest()}
                for name in ("volume_index.parquet", "volume.json")
            },
        },
        "benchmark_id": C3_RANDOM80_POC_BENCHMARK_ID,
        "case_id": "synthetic_case",
        "dataset_id": C3_RANDOM80_POC_DATASET_ID,
        "mask": {
            "kind": RANDOM_TRACE_MASK_KIND,
            "missing_fraction": 0.8,
            "random_seed": 42,
            "unit": "complete_trace",
            "files": case["input_files"]["mask"],
        },
        "volume_id": "synthetic_volume",
        "selection": artifacts.volume_metadata["selection"],
        "shape": artifacts.volume_metadata["shape"],
        "observed_trace_count": int(observed.observed_trace_mask.sum()),
        "target_trace_count": int(observed.evaluation_target_trace_mask.sum()),
    }
    assert len(progress) == 4


def test_non_poc_mask_contract_is_rejected_before_output(tmp_path: Path) -> None:
    artifacts = _prepare_poc_artifacts(tmp_path, mask_kind=RANDOM_WHOLE_FFID_MASK_KIND)
    config = _write_config(tmp_path / "config.yaml", artifacts)
    output = tmp_path / "run"

    with pytest.raises(ValueError, match="PoC mask kind"):
        _run(artifacts, config, output)

    assert not output.exists()


def test_pocs_core_receives_unchanged_physical_observations_without_global_rms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = _prepare_poc_artifacts(tmp_path)
    config = _write_config(tmp_path / "config.yaml", artifacts)
    observed = load_observed_c3_volume(
        interim_dir=artifacts.interim,
        processed_dir=artifacts.processed,
        mask_dir=artifacts.mask,
        case_dir=artifacts.case,
        volume_dir=artifacts.volume,
    )
    original_interpolate = pocs_pipeline.interpolate_pocs_volume
    received_core_input: list[np.ndarray] = []

    def recorded_interpolate(
        values: np.ndarray,
        observed_trace_mask: np.ndarray,
        **kwargs: object,
    ) -> object:
        received_core_input.append(np.array(values, copy=True))
        assert np.array_equal(values, observed.values)
        assert np.array_equal(observed_trace_mask, observed.observed_trace_mask)
        return original_interpolate(values, observed_trace_mask, **kwargs)

    def unexpected_global_rms(*args: object, **kwargs: object) -> float:
        raise AssertionError("benchmark Global RMS must not be computed by POCS")

    monkeypatch.setattr(pocs_pipeline, "interpolate_pocs_volume", recorded_interpolate)
    monkeypatch.setattr(
        amplitude_scaling,
        "compute_observed_global_rms",
        unexpected_global_rms,
    )

    _run(artifacts, config, tmp_path / "run")

    assert len(received_core_input) == 1


def test_incomplete_window_coverage_is_rejected_before_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = _prepare_poc_artifacts(tmp_path)
    config = _write_config(
        tmp_path / "config.yaml",
        artifacts,
        changes={
            "window_shape": [4, 1, 1, 1, 1],
            "overlap": [0, 0, 0, 0, 0],
        },
    )
    output = tmp_path / "run"
    evaluation_called = False

    def unexpected_evaluation(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal evaluation_called
        evaluation_called = True
        raise AssertionError("evaluation must not run for an incomplete prediction")

    monkeypatch.setattr(
        pocs_pipeline,
        "evaluate_c3_volume_prediction",
        unexpected_evaluation,
    )

    with pytest.raises(ValueError, match="cover every target sample"):
        _run(artifacts, config, output)

    assert evaluation_called is False
    assert not output.exists()


@pytest.mark.parametrize(
    ("invalid_value", "match"),
    [("nonfinite_target", "non-finite"), ("changed_observation", "preserve observed")],
)
def test_invalid_pocs_output_is_rejected_before_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_value: str,
    match: str,
) -> None:
    artifacts = _prepare_poc_artifacts(tmp_path)
    config = _write_config(tmp_path / "config.yaml", artifacts)
    observed = load_observed_c3_volume(
        interim_dir=artifacts.interim,
        processed_dir=artifacts.processed,
        mask_dir=artifacts.mask,
        case_dir=artifacts.case,
        volume_dir=artifacts.volume,
    )
    original_interpolate = pocs_pipeline.interpolate_pocs_volume

    def invalid_interpolate(*args: object, **kwargs: object) -> WindowedPocsResult:
        result = original_interpolate(*args, **kwargs)
        values = result.values.copy()
        if invalid_value == "nonfinite_target":
            target_index = tuple(np.argwhere(observed.evaluation_target_trace_mask)[0])
            values[(0, *target_index)] = np.nan
        else:
            observed_index = tuple(np.argwhere(observed.observed_trace_mask)[0])
            values[(0, *observed_index)] += 1.0
        return WindowedPocsResult(
            values,
            result.block_count,
            result.empty_block_count,
            result.uncovered_sample_count,
        )

    monkeypatch.setattr(pocs_pipeline, "interpolate_pocs_volume", invalid_interpolate)
    monkeypatch.setattr(
        pocs_pipeline,
        "evaluate_c3_volume_prediction",
        lambda *args, **kwargs: pytest.fail("invalid output reached target evaluator"),
    )
    output = tmp_path / "run"

    with pytest.raises(ValueError, match=match):
        _run(artifacts, config, output)

    assert not output.exists()


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
    artifacts = _prepare_poc_artifacts(tmp_path)
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
    artifacts = _prepare_poc_artifacts(tmp_path)
    config_path = _write_config(tmp_path / "config.yaml", artifacts)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["evaluation"][key] = bad_value
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    output = tmp_path / "run"

    with pytest.raises(ValueError, match=f"evaluation.{key}"):
        _run(artifacts, config_path, output)

    assert not output.exists()


def test_declared_volume_mismatch_and_broken_binding_create_no_output(tmp_path: Path) -> None:
    artifacts = _prepare_poc_artifacts(tmp_path)
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
    artifacts = _prepare_poc_artifacts(tmp_path)
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
    artifacts = _prepare_poc_artifacts(tmp_path)
    config = _write_config(tmp_path / "config.yaml", artifacts)
    output = tmp_path / "run"
    output.mkdir()
    marker = output / "marker.txt"
    marker.write_text("unchanged", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        _run(artifacts, config, output)

    assert marker.read_text(encoding="utf-8") == "unchanged"


def test_prediction_is_reproducible_and_target_amplitudes_do_not_leak(tmp_path: Path) -> None:
    first = _prepare_poc_artifacts(tmp_path / "first")
    changed_truth = _prepare_poc_artifacts(tmp_path / "changed", target_offset=5000.0)
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
assert 'seis_interp.training.amplitude_scaling' not in sys.modules
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
    artifacts = _prepare_poc_artifacts(tmp_path)
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
    assert (
        summary["evaluation_target"]
        == json.loads((output / "metrics.json").read_text(encoding="utf-8"))["evaluation_target"]
    )
    assert (output / PREDICTION_RELATIVE_PATH).is_file()
    assert "Loading and verifying C3 inputs" in captured.err


def test_saved_volume_metadata_still_matches_loaded_artifact(tmp_path: Path) -> None:
    artifacts = _prepare_poc_artifacts(tmp_path)

    _, loaded = load_c3_volume_index(artifacts.volume)

    assert loaded == artifacts.volume_metadata
