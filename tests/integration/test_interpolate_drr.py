from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from seis_interp import run_records
from seis_interp.cli import main
from seis_interp.data.c3_volume_adapter import load_observed_c3_volume
from seis_interp.pipelines.interpolate_drr import (
    METHOD,
    PREDICTION_RELATIVE_PATH,
    interpolate_drr_run,
)
from seis_interp.processing.interpolation_masks import (
    RANDOM_TRACE_MASK_KIND,
    RANDOM_WHOLE_FFID_MASK_KIND,
)
from tests.fixtures.c3_volume_run_artifacts import (
    PreparedC3VolumeRunArtifacts,
    prepare_c3_volume_run_artifacts,
)


def _write_config(
    path: Path,
    artifacts: PreparedC3VolumeRunArtifacts,
    *,
    windowed: bool = False,
    changes: dict[str, object] | None = None,
) -> Path:
    drr: dict[str, object] = {
        "rank": 1,
        "damping_power": 3,
        "n_iterations": 1,
        "frequency_min_hz": 0.0,
        "frequency_max_hz": None,
        "spatial_window_shape": [2, 2, 2, 3] if windowed else None,
        "spatial_overlap": [0, 1, 0, 0] if windowed else None,
    }
    if changes:
        drr.update(changes)
    case = json.loads((artifacts.case / "benchmark_case.json").read_text(encoding="utf-8"))
    config = {
        "project": {"random_seed": 42},
        "interpolation_mask": {
            "partition": "test",
            "kind": artifacts.mask_kind,
            "missing_fraction": case["mask"]["missing_fraction"],
        },
        "benchmark_case": {"id": "synthetic_case"},
        "benchmark_volume": {
            "id": "synthetic_volume",
            "selection": artifacts.volume_metadata["selection"],
        },
        "drr": drr,
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
    return interpolate_drr_run(
        config_path=config,
        interim_dir=artifacts.interim,
        processed_dir=artifacts.processed,
        mask_dir=artifacts.mask,
        case_dir=artifacts.case,
        volume_dir=artifacts.volume,
        output_dir=output,
    )


@pytest.mark.parametrize(
    ("windowed", "time_sample_count"),
    [(False, 3), (True, 4)],
)
def test_run_writes_prediction_metrics_and_complete_drr_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    windowed: bool,
    time_sample_count: int,
) -> None:
    git_metadata = {"git_commit": "d" * 40, "git_worktree_dirty": windowed}
    monkeypatch.setattr(run_records, "current_git_metadata", lambda: dict(git_metadata))
    artifacts = prepare_c3_volume_run_artifacts(tmp_path, time_sample_count=time_sample_count)
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
            assert not (output / "run.json").exists()
            value = "2026-09-07T10:00:10Z"
        timestamps.append(value)
        return value

    monkeypatch.setattr(run_records, "utc_timestamp", recorded_timestamp)

    metrics = interpolate_drr_run(
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
    json.dumps(stored_metrics, allow_nan=False)
    json.dumps(run, allow_nan=False)
    json.dumps(inputs_lock, allow_nan=False)
    assert metrics["method"] == METHOD
    assert metrics["method_variant"] == "reconstruction_only_hard_consistency"
    assert metrics["evaluation_domain"] == "evaluation_target"
    assert metrics["amplitude_domain"] == "physical"
    assert metrics["evaluation_target"]["trace_count"] == int(
        np.count_nonzero(observed.evaluation_target_trace_mask)
    )
    assert metrics["observed_max_abs_error"] == 0.0
    assert metrics["uncovered_trace_count"] == 0
    assert metrics["uncovered_sample_count"] == 0
    assert metrics["warnings"] == []
    assert prediction.shape == observed.values.shape
    assert prediction.dtype == observed.values.dtype
    assert np.all(np.isfinite(prediction))
    np.testing.assert_array_equal(
        prediction[:, observed.observed_trace_mask],
        observed.values[:, observed.observed_trace_mask],
    )

    assert run["method"] == METHOD
    assert run["method_variant"] == "reconstruction_only_hard_consistency"
    assert run["case_id"] == "synthetic_case"
    assert run["volume_id"] == "synthetic_volume"
    assert run["git_commit"] == git_metadata["git_commit"]
    assert run["git_worktree_dirty"] is git_metadata["git_worktree_dirty"]
    assert run["status"] == "success"
    assert timestamps == ["2026-09-07T10:00:00Z", "2026-09-07T10:00:10Z"]
    assert run["started_at_utc"] == timestamps[0]
    assert run["finished_at_utc"] == timestamps[1]
    assert run["device"] == "cpu"
    assert run["python_version"]
    assert run["numpy_version"] == np.__version__
    assert run["random_seed"] == 42
    assert run["input"]["mask"]["kind"] == RANDOM_TRACE_MASK_KIND
    assert run["input"]["selected_volume"]["selection"] == artifacts.volume_metadata["selection"]
    assert run["drr"] == {
        "mode": "reconstruction_only",
        "rank": 1,
        "damping_power": 3,
        "n_iterations": 1,
        "level_count": 4,
        "embedding_rule": "floor_axis_length_over_2_plus_1",
        "damping_reference": "rank_plus_one_singular_value",
        "svd_backend": "numpy.linalg.svd",
        "full_matrices": False,
        "observed_data_consistency": "hard_reinsertion_each_iteration_and_time_domain",
        "amplitude_normalization": "none",
    }
    assert run["fft"] == {
        "norm": "ortho",
        "padding": "next_power_of_two",
        "internal_dtype": "float64_and_complex128",
        "frequency_min_hz": 0.0,
        "frequency_max_hz": None,
        "sample_interval_s": pytest.approx(0.008),
        "length": 4,
        "nyquist_hz": pytest.approx(62.5),
        "processed_bin_count": 3,
        "first_bin_index": 0,
        "last_bin_index": 2,
        "first_frequency_hz": 0.0,
        "last_frequency_hz": pytest.approx(62.5),
        "outside_band_missing_prediction": "zero_before_time_truncation",
    }
    expected_spatial_shape = [2, 2, 2, 3] if windowed else None
    expected_overlap = [0, 1, 0, 0] if windowed else None
    expected_maximum_shape = [time_sample_count, 2, 2 if windowed else 3, 2, 3]
    expected_hankel_shape = [16, 2 if windowed else 4]
    assert run["window"]["requested_spatial_shape"] == expected_spatial_shape
    assert run["window"]["requested_spatial_overlap"] == expected_overlap
    assert run["window"]["maximum_block_shape"] == expected_maximum_shape
    assert run["window"]["maximum_hankel_matrix_shape"] == expected_hankel_shape
    assert run["window"]["blend"] == "positive_hann_interior_synthesis_only"
    assert run["window"]["block_count"] == (2 if windowed else 1)
    assert run["window"]["empty_block_count"] == 0
    assert run["window"]["uncovered_trace_count"] == 0
    assert run["window"]["uncovered_sample_count"] == 0
    assert run["resources"]["load_and_verification_seconds"] >= 0.0
    assert run["resources"]["reconstruction_seconds"] >= 0.0
    assert run["resources"]["evaluation_seconds"] >= 0.0
    assert run["resources"]["process_max_rss_kib"] > 0
    assert run["prediction"] == {
        "artifact": "artifacts/prediction.npy",
        "axis_order": list(artifacts.volume_metadata["axis_order"]),
        "shape": list(prediction.shape),
        "dtype": prediction.dtype.name,
    }
    assert inputs_lock["benchmark_case"]["case_id"] == "synthetic_case"
    assert inputs_lock["benchmark_volume"]["volume_id"] == "synthetic_volume"
    assert len(progress) == 4


def test_random_whole_ffid_mask_runs_and_preserves_observations(tmp_path: Path) -> None:
    artifacts = prepare_c3_volume_run_artifacts(tmp_path, mask_kind=RANDOM_WHOLE_FFID_MASK_KIND)
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
    np.testing.assert_array_equal(
        prediction[:, observed.observed_trace_mask],
        observed.values[:, observed.observed_trace_mask],
    )


def test_target_truth_does_not_leak_and_repeated_runs_are_numerically_stable(
    tmp_path: Path,
) -> None:
    first = prepare_c3_volume_run_artifacts(tmp_path / "first")
    changed_truth = prepare_c3_volume_run_artifacts(tmp_path / "changed", target_offset=5000.0)
    first_config = _write_config(tmp_path / "first.yaml", first)
    changed_config = _write_config(tmp_path / "changed.yaml", changed_truth)

    first_metrics = _run(first, first_config, tmp_path / "first-run")
    repeated_metrics = _run(first, first_config, tmp_path / "repeated-run")
    changed_metrics = _run(changed_truth, changed_config, tmp_path / "changed-run")
    first_prediction = np.load(tmp_path / "first-run" / PREDICTION_RELATIVE_PATH)
    repeated_prediction = np.load(tmp_path / "repeated-run" / PREDICTION_RELATIVE_PATH)
    changed_prediction = np.load(tmp_path / "changed-run" / PREDICTION_RELATIVE_PATH)

    np.testing.assert_allclose(repeated_prediction, first_prediction, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(changed_prediction, first_prediction, rtol=1e-6, atol=1e-6)
    assert first_metrics["evaluation_target"] == repeated_metrics["evaluation_target"]
    assert first_metrics["evaluation_target"] != changed_metrics["evaluation_target"]


def test_declared_input_contradictions_and_broken_binding_create_no_output(
    tmp_path: Path,
) -> None:
    artifacts = prepare_c3_volume_run_artifacts(tmp_path)
    base_path = _write_config(tmp_path / "base.yaml", artifacts)
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    contradictions = [
        (("benchmark_case", "id"), "wrong_case", "benchmark_case.id"),
        (("benchmark_volume", "id"), "wrong_volume", "benchmark_volume.id"),
        (("benchmark_volume", "selection", "time"), [1, 4], "benchmark_volume.selection"),
        (("interpolation_mask", "partition"), "validation", "interpolation_mask.partition"),
        (("interpolation_mask", "kind"), "random_whole_ffid", "interpolation_mask.kind"),
        (("interpolation_mask", "missing_fraction"), 0.25, "missing_fraction"),
        (("project", "random_seed"), 7, "project.random_seed"),
    ]
    for index, (keys, bad_value, match) in enumerate(contradictions):
        config = deepcopy(base)
        parent = config
        for key in keys[:-1]:
            parent = parent[key]
        parent[keys[-1]] = bad_value
        variant = tmp_path / f"contradiction-{index}.yaml"
        variant.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        output = tmp_path / f"contradiction-run-{index}"

        with pytest.raises(ValueError, match=match):
            _run(artifacts, variant, output)
        assert not output.exists()

    amplitudes = np.load(artifacts.interim / "amplitudes.npy", allow_pickle=False)
    changed = np.array(amplitudes, copy=True)
    changed[0, 0] += 1.0
    np.save(artifacts.interim / "amplitudes.npy", changed, allow_pickle=False)
    binding_output = tmp_path / "binding-run"
    with pytest.raises(ValueError, match="input_files"):
        _run(artifacts, base_path, binding_output)
    assert not binding_output.exists()


def test_invalid_drr_and_evaluation_configurations_create_no_output(tmp_path: Path) -> None:
    artifacts = prepare_c3_volume_run_artifacts(tmp_path)
    base_path = _write_config(tmp_path / "base.yaml", artifacts)
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    variants: list[tuple[dict[str, object], str]] = []

    extra = deepcopy(base)
    extra["drr"]["extra"] = 1
    variants.append((extra, "unexpected"))
    missing = deepcopy(base)
    del missing["drr"]["rank"]
    variants.append((missing, "missing"))
    for key in ("rank", "damping_power", "n_iterations"):
        invalid = deepcopy(base)
        invalid["drr"][key] = 0
        variants.append((invalid, f"drr.{key}"))
    invalid_minimum = deepcopy(base)
    invalid_minimum["drr"]["frequency_min_hz"] = -1.0
    variants.append((invalid_minimum, "drr.frequency_min_hz"))
    invalid_maximum = deepcopy(base)
    invalid_maximum["drr"]["frequency_max_hz"] = 0.0
    variants.append((invalid_maximum, "drr.frequency_max_hz"))
    reversed_range = deepcopy(base)
    reversed_range["drr"]["frequency_min_hz"] = 2.0
    reversed_range["drr"]["frequency_max_hz"] = 1.0
    variants.append((reversed_range, "greater than"))
    bad_shape = deepcopy(base)
    bad_shape["drr"]["spatial_window_shape"] = [2, 2, 2]
    bad_shape["drr"]["spatial_overlap"] = [0, 0, 0, 0]
    variants.append((bad_shape, "four integers"))
    mismatched_window = deepcopy(base)
    mismatched_window["drr"]["spatial_window_shape"] = [2, 2, 2, 3]
    variants.append((mismatched_window, "both be null or lists"))
    bad_overlap = deepcopy(base)
    bad_overlap["drr"]["spatial_window_shape"] = [2, 2, 2, 3]
    bad_overlap["drr"]["spatial_overlap"] = [0, 2, 0, 0]
    variants.append((bad_overlap, "less than"))
    oversized_rank = deepcopy(base)
    oversized_rank["drr"]["rank"] = 4
    variants.append((oversized_rank, "rank"))
    above_nyquist = deepcopy(base)
    above_nyquist["drr"]["frequency_max_hz"] = 100.0
    variants.append((above_nyquist, "Nyquist"))
    bad_evaluation = deepcopy(base)
    bad_evaluation["evaluation"]["domain"] = "all_traces"
    variants.append((bad_evaluation, "evaluation.domain"))
    bad_primary_metric = deepcopy(base)
    bad_primary_metric["evaluation"]["primary_metric"] = "correlation"
    variants.append((bad_primary_metric, "evaluation.primary_metric"))

    for index, (config, match) in enumerate(variants):
        variant = tmp_path / f"invalid-{index}.yaml"
        variant.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        output = tmp_path / f"invalid-run-{index}"

        with pytest.raises(ValueError, match=match):
            _run(artifacts, variant, output)
        assert not output.exists()


def test_existing_output_directory_is_not_modified(tmp_path: Path) -> None:
    artifacts = prepare_c3_volume_run_artifacts(tmp_path)
    config = _write_config(tmp_path / "config.yaml", artifacts)
    output = tmp_path / "run"
    output.mkdir()
    marker = output / "marker.txt"
    marker.write_text("unchanged", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        _run(artifacts, config, output)

    assert marker.read_text(encoding="utf-8") == "unchanged"
    assert list(output.iterdir()) == [marker]


def test_empty_windows_remain_zero_are_evaluated_and_save_warning(tmp_path: Path) -> None:
    artifacts = prepare_c3_volume_run_artifacts(tmp_path, mask_kind=RANDOM_WHOLE_FFID_MASK_KIND)
    config = _write_config(
        tmp_path / "config.yaml",
        artifacts,
        changes={
            "spatial_window_shape": [1, 1, 2, 3],
            "spatial_overlap": [0, 0, 0, 0],
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
    expected_uncovered_traces = int(np.count_nonzero(observed.evaluation_target_trace_mask))
    expected_uncovered_samples = observed.values.shape[0] * expected_uncovered_traces
    np.testing.assert_array_equal(
        prediction[:, observed.evaluation_target_trace_mask],
        np.zeros((observed.values.shape[0], expected_uncovered_traces), dtype=prediction.dtype),
    )
    assert (
        metrics["evaluation_target"]["error_energy"]
        == metrics["evaluation_target"]["reference_energy"]
    )
    assert metrics["evaluation_target"]["relative_l2"] == 1.0
    assert metrics["evaluation_target"]["snr_db"] == metrics["zero_fill"]["snr_db"]
    assert metrics["evaluation_target"]["rmse"] == metrics["zero_fill"]["rmse"]
    assert metrics["uncovered_trace_count"] == expected_uncovered_traces
    assert metrics["uncovered_sample_count"] == expected_uncovered_samples
    assert len(metrics["warnings"]) == 1
    assert "remain zero for evaluation" in metrics["warnings"][0]
    assert run["window"]["empty_block_count"] > 0
    assert run["window"]["uncovered_trace_count"] == expected_uncovered_traces
    assert run["window"]["uncovered_sample_count"] == expected_uncovered_samples
    assert run["warnings"] == metrics["warnings"]


def test_importing_cpu_drr_pipeline_does_not_import_torch() -> None:
    code = """
import builtins
import sys
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise RuntimeError('torch import attempted')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import seis_interp.pipelines.interpolate_drr
assert 'torch' not in sys.modules
"""

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_real_drr_cli_runs_real_pipeline_end_to_end(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifacts = prepare_c3_volume_run_artifacts(tmp_path)
    config = _write_config(tmp_path / "config.yaml", artifacts)
    output = tmp_path / "cli-run"

    exit_code = main(
        [
            "interpolate",
            "drr",
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
    assert summary["method"] == METHOD
    assert (output / PREDICTION_RELATIVE_PATH).is_file()
    assert "Loading and verifying C3 inputs" in captured.err
