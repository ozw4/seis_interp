"""PoC comparisons rely on complete, mutually consistent run evidence."""

import csv
import json
import subprocess
import sys
from copy import deepcopy

import pytest

from seis_interp.evaluation.c3_poc_comparison import (
    SUMMARY_COLUMNS,
    empty_poc_summary_row,
    matching_poc_input_lock,
    read_poc_json,
    validate_poc_run_artifacts,
    write_poc_comparison,
)
from tests.fixtures.c3_poc_runs import poc_input_lock, write_poc_run_artifacts

METHODS = ("pocs", "drr", "nersi", "ccnet5d", "relational_trace_graph")


def _replace_record_value(directory, file_name, keys, value):
    path = directory / file_name
    record = read_poc_json(path)
    target = record
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    path.write_text(json.dumps(record, allow_nan=False), encoding="utf-8")


def test_five_methods_have_comparable_rows_and_full_precision_json_csv(tmp_path):
    rows, locks = [], []
    for method in METHODS:
        directory = tmp_path / method
        write_poc_run_artifacts(directory, method)
        row, lock = validate_poc_run_artifacts(method, directory)
        assert row["status"] == "success"
        assert set(row) == set(SUMMARY_COLUMNS) | {"coverage"}
        assert row["coverage"]["complete"] is True
        assert row["training_or_reconstruction_seconds"] == 2.0
        assert row["process_max_rss_kib"] == 1024
        assert row["cuda_max_memory_allocated_bytes"] is None
        if method in {"pocs", "drr"}:
            assert row["prediction_seconds"] is None
            assert row["checkpoint_path"] is None
            assert row["parameter_count"] is None
        else:
            assert row["prediction_seconds"] == 1.0
            assert row["checkpoint_path"] == str(directory / "final.pt")
            assert row["supervised_trace_presentations"] == 10
        rows.append(row)
        locks.append(lock)
    assert matching_poc_input_lock(locks) == poc_input_lock()
    summary = {"status": "success", "inputs_lock": locks[0], "runs": rows}
    write_poc_comparison(tmp_path, summary)
    assert read_poc_json(tmp_path / "summary.json") == summary
    with (tmp_path / "summary.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        csv_rows = list(reader)
        assert tuple(reader.fieldnames) == SUMMARY_COLUMNS
    assert [row["method"] for row in csv_rows] == list(METHODS)
    assert csv_rows[0]["checkpoint_path"] == ""
    assert csv_rows[0]["relative_l2"] == "0.3162"


@pytest.mark.parametrize(
    ("value", "display"),
    [(1.123456, "1.1235"), (1e-8, "approximately 0"), (-1e-8, "approximately 0"), (-0.0, "0.0000")],
)
def test_csv_rounding_preserves_formal_json_precision(tmp_path, value, display):
    row = empty_poc_summary_row("pocs", tmp_path)
    row["snr_db"] = value
    write_poc_comparison(tmp_path, {"runs": [row]})
    assert read_poc_json(tmp_path / "summary.json")["runs"][0]["snr_db"] == value
    with (tmp_path / "summary.csv").open(newline="", encoding="utf-8") as stream:
        saved = next(csv.DictReader(stream))
    assert saved["snr_db"] == display
    assert saved["prediction_path"] == ""


def test_empty_row_contains_all_columns_for_individual_failure(tmp_path):
    row = empty_poc_summary_row("nersi", tmp_path)
    assert tuple(row) == SUMMARY_COLUMNS
    assert row["method"] == "nersi"
    assert row["output_directory"] == str(tmp_path)
    assert row["status"] == "not_run"
    assert row["snr_db"] is row["error_message"] is None
    assert matching_poc_input_lock([]) is None


@pytest.mark.parametrize(
    ("keys", "value"),
    [
        (("benchmark_id",), "different"),
        (("dataset_id",), "different"),
        (("case_id",), "different"),
        (("volume_id",), "different"),
        (("selection", "relative_receiver_y"), [19, 51]),
        (("shape",), [384, 16, 32, 8, 31]),
        (("mask", "random_seed"), 43),
        (("mask", "files", "observation_mask.parquet", "sha256"), "d" * 64),
        (("observed_trace_count",), 26213),
        (("target_trace_count",), 104859),
        (("benchmark_case", "sha256"), "d" * 64),
        (("benchmark_volume", "files", "volume_index.parquet", "sha256"), "d" * 64),
    ],
)
def test_input_lock_comparison_includes_nested_hashes_and_all_fields(keys, value):
    original = poc_input_lock()
    changed = deepcopy(original)
    target = changed
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    with pytest.raises(ValueError, match="PoC methods used different verified input locks"):
        matching_poc_input_lock([original, deepcopy(original), changed])


@pytest.mark.parametrize(
    ("file_name", "keys", "value", "message"),
    [
        ("metadata.json", ("status",), "failed", "metadata.status"),
        ("metadata.json", ("method",), "nersi", "metadata.method"),
        ("metadata.json", ("case_id",), "wrong", "metadata.case_id"),
        ("metadata.json", ("coverage", "complete"), False, "coverage.complete"),
        ("metadata.json", ("coverage", "target_trace_count"), 1, "target_trace_count"),
        ("metadata.json", ("coverage", "covered_target_trace_count"), 1, "covered_target"),
        ("metadata.json", ("coverage", "uncovered_trace_count"), 1, "uncovered_trace"),
        ("metadata.json", ("coverage", "uncovered_sample_count"), 1, "uncovered_sample"),
        ("metadata.json", ("coverage", "target_coverage_fraction"), 0.9, "fraction"),
        ("metadata.json", ("coverage", "boundary_targets_included"), False, "boundary"),
        ("metadata.json", ("compute", "optimizer_updates"), True, "nonnegative integer"),
        ("metadata.json", ("compute", "supervised_trace_presentations"), None, "integer"),
        ("metadata.json", ("timing", "training_seconds"), -1, "nonnegative"),
        ("metadata.json", ("resource_usage", "process_max_rss_kib"), None, "integer"),
        ("metrics.json", ("evaluation_domain",), "observed", "evaluation_domain"),
        ("metrics.json", ("amplitude_domain",), "normalized", "amplitude_domain"),
        ("metrics.json", ("evaluation_target", "covered_target_trace_count"), 1, "count"),
        ("metrics.json", ("evaluation_target", "snr_status"), "unknown", "snr_status"),
        ("metrics.json", ("evaluation_target", "snr_db"), None, "finite number"),
        ("metrics.json", ("evaluation_target", "relative_l2"), None, "finite number"),
        ("metrics.json", ("evaluation_target", "rmse"), -1, "nonnegative"),
    ],
)
def test_incomplete_or_inconsistent_success_evidence_is_rejected(
    tmp_path, file_name, keys, value, message
):
    write_poc_run_artifacts(tmp_path, "ccnet5d")
    _replace_record_value(tmp_path, file_name, keys, value)
    with pytest.raises(ValueError, match=message):
        validate_poc_run_artifacts("ccnet5d", tmp_path)


@pytest.mark.parametrize("file_name", ["prediction.npy", "final.pt"])
@pytest.mark.parametrize("corruption", ["missing", "content", "path", "digest"])
def test_artifacts_require_canonical_paths_existing_files_and_matching_hashes(
    tmp_path, file_name, corruption
):
    write_poc_run_artifacts(tmp_path, "nersi")
    artifact_name = "prediction" if file_name == "prediction.npy" else "checkpoint"
    if corruption == "missing":
        (tmp_path / file_name).unlink()
    elif corruption == "content":
        (tmp_path / file_name).write_bytes(b"changed artifact")
    else:
        key = "path" if corruption == "path" else "sha256"
        value = "../elsewhere/" + file_name if key == "path" else "f" * 64
        _replace_record_value(tmp_path, "metadata.json", ("artifacts", artifact_name, key), value)
    with pytest.raises(ValueError, match="artifact|SHA-256"):
        validate_poc_run_artifacts("nersi", tmp_path)


@pytest.mark.parametrize("corruption", ["checkpoint", "file", "compute"])
def test_classical_runs_require_null_checkpoint_and_compute(tmp_path, corruption):
    write_poc_run_artifacts(tmp_path, "pocs")
    if corruption == "checkpoint":
        _replace_record_value(tmp_path, "metadata.json", ("artifacts", "checkpoint"), {})
    elif corruption == "file":
        (tmp_path / "final.pt").write_bytes(b"unexpected model")
    else:
        _replace_record_value(tmp_path, "metadata.json", ("compute", "parameter_count"), 0)
    with pytest.raises(ValueError, match="classical methods"):
        validate_poc_run_artifacts("pocs", tmp_path)


@pytest.mark.parametrize("status", ["perfect_reconstruction", "undefined_zero_reference"])
def test_nullable_degenerate_metrics_are_preserved(tmp_path, status):
    write_poc_run_artifacts(tmp_path, "pocs")
    path = tmp_path / "metrics.json"
    metrics = read_poc_json(path)
    target = metrics["evaluation_target"]
    target.update(snr_status=status, snr_db=None)
    target["relative_l2"] = None if status == "undefined_zero_reference" else 0.0
    if status == "perfect_reconstruction":
        target.update(rmse=0.0, mean_trace_relative_mse=0.0)
    path.write_text(json.dumps(metrics, allow_nan=False), encoding="utf-8")
    row, _ = validate_poc_run_artifacts("pocs", tmp_path)
    write_poc_comparison(tmp_path, {"runs": [row]})
    saved = read_poc_json(tmp_path / "summary.json")["runs"][0]
    assert saved["snr_db"] is None
    assert saved["snr_status"] == status
    assert saved["relative_l2"] == target["relative_l2"]


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e999", "-1e999"])
def test_json_reader_rejects_nonfinite_values_at_any_depth(tmp_path, token):
    path = tmp_path / "record.json"
    path.write_text('{"nested": [{"value": ' + token + "}]}", encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        read_poc_json(path)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_summary_writer_never_emits_nonfinite_values(tmp_path, value):
    row = empty_poc_summary_row("pocs", tmp_path)
    row["snr_db"] = value
    with pytest.raises(ValueError, match="Out of range float"):
        write_poc_comparison(tmp_path, {"runs": [row]})
    assert not (tmp_path / "summary.json").exists()
    assert not (tmp_path / "summary.csv").exists()


def test_validation_does_not_read_arrays_or_deserialize_opaque_artifacts(tmp_path):
    write_poc_run_artifacts(tmp_path, "nersi")
    script = """
import builtins
import sys
from pathlib import Path
import seis_interp.evaluation
import numpy
def reject_array_read(*args, **kwargs):
    raise AssertionError('comparison must not read amplitude arrays')
numpy.load = reject_array_read
original_import = builtins.__import__
def reject_array_backend(name, *args, **kwargs):
    if name.split('.')[0] in {'numpy', 'torch'}:
        raise AssertionError('comparison must not import an array backend')
    return original_import(name, *args, **kwargs)
builtins.__import__ = reject_array_backend
from seis_interp.evaluation.c3_poc_comparison import validate_poc_run_artifacts
row, lock = validate_poc_run_artifacts('nersi', Path(sys.argv[1]))
assert row['status'] == 'success'
"""
    subprocess.run([sys.executable, "-c", script, str(tmp_path)], check=True)


def test_missing_required_common_metric_is_rejected(tmp_path):
    write_poc_run_artifacts(tmp_path, "nersi")
    path = tmp_path / "metrics.json"
    metrics = read_poc_json(path)
    del metrics["evaluation_target"]["mean_trace_relative_mse"]
    path.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ValueError, match="missing common evaluation metrics"):
        validate_poc_run_artifacts("nersi", tmp_path)
