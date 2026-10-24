import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from seis_interp.data.study067_analysis_inputs import require_output_directory, verify_run_rows
from seis_interp.evaluation.study067_report import shadow_rows
from seis_interp.evaluation.study067_seed_audit import audit_seed_namespaces, seed_occurrences
from seis_interp.evaluation.study067_trace_metrics import (
    derived_traces,
    population_summaries,
    waveform_columns,
)


def test_derived_columns_align_saved_arrays_and_apply_idw_once(tmp_path):
    target = np.array([[1, 2, -1, 3], [2, 1, -2, 2], [3, 2, -1, 1]], dtype=np.float32)
    prediction = target * 0.7 + np.array([0, 1, 0, 0.5], dtype=np.float32)
    np.save(tmp_path / "prediction.npy", prediction[[2, 0, 1]])
    np.save(tmp_path / "cell_ids.npy", [30, 10, 20])
    saved = pd.DataFrame(
        {
            "cell_id": [20, 10],
            "trace_id": ["b", "a"],
            "source_cell": [1, 0],
            "clean_target": [True, False],
            **waveform_columns(prediction[[1, 0]], target[[1, 0]]),
        }
    )
    scales = pd.DataFrame(
        {"cell_id": [10, 20], "oracle_rms": [2, 3], "idw_rms": [4, 1.5], "idw_scale_k_i": [2, 0.5]}
    )
    source = target, pd.Series([0, 1, 2], index=[10, 20, 30]), 4
    result = derived_traces(tmp_path, saved, source, scales)
    np.testing.assert_allclose(
        result.prediction_energy, np.square(prediction[[1, 0]].astype(float)).sum(axis=1)
    )
    np.testing.assert_allclose(result.idw_scale_k_i, [0.5, 2])
    nonoracle = (prediction[[1, 0]].astype(float) * np.array([0.5, 2])[:, None]).astype(np.float32)
    np.testing.assert_allclose(
        result.idw_error_energy, np.square(nonoracle.astype(float) - target[[1, 0]]).sum(axis=1)
    )
    scalar = (
        np.sum(prediction[[1, 0]].astype(float) * target[[1, 0]], axis=1)
        / result.prediction_energy.to_numpy()
    )
    expected_si = (
        np.square(scalar[:, None] * prediction[[1, 0]] - target[[1, 0]]).sum(axis=1)
        / result.target_energy.to_numpy()
    )
    np.testing.assert_allclose(result.scale_invariant_relative_mse, expected_si)
    summaries = population_summaries(result, "forge")
    assert len(summaries) == 6
    assert summaries[0]["trace_count"] == 1
    assert summaries[3]["trace_count"] == 2
    saved.loc[0, "error_energy"] += 0.1
    with pytest.raises(ValueError, match="error_energy"):
        derived_traces(tmp_path, saved, source, scales)


def test_descriptive_trace_quantiles_global_metrics_and_zero_energy():
    p = np.array([[1, 1, 1], [2, 0, 0]], dtype=np.float32)
    y = np.array([[2, 2, 2], [1, 0, 0]], dtype=np.float32)
    table = pd.DataFrame({"clean_target": True, **waveform_columns(p, y)})
    summary = population_summaries(table, "c3")[0]
    assert summary["relative_mse"] == pytest.approx((0.25 + 1) / 2)
    assert summary["global_nmse"] == pytest.approx(4 / 13)
    assert summary["mean_trace_snr_db"] == pytest.approx((10 * np.log10(4) + 0) / 2)
    assert summary["trace_snr_p10"] == pytest.approx(np.percentile([10 * np.log10(4), 0], 10))
    assert waveform_columns(np.ones((1, 3)), np.zeros((1, 3)))["relative_mse"][0] == 1


def test_missing_imported_ema_is_explicit(tmp_path):
    (tmp_path / "config.resolved.yaml").write_text(
        yaml.safe_dump({"training": {"ema_eval_decays": [0.999, 0.9995]}})
    )
    rows = shadow_rows(tmp_path, {"stage": "S1", "imported_from": "pilot"})
    assert len(rows) == 2
    assert {r["status"] for r in rows} == {"missing_saved_shadow"}


def test_saved_rows_must_match_the_seed_partition():
    assignment = pd.DataFrame(
        {"cell_id": [0, 1], "trace_id": ["a", "b"], "trace_role": ["validation", "test"]}
    )
    verify_run_rows(assignment.iloc[[1]], assignment, "test")
    with pytest.raises(ValueError, match="partition"):
        verify_run_rows(assignment.iloc[[0]], assignment, "test")


def test_analysis_output_refuses_runs_including_symlinks(tmp_path):
    (tmp_path / "runs").mkdir()
    (tmp_path / "alias").symlink_to(tmp_path / "runs", target_is_directory=True)
    for path in (tmp_path / "runs/out", tmp_path / "alias/out"):
        with pytest.raises(ValueError, match="outside runs"):
            require_output_directory(tmp_path, path)


def test_namespace_audit_prunes_blinded_tree_before_any_read(tmp_path, monkeypatch):
    blinded = tmp_path / "runs/production"
    (blinded / "evaluation/seed_7321").mkdir(parents=True)
    (blinded / "evaluation/seed_7321/manifest.json").write_text('{"trace_seed": 7321}')
    (tmp_path / "studies").mkdir()
    (tmp_path / "studies/config.yaml").write_text(
        "trace_split:\n  evaluation_seeds: [7321, 7322, 7323]\n"
    )
    original = Path.read_text

    def read(path, *args, **kwargs):
        assert blinded not in path.parents
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    result = audit_seed_namespaces(tmp_path, [blinded])
    assert result["passed"]
    assert result["scanned_file_count"] == 1
    (tmp_path / "runs/old").mkdir()
    (tmp_path / "runs/old/experiment.json").write_text(
        json.dumps({"partitions": {"seed_7322": {"train": 2}}})
    )
    (tmp_path / "studies/config.yaml").write_text("mask_seed: 7321\n")
    result = audit_seed_namespaces(tmp_path, [blinded])
    assert not result["passed"]
    assert {r["seed"] for r in result["partition_seed_uses"]} == {7321, 7322}


def test_seed_audit_distinguishes_model_and_partition_seeds():
    rows = seed_occurrences(
        {"model_seed": 7321, "trace_split": {"search_seed": 7322}}, [7321, 7322]
    )
    assert [r["kind"] for r in rows] == ["non_partition_seed", "partition_seed_use"]
