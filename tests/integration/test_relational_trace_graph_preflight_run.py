"""Artifact preflight reports explicit input and resource blockers without training."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest
import torch

from seis_interp.configuration import REPOSITORY_ROOT
from seis_interp.data.trace_graph_domain import load_training_trace_graph_domain
from seis_interp.models.relational_trace_graph import RelationalTraceGraphInterpolator
from seis_interp.pipelines.preflight_relational_trace_graph import (
    preflight_relational_trace_graph_run,
)
from seis_interp.processing.trace_graph_preprocessing import fit_trace_graph_preprocessing
from seis_interp.relational_trace_graph_config import (
    validate_relational_trace_graph_training_config,
)
from seis_interp.training.relational_trace_graph_checkpoints import (
    save_relational_trace_graph_checkpoint,
)
from tests.fixtures.relational_trace_graph_runs import (
    prepare_trace_graph_run_artifacts,
    trace_graph_prediction_config,
    trace_graph_training_config,
    write_trace_graph_config,
)


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    return prepare_trace_graph_run_artifacts(
        tmp_path_factory.mktemp("preflight_inputs"), dense=True
    )


def _preflight(data, config_path, **kwargs):
    return preflight_relational_trace_graph_run(
        config_path=config_path,
        interim_dir=data.interim,
        processed_dir=data.processed,
        mask_dir=data.masks["validation"],
        case_dir=data.cases["validation"],
        volume_dir=data.volumes["validation"],
        train_mask_dir=data.masks["train"],
        train_case_dir=data.cases["train"],
        train_volume_dir=data.volumes["train"],
        query_count=2,
        **kwargs,
    )


def test_training_preflight_measures_selected_queries_without_optimizer(
    artifacts, tmp_path, monkeypatch
):
    config = trace_graph_training_config()
    path = write_trace_graph_config(tmp_path / "training.yaml", config)
    monkeypatch.setattr(
        torch.optim,
        "AdamW",
        lambda *args, **kwargs: pytest.fail("preflight must not start training"),
    )

    report = _preflight(artifacts, path, evaluate_baselines=True)

    assert report["status"] == "success", report
    assert report["training_started"] is False
    assert report["input"]["model_state"] == "untrained"
    assert len(report["sampled_query_trace_ids"]) == 2
    assert report["sampled_query_trace_ids"] == sorted(report["sampled_query_trace_ids"])
    assert report["total_case_query_count"] > 2
    assert report["input"]["benchmark_inputs_lock"]["benchmark_case"]["input_files"]
    assert report["resources"]["process_max_rss_bytes"] > 0
    assert report["prediction_diagnostics"]["timings"]["forward_seconds"] >= 0
    assert report["geometry"]["query_count"] == 2
    assert report["model_metrics"]["evaluation_target"]["trace_count"] == 2
    assert set(report["baselines"]) >= {"zero", "idw"}
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    assert sorted(path.name for path in tmp_path.iterdir()) == ["training.yaml"]


def test_preflight_reports_resource_limit_without_changing_sample_or_graph(artifacts, tmp_path):
    path = write_trace_graph_config(tmp_path / "training.yaml", trace_graph_training_config())
    report = _preflight(artifacts, path, max_process_rss_bytes=1)
    assert report["status"] == "blocked"
    assert report["training_started"] is False
    assert len(report["sampled_query_trace_ids"]) == 2
    assert report["blockers"][0]["measurement"] == "process_max_rss_bytes"
    assert report["blockers"][0]["reason"] == "measured_limit_exceeded"


@pytest.mark.parametrize("problem", ["missing", "time", "hash"])
def test_input_failures_are_reported_before_measurement(artifacts, tmp_path, problem):
    config = trace_graph_training_config()
    if problem == "time":
        config["training_data"]["time_samples"] = [0, 3]
    path = write_trace_graph_config(tmp_path / "training.yaml", config)
    case_dir = artifacts.cases["validation"]
    if problem == "missing":
        case_dir = tmp_path / "missing_case"
    elif problem == "hash":
        case_dir = tmp_path / "changed_case"
        case_dir.mkdir()
        case = json.loads((artifacts.cases["validation"] / "benchmark_case.json").read_text())
        case["input_files"]["mask"]["observation_mask.parquet"]["sha256"] = "0" * 64
        (case_dir / "benchmark_case.json").write_text(json.dumps(case))
    report = preflight_relational_trace_graph_run(
        config_path=path,
        interim_dir=artifacts.interim,
        processed_dir=artifacts.processed,
        mask_dir=artifacts.masks["validation"],
        case_dir=case_dir,
        volume_dir=artifacts.volumes["validation"],
    )
    assert report["status"] == "blocked"
    assert report["stage"] == "inputs"
    assert report["blockers"][0]["message"]
    assert "prediction_diagnostics" not in report
    assert report["training_started"] is False


def test_frozen_preflight_uses_checkpoint_scales_and_validates_provenance(artifacts, tmp_path):
    config = trace_graph_training_config()
    model_config, settings, _ = validate_relational_trace_graph_training_config(config)
    training = load_training_trace_graph_domain(
        interim_dir=artifacts.interim,
        processed_dir=artifacts.processed,
        pool="all_train_traces",
        time_samples=(1, 4),
        mask_dir=artifacts.masks["train"],
        case_dir=artifacts.cases["train"],
        volume_dir=artifacts.volumes["train"],
    )
    fixed = fit_trace_graph_preprocessing(training, **config["geometry_features"])
    model = RelationalTraceGraphInterpolator(**model_config)
    checkpoint = tmp_path / "model.pt"
    save_relational_trace_graph_checkpoint(
        checkpoint,
        model_config=model.constructor_config(),
        state_dict=model.state_dict(),
        preprocessing=fixed,
        graph_settings=settings,
        training_mask=config["training_mask"],
        training_provenance={"source_inputs_lock": training.inputs_lock},
        training_random_seed=7,
        checkpoint_role="best_validation",
        global_step=0,
        selection_metrics={},
    )
    frozen = trace_graph_prediction_config()
    frozen["benchmark_case"]["id"] = "validation"
    path = write_trace_graph_config(tmp_path / "frozen.yaml", frozen)
    report = preflight_relational_trace_graph_run(
        config_path=path,
        checkpoint_path=checkpoint,
        interim_dir=artifacts.interim,
        processed_dir=artifacts.processed,
        mask_dir=artifacts.masks["validation"],
        case_dir=artifacts.cases["validation"],
        volume_dir=artifacts.volumes["validation"],
        query_count=2,
    )
    assert report["status"] == "success", report
    assert report["input"]["checkpoint_step"] == 0
    assert report["input"]["checkpoint_role"] == "best_validation"
    np.testing.assert_array_equal(report["input"]["benchmark"]["time_s"], fixed.time_s)


def test_preflight_query_limit_rejects_request_without_silent_truncation(tmp_path):
    report = preflight_relational_trace_graph_run(
        config_path=tmp_path / "unused",
        interim_dir=tmp_path,
        processed_dir=tmp_path,
        mask_dir=tmp_path,
        case_dir=tmp_path,
        query_count=33,
        query_limit=32,
    )
    assert report["status"] == "blocked"
    assert "query_count exceeds" in report["blockers"][0]["message"]


def test_thin_preflight_script_returns_strict_json_and_nonzero_exit_for_missing_inputs(tmp_path):
    command = [sys.executable, str(REPOSITORY_ROOT / "scripts/preflight_grid_free_trace_graph.py")]
    for name in ("config", "interim", "processed", "mask", "case"):
        command.extend([f"--{name}", str(tmp_path / name)])
    completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 1, completed.stderr
    report = json.loads(completed.stdout)
    assert report["status"] == "blocked"
    assert report["blockers"][0]["type"] == "FileNotFoundError"
    assert report["training_started"] is False
