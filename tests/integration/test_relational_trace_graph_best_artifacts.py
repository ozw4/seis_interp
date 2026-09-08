"""Saved validation predictions, best checkpoints and physical metrics agree."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seis_interp.data.c3_volume_index_store import load_c3_volume_index
from seis_interp.data.trace_graph_domain import load_benchmark_trace_graph_domain
from seis_interp.evaluation.trace_graph_metrics import evaluate_trace_graph_prediction
from seis_interp.pipelines.train_relational_trace_graph import train_relational_trace_graph_run
from seis_interp.training.relational_trace_graph_checkpoints import (
    load_relational_trace_graph_checkpoint,
)
from seis_interp.training.relational_trace_graph_prediction import predict_relational_trace_graph
from tests.fixtures import ccnet5d_artifacts
from tests.fixtures.relational_trace_graph_runs import (
    prepare_trace_graph_run_artifacts,
    trace_graph_training_config,
    write_trace_graph_config,
)


@pytest.fixture
def compact_artifacts(tmp_path, monkeypatch):
    original = ccnet5d_artifacts.make_c3_trace_table

    def compact_table(**kwargs):
        table = original(**kwargs)
        # One train shot exposes the required global 8 x 68 offsets. The other
        # shots retain a complete 2 x 2 crop, keeping native validation small.
        crop = (table["receiver_x_m"] - table["source_x_m"]).le(-100) & (
            table["receiver_y_m"] - table["source_y_m"]
        ).le(-2640)
        table = table.loc[table["ffid"].eq(table["ffid"].iloc[0]) | crop].reset_index(drop=True)
        table["array_row"] = np.arange(len(table), dtype=np.int64)
        return table

    monkeypatch.setattr(ccnet5d_artifacts, "make_c3_trace_table", compact_table)
    return prepare_trace_graph_run_artifacts(tmp_path / "data", dense=True)


def _saved_query_predictions(
    output: Path, query_index: pd.DataFrame, volume_dir: Path | None
) -> np.ndarray:
    values = np.load(output / "artifacts/prediction.npy", allow_pickle=False)
    if volume_dir is None:
        assert values.ndim == 2
        return values
    assert values.ndim == 5
    volume_index, _ = load_c3_volume_index(volume_dir)
    positions = {
        int(array_row): position for position, array_row in enumerate(volume_index["array_row"])
    }
    query_positions = [positions[int(row)] for row in query_index["array_row"]]
    return values.reshape(values.shape[0], -1).T[query_positions]


def _assert_metrics_match(actual: dict[str, object], recorded: dict[str, object]) -> None:
    assert actual.keys() == recorded.keys()
    for name, value in actual.items():
        if isinstance(value, dict):
            _assert_metrics_match(value, recorded[name])
        elif isinstance(value, float):
            assert recorded[name] == pytest.approx(value, rel=1e-12, abs=1e-12)
        else:
            assert recorded[name] == value


@pytest.mark.parametrize("dense", [False, True], ids=["native", "dense"])
def test_saved_best_prediction_matches_checkpoint_and_recorded_selection(
    compact_artifacts, tmp_path, dense
):
    data = compact_artifacts
    config = trace_graph_training_config()
    # Actual optimizer updates overshoot the first validation optimum, so a
    # final-state prediction cannot accidentally satisfy the best-state check.
    config["training"].update(max_steps=4, validation_interval=1, learning_rate=0.5)
    config_path = write_trace_graph_config(tmp_path / "train.yaml", config)
    output = tmp_path / "train"
    volume_dir = data.volumes["validation"] if dense else None
    train_relational_trace_graph_run(
        config_path=config_path,
        interim_dir=data.interim,
        processed_dir=data.processed,
        validation_mask_dir=data.masks["validation"],
        validation_case_dir=data.cases["validation"],
        validation_volume_dir=volume_dir,
        train_mask_dir=data.masks["train"],
        train_case_dir=data.cases["train"],
        train_volume_dir=data.volumes["train"],
        output_dir=output,
    )
    metrics = json.loads((output / "metrics.json").read_text())
    metadata = json.loads((output / "run.json").read_text())
    best = load_relational_trace_graph_checkpoint(output / "artifacts/best.pt")
    final = load_relational_trace_graph_checkpoint(output / "artifacts/final.pt")
    assert best.checkpoint_role == "best_validation"
    assert final.checkpoint_role == "final"
    assert (
        best.global_step == metrics["best_step"] < final.global_step == metrics["steps_completed"]
    )
    assert (
        metrics["best_validation_metrics"]["evaluation_target"]["error_energy"]
        < (metrics["final_validation_metrics"]["evaluation_target"]["error_energy"])
    )
    assert metadata["checkpoints"]["best"]["step"] == best.global_step
    assert metadata["prediction"]["checkpoint_role"] == "best_validation"
    assert metadata["prediction"]["layout"] == ("dense_volume" if dense else "native_trace_list")

    query_index = pd.read_parquet(output / "artifacts/query_index.parquet")
    query_ids = query_index["trace_id"].to_numpy(dtype=np.int64)
    flags = query_index["has_observed_context"].to_numpy(dtype=bool)
    saved_queries = _saved_query_predictions(output, query_index, volume_dir)
    domain = load_benchmark_trace_graph_domain(
        interim_dir=data.interim,
        processed_dir=data.processed,
        mask_dir=data.masks["validation"],
        case_dir=data.cases["validation"],
        volume_dir=volume_dir,
        time_samples=tuple(best.preprocessing.fit_domain["time_samples"]),
    )
    assert len(query_ids) == int(domain.query_mask.sum())
    np.testing.assert_array_equal(query_ids, domain.trace_ids[domain.query_mask])
    predictions = [
        predict_relational_trace_graph(
            checkpoint.model,
            domain,
            checkpoint.preprocessing,
            graph_settings=checkpoint.graph_settings,
            query_trace_ids=query_ids,
            query_batch_size=config["evaluation"]["query_batch_size"],
        )
        for checkpoint in (best, final)
    ]
    assert np.abs(predictions[0].prediction).max() > 1e-4
    np.testing.assert_allclose(saved_queries, predictions[0].prediction, atol=1e-6, rtol=1e-5)
    np.testing.assert_array_equal(flags, predictions[0].has_observed_context)
    assert np.max(np.abs(predictions[1].prediction - predictions[0].prediction)) > 1e-3

    measured = evaluate_trace_graph_prediction(
        saved_queries,
        domain,
        query_trace_ids=query_ids,
        has_observed_context=flags,
    )
    assert measured["evaluation_target"]["trace_count"] == len(query_ids)
    assert measured["evaluation_target"]["sample_count"] == saved_queries.size
    _assert_metrics_match(measured, metrics["best_validation_metrics"])
    _assert_metrics_match(measured, best.selection_metrics)
