"""Explicit pilot selection, physical re-scoring and fixed-geometry figures."""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seis_interp.data.c3_volume_adapter import ObservedC3Volume
from seis_interp.data.c3_volume_run_inputs import C3VolumeRunInputs
from seis_interp.data.file_checksums import file_sha256
from seis_interp.evaluation import c3_first_results as collector
from seis_interp.evaluation.c3_volume_metrics import evaluate_c3_volume_prediction
from seis_interp.visualization.c3_first_results import plot_c3_first_results


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, allow_nan=False))


@pytest.fixture
def pilot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    shape = (4, 2, 2, 2, 3)
    spatial = shape[1:]
    rows = np.arange(np.prod(spatial), dtype=np.int64).reshape(spatial)
    amplitudes = np.arange(rows.size * shape[0], dtype=np.float32).reshape(-1, shape[0]) / 17
    observed = rows % 3 == 0
    values = amplitudes.T.reshape(shape).copy()
    values[:, ~observed] = 0
    times = np.arange(shape[0]) * 0.008
    volume = ObservedC3Volume(values, times, rows, observed, ~observed)
    index = pd.DataFrame({"array_row": rows.ravel()})
    for name, coords in zip(
        ("source_line", "shot_in_line", "relative_receiver_x", "relative_receiver_y"),
        np.indices(spatial),
        strict=True,
    ):
        index[f"{name}_index"] = coords.ravel()
    index["source_x_m"] = index.source_line_index * 100.0
    index["source_y_m"] = index.shot_in_line_index * 20.0 + index.source_line_index * 10.0
    index["relative_receiver_x_m"] = index.relative_receiver_x_index * 50.0
    index["relative_receiver_y_m"] = index.relative_receiver_y_index * 25.0
    selection = {
        "time": [0, 4],
        "source_line": [41, 43],
        "shot_in_line": [32, 34],
        "relative_receiver_x": [0, 2],
        "relative_receiver_y": [18, 21],
    }
    metadata = {
        "volume_id": "validation_volume",
        "selection": selection,
        "shape": list(shape),
        "trace_count": int(rows.size),
        "role_counts": {
            "observed": int(observed.sum()),
            "evaluation_target": int((~observed).sum()),
        },
    }
    case = {"case_id": "validation_case", "partition": "validation", "mask": {"random_seed": 142}}
    lock = {
        "benchmark_case": {
            "case_id": case["case_id"],
            "sha256": "case-hash",
            "input_files": {"mask": {"sha256": "mask-hash"}},
        },
        "benchmark_volume": {
            "volume_id": metadata["volume_id"],
            "files": {"volume.json": {"sha256": "volume-hash"}},
        },
    }
    inputs = C3VolumeRunInputs(volume, index, case, metadata, lock)
    interim = tmp_path / "interim"
    interim.mkdir()
    np.save(interim / "amplitudes.npy", amplitudes)
    manifest = tmp_path / "benchmark_suite.json"
    _json(manifest, {"synthetic_fixture": True})
    suite = {"interim": "interim", "cases": [{"case_id": case["case_id"], "volume_dir": "volume"}]}
    monkeypatch.setattr(
        collector, "load_c3_benchmark_input_manifest", lambda *_args, **_kwargs: suite
    )
    monkeypatch.setattr(
        collector, "load_c3_benchmark_volume_inputs", lambda *_args, **_kwargs: inputs
    )
    monkeypatch.setattr(
        collector, "plot_c3_first_results", lambda **_kwargs: {"status": "fixture_skipped"}
    )
    return {
        "inputs": inputs,
        "interim": interim,
        "manifest": manifest,
        "root": tmp_path,
        "reference": amplitudes.T.reshape(shape),
    }


def _native(
    pilot: dict, method: str, *, prediction: np.ndarray | None = None, suffix: str = ""
) -> Path:
    inputs = pilot["inputs"]
    native = pilot["root"] / f"{method}{suffix}"
    (native / "artifacts").mkdir(parents=True)
    if prediction is None:
        prediction = pilot["reference"].copy()
        prediction[:, inputs.observed_volume.evaluation_target_trace_mask] *= 0.75
    np.save(native / "artifacts" / "prediction.npy", prediction)
    metrics = evaluate_c3_volume_prediction(
        prediction,
        inputs.observed_volume,
        interim_dir=pilot["interim"],
        volume_metadata=inputs.volume_metadata,
    )
    metrics.update(
        method=collector._NATIVE_METHODS[method],
        case_id=inputs.case["case_id"],
        volume_id=inputs.volume_metadata["volume_id"],
    )
    metadata = {
        "method": metrics["method"],
        "case_id": inputs.case["case_id"],
        "volume_id": inputs.volume_metadata["volume_id"],
        "status": "success",
        "input": {
            "partition": "validation",
            "mask": {"random_seed": 142},
            "selected_volume": {"selection": inputs.volume_metadata["selection"]},
        },
        "prediction": {"shape": list(prediction.shape)},
        "resources": {"reconstruction_seconds": 0.125} if method in ("pocs", "drr") else {},
    }
    lock = deepcopy(inputs.inputs_lock)
    if method in ("siren5d", "nersi", "ccnet5d", "relational_trace_graph"):
        checkpoint = native / "artifacts" / "final.pt"
        checkpoint.write_bytes(b"synthetic checkpoint; collector must never deserialize")
        metadata["checkpoint"] = {
            "role": "fixed_step_final" if method in ("siren5d", "nersi") else "final",
            "path": str(checkpoint),
            "sha256": file_sha256(checkpoint),
        }
    if method in ("siren5d", "nersi"):
        metadata["training"] = {"max_steps": 2, "steps_completed": 2}
        metadata["resources"] = {"training_seconds": 0.25, "prediction_seconds": 0.05}
        metrics["training"] = {
            "history": [{"step": 1, "train_loss": 2.0}, {"step": 2, "train_loss": 1.0}]
        }
    if method == "nersi":
        metadata["prediction"]["sha256"] = file_sha256(native / "artifacts" / "prediction.npy")
    if method == "relational_trace_graph":
        targets = inputs.observed_volume.array_rows[
            inputs.observed_volume.evaluation_target_trace_mask
        ][::-1]
        context = np.arange(len(targets)) % 2 == 0
        pd.DataFrame(
            {
                "prediction_row": np.arange(len(targets)),
                "trace_id": targets,
                "array_row": targets,
                "has_observed_context": context,
            }
        ).to_parquet(native / "artifacts" / "query_index.parquet")
        metrics.update(
            zero_context_query_count=int((~context).sum()),
            without_observed_context={"trace_count": int((~context).sum())},
        )
        metadata["input"].update(time_samples=[0, 4], time_s=inputs.observed_volume.time_s.tolist())
        metadata["prediction"].update(layout="dense_volume", query_batch_size=32)
        metadata["graph"] = {"neighbors_per_relation": 2, "candidate_chunk_size": 4096}
        metadata["model"] = {"width": 32, "message_passing_rounds": 2}
        metadata["parameter_count"] = 123
        lock["benchmark_volume"]["array_rows"] = inputs.index_table.array_row.tolist()
    _json(native / "metrics.json", metrics)
    _json(native / "run.json", metadata)
    _json(native / "inputs.lock.json", lock)
    return native


def _summarize(pilot: dict, selected: dict, *, suffix: str = "") -> dict:
    return collector.summarize_c3_first_results(
        suite_path=pilot["manifest"],
        case_id="validation_case",
        method_runs={**dict.fromkeys(collector.METHODS), **selected},
        output_dir=pilot["root"] / f"summary{suffix}",
    )


def test_five_native_formats_reconcile_and_keep_unmeasured_null(pilot: dict) -> None:
    paths = {method: _native(pilot, method) for method in collector.METHODS}
    result = _summarize(pilot, paths)
    assert result["status"] == "first_results_complete"
    assert len(result["rows"]) == 6
    baseline, *methods = result["rows"]
    assert baseline["snr_db"] == 0
    assert all(row["metric_comparison"]["status"] == "matched" for row in methods)
    assert len({row["reference_energy"] for row in result["rows"]}) == 1
    assert methods[-1]["no_context_query_count"] == 8
    assert methods[-1]["graph"] == {"neighbors_per_relation": 2, "candidate_chunk_size": 4096}
    assert methods[-1]["neighbors_per_relation"] == 2
    assert methods[-1]["prediction_settings"]["query_batch_size"] == 32
    assert methods[-1]["model"]["width"] == 32
    assert methods[-1]["parameter_count"] == 123
    assert methods[-1]["uncovered_sample_count"] is None
    assert methods[0]["reconstruction_seconds"] == 0.125
    assert methods[0]["pretraining_seconds"] is None
    assert methods[0]["null_reasons"]["pretraining_seconds"]
    assert methods[0]["process_max_rss_kib"] is None
    assert methods[2]["checkpoint_role"] == "fixed_step_final"
    assert methods[3]["checkpoint_role"] == "final"
    assert (
        json.loads((pilot["root"] / "summary/first_results.json").read_text())["rows"]
        == result["rows"]
    )
    with (pilot["root"] / "summary/first_results.csv").open() as stream:
        table = list(csv.DictReader(stream))
    assert len(table) == 6
    assert table[1]["pretraining_seconds"] == ""
    assert "training_regime" not in result["rows"][0]
    assert "prediction_seconds" not in result["rows"][0]
    assert "total_method_seconds" not in result["rows"][0]


def test_optional_nersi_adds_one_validated_internal_learning_row(pilot: dict) -> None:
    native = _native(pilot, "nersi")

    result = _summarize(pilot, {"nersi": native}, suffix="_nersi")

    assert [row["method"] for row in result["rows"]] == [
        "zero_fill",
        *collector.METHODS,
        "nersi",
    ]
    row = result["rows"][-1]
    assert row["status"] == "success"
    assert row["checkpoint_role"] == "fixed_step_final"
    assert row["training_regime"] == "per_volume_observed_only_internal_learning"
    assert not row["additional_supervised_training_data"]
    assert row["target_volume_model_optimization"]
    assert row["expected_target_count"] == row["trace_count"]
    assert row["expected_sample_count"] == row["sample_count"]
    assert row["volume_fit_seconds"] == 0.25
    assert row["prediction_seconds"] == row["frozen_inference_seconds"] == 0.05
    assert row["total_method_seconds"] == 0.3


def test_optional_nersi_rejects_a_run_bound_to_another_volume(pilot: dict) -> None:
    native = _native(pilot, "nersi")
    lock_path = native / "inputs.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["benchmark_volume"]["files"]["volume.json"]["sha256"] = "another-volume"
    _json(lock_path, lock)

    result = _summarize(pilot, {"nersi": native}, suffix="_wrong_nersi_volume")

    row = result["rows"][-1]
    assert row["method"] == "nersi"
    assert row["status"] == "invalid_result"
    assert "benchmark_volume" in row["reason"]


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("metadata_volume", "case or volume"),
        ("checkpoint_hash", "checkpoint metadata"),
        ("prediction", "prediction SHA-256"),
    ],
)
def test_optional_nersi_rejects_cross_record_identity_or_changed_artifacts(
    pilot: dict,
    corruption: str,
    message: str,
) -> None:
    native = _native(pilot, "nersi")
    run_path = native / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if corruption == "metadata_volume":
        run["volume_id"] = "another-volume"
        _json(run_path, run)
    elif corruption == "checkpoint_hash":
        del run["checkpoint"]["sha256"]
        _json(run_path, run)
    else:
        prediction_path = native / "artifacts" / "prediction.npy"
        prediction = np.load(prediction_path, allow_pickle=False)
        np.save(prediction_path, prediction + np.float32(0.25))

    result = _summarize(pilot, {"nersi": native}, suffix=f"_{corruption}")

    row = result["rows"][-1]
    assert row["status"] == "invalid_result"
    assert message in row["reason"]


def test_explicit_path_wins_without_directory_discovery(pilot: dict) -> None:
    selected = _native(pilot, "pocs")
    _native(pilot, "pocs", prediction=pilot["reference"], suffix="_better")
    result = _summarize(pilot, {"pocs": selected})
    assert result["status"] == "partial_results"
    assert result["rows"][1]["run_path"] == str(selected)
    assert result["rows"][1]["snr_status"] == "finite"
    assert all(row["status"] == "not_run" for row in result["rows"][2:])


@pytest.mark.parametrize("field", ["case", "mask", "volume_hash", "energy", "best", "budget"])
def test_rejects_mismatched_binding_energy_or_checkpoint(pilot: dict, field: str) -> None:
    native = _native(pilot, "siren5d")
    if field in ("case", "energy"):
        path = native / "metrics.json"
        record = json.loads(path.read_text())
        if field == "case":
            record["case_id"] = "test_case"
        else:
            record["evaluation_target"]["error_energy"] *= 1.1
    elif field == "volume_hash":
        path = native / "inputs.lock.json"
        record = json.loads(path.read_text())
        record["benchmark_volume"]["files"]["volume.json"]["sha256"] = "changed"
    else:
        path = native / "run.json"
        record = json.loads(path.read_text())
        if field == "mask":
            record["input"]["mask"]["random_seed"] = 42
        elif field == "best":
            record["checkpoint"]["role"] = "best"
        else:
            record["training"]["steps_completed"] = 1
    _json(path, record)
    result = _summarize(pilot, {"siren5d": native})
    row = result["rows"][3]
    assert result["status"] == "partial_results"
    assert row["status"] == "invalid_result"
    assert row["reason"]


@pytest.mark.parametrize("subset", ["dense", "query", "query_duplicate", "context"])
def test_subset_and_incorrect_context_are_not_full_volume(pilot: dict, subset: str) -> None:
    native = _native(pilot, "relational_trace_graph")
    if subset == "dense":
        np.save(native / "artifacts/prediction.npy", pilot["reference"][:-1])
    else:
        path = native / "artifacts/query_index.parquet"
        query = pd.read_parquet(path)
        if subset == "query":
            query = query.iloc[:-1]
        elif subset == "query_duplicate":
            query.loc[0, ["trace_id", "array_row"]] = query.loc[1, ["trace_id", "array_row"]]
        else:
            query["has_observed_context"] = True
        query.to_parquet(path)
    result = _summarize(pilot, {"relational_trace_graph": native})
    assert result["rows"][-1]["status"] == "invalid_result"


@pytest.mark.parametrize("zero_reference", [False, True])
def test_degenerate_snr_remains_null_with_native_status(pilot: dict, zero_reference: bool) -> None:
    if zero_reference:
        np.save(pilot["interim"] / "amplitudes.npy", np.zeros((24, 4), dtype=np.float32))
    native = _native(pilot, "pocs", prediction=pilot["reference"])
    result = _summarize(pilot, {"pocs": native})
    row = result["rows"][1]
    assert row["status"] == "success"
    assert row["snr_db"] is None
    assert row["snr_status"] == (
        "undefined_zero_reference" if zero_reference else "perfect_reconstruction"
    )
    json.dumps(result, allow_nan=False)


def test_failed_outer_run_is_retained(pilot: dict) -> None:
    outer = pilot["root"] / "failed"
    outer.mkdir()
    _json(outer / "request.json", {"case_id": "validation_case", "action": "pocs"})
    _json(outer / "result.json", {"status": "timeout", "reason": "Declared time budget exceeded."})
    result = _summarize(pilot, {"pocs": outer})
    assert result["rows"][1]["status"] == "timeout"
    assert "budget" in result["rows"][1]["reason"]


@pytest.mark.parametrize("invalid", ["suite", "preflight"])
def test_outer_suite_hash_and_preflight_are_checked(pilot: dict, invalid: str) -> None:
    native = _native(pilot, "pocs")
    outer = pilot["root"] / "outer"
    outer.mkdir()
    request = {
        "case_id": "validation_case",
        "action": "pocs",
        "input_hashes": {
            "suite": "wrong" if invalid == "suite" else file_sha256(pilot["manifest"])
        },
        "preflight": invalid == "preflight",
    }
    _json(outer / "request.json", request)
    _json(outer / "result.json", {"status": "success", "native_run_directory": str(native)})
    result = _summarize(pilot, {"pocs": outer})
    assert result["rows"][1]["status"] == "invalid_result"
    assert invalid in result["rows"][1]["reason"]


def test_duplicate_run_and_incomplete_method_mapping_rejected(pilot: dict) -> None:
    native = _native(pilot, "pocs")
    with pytest.raises(ValueError, match="same run path"):
        _summarize(pilot, {"pocs": native, "drr": native})
    with pytest.raises(ValueError, match="all five"):
        collector.summarize_c3_first_results(
            suite_path=pilot["manifest"],
            case_id="validation_case",
            method_runs={"pocs": native},
            output_dir=pilot["root"] / "output",
        )


@pytest.mark.parametrize("extreme_prediction", [False, True])
def test_fixed_section_and_id_trace_figures(pilot: dict, extreme_prediction: bool) -> None:
    output = pilot["root"] / "figures"
    output.mkdir()
    predictions = {"pocs": pilot["reference"]}
    if extreme_prediction:
        predictions["relational_trace_graph"] = np.full_like(pilot["reference"], 1e30)
    figures = plot_c3_first_results(
        inputs=pilot["inputs"],
        interim_dir=pilot["interim"],
        predictions=predictions,
        histories={"siren5d": [{"step": 1, "train_loss": 2.0}, {"step": 2, "train_loss": 1.0}]},
        output_dir=output,
    )
    assert [item["varying_axis"] for item in figures["sections"]] == [
        "relative_receiver_y",
        "shot_in_line",
        "source_line",
    ]
    assert figures["sections"][0]["fixed_local_indices"] == {
        "source_line": 1,
        "shot_in_line": 1,
        "relative_receiver_x": 1,
    }
    assert figures["traces"]["target_ids"] == [1, 2, 4]
    reference_limit = 1.05 * np.max(np.abs(pilot["reference"].reshape(4, -1)[:, [1, 2, 4]]))
    trace_view = figures["traces"]["reference_range_view"]
    assert trace_view["ylim"] == pytest.approx([-reference_limit, reference_limit])
    assert trace_view["out_of_range_sample_counts"]["pocs"] == {"1": 0, "2": 0, "4": 0}
    if extreme_prediction:
        assert trace_view["out_of_range_sample_counts"]["relational_trace_graph"] == {
            "1": 4,
            "2": 4,
            "4": 4,
        }
        assert trace_view["out_of_range_total_sample_counts"]["relational_trace_graph"] == 12
    assert figures["traces"]["amplitude_transformation"].startswith("none;")
    assert len(figures["sections"][0]["physical_coordinates_m"]["source_y_m"]) == 3
    assert (
        figures["learning_curves"]["siren5d"]["loss_meaning"]
        != figures["learning_curves"]["ccnet5d"]["loss_meaning"]
    )
    for name in [
        *(item["file"] for item in figures["sections"]),
        "target_traces.png",
        "learning_curve_siren5d.png",
    ]:
        assert (output / name).stat().st_size > 1000
    json.dumps(figures, allow_nan=False)


def _recorded_zero_fill(pilot: dict) -> Path:
    native = pilot["root"] / "zero"
    (native / "artifacts").mkdir(parents=True)
    np.save(native / "artifacts/prediction.npy", pilot["inputs"].observed_volume.values)
    metrics = evaluate_c3_volume_prediction(
        pilot["inputs"].observed_volume.values,
        pilot["inputs"].observed_volume,
        interim_dir=pilot["interim"],
        volume_metadata=pilot["inputs"].volume_metadata,
    )
    metrics.update(
        method="zero_fill",
        scope="full_volume",
        case_id="validation_case",
        suite_sha256=file_sha256(pilot["manifest"]),
    )
    _json(native / "metrics.json", metrics)
    _json(native / "inputs.lock.json", pilot["inputs"].inputs_lock)
    _json(
        native / "run.json",
        {
            "method": "zero_fill",
            "timings": {"evaluation_seconds": 0.25},
            "resources": {"process_max_rss_kib": 1024},
        },
    )
    return native


def test_recorded_zero_fill_preserves_original_measurements(pilot: dict) -> None:
    native = _recorded_zero_fill(pilot)
    before = (native / "metrics.json").read_bytes()
    prediction_before = (native / "artifacts/prediction.npy").read_bytes()
    row = _summarize(pilot, {"zero_fill": native})["rows"][0]
    assert row["evaluation_seconds"] == 0.25
    assert row["process_max_rss_kib"] == 1024
    assert row["metric_comparison"]["status"] == "matched"
    assert row["native_run_path"] == str(native)
    assert (native / "metrics.json").read_bytes() == before
    assert row["prediction_sha256"] == file_sha256(native / "artifacts/prediction.npy")
    assert (native / "artifacts/prediction.npy").read_bytes() == prediction_before
    assert "prediction_sha256" not in row["null_reasons"]
    assert "saved baseline" in row["measurement_scopes"]["summary_re_evaluation_seconds"]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_energy_comparison_uses_stored_dtype_tolerance(pilot: dict, dtype) -> None:
    values = pilot["reference"].astype(dtype) * 0.75
    values[:, pilot["inputs"].observed_volume.observed_trace_mask] = pilot[
        "inputs"
    ].observed_volume.values[:, pilot["inputs"].observed_volume.observed_trace_mask]
    native = _native(pilot, "pocs", prediction=values)
    path = native / "metrics.json"
    metrics = json.loads(path.read_text())
    metrics["evaluation_target"]["error_energy"] *= 1.0 + np.finfo(dtype).eps
    _json(path, metrics)
    row = _summarize(pilot, {"pocs": native})["rows"][1]
    assert row["status"] == "success"
    assert row["metric_comparison"]["relative_tolerance"] == 32 * np.finfo(dtype).eps


@pytest.mark.parametrize("method", ["ccnet5d", "relational_trace_graph"])
def test_final_checkpoint_training_budget_checked(pilot: dict, method: str) -> None:
    native = _native(pilot, method)
    training = pilot["root"] / "training"
    (training / "artifacts").mkdir(parents=True)
    checkpoint = training / "artifacts/final.pt"
    checkpoint.write_bytes(b"synthetic final checkpoint")
    prediction_metadata = json.loads((native / "run.json").read_text())
    prediction_metadata["checkpoint"].update(
        path=str(checkpoint),
        sha256=file_sha256(checkpoint),
        epoch=1,
        global_step=2,
        step=2,
    )
    _json(native / "run.json", prediction_metadata)
    _json(
        training / "run.json",
        {
            "status": "success",
            "checkpoints": {"final": {"role": "final"}},
            "training": {"max_epochs": 2, "max_steps": 3},
        },
    )
    _json(training / "metrics.json", {"epochs_completed": 1, "steps_completed": 2})
    row = _summarize(pilot, {method: native})["rows"][1 + collector.METHODS.index(method)]
    assert row["status"] == "invalid_result"
    assert "training budget" in row["reason"]


def test_blocked_graph_prediction_retains_reason_without_native_run(pilot: dict) -> None:
    outer = pilot["root"] / "blocked_graph"
    outer.mkdir()
    _json(
        outer / "request.json",
        {
            "action": "gnn-predict",
            "case_id": "validation_case",
            "preflight": False,
        },
    )
    _json(
        outer / "result.json",
        {
            "status": "blocked",
            "reason": "gnn-predict requires an explicit final checkpoint",
            "native_run_directory": str(outer / "native"),
        },
    )
    result = _summarize(pilot, {"relational_trace_graph": outer})
    row = result["rows"][-1]
    assert result["status"] == "partial_results"
    assert row["status"] == "blocked"
    assert "final checkpoint" in row["reason"]
    assert row["trace_count"] is None
    assert row["expected_target_count"] == 16
    assert row["expected_sample_count"] == 64
    assert row["snr_db"] is None
    assert not (outer / "native").exists()


def test_ccnet_exposure_counts_spatial_shape_from_native_source_lock(pilot: dict) -> None:
    native = _native(pilot, "ccnet5d")
    training = pilot["root"] / "training"
    (training / "artifacts").mkdir(parents=True)
    checkpoint = training / "artifacts/final.pt"
    checkpoint.write_bytes(b"synthetic final checkpoint")
    metadata = json.loads((native / "run.json").read_text())
    metadata["checkpoint"].update(
        path=str(checkpoint),
        sha256=file_sha256(checkpoint),
        epoch=2,
        global_step=128,
    )
    _json(native / "run.json", metadata)
    _json(
        training / "run.json",
        {
            "status": "success",
            "checkpoints": {"final": {"role": "final"}},
            "training": {"max_epochs": 2},
        },
    )
    _json(training / "metrics.json", {"epochs_completed": 2, "steps_completed": 128})
    _json(
        training / "inputs.lock.json",
        {
            "source_inputs_lock": {
                "regions": {
                    "fit": {"shape": [384, 8, 32, 8, 32]},
                    "selection": {"shape": [384, 2, 32, 8, 32]},
                }
            }
        },
    )
    row = _summarize(pilot, {"ccnet5d": native})["rows"][4]
    assert row["status"] == "success"
    assert row["training_exposure"]["region_trace_counts"] == {"fit": 65536, "selection": 16384}


@pytest.mark.parametrize("changed_role", ["observed", "target", "shape"])
def test_recorded_zero_fill_rejects_changed_saved_prediction(
    pilot: dict, changed_role: str
) -> None:
    native = _recorded_zero_fill(pilot)
    prediction_path = native / "artifacts/prediction.npy"
    values = np.load(prediction_path)
    if changed_role == "shape":
        values = values[:-1]
    else:
        mask = pilot["inputs"].observed_volume.observed_trace_mask
        if changed_role == "target":
            mask = ~mask
        values[:, mask] += 1
    np.save(prediction_path, values)
    with pytest.raises(ValueError, match="saved zero-fill prediction differs"):
        _summarize(pilot, {"zero_fill": native})
    assert not (pilot["root"] / "summary").exists()


def test_siren_resource_scope_covers_fit_and_prediction_process(pilot: dict) -> None:
    native = _native(pilot, "siren5d")
    path = native / "run.json"
    metadata = json.loads(path.read_text())
    amplitude = {"scale_source": "observed_volume", "amplitude_rms": 1.25}
    metadata.update(
        amplitude=amplitude,
        resources={"process_max_rss_kib": 1024, "cuda_max_memory_allocated_bytes": 2048},
    )
    _json(path, metadata)
    row = _summarize(pilot, {"siren5d": native})["rows"][3]
    assert row["status"] == "success"
    assert row["amplitude"] == amplitude
    for field in ("process_max_rss_kib", "cuda_max_memory_allocated_bytes"):
        assert row["measurement_scopes"][field] == "whole native process lifetime peak"


@pytest.mark.parametrize("method", ["ccnet5d", "relational_trace_graph"])
def test_training_and_frozen_amplitude_records_survive_summary(pilot: dict, method: str) -> None:
    native = _native(pilot, method)
    training = pilot["root"] / "training"
    (training / "artifacts").mkdir(parents=True)
    checkpoint = training / "artifacts/final.pt"
    checkpoint.write_bytes(b"synthetic final checkpoint")
    path = native / "run.json"
    metadata = json.loads(path.read_text())
    metadata["checkpoint"].update(
        path=str(checkpoint),
        sha256=file_sha256(checkpoint),
        epoch=1,
        global_step=2,
        step=2,
    )
    scale = (
        {"amplitude_rms": 8.25} if method == "ccnet5d" else {"amplitude_scale": 4.1920812871848e34}
    )
    metadata["amplitude"] = {"scale_source": "checkpoint", **scale}
    _json(path, metadata)
    training_amplitude = {"scale_source": "training_pool", **scale}
    _json(
        training / "run.json",
        {
            "status": "success",
            "checkpoints": {"final": {"role": "final"}},
            "training": {"max_epochs": 1, "max_steps": 2},
            "amplitude": training_amplitude,
        },
    )
    _json(training / "metrics.json", {"epochs_completed": 1, "steps_completed": 2})
    _json(
        training / "inputs.lock.json",
        {
            "source_inputs_lock": {
                "regions": {
                    "fit": {"shape": [4, 1, 1, 2, 4]},
                    "selection": {"shape": [4, 1, 1, 2, 4]},
                }
            }
        },
    )
    result = _summarize(pilot, {method: native})
    row = result["rows"][1 + collector.METHODS.index(method)]
    assert row["status"] == "success"
    assert row["amplitude"] == metadata["amplitude"]
    assert row["training_amplitude"] == training_amplitude
    with (pilot["root"] / "summary/first_results.csv").open() as stream:
        stored = next(row for row in csv.DictReader(stream) if row["method"] == method)
    assert json.loads(stored["amplitude"]) == metadata["amplitude"]
    assert json.loads(stored["training_amplitude"]) == training_amplitude


def test_zero_reference_waveform_view_uses_finite_fallback(pilot: dict) -> None:
    import matplotlib.pyplot as plt

    from seis_interp.visualization.c3_first_results import _plot_traces

    output = pilot["root"] / "zero_reference_traces"
    output.mkdir()
    metadata = _plot_traces(
        plt,
        pilot["inputs"],
        np.zeros((24, 4), dtype=np.float32),
        {"relational_trace_graph": np.full_like(pilot["reference"], 1e30)},
        output,
    )
    view = metadata["reference_range_view"]
    assert metadata["target_ids"] == [1, 2, 4]
    assert view["reference_max_abs_amplitude"] == 0
    assert view["ylim"] == [-1.05, 1.05]
    assert view["out_of_range_total_sample_counts"]["relational_trace_graph"] == 12
    json.dumps(metadata, allow_nan=False)
