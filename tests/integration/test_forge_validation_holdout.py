import hashlib
import json
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import segyio
import torch
import yaml

from seis_interp.data.forge_aligned_inputs import moveout_canvas
from seis_interp.data.forge_headers import sha256_file
from seis_interp.data.forge_mvp_artifacts import load_model_inputs, write_json
from seis_interp.data.forge_mvp_run_records import execution_identity
from seis_interp.pipelines.evaluate_forge_mvp import evaluate_forge_mvp
from seis_interp.pipelines.fair_grid_stage import load_fair_grid_context
from seis_interp.pipelines.prepare_forge_mvp import prepare_forge_mvp
from seis_interp.pipelines.run_forge_mvp import run_forge_method
from seis_interp.pipelines.tune_forge_matched import tune_forge_matched
from seis_interp.pipelines.visualize_forge_mvp import visualize_forge_mvp
from seis_interp.processing.forge_mvp_contract import METHODS
from seis_interp.processing.forge_time_sampling import resample_forge_traces
from seis_interp.training.forge_mvp_methods import graph_data
from seis_interp.training.forge_mvp_preflight import check_method_shapes
from tests.fixtures.forge_mvp import tiny_forge_study


class _RecordingRaw:
    def __init__(self, raw, name, log):
        self._raw, self._name, self._log = raw, name, log

    def __getitem__(self, index):
        self._log.append((self._name, int(index)))
        return self._raw[index]


@pytest.fixture
def segy_reads(monkeypatch):
    """Record every (file, trace index) read through the SEG-Y reader."""
    log = []
    original = segyio.open

    @contextmanager
    def recording_open(path, *args, **kwargs):
        with original(path, *args, **kwargs) as handle:
            yield SimpleNamespace(
                bin=handle.bin,
                samples=handle.samples,
                trace=SimpleNamespace(raw=_RecordingRaw(handle.trace.raw, Path(path).name, log)),
            )

    monkeypatch.setattr(segyio, "open", recording_open)
    return log


@pytest.fixture
def git_in_repository(monkeypatch):
    repo = Path(__file__).resolve().parents[2]
    original = subprocess.check_output

    def check_output(args, **kwargs):
        if args[:2] == ["git", "-C"]:
            args = [*args[:2], str(repo), *args[3:]]
        return original(args, **kwargs)

    monkeypatch.setattr(subprocess, "check_output", check_output)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _holdout_study(tmp_path):
    """21 outer-observed cells; 10 become validation targets, 14 outer test cells excluded."""
    fixture = tiny_forge_study(tmp_path)
    study, config, _, configs = fixture[:4]
    config.update(
        test_trace_count=14,
        observed_trace_count=21,
        missing_fraction=0.4,
        time_sample_count=5,
        sample_interval_us=2000,
    )
    config["time_sampling"] = {
        "native_sample_count": 9,
        "native_sample_interval_us": 1000,
        "decimation_factor": 2,
    }
    config["preprocessing"].update(
        amplitude_scaling="trace_oracle",
        trace_rms=True,
        alignment={"velocity_m_per_s": 50000.0},
    )
    config["validation_holdout"] = {"fraction": 0.5, "mask_id": "tiny_validation", "mask_seed": 11}
    (study / "config.yaml").write_text(yaml.safe_dump(config))
    configs["nersi_real"]["model"]["time_sample_count"] = 5
    configs["ccnet5d_grid"]["patches"]["shape"][0] = 5
    for method in ("nersi_real", "ccnet5d_grid"):
        (study / f"{method}.yaml").write_text(yaml.safe_dump(configs[method]))
    return study, config, fixture[-1]


@pytest.fixture
def preparations(tmp_path, git_in_repository, segy_reads):
    study, config, values = _holdout_study(tmp_path)
    final = prepare_forge_mvp(tmp_path, study)
    segy_reads.clear()
    validation = prepare_forge_mvp(tmp_path, study, validation_holdout=True)
    reads = list(segy_reads)
    return final, validation, config, resample_forge_traces(values, config), reads


def _cells_read(preparation, reads):
    manifest = pd.read_parquet(preparation / "input/m1_trace_manifest.parquet")
    cells = manifest.set_index(["source_file", "trace_index"]).cell_id
    return [int(cells.loc[key]) for key in reads]


def _split(preparation, name):
    mask = pd.read_parquet(preparation / "masks/m1_random80_mvp_v1.parquet")
    return mask.loc[mask.split.eq(name), "cell_id"].to_numpy()


def test_validation_preparation_reads_outer_observed_traces_only(preparations):
    final, validation, config, values, reads = preparations
    assert validation.name.endswith("_validation") and not final.name.endswith("_validation")
    final_observed, final_test = _split(final, "observed"), _split(final, "test")
    observed, held_out = _split(validation, "observed"), _split(validation, "test")
    excluded = _split(validation, "excluded")
    assert (len(observed), len(held_out), len(excluded)) == (11, 10, 14)
    np.testing.assert_array_equal(np.sort(np.concatenate((observed, held_out))), final_observed)
    np.testing.assert_array_equal(excluded, final_test)
    # Each outer-observed waveform is read exactly once; no outer test waveform is read.
    cells = _cells_read(validation, reads)
    assert sorted(cells) == sorted(final_observed.tolist())
    assert not set(cells) & set(excluded)
    seal = json.loads((validation / "preparation_manifest.json").read_text())
    final_seal = json.loads((final / "preparation_manifest.json").read_text())
    gates = json.loads((validation / "preparation_gates.json").read_text())
    assert gates["test_waveforms_read"] == gates["excluded_waveforms_read"] == 0
    assert gates["outer_observed_waveforms_read"] == 21
    assert gates["validation_targets_from_outer_observed"] == 10
    assert seal["selection_domain"] == "validation_holdout"
    assert (seal["observed_trace_count"], seal["test_trace_count"]) == (11, 10)
    assert seal["excluded_trace_count"] == 14 and seal["excluded_waveforms_read"] == 0
    assert (seal["parent_observed_trace_count"], seal["parent_test_trace_count"]) == (21, 14)
    assert seal["parent_mask_hash"] == final_seal["mask_hash"]
    assert seal["parent_mask_hash"] == sha256_file(validation / "masks/parent_mask.parquet")
    assert seal["validation_mask_hash"] == seal["mask_hash"]
    assert seal["mask_hash"] == sha256_file(validation / "masks/m1_random80_mvp_v1.parquet")
    assert (seal["mask_id"], seal["mask_seed"]) == ("tiny_validation", 11)
    assert (seal["parent_mask_id"], seal["parent_mask_seed"]) == (
        config["mask_id"],
        config["mask_seed"],
    )
    assert "selection_domain" not in final_seal
    contract = json.loads((validation / "input/input_contract.json").read_text())
    assert (contract["observed_trace_count"], contract["test_trace_count"]) == (11, 10)
    # Only O \\ V enters the global RMS and the stored model inputs.
    stored = np.load(validation / "input/observed_waveforms.npy")
    np.testing.assert_array_equal(stored, values[observed])
    final_stored = np.load(final / "input/observed_waveforms.npy")
    np.testing.assert_array_equal(stored, final_stored[np.isin(final_observed, observed)])
    assert seal["global_rms"] == pytest.approx(
        np.sqrt(np.square(values[observed].astype(float)).mean())
    )
    scales = pd.read_parquet(validation / "input/trace_scales.parquet").set_index("cell_id")
    final_scales = pd.read_parquet(final / "input/trace_scales.parquet").set_index("cell_id")
    assert set(scales.index) == set(observed) | set(held_out)
    assert scales.loc[held_out, "scale_source"].eq("oracle_test").all()
    np.testing.assert_array_equal(
        scales.loc[held_out, "rms"],
        np.sqrt(np.mean(np.square(values[held_out].astype(np.float64)), axis=1)),
    )
    np.testing.assert_array_equal(scales.rms, final_scales.loc[scales.index, "rms"])


def test_validation_preparation_keeps_final_geometry_and_canvas(preparations):
    final, validation, _, _, _ = preparations
    seal = json.loads((validation / "preparation_manifest.json").read_text())
    final_seal = json.loads((final / "preparation_manifest.json").read_text())
    for key in ("mapping_hash", "alignment", "canvas_sample_count", "time_sample_count"):
        assert seal[key] == final_seal[key]
    assert seal["canvas_sample_count"] > seal["time_sample_count"]
    for name in ("features/feature_hashes.json", "input/m1_trace_manifest.parquet"):
        assert sha256_file(validation / name) == sha256_file(final / name)
    for method in METHODS:
        inputs, config = load_model_inputs(validation, method)
        final_inputs, final_config = load_model_inputs(final, method)
        assert config["alignment"] == final_config["alignment"]
        pd.testing.assert_frame_equal(inputs.geometry, final_inputs.geometry)
        np.testing.assert_array_equal(inputs.test_cells, _split(validation, "test"))
        np.testing.assert_array_equal(inputs.excluded_cells, _split(validation, "excluded"))
        canvas = moveout_canvas(inputs, config["alignment"])
        assert canvas.length == seal["canvas_sample_count"]
        np.testing.assert_array_equal(
            canvas.delays, moveout_canvas(final_inputs, config["alignment"]).delays
        )
        volume = inputs.volume()
        flat = volume.values.reshape(volume.values.shape[0], -1)
        assert np.all(flat[:, inputs.excluded_cells] == 0)
        assert not volume.observed_trace_mask.reshape(-1)[inputs.excluded_cells].any()
        assert not volume.evaluation_target_trace_mask.reshape(-1)[inputs.excluded_cells].any()
        with pytest.raises(ValueError, match="outside outer observed"):
            inputs.read_observed(inputs.test_cells[:1])
        assert len(inputs.model_space().observed_values) == 11
        if method.startswith("regsi"):
            domain, training, _ = graph_data(inputs, config)
            final_domain, final_training, _ = graph_data(final_inputs, final_config)
            np.testing.assert_array_equal(domain.trace_ids, final_domain.trace_ids)
            assert (
                training.preprocessing.midpoint_origin_m
                == final_training.preprocessing.midpoint_origin_m
            )
            assert int(domain.observed_mask.sum()) == 11


def test_every_method_predicts_exactly_the_validation_cells(preparations, tmp_path, segy_reads):
    _, validation, _, _, _ = preparations
    held_out = _split(validation, "test")
    excluded = set(_split(validation, "excluded").tolist())
    segy_reads.clear()
    output = tmp_path / "validation_matrix"
    output.mkdir()
    write_json(
        output / "matrix_manifest.json",
        {
            "methods": list(METHODS),
            "preparation": str(validation.relative_to(tmp_path)),
            "preparation_hash": sha256_file(validation / "preparation_manifest.json"),
            **execution_identity(tmp_path),
        },
    )
    for method in METHODS:
        inputs, config = load_model_inputs(validation, method)
        check_method_shapes(inputs, config)
        run_forge_method(tmp_path, validation, output / "runs" / method, method)
        index = pd.read_parquet(output / "runs" / method / "prediction_index.parquet")
        np.testing.assert_array_equal(index.cell_id, held_out)
        assert index.split.eq("test").all()
        prediction = np.load(output / "runs" / method / "prediction.npy")
        assert prediction.shape == (10, 5) and np.isfinite(prediction).all()
    assert segy_reads == []
    result = evaluate_forge_mvp(tmp_path, validation, output)
    assert result["status"] == "accepted", result
    # Scoring reads the validation targets only, never an excluded trace.
    assert sorted(_cells_read(validation, segy_reads)) == sorted(held_out.tolist())
    for method in METHODS:
        record = json.loads((output / "runs" / method / "run_manifest.json").read_text())
        assert record["status"] == "evaluated" and record["test_trace_count"] == 10
        traces = pd.read_parquet(output / "runs" / method / "metrics_by_trace.parquet")
        np.testing.assert_array_equal(np.sort(traces.cell_id), held_out)
    segy_reads.clear()
    visualize_forge_mvp(tmp_path, output)
    assert segy_reads and not set(_cells_read(validation, segy_reads)) & excluded


def test_validation_flag_requires_declared_holdout_before_reading(
    tmp_path, git_in_repository, segy_reads
):
    study = tiny_forge_study(tmp_path)[0]
    with pytest.raises(ValueError, match="validation_holdout requires"):
        prepare_forge_mvp(tmp_path, study, validation_holdout=True)
    assert segy_reads == []
    assert not (tmp_path / "runs").exists()


def test_matched_tuner_scores_every_shadow_on_the_validation_cells_only(
    tmp_path, git_in_repository, segy_reads
):
    study, _, _ = _holdout_study(tmp_path)
    for method in ("nersi_real", "regsi_real", "regsi_grid"):
        config = yaml.safe_load((study / f"{method}.yaml").read_text())
        config["training"].update(max_steps=4, report_interval=2)
        (study / f"{method}.yaml").write_text(yaml.safe_dump(config))
    validation = prepare_forge_mvp(tmp_path, study, validation_holdout=True)
    script = Path(__file__).resolve().parents[2] / "scripts/forge_matched_tuning.py"
    (tmp_path / "scripts" / script.name).write_bytes(script.read_bytes())
    held_out = _split(validation, "test")
    for method in ("nersi_real", "regsi_real"):
        segy_reads.clear()
        search = tmp_path / f"{method}_search.yaml"
        candidate = {"name": "grid", "training": {"ema_eval_decays": [0.5, 0.9, 0.999]}}
        search.write_text(
            yaml.safe_dump(
                {
                    "method_id": method,
                    "target_relative_mse": 0.0,
                    "device": "cpu",
                    "candidates": [candidate, {"name": "seed", "model_seed": 7302}],
                }
            )
        )
        output = tune_forge_matched(tmp_path, validation, search)
        # Scoring reads the validation targets once; no excluded outer test trace is read.
        assert sorted(_cells_read(validation, segy_reads)) == sorted(held_out.tolist())
        record = json.loads((output / "manifest.json").read_text())
        assert record["status"] == "completed" and record["target_met"] is False
        assert record["selection_domain"] == "validation_holdout"
        assert record["independent_test_estimate"] is False
        index = pd.read_parquet(output / "prediction_index.parquet")
        np.testing.assert_array_equal(index.cell_id, held_out)
        rows = pd.read_csv(output / "search_results.csv")
        assert rows.name.tolist() == ["grid", "grid", "grid", "seed"]
        assert rows.ema_decay.astype(str).tolist()[:3] == ["0.5", "0.9", "0.999"]
        assert rows.model_seed.tolist() == [7301, 7301, 7301, 7302]
        assert rows.primary_relative_mse.nunique() == 4
        for decay in ("0.5", "0.9", "0.999"):
            traces = pd.read_parquet(output / f"grid/ema_{decay}/metrics_by_trace.parquet")
            np.testing.assert_array_equal(np.sort(traces.cell_id), held_out)
            assert np.load(output / f"grid/ema_{decay}/prediction.npy").shape == (10, 5)


def _registered_grid(final, validation):
    """A fair grid registering the two tiny preparations, as grid.yaml does for study_060."""
    seal = json.loads((validation / "preparation_manifest.json").read_text())
    final_seal = json.loads((final / "preparation_manifest.json").read_text())
    cells = np.sort(_split(validation, "test")).astype(np.int64)
    return {
        "study_id": seal["study_id"],
        "preparations": {
            "final_mask_hash": final_seal["mask_hash"],
            "validation_mask_hash": seal["mask_hash"],
            "validation_cells_sha256": hashlib.sha256(cells.tobytes()).hexdigest(),
            "observed_trace_counts": {"validation": 11, "final": 21},
        },
        "methods": {
            "regsi": {"method_id": "regsi_real", "pair_method_id": "regsi_grid"},
            "nersi_real": {"method_id": "nersi_real"},
        },
    }


def _edited(grid, path, value):
    edited = json.loads(json.dumps(grid))
    *parents, leaf = path
    node = edited
    for key in parents:
        node = node[key]
    node[leaf] = value
    return edited


def _with_budget(preparation, budget):
    """Add a matched budget to the sealed contract (the tiny study has none) and reseal it."""
    path = preparation / "input/input_contract.json"
    contract = json.loads(path.read_text())
    _rewrite(path, {**contract, "matched_training": budget})
    seal_path = preparation / "preparation_manifest.json"
    seal = json.loads(seal_path.read_text())
    seal["artifacts"]["input/input_contract.json"] = sha256_file(path)
    _rewrite(seal_path, seal)


def _rewrite(path, value):
    """Replace a sealed (read-only) JSON file of a test preparation."""
    path.chmod(0o644)
    path.write_text(json.dumps(value))


def test_fair_grid_context_accepts_only_the_registered_preparations(preparations, tmp_path):
    final, validation, *_ = preparations
    study = Path(__file__).resolve().parents[2] / "studies/study_060_forge_fair_grid"
    budget = yaml.safe_load((study / "config.yaml").read_text())["matched_training"]
    for preparation in (final, validation):
        _with_budget(preparation, budget)
    grid = _registered_grid(final, validation)
    context = load_fair_grid_context(tmp_path, grid, validation, final)
    assert context.preparations == {
        domain: {
            "path": str(path.relative_to(tmp_path)),
            "manifest_sha256": sha256_file(path / "preparation_manifest.json"),
        }
        for domain, path in (("validation", validation), ("final", final))
    }
    assert {d: int(m.sum()) for d, m in context.observed_masks.items()} == {
        "validation": 11,
        "final": 21,
    }
    assert set(context.base_configs["final"]) == {"regsi_real", "regsi_grid", "nersi_real"}
    seal = json.loads((final / "preparation_manifest.json").read_text())
    assert context.canvas_sample_count == seal["canvas_sample_count"]
    assert context.budget == budget
    faults = [
        (("preparations", "validation_mask_hash"), "0" * 64, "pre-registered masks"),
        (("preparations", "final_mask_hash"), "0" * 64, "pre-registered masks"),
        (("preparations", "validation_cells_sha256"), "0" * 64, "validation cells differ"),
        (("preparations", "observed_trace_counts", "validation"), 12, "validation observed"),
        (("preparations", "observed_trace_counts", "final"), 20, "final observed count"),
        (("study_id",), "other_study", "another study"),
    ]
    for path, value, message in faults:
        with pytest.raises(ValueError, match=message):
            load_fair_grid_context(tmp_path, _edited(grid, path, value), validation, final)
    with pytest.raises(ValueError, match="pre-registered masks"):
        load_fair_grid_context(tmp_path, grid, final, validation)
    # A final preparation on another alignment canvas is refused.
    other = tmp_path / "runs" / "other_alignment"
    shutil.copytree(final, other)
    manifest = json.loads((other / "preparation_manifest.json").read_text())
    manifest["alignment"] = {**manifest["alignment"], "velocity_m_per_s": 1.0}
    _rewrite(other / "preparation_manifest.json", manifest)
    with pytest.raises(ValueError, match="differ in alignment"):
        load_fair_grid_context(tmp_path, grid, validation, other)
    other = tmp_path / "runs" / "other_budget"
    shutil.copytree(final, other)
    _with_budget(other, {**budget, "max_steps": 1})
    with pytest.raises(ValueError, match="differ in the matched budget"):
        load_fair_grid_context(tmp_path, grid, validation, other)


def test_matched_tuner_refuses_to_score_the_final_test_of_a_holdout_study(
    preparations, tmp_path, segy_reads
):
    final, *_ = preparations
    script = Path(__file__).resolve().parents[2] / "scripts/forge_matched_tuning.py"
    (tmp_path / "scripts" / script.name).write_bytes(script.read_bytes())
    runs = final.parent
    segy_reads.clear()
    refused = [
        {"target_relative_mse": 1.0},
        {"purpose": "final_evaluation", "target_relative_mse": 1.0},
        {"evaluate": False},
    ]
    for number, body in enumerate(refused):
        search = tmp_path / f"refused_{number}.yaml"
        body = {"method_id": "pocs_grid", "device": "cpu", "candidates": [{"name": "c"}], **body}
        search.write_text(yaml.safe_dump(body))
        with pytest.raises(ValueError, match="selects on its validation holdout"):
            tune_forge_matched(tmp_path, final, search)
    assert segy_reads == [] and not list(runs.glob("*_matched_tuning"))
    search = tmp_path / "final.yaml"
    body = {"purpose": "final_evaluation", "evaluate": False}
    search.write_text(
        yaml.safe_dump(
            {"method_id": "pocs_grid", "device": "cpu", "candidates": [{"name": "c"}], **body}
        )
    )
    output = tune_forge_matched(tmp_path, final, search)
    record = json.loads((output / "manifest.json").read_text())
    assert record["status"] == "completed" and record["independent_test_estimate"] is True
    assert segy_reads == [] and not list(output.rglob("metrics*"))
