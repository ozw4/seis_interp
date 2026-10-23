"""C3 fair-grid preparations in the FORGE layout, read by the FORGE planner and tuner.

A tiny synthetic C3 crop (``tests/fixtures/c3_fair_grid.py``) is prepared once per
module as a final (51 observed / 205 test) and a validation (41 observed / 10 V /
205 excluded) preparation.
"""

import hashlib
import json
import shutil
import stat
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from seis_interp.data.c3_fair_grid_truth import read_c3_scored_targets
from seis_interp.data.c3_volume_index_store import load_c3_volume_index
from seis_interp.data.forge_headers import sha256_file
from seis_interp.data.forge_mvp_artifacts import load_model_inputs, verify_hashes
from seis_interp.pipelines import prepare_c3_fair_grid as preparation_pipeline
from seis_interp.pipelines import tune_forge_matched as tuner
from seis_interp.pipelines.fair_grid_stage import load_fair_grid_context, plan_fair_grid_stage
from seis_interp.pipelines.prepare_c3_fair_grid import prepare_c3_fair_grid
from seis_interp.pipelines.tune_forge_matched import (
    _classical_batched,
    matched_candidate_config,
    tune_forge_matched,
)
from seis_interp.processing.c3_fair_grid_contract import (
    c3_geometry_table,
    c3_trace_manifest,
    make_c3_outer_mask,
)
from seis_interp.processing.fair_grid_rules import batch_aligned_fraction
from seis_interp.processing.forge_mvp_contract import make_validation_holdout
from seis_interp.processing.forge_nersi_profiles import nersi_profile_layout
from seis_interp.training.forge_mvp_methods import model_parameter_count, run_method
from tests.fixtures.c3_fair_grid import (
    BUDGET,
    DECAYS,
    INTERIM,
    SPATIAL_SHAPE,
    STUDY_ID,
    TIME_COUNT,
    TRACES,
    tiny_c3_study,
)

SEAL = "preparation_manifest.json"
MASK = "masks/m1_random80_mvp_v1.parquet"
METHODS = ("pocs_grid", "drr_grid", "ccnet5d_grid", "regsi_real", "nersi_profile")
NEURAL = ("ccnet5d_grid", "regsi_real", "nersi_profile")
FINAL_SEAL_KEYS = {
    "amplitude_scaling",
    "dataset_id",
    "study_id",
    "candidate_id",
    "mask_id",
    "mask_seed",
    "git_commit",
    "dirty_worktree",
    "status",
    "start_time",
    "end_time",
    "input_hashes",
    "mapping_hash",
    "mask_hash",
    "global_rms",
    "observed_trace_count",
    "test_trace_count",
    "clean_test_trace_count",
    "time_sample_count",
    "test_waveforms_read",
    "alignment",
    "canvas_sample_count",
    "parameter_counts",
    "implementation_hashes",
    "artifacts",
}
VALIDATION_SEAL_KEYS = {
    "selection_domain",
    "validation_fraction",
    "validation_mask_hash",
    "parent_mask_hash",
    "parent_mask_id",
    "parent_mask_seed",
    "parent_mask_study_id",
    "parent_observed_trace_count",
    "parent_test_trace_count",
    "excluded_trace_count",
    "excluded_waveforms_read",
}
COUNTS = {
    "final": {"observed": 51, "test": 205},
    "validation": {"observed": 41, "test": 10, "excluded": 205},
}
REAL_VOLUME = "data/processed/c3_na/study_041_c3_nersi_translated_window/shot18_ry18/volume"


def _git(args, **kwargs):
    return "testcommit\n" if args[-1] == "HEAD" else ""


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("c3_fair_grid")
    reads = []
    original = preparation_pipeline.read_c3_amplitude_rows

    def recorded(path, rows, count):
        reads.append(np.asarray(rows).copy())
        return original(path, rows, count)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(preparation_pipeline.subprocess, "check_output", _git)
        patch.setattr(preparation_pipeline, "read_c3_amplitude_rows", recorded)
        study, contract, configs = tiny_c3_study(root)
        runs = {
            "final": prepare_c3_fair_grid(root, study),
            "validation": prepare_c3_fair_grid(root, study, validation_holdout=True),
        }
    script = Path(__file__).resolve().parents[2] / "scripts/forge_matched_tuning.py"
    (root / "scripts" / script.name).write_bytes(script.read_bytes())
    amplitudes = np.load(root / INTERIM / "amplitudes.npy")
    return SimpleNamespace(
        root=root,
        study=study,
        contract=contract,
        configs=configs,
        runs=runs,
        reads=reads,
        amplitudes=amplitudes,
    )


@pytest.fixture(autouse=True)
def single_thread(monkeypatch):
    monkeypatch.setattr(
        tuner, "execution_identity", lambda _: {"git_commit": "tunecommit", "dirty_worktree": True}
    )
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _seal(run):
    return json.loads((run / SEAL).read_text())


def _rows(prepared, cells):
    """Interim rows of the given cells, samples 0:TIME_COUNT."""
    manifest = pd.read_parquet(prepared.runs["final"] / "input/c3_trace_manifest.parquet")
    rows = manifest.set_index("cell_id").loc[np.asarray(cells), "array_row"].to_numpy()
    return prepared.amplitudes[rows, :TIME_COUNT]


# --- preparation --------------------------------------------------------------------------


@pytest.mark.parametrize("domain", ["final", "validation"])
def test_preparation_has_the_forge_layout_and_seal(prepared, domain):
    run = prepared.runs[domain]
    seal = _seal(run)
    validation = domain == "validation"
    assert run.parent == prepared.root / "runs" / STUDY_ID
    assert run.name.endswith("_testcommit" + ("_validation" if validation else ""))
    assert set(seal) == FINAL_SEAL_KEYS | (VALIDATION_SEAL_KEYS if validation else set())
    files = {str(path.relative_to(run)) for path in run.rglob("*") if path.is_file()}
    assert files == set(seal["artifacts"]) | {SEAL}
    verify_hashes(run, seal["artifacts"])
    for path in run.rglob("*"):
        if path.is_file():
            assert stat.S_IMODE(path.stat().st_mode) == 0o444, path
    expected = {
        "input/input_contract.json",
        "input/input_hashes.json",
        "input/c3_truth.json",
        "input/c3_trace_manifest.parquet",
        "input/observed_index.parquet",
        "input/observed_waveforms.npy",
        MASK,
        "masks/mask_manifest.json",
        "features/real_geometry.parquet",
        "features/grid_geometry.parquet",
        "features/feature_hashes.json",
        "preparation_gates.json",
        "tables/run_matrix.csv",
        "source/scripts/c3_fair_grid_prepare.py",
        *(f"configs/{method}.yaml" for method in METHODS),
    }
    assert expected <= files
    assert ("masks/parent_mask.parquet" in files) == validation
    assert seal["status"] == "prepared" and seal["dataset_id"] == "seg_c3_na"
    assert seal["amplitude_scaling"] == "observed_global_rms"
    assert seal["alignment"] is None
    assert seal["canvas_sample_count"] == seal["time_sample_count"] == TIME_COUNT
    assert seal["test_waveforms_read"] == 0
    mask = pd.read_parquet(run / MASK)
    assert mask.split.value_counts().to_dict() == COUNTS[domain]
    assert seal["observed_trace_count"] == COUNTS[domain]["observed"]
    assert seal["test_trace_count"] == seal["clean_test_trace_count"] == COUNTS[domain]["test"]
    assert seal["mask_hash"] == sha256_file(run / MASK)
    assert seal["mapping_hash"] == sha256_file(run / "input/c3_trace_manifest.parquet")
    inputs = yaml.safe_load((prepared.study / "inputs.yaml").read_text())
    assert seal["input_hashes"] == {
        inputs[name]: inputs[f"{name}_sha256"]
        for name in ("volume_index", "volume_metadata", "amplitudes", "time_s")
    }
    assert seal["parameter_counts"] == {
        name: model_parameter_count(prepared.configs[name], TIME_COUNT) for name in NEURAL
    }
    # Observed waveforms are the interim rows of the mask's observed cells, in mask order.
    observed = mask.cell_id[mask.split.eq("observed")].to_numpy()
    index = pd.read_parquet(run / "input/observed_index.parquet")
    np.testing.assert_array_equal(index.cell_id, observed)
    values = np.load(run / "input/observed_waveforms.npy")
    assert values.dtype == np.float32 and values.shape == (len(observed), TIME_COUNT)
    np.testing.assert_array_equal(values, _rows(prepared, observed))
    rms = np.sqrt(np.mean(np.square(values.astype(np.float64))))
    assert seal["global_rms"] == pytest.approx(rms, rel=1e-12)
    # One exact-lattice geometry table for both modes; array_row stays out of it.
    real = (run / "features/real_geometry.parquet").read_bytes()
    assert real == (run / "features/grid_geometry.parquet").read_bytes()
    volume, _ = load_c3_volume_index(prepared.root / "data/processed/tiny_c3/volume")
    pd.testing.assert_frame_equal(
        pd.read_parquet(run / "features/real_geometry.parquet"),
        c3_geometry_table(volume, SPATIAL_SHAPE),
    )
    assert json.loads((run / "input/c3_truth.json").read_text()) == {
        "amplitudes": inputs["amplitudes"],
        "amplitudes_sha256": inputs["amplitudes_sha256"],
        "time_s": inputs["time_s"],
        "time_s_sha256": inputs["time_s_sha256"],
        "time_sample_count": TIME_COUNT,
        "sample_interval_us": 8000,
    }
    for method in METHODS:
        config = yaml.safe_load((run / "configs" / f"{method}.yaml").read_text())
        assert config == {
            **prepared.configs[method],
            "amplitude_scaling": "observed_global_rms",
            "mask_hash": seal["mask_hash"],
            "mask_id": seal["mask_id"],
            "mask_seed": seal["mask_seed"],
            "global_rms": seal["global_rms"],
        }


def test_validation_preparation_nests_in_the_final_mask(prepared):
    final, validation = (prepared.runs[d] for d in ("final", "validation"))
    final_seal, seal = _seal(final), _seal(validation)
    assert seal["parent_mask_hash"] == final_seal["mask_hash"]
    assert (validation / "masks/parent_mask.parquet").read_bytes() == (final / MASK).read_bytes()
    assert seal["validation_mask_hash"] == seal["mask_hash"] != final_seal["mask_hash"]
    assert seal["mapping_hash"] == final_seal["mapping_hash"]
    assert seal["parent_mask_study_id"] == STUDY_ID
    assert (seal["parent_observed_trace_count"], seal["parent_test_trace_count"]) == (51, 205)
    assert seal["excluded_trace_count"] == 205 and seal["excluded_waveforms_read"] == 0
    assert seal["global_rms"] != final_seal["global_rms"]
    parent, mask = pd.read_parquet(final / MASK), pd.read_parquet(validation / MASK)
    expected = make_validation_holdout(parent, prepared.contract)
    pd.testing.assert_frame_equal(mask.fillna(""), expected.fillna(""))
    assert set(mask.cell_id[mask.split.ne("excluded")]) == set(
        parent.cell_id[parent.split.eq("observed")]
    )
    contract = json.loads((validation / "input/input_contract.json").read_text())
    assert contract["selection_domain"] == "validation_holdout"
    assert contract["observed_trace_count"] == 41 and contract["test_trace_count"] == 10


def test_only_the_observed_rows_of_each_preparation_are_read(prepared):
    assert len(prepared.reads) == 2
    manifest = pd.read_parquet(prepared.runs["final"] / "input/c3_trace_manifest.parquet")
    for rows, domain in zip(prepared.reads, ("final", "validation"), strict=True):
        mask = pd.read_parquet(prepared.runs[domain] / MASK)
        observed = mask.cell_id[mask.split.eq("observed")]
        expected = manifest.set_index("cell_id").loc[observed, "array_row"].to_numpy()
        np.testing.assert_array_equal(rows, expected)


def test_preparation_rejects_inputs_that_differ_from_the_contract(prepared, tmp_path):
    root = tmp_path / "copy"
    shutil.copytree(prepared.root / "data", root / "data")
    shutil.copytree(prepared.root / "scripts", root / "scripts")
    study = root / "studies" / STUDY_ID
    shutil.copytree(prepared.study, study)
    config = yaml.safe_load((study / "config.yaml").read_text())
    (study / "config.yaml").write_text(yaml.safe_dump({**config, "time_sample_count": 16}))
    for method in ("ccnet5d_grid", "nersi_profile"):
        body = yaml.safe_load((study / f"{method}.yaml").read_text())
        if method == "ccnet5d_grid":
            body["patches"]["shape"][0] = 16
            body["prediction"]["core_shape"][0] = 16
        else:
            body["model"]["profile_shape"][0] = 16
        (study / f"{method}.yaml").write_text(yaml.safe_dump(body))
    with pytest.raises(ValueError, match="does not match the C3 contract crop"):
        prepare_c3_fair_grid(root, study)
    for name in ("config", "ccnet5d_grid", "nersi_profile"):
        shutil.copyfile(prepared.study / f"{name}.yaml", study / f"{name}.yaml")
    inputs = yaml.safe_load((study / "inputs.yaml").read_text())
    (study / "inputs.yaml").write_text(yaml.safe_dump({**inputs, "amplitudes_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="input hash mismatch"):
        prepare_c3_fair_grid(root, study)
    assert not (root / "runs").exists()


# --- model inputs and the truth reader ----------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
def test_every_method_loads_and_predicts_physical_test_traces(prepared, method):
    inputs, config = load_model_inputs(prepared.runs["validation"], method)
    assert inputs.observed_rms is None and inputs.test_rms is None
    assert inputs.global_rms == _seal(prepared.runs["validation"])["global_rms"]
    assert (len(inputs.observed_cells), len(inputs.test_cells), len(inputs.excluded_cells)) == (
        41,
        10,
        205,
    )
    np.testing.assert_allclose(inputs.time_s, np.arange(TIME_COUNT) * 0.008)
    result = run_method(inputs, config, reporter=lambda message: None)
    assert result.prediction.shape == (10, TIME_COUNT) and result.prediction.dtype == np.float32
    assert np.isfinite(result.prediction).all()
    if method in NEURAL:
        assert (
            result.diagnostics["parameter_count"]
            == _seal(prepared.runs["validation"])["parameter_counts"][method]
        )


@pytest.mark.parametrize("method", ["pocs_grid", "drr_grid"])
def test_batched_classical_search_matches_the_global_rms_run(prepared, method):
    # Both paths fit the observed traces over the global RMS and restore it once.
    inputs, config = load_model_inputs(prepared.runs["validation"], method)
    expected = run_method(inputs, config, reporter=lambda message: None).prediction
    batched = _classical_batched(inputs, config, {"device": "cpu", "workers": 1})
    np.testing.assert_array_equal(batched.prediction, expected)


def test_profile_coordinates_are_the_c3_index_coordinates(prepared):
    inputs, _ = load_model_inputs(prepared.runs["final"], "nersi_profile")
    layout = nersi_profile_layout(inputs.geometry, inputs.spatial_shape)
    lines, shots, rx, _ = SPATIAL_SHAPE
    assert layout.receiver_points == SPATIAL_SHAPE[3]
    expected = np.column_stack(
        [
            axis.ravel() / (n - 1)
            for axis, n in zip(np.indices((lines, shots, rx)), (lines, shots, rx), strict=True)
        ]
    )
    np.testing.assert_array_equal(layout.coordinates, expected)


@pytest.mark.parametrize("domain", ["final", "validation"])
def test_truth_reader_returns_the_scored_rows_in_mask_order(prepared, domain):
    run = prepared.runs[domain]
    mask = pd.read_parquet(run / MASK)
    target = read_c3_scored_targets(prepared.root, run, mask)
    scored = mask.cell_id[mask.split.eq("test")].to_numpy()
    assert target.dtype == np.float32 and target.shape == (COUNTS[domain]["test"], TIME_COUNT)
    np.testing.assert_array_equal(target, _rows(prepared, scored))


def test_truth_reader_verifies_the_interim_arrays(prepared, tmp_path):
    root = tmp_path / "copy"
    shutil.copytree(prepared.root / "data", root / "data")
    run = prepared.runs["final"]
    mask = pd.read_parquet(run / MASK)
    path = root / INTERIM / "amplitudes.npy"
    values = np.load(path)
    values[0, 0] += 1.0
    np.save(path, values, allow_pickle=False)
    with pytest.raises(ValueError, match="input hash mismatch"):
        read_c3_scored_targets(root, run, mask)


# --- matched tuning -----------------------------------------------------------------------


def _search(root, method, candidates, **options):
    path = root / f"search_{method}_{len(list(root.glob('search_*.yaml')))}.yaml"
    body = {"method_id": method, "device": "cpu", "torch_threads": 1, "candidates": candidates}
    path.write_text(yaml.safe_dump({**body, **options}))
    return path


def _counted_reads(monkeypatch):
    calls = []

    def counted(repo, preparation, mask):
        values = read_c3_scored_targets(repo, preparation, mask)
        calls.append(len(values))
        return values

    monkeypatch.setattr(tuner, "read_c3_scored_targets", counted)
    return calls


@pytest.mark.parametrize(
    "method,candidates,options",
    [
        (
            "pocs_grid",
            [{"name": "it2"}, {"name": "it3", "algorithm": {"n_iterations": 3}}],
            {"backend": "batched"},
        ),
        (
            "nersi_profile",
            [
                {
                    "name": "shadows",
                    "training": {"ema_eval_decays": DECAYS, "ema_snapshot_updates": [2]},
                }
            ],
            {},
        ),
    ],
)
def test_matched_search_scores_the_holdout_from_the_interim_rows(
    prepared, monkeypatch, method, candidates, options
):
    root, run = prepared.root, prepared.runs["validation"]
    reads = _counted_reads(monkeypatch)
    search = _search(root, method, candidates, target_relative_mse=1e9, **options)
    output = tune_forge_matched(root, run, search)
    record = json.loads((output / "manifest.json").read_text())
    verify_hashes(output, record["artifacts"])
    assert record["status"] == "completed" and record["selection_domain"] == "validation_holdout"
    assert record["alignment"] is None and record["matched_training"] == BUDGET
    assert reads == [10]
    mask = pd.read_parquet(run / MASK)
    target = _rows(prepared, mask.cell_id[mask.split.eq("test")]).astype(np.float64)
    results = pd.read_csv(output / "search_results.csv", dtype={"ema_decay": str})
    assert len(results) == (len(candidates) if method == "pocs_grid" else len(DECAYS))
    for row in results.itertuples():
        directory = output / row.name / (f"ema_{row.ema_decay}" if method != "pocs_grid" else "")
        prediction = np.load(directory / "prediction.npy").astype(np.float64)
        expected = np.mean(((prediction - target) ** 2).sum(1) / (target**2).sum(1))
        assert row.primary_relative_mse == pytest.approx(expected, rel=1e-10)


def test_final_retraining_reads_no_target(prepared, monkeypatch):
    reads = _counted_reads(monkeypatch)
    search = _search(
        prepared.root,
        "pocs_grid",
        [{"name": "final"}],
        purpose="final_evaluation",
        evaluate=False,
        backend="batched",
    )
    output = tune_forge_matched(prepared.root, prepared.runs["final"], search)
    assert reads == []
    assert json.loads((output / "manifest.json").read_text())["status"] == "completed"
    assert np.load(output / "final/prediction.npy").shape == (205, TIME_COUNT)


# --- stage planning -----------------------------------------------------------------------


def _values(path, values, prefix):
    return [{"id": f"{prefix}{i}", "set": {path: value}} for i, value in enumerate(values)]


def _neural(method_id, axis1, axis2, **extra):
    return {
        "method_id": method_id,
        "device": "cpu",
        "fixed": {"training.weight_decay": 0.0},
        "axis1": {"name": "first", "ordered": True, "values": axis1},
        "axis2": {"name": "second", "ordered": True, "values": axis2},
        "s2": [],
        "weight_decay": {"path": "training.weight_decay", "s3": [1e-4], "extension": {}},
        **extra,
    }


def _grid(prepared):
    final, validation = (_seal(prepared.runs[d]) for d in ("final", "validation"))
    mask = pd.read_parquet(prepared.runs["validation"] / MASK)
    cells = np.sort(mask.cell_id[mask.split.eq("test")].to_numpy()).astype(np.int64)
    pairs = [
        {
            "nominal": nominal,
            "search": batch_aligned_fraction(nominal, 41, TRACES),
            "final": batch_aligned_fraction(nominal, 51, TRACES),
        }
        for nominal in (0.25, 0.5, 0.75)
    ]
    frequency = [1.05, 1.1, 1.15]
    return {
        "study_id": STUDY_ID,
        "preparations": {
            "final_mask_hash": final["mask_hash"],
            "validation_mask_hash": validation["mask_hash"],
            "validation_cells_sha256": hashlib.sha256(cells.tobytes()).hexdigest(),
            "observed_trace_counts": {"validation": 41, "final": 51},
        },
        "selection": {
            "score": "primary_relative_mse",
            "train_loss": "final_training_loss_window_mean",
            "wd_calibration_max_relative_increase": 0.05,
            "reseed_top_runs": 2,
            "diverged_runs": {"failure_reason_pattern": "non-?finite", "reported_score": 1.0},
        },
        "ema_eval_decays": DECAYS,
        "ema_snapshot_updates": [2],
        "seeds": {"search": 7301, "reseed": [7302, 7303], "final": [7311, 7312, 7313]},
        "methods": {
            "regsi": _neural(
                "regsi_real",
                _values("graph.neighbors_per_relation", [1, 2, 3], "k"),
                _values("training.inner_mask_fraction", [p["search"] for p in pairs], "imf"),
                inner_mask_pairs=pairs,
            ),
            "ccnet5d_grid": _neural(
                "ccnet5d_grid",
                _values("patches.inner_mask_fraction", [0.3, 0.4, 0.5], "f"),
                _values("model.time_dilation", [1, 2, 3], "d"),
            ),
            "nersi_profile": _neural(
                "nersi_profile",
                [
                    {
                        "id": f"src{i}",
                        "set": {"model.frequency_base.0": v, "model.frequency_base.1": v},
                    }
                    for i, v in enumerate(frequency)
                ],
                _values("model.frequency_base.2", frequency, "rx"),
            ),
            "pocs_grid": {
                "method_id": "pocs_grid",
                "device": "cpu",
                "execution": {"backend": "batched"},
                "fixed": {"algorithm.threshold_start": 1.0},
                "factors": [
                    {"values": _values("algorithm.n_iterations", [2, 3], "it")},
                    {"values": _values("algorithm.threshold_end", [0.1, 0.2], "end")},
                ],
            },
            "drr_grid": {
                "method_id": "drr_grid",
                "device": "cpu",
                "execution": {"backend": "batched", "workers": 1},
                "fixed": {},
                "factors": [{"values": _values("algorithm.rank", [1, 2], "r")}],
            },
        },
    }


def test_stage_planner_writes_s1_from_the_c3_preparations(prepared):
    root = prepared.root
    grid = _grid(prepared)
    grid_path = root / "studies" / STUDY_ID / "grid.yaml"
    grid_path.write_text(yaml.safe_dump(grid, sort_keys=False))
    context = load_fair_grid_context(
        root, grid, prepared.runs["validation"], prepared.runs["final"]
    )
    assert context.canvas_sample_count == TIME_COUNT and context.budget == BUDGET
    assert set(context.base_configs["validation"]) == set(METHODS)
    sizes = {"regsi": 9, "ccnet5d_grid": 9, "nersi_profile": 9, "pocs_grid": 4, "drr_grid": 2}
    for method, size in sizes.items():
        directory = plan_fair_grid_stage(root, grid_path, method, "S1", context)
        document = json.loads((directory / "stage.json").read_text())
        assert len(document["runs"]) == size
        assert document["preparations"]["validation"]["path"] == str(
            prepared.runs["validation"].relative_to(root)
        )
        for run in document["runs"]:
            search = yaml.safe_load((root / run["yaml"]).read_text())
            assert search["purpose"] == "selection" and search["evaluate"] is True
            checks = run["feasibility"]
            assert {checks[d]["observed_trace_count"] for d in checks} == {41, 51}
            base = context.base_configs["validation"][run["method_id"]]
            config, parameters = matched_candidate_config(
                base, search["candidates"][0], "cpu", BUDGET, TIME_COUNT, 41
            )
            assert checks["validation"]["parameter_count"] == parameters
            if method == "regsi":
                assert all(checks[d]["hidden_traces_per_episode"] % TRACES == 0 for d in checks)
            if method == "ccnet5d_grid":
                assert all(checks[d]["min_hidden_traces_per_patch"] >= TRACES for d in checks)
            if method in ("regsi", "ccnet5d_grid", "nersi_profile"):
                assert config["training"]["ema_eval_decays"] == DECAYS


# --- the real C3 crop (skipped without the local data) ------------------------------------


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / REAL_VOLUME).is_dir(),
    reason="local SEG C3 NA volume index is not available",
)
def test_real_c3_crop_masks_have_the_registered_counts():
    volume, metadata = load_c3_volume_index(Path(__file__).resolve().parents[2] / REAL_VOLUME)
    shape = metadata["shape"][1:]
    assert shape == [16, 32, 8, 32]
    config = {
        "study_id": "study_062_c3_v4_fair_grid",
        "candidate_id": "c3_v3_shot18_ry18",
        "mask_id": "c3_v4_random80_fair_v1",
        "mask_seed": 620701,
        "eligible_trace_count": 131072,
        "test_trace_count": 104858,
        "observed_trace_count": 26214,
        "validation_holdout": {
            "fraction": 0.1,
            "mask_id": "c3_v4_random80_validation_v1",
            "mask_seed": 620702,
        },
    }
    manifest = c3_trace_manifest(volume, shape, config["candidate_id"])
    np.testing.assert_array_equal(manifest.cell_id, np.arange(131072))
    mask = make_c3_outer_mask(manifest, config)
    holdout = make_validation_holdout(mask, config)
    assert holdout.split.value_counts().to_dict() == {
        "excluded": 104858,
        "observed": 23593,
        "test": 2621,
    }
    assert mask.source_cell.nunique() == 512
