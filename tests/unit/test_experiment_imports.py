import json
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import yaml

from seis_interp.data.experiment_imports import (
    find_prediction_import,
    imported_prediction,
    method_config_hash,
    saved_input_fingerprint,
)
from seis_interp.data.forge_headers import sha256_file


def source_run(root, *, forge=False, method="pocs_grid", time_count=2):
    source = root / "source_experiment"
    (source / "study").mkdir(parents=True)
    (source / "source/src").mkdir(parents=True)
    (source / "source/src/fitter.py").write_text("# frozen fitting implementation\n")
    contract = {
        "spatial_shape": [1, 1, 1, 4],
        "time_sample_count": time_count,
        "sample_interval_us": 4000,
        "validation_holdout": {"mask_seed": 19},
    }
    (source / "study/config.yaml").write_text(yaml.safe_dump(contract))
    directory = source / "input" if forge else source / "input/seed_19"
    directory.mkdir(parents=True)
    partition = pd.DataFrame(
        {
            "cell_id": [0, 1, 2, 3],
            "role" if forge else "trace_role": ["train", "train", "validation", "test"],
        }
    )
    partition.to_parquet(directory / "partition.parquet", index=False)
    geometry = "grid" if method in ("pocs_grid", "drr_grid") else "real"
    pd.DataFrame({"cell_id": [0, 1, 2, 3], "x": [0.0, 1.0, 2.0, 3.0]}).to_parquet(
        source / "input" / (f"{geometry}_geometry.parquet" if forge else "geometry.parquet"),
        index=False,
    )
    np.save(directory / "observed_waveforms.npy", np.ones((2, time_count), dtype=np.float32))
    if forge:
        pd.DataFrame({"cell_id": [0, 1, 2, 3], "rms": [1.0, 2.0, 3.0, 4.0]}).to_parquet(
            directory / "trace_scales.parquet", index=False
        )
    else:
        np.save(source / "input/time_s.npy", np.arange(time_count, dtype=np.float64) * 4000 / 1e6)
    (source / "experiment.json").write_text(
        json.dumps({"global_rms": 1.0} if forge else {"partitions": {"19": {"global_rms": 1.0}}})
    )
    config = {
        "method_id": method,
        "device": "cpu",
        "algorithm": {"n_iterations": 16},
        "execution": {"backend": "batched"},
    }
    if method == "regsi_real":
        config.pop("algorithm")
        config.update(model={"node_fourier_components": 0}, training={"inner_mask_fraction": 0.25})
    elif method == "nersi_profile":
        config.pop("algorithm")
        config["training"] = {"coordinate_jitter_cells": 0.5}
    run = source / "method_runs" / method
    if not forge:
        run = run / "S1/a01_b01"
    run.mkdir(parents=True)
    (run / "config.resolved.yaml").write_text(yaml.safe_dump(config))
    np.save(run / "cell_ids.npy", np.array([2, 3] if forge else [2]))
    np.save(run / "prediction.npy", np.zeros((2 if forge else 1, time_count), dtype=np.float32))
    record = {
        "method_id": method,
        "status": "completed",
        "artifacts": {
            name: sha256_file(run / name)
            for name in ["config.resolved.yaml", "prediction.npy", "cell_ids.npy"]
        },
    }
    if not forge:
        record["trace_seed"] = 19
    (run / "manifest.json").write_text(json.dumps(record))
    return source, run, directory, config


@pytest.mark.parametrize("forge,time_count", [(True, 1001), (False, 384)])
def test_nersi_prediction_import_uses_input_canvas_not_yaml_length(tmp_path, forge, time_count):
    source, run, _, config = source_run(
        tmp_path, forge=forge, method="nersi_profile", time_count=time_count
    )
    config["model"] = {"profile_shape": [384 if forge else 2176, 32]}
    if forge:
        config["alignment"] = {"velocity_m_per_s": 500.0, "max_delay_samples": 1171}
    (run / "config.resolved.yaml").write_text(yaml.safe_dump(config))
    record = json.loads((run / "manifest.json").read_text())
    record["artifacts"]["config.resolved.yaml"] = sha256_file(run / "config.resolved.yaml")
    (run / "manifest.json").write_text(json.dumps(record))
    fingerprint = saved_input_fingerprint(source, 19, "nersi_profile", ("validation",))
    assert find_prediction_import(tmp_path, [source], config, fingerprint, 19, "S1") is not None
    enabled = deepcopy(config)
    enabled["canvas_time_upsampling"] = True
    match = find_prediction_import(tmp_path, [source], enabled, fingerprint, 19, "S1")
    assert (match is None) == forge


def test_forge_upsampling_import_requires_matching_jitter_loss_and_inputs(tmp_path):
    source, run, _, config = source_run(
        tmp_path, forge=True, method="nersi_profile", time_count=1001
    )
    config.update(
        canvas_time_upsampling=True,
        alignment={"velocity_m_per_s": 500.0, "max_delay_samples": 1171},
        model={"profile_shape": [2176, 32]},
    )
    saved = deepcopy(config)
    saved["training"]["loss_support_mask"] = False
    (run / "config.resolved.yaml").write_text(yaml.safe_dump(saved))
    record = json.loads((run / "manifest.json").read_text())
    record["artifacts"]["config.resolved.yaml"] = sha256_file(run / "config.resolved.yaml")
    (run / "manifest.json").write_text(json.dumps(record))
    fingerprint = saved_input_fingerprint(source, 19, "nersi_profile", ("validation",))

    assert find_prediction_import(tmp_path, [source], config, fingerprint, 19, "S1") is not None
    for key, value in [("coordinate_jitter_cells", None), ("loss_support_mask", True)]:
        changed = deepcopy(config)
        changed["training"][key] = value
        assert find_prediction_import(tmp_path, [source], changed, fingerprint, 19, "S1") is None
    assert find_prediction_import(tmp_path, [source], config, "different inputs", 19, "S1") is None
    config.pop("canvas_time_upsampling")
    assert find_prediction_import(tmp_path, [source], config, fingerprint, 19, "S1") is None


def test_execution_allowlist_preserves_scientific_and_precision_settings():
    first = {
        "device": "cuda:0",
        "training": {"mixed_precision": "bf16", "max_steps": 50000},
        "execution": {"saved_tensors_on_cpu": False},
        "prediction": {"query_batch_size": 16, "core_shape": [4, 8]},
    }
    second = deepcopy(first)
    second.update(device="cuda:1", execution={"saved_tensors_on_cpu": True})
    second["prediction"]["query_batch_size"] = 32
    assert method_config_hash(first) == method_config_hash(second)
    for key, value in [
        ("mixed_precision", "off"),
        ("max_steps", 49999),
        ("coordinate_jitter_cells", 0.5),
    ]:
        changed = deepcopy(second)
        changed["training"][key] = value
        assert method_config_hash(changed) != method_config_hash(first)
    second["prediction"]["core_shape"] = [8, 8]
    assert method_config_hash(second) != method_config_hash(first)


@pytest.mark.parametrize("forge", [False, True])
def test_native_and_single_forge_matching_uses_saved_inputs_and_prediction_completion(
    tmp_path, forge
):
    source, run, directory, config = source_run(tmp_path, forge=forge)
    fingerprint = saved_input_fingerprint(source, 19, "pocs_grid", ("validation",))
    provenance = find_prediction_import(tmp_path, [source], config, fingerprint, 19, "S1")
    assert provenance["source_path"] == str(run)
    assert len(provenance["source_snapshot_sha256"]) == 64
    changed = deepcopy(config)
    changed["algorithm"]["n_iterations"] = 32
    assert find_prediction_import(tmp_path, [source], changed, fingerprint, 19, "S1") is None
    np.save(directory / "observed_waveforms.npy", np.full((2, 2), 2, dtype=np.float32))
    assert saved_input_fingerprint(source, 19, "pocs_grid", ("validation",)) != fingerprint
    assert find_prediction_import(tmp_path, [source], config, fingerprint, 19, "S1") is None
    np.save(directory / "observed_waveforms.npy", np.ones((2, 2), dtype=np.float32))
    record = json.loads((run / "manifest.json").read_text())
    record["status"] = "running"
    (run / "manifest.json").write_text(json.dumps(record))
    assert find_prediction_import(tmp_path, [source], config, fingerprint, 19, "S1") is None
    record["status"] = "completed"
    (run / "manifest.json").write_text(json.dumps(record))
    (run / "prediction.npy").unlink()
    assert find_prediction_import(tmp_path, [source], config, fingerprint, 19, "S1") is None


def test_partition_and_scale_changes_reject_reuse(tmp_path):
    source, _, directory, config = source_run(tmp_path, forge=True)
    initial = saved_input_fingerprint(source, 19, "pocs_grid", ("validation",))
    scales = pd.read_parquet(directory / "trace_scales.parquet")
    scales.loc[2, "rms"] = 9
    scales.to_parquet(directory / "trace_scales.parquet", index=False)
    assert find_prediction_import(tmp_path, [source], config, initial, 19, "S1") is None
    scaled = saved_input_fingerprint(source, 19, "pocs_grid", ("validation",))
    partition = pd.read_parquet(directory / "partition.parquet")
    partition.loc[[2, 3], "role"] = ["test", "validation"]
    partition.to_parquet(directory / "partition.parquet", index=False)
    assert find_prediction_import(tmp_path, [source], config, scaled, 19, "S1") is None


def test_handmade_grid_candidate_ids_are_adapted_from_candidates_json(tmp_path):
    source, single, _, config = source_run(tmp_path, forge=True)
    run = single / "S1/legacy_candidate"
    run.mkdir(parents=True)
    for path in list(single.iterdir()):
        if path.is_file():
            path.rename(run / path.name)
    (source / "candidates.json").write_text(
        json.dumps({"pocs_grid": [{"id": "legacy_candidate", "config": config}]})
    )
    fingerprint = saved_input_fingerprint(source, 19, "pocs_grid", ("validation",))
    match = find_prediction_import(tmp_path, [source], config, fingerprint, 19, "S1")
    assert match["source_path"] == str(run)
    # A summary alone cannot make an unfinished candidate reusable.
    (source / "pocs_grid_summary.csv").write_text("id,validation_score\nlegacy_candidate,0.1\n")
    (run / "manifest.json").write_text(json.dumps({"status": "running"}))
    assert find_prediction_import(tmp_path, [source], config, fingerprint, 19, "S1") is None


def test_completed_prediction_must_cover_current_domains(tmp_path):
    source, run, _, config = source_run(tmp_path)
    fingerprint = saved_input_fingerprint(source, 19, "pocs_grid", ("validation", "test"))
    assert find_prediction_import(tmp_path, [run], config, fingerprint, 19, "evaluation") is None


@pytest.mark.parametrize("forge", [False, True])
@pytest.mark.parametrize("method", ["pocs_grid", "regsi_real", "nersi_profile"])
def test_pilot_candidate_layout_requires_config_inputs_and_completed_prediction(
    tmp_path, forge, method
):
    source, original, directory, config = source_run(tmp_path, forge=forge, method=method)
    single = forge and method == "nersi_profile"
    run = original
    if forge and not single:
        run = original / "S1/a02_b04"
        run.mkdir(parents=True)
        for path in list(original.iterdir()):
            if path.is_file():
                path.rename(run / path.name)
    seal_path = source / "experiment.json"
    pilot = {"pocs_grid": "P1b", "regsi_real": "P2", "nersi_profile": "P3"}[method]
    seal = json.loads(seal_path.read_text()) | {"pilot": pilot, "validation_only": True}
    seal_path.write_text(json.dumps(seal))
    (source / "candidates.json").write_text(
        json.dumps({method: [{"id": run.name if not single else "a02_b04", "config": config}]})
    )
    (source / f"{method}_summary.csv").write_text(
        "id,status,validation_score\na02_b04,completed,0.001\n"
    )
    pd.DataFrame({"cell_id": [2], "snr_db": [99.0]}).to_parquet(
        run / "validation_per_trace.parquet", index=False
    )
    fingerprint = saved_input_fingerprint(source, 19, method, ("validation",))
    current = deepcopy(config)
    current["execution"]["cpu_threads"] = 4
    match = find_prediction_import(tmp_path, [source], current, fingerprint, 19, "S1")
    assert match["source_path"] == str(run)
    np.testing.assert_array_equal(imported_prediction(match, [2]), np.zeros((1, 2)))
    if method == "pocs_grid":
        current["algorithm"]["n_iterations"] = 32
    elif method == "regsi_real":
        current["training"]["inner_mask_fraction"] = 0.1
    else:
        current["training"].pop("coordinate_jitter_cells")
    assert find_prediction_import(tmp_path, [source], current, fingerprint, 19, "S1") is None
    np.save(directory / "observed_waveforms.npy", np.zeros((2, 2), dtype=np.float32))
    assert find_prediction_import(tmp_path, [source], config, fingerprint, 19, "S1") is None
    np.save(directory / "observed_waveforms.npy", np.ones((2, 2), dtype=np.float32))
    record = json.loads((run / "manifest.json").read_text())
    (run / "manifest.json").write_text(json.dumps(record | {"status": "running"}))
    assert find_prediction_import(tmp_path, [source], config, fingerprint, 19, "S1") is None
    (run / "manifest.json").write_text(json.dumps(record))
    np.save(run / "prediction.npy", np.full((2 if forge else 1, 2), 9, dtype=np.float32))
    with pytest.raises(ValueError, match="input hash mismatch: prediction.npy"):
        find_prediction_import(tmp_path, [source], config, fingerprint, 19, "S1")
    (run / "prediction.npy").unlink()
    assert find_prediction_import(tmp_path, [source], config, fingerprint, 19, "S1") is None


def test_validation_only_pilot_cannot_supply_evaluation_even_with_full_predictions(tmp_path):
    source, run, _, config = source_run(tmp_path, forge=True)
    path = source / "experiment.json"
    path.write_text(json.dumps(json.loads(path.read_text()) | {"validation_only": True}))
    fingerprint = saved_input_fingerprint(source, 19, "pocs_grid", ("validation", "test"))
    assert find_prediction_import(tmp_path, [run], config, fingerprint, 19, "evaluation") is None


def test_pilot_summary_cannot_authorize_an_undeclared_candidate(tmp_path):
    source, _, _, config = source_run(tmp_path)
    (source / "candidates.json").write_text(json.dumps({"pocs_grid": []}))
    (source / "pocs_grid_summary.csv").write_text("id,status\na01_b01,completed\n")
    fingerprint = saved_input_fingerprint(source, 19, "pocs_grid", ("validation",))
    assert find_prediction_import(tmp_path, [source], config, fingerprint, 19, "S1") is None


@pytest.mark.parametrize("overlap_policy", ["explicit", "point_half"])
def test_legacy_drr_prediction_import_requires_zero_overlap(tmp_path, overlap_policy):
    from seis_interp.processing.c3_region_search import resolve_c3_candidate

    source, run, _, legacy = source_run(tmp_path, method="drr_grid")
    legacy["algorithm"].update(
        rank=1,
        damping_power=2,
        frequency_min_hz=0.0,
        frequency_max_hz=None,
        spatial_window_shape=[1, 2, 1, 4],
        spatial_overlap=[0] * 4,
    )
    (run / "config.resolved.yaml").write_text(yaml.safe_dump(legacy))
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["artifacts"]["config.resolved.yaml"] = sha256_file(run / "config.resolved.yaml")
    (run / "manifest.json").write_text(json.dumps(manifest))
    fingerprint = saved_input_fingerprint(source, 19, "drr_grid", ("validation",))
    current = resolve_c3_candidate(legacy, 2, 64)
    match = find_prediction_import(tmp_path, [source], current, fingerprint, 19, "S1")
    assert match["source_path"] == str(run)
    current = deepcopy(legacy)
    current["algorithm"].update(
        overlap_policy=overlap_policy,
        spatial_overlap=[0, 1, 0, 2],
    )
    assert find_prediction_import(tmp_path, [source], current, fingerprint, 19, "S1") is None
    # A saved clip-and-shift run at overlap zero is a different computation;
    # skip it without aborting the scan of historical import sources.
    legacy["algorithm"]["boundary_rule"] = "clip_and_shift"
    (run / "config.resolved.yaml").write_text(yaml.safe_dump(legacy))
    manifest["artifacts"]["config.resolved.yaml"] = sha256_file(run / "config.resolved.yaml")
    (run / "manifest.json").write_text(json.dumps(manifest))
    current = deepcopy(legacy)
    current["algorithm"].pop("boundary_rule")
    assert find_prediction_import(tmp_path, [source], current, fingerprint, 19, "S1") is None
