"""Prepare immutable M1 data, one outer mask, two geometries, and six configs."""

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from seis_interp.data.forge_aligned_inputs import shared_alignment
from seis_interp.data.forge_headers import sha256_file
from seis_interp.data.forge_mvp_artifacts import (
    load_candidate_inputs,
    mvp_implementation_files,
    observed_global_rms,
    read_selected_traces,
    verify_hashes,
    write_json,
)
from seis_interp.forge_mvp_config import (
    NEURAL_METHODS,
    validate_mvp_configs,
    validate_parameter_counts,
)
from seis_interp.processing.forge_mvp_contract import (
    METHODS,
    make_geometry,
    make_outer_mask,
    validate_mapping,
)
from seis_interp.processing.forge_time_sampling import native_time_contract, resample_forge_traces
from seis_interp.processing.forge_trace_scaling import (
    prepare_oracle_trace_scales,
    prepare_trace_scales,
)
from seis_interp.training.forge_mvp_methods import model_parameter_count
from seis_interp.visualization.forge_mvp import plot_preparation


def prepare_forge_mvp(repo: Path, study: Path) -> Path:
    config = yaml.safe_load((study / "config.yaml").read_text())
    inputs = yaml.safe_load((study / "inputs.yaml").read_text())
    configs = {method: yaml.safe_load((study / f"{method}.yaml").read_text()) for method in METHODS}
    validate_mvp_configs(configs, config)
    native_contract = native_time_contract(config)
    tables, grids, hashes = load_candidate_inputs(repo, inputs, config["candidate_id"])
    manifest = validate_mapping(
        tables["trace_mapping"],
        {r: tables[f"{r}_stations"] for r in ("source", "receiver")},
        tables["grid_cells"],
        grids,
        native_contract,
    )
    mask_config = {**config, "study_id": config.get("mask_study_id", config["study_id"])}
    mask = make_outer_mask(manifest, mask_config)
    reordered = make_outer_mask(manifest.sample(frac=1, random_state=17), mask_config)
    if "outer_mask" in inputs:
        verify_hashes(repo, {inputs["outer_mask"]: inputs["outer_mask_sha256"]})
        pd.testing.assert_frame_equal(mask, pd.read_parquet(repo / inputs["outer_mask"]))
    pd.testing.assert_frame_equal(mask, reordered)
    features = {mode: make_geometry(manifest, grids, mode) for mode in ("real", "grid")}
    # Poisoning actual coordinates must leave the entire grid feature table identical.
    poisoned = manifest.copy()
    poisoned[["source_x_m", "source_y_m", "receiver_x_m", "receiver_y_m"]] = np.nan
    pd.testing.assert_frame_equal(features["grid"], make_geometry(poisoned, grids, "grid"))
    if features["real"].equals(features["grid"]):
        raise ValueError("real and projected geometry are identical")
    alignment = None
    canvas_length = config["time_sample_count"]
    if config["preprocessing"].get("alignment") is not None:
        interval = np.arange(2, dtype=np.float64) * config["sample_interval_us"] / 1e6
        alignment = shared_alignment(
            config["preprocessing"]["alignment"]["velocity_m_per_s"],
            [features["real"], features["grid"]],
            float(interval[1] - interval[0]),
        )
        canvas_length += alignment["max_delay_samples"]
    parameter_counts = {}
    if "matched_training" in config:
        parameter_counts = {
            name: model_parameter_count(configs[name], canvas_length) for name in NEURAL_METHODS
        }
        validate_parameter_counts(parameter_counts, config["matched_training"])
    raw_lock_path = repo / inputs["raw_input_lock"]
    verify_hashes(repo, {inputs["raw_input_lock"]: inputs["raw_input_lock_sha256"]})
    raw_inventory = {r["path"]: r["sha256"] for r in json.loads(raw_lock_path.read_text())}
    raw_hashes = {name: raw_inventory[name] for name in sorted(manifest.source_file.unique())}
    raw_root = repo / inputs["dataset_root"]
    print(
        f"Verifying {len(raw_hashes)} SEG-Y identities; reading observed trace indices only",
        flush=True,
    )
    verify_hashes(raw_root, raw_hashes)
    observed_rows = manifest.loc[mask.split.eq("observed")].reset_index(drop=True)
    values = read_selected_traces(
        raw_root,
        observed_rows,
        native_contract["time_sample_count"],
        native_contract["sample_interval_us"],
    )
    values = resample_forge_traces(values, config)
    rms = observed_global_rms(values)
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain"], text=True
        ).strip()
    )
    started = datetime.now(timezone.utc)
    run_id = started.strftime("%Y%m%dT%H%M%S%fZ") + "_" + commit[:12]
    run = repo / "runs" / config["study_id"] / run_id
    for directory in ("input", "masks", "features", "configs", "figures", "tables"):
        (run / directory).mkdir(parents=True, exist_ok=False)
    write_json(run / "input/input_contract.json", config)
    write_json(run / "input/input_hashes.json", hashes)
    write_json(
        run / "input/raw_input_hashes.json",
        {
            "dataset_root": inputs["dataset_root"],
            "hashes": raw_hashes,
            "inventory": inputs["raw_input_lock"],
            "inventory_sha256": inputs["raw_input_lock_sha256"],
        },
    )
    write_json(run / "input/candidate_grids.json", grids)
    manifest.to_parquet(run / "input/m1_trace_manifest.parquet", index=False)
    observed_rows[["trace_id", "cell_id"]].to_parquet(
        run / "input/observed_index.parquet", index=False
    )
    np.save(run / "input/observed_waveforms.npy", values, allow_pickle=False)
    mask_path = run / "masks/m1_random80_mvp_v1.parquet"
    if "outer_mask" in inputs:
        shutil.copyfile(repo / inputs["outer_mask"], mask_path)
    else:
        mask.to_parquet(mask_path, index=False)
    mask_hash = sha256_file(mask_path)
    write_json(
        run / "masks/mask_manifest.json",
        {
            "mask_hash": mask_hash,
            "mask_id": config["mask_id"],
            "mask_seed": config["mask_seed"],
            "study_id": mask_config["study_id"],
            "candidate_id": config["candidate_id"],
            "algorithm": (
                "SHA256(UTF-8 canonical compact JSON list "
                "[study_id,candidate_id,mask_id,integer mask_seed,integer cell_id]); "
                "digest ascending, then cell_id ascending; hash_rank is 1-based"
            ),
            "counts": mask.split.value_counts().to_dict(),
            "row_order_invariant": True,
        },
    )
    for mode, table in features.items():
        table.to_parquet(run / "features" / f"{mode}_geometry.parquet", index=False)
    write_json(
        run / "features/feature_hashes.json",
        {mode: sha256_file(run / "features" / f"{mode}_geometry.parquet") for mode in features},
    )
    amplitude_binding = {}
    test_waveforms_read = 0
    if config["preprocessing"]["amplitude_scaling"] == "observed_trace_rms_idw":
        scales, spacing = prepare_trace_scales(
            values,
            observed_rows.cell_id.to_numpy(),
            manifest.cell_id.to_numpy(),
            tuple(config["spatial_shape"]),
            grids,
            config["preprocessing"]["idw"],
        )
        scales.to_parquet(run / "input/trace_scales.parquet", index=False)
        amplitude_binding = {
            "amplitude_scaling": "observed_trace_rms_idw",
            "trace_scale_hash": sha256_file(run / "input/trace_scales.parquet"),
            "idw": {**config["preprocessing"]["idw"], "axis_scales_m": list(spacing)},
        }
        write_json(
            run / "input/amplitude_contract.json",
            {
                **amplitude_binding,
                "global_rms_role": "diagnostic_only",
                "training_targets": "unit_trace_rms",
                "model_scalar_scale": 1.0,
                "inner_target_rms_usage": "label_normalization_only",
                "outer_target_amplitudes_read": 0,
                "prediction_restoration": "model_output_times_idw_rms_once",
                "zero_rms_policy": "stop",
                "uncovered_idw_policy": "stop",
            },
        )
    elif config["preprocessing"]["amplitude_scaling"] == "trace_oracle":
        test_rows = manifest.loc[mask.split.eq("test")].reset_index(drop=True)
        print("Computing authorized oracle test RMS after time resampling", flush=True)
        test_values = read_selected_traces(
            raw_root,
            test_rows,
            native_contract["time_sample_count"],
            native_contract["sample_interval_us"],
        )
        test_values = resample_forge_traces(test_values, config)
        test_waveforms_read = len(test_values)
        scales = prepare_oracle_trace_scales(
            values,
            observed_rows.cell_id.to_numpy(),
            test_values,
            test_rows.cell_id.to_numpy(),
        )
        oracle_trace_count = len(test_values)
        del test_values
        scales.to_parquet(run / "input/trace_scales.parquet", index=False)
        amplitude_binding = {
            "amplitude_scaling": "trace_oracle",
            "trace_scale_hash": sha256_file(run / "input/trace_scales.parquet"),
        }
        write_json(
            run / "input/amplitude_contract.json",
            {
                **amplitude_binding,
                "global_rms_role": "diagnostic_only",
                "training_targets": "unit_trace_rms",
                "model_scalar_scale": 1.0,
                "inner_target_rms_usage": "label_normalization_only",
                "outer_target_amplitudes_read": oracle_trace_count,
                "outer_target_usage": "oracle_rms_only; no held-out waveform enters training",
                "prediction_restoration": "model_output_times_oracle_rms_once",
                "zero_rms_policy": "stop",
            },
        )
    for method, resolved in configs.items():
        resolved.update(amplitude_binding)
        if alignment is not None:
            resolved["alignment"] = dict(alignment)
        resolved.update(
            mask_hash=mask_hash,
            mask_id=config["mask_id"],
            mask_seed=config["mask_seed"],
            global_rms=rms,
        )
        (run / "configs" / f"{method}.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
    pd.DataFrame(
        [
            {
                "method_id": method,
                "geometry_mode": cfg["geometry_mode"],
                "model_seed": cfg["model_seed"],
                "mask_hash": mask_hash,
                "status": "not_started",
            }
            for method, cfg in configs.items()
        ]
    ).to_csv(run / "tables/run_matrix.csv", index=False)
    # Every cell carries a unique synthetic signature: no averaging or permutation.
    cells = manifest.cell_id.to_numpy()
    signatures = np.column_stack((cells, -cells)).astype(np.float32)
    arranged = np.zeros((int(np.prod(config["spatial_shape"])), 2), dtype=np.float32)
    arranged[cells] = signatures
    if not np.array_equal(arranged[cells], signatures):
        raise ValueError("no-mask mapping round trip failed")
    observed_arranged = np.zeros(
        (int(np.prod(config["spatial_shape"])), values.shape[1]), dtype=np.float32
    )
    observed_arranged[observed_rows.cell_id] = values
    if not np.array_equal(observed_arranged[observed_rows.cell_id], values):
        raise ValueError("observed waveform round trip failed")
    del observed_arranged
    plot_preparation(manifest, mask, features, run / "figures")
    gates = {
        "mapping": True,
        "mask": True,
        "row_order_invariant": True,
        "grid_actual_coordinate_poisoning": True,
        "geometry_differs": True,
        "no_mask_round_trip_synthetic_all_cells": True,
        "waveform_round_trip_observed_only": True,
        "test_waveforms_read": test_waveforms_read,
        "observed_waveforms_stored": len(values),
        "regsi_configs_equal_except_geometry": True,
        "shared_alignment": alignment,
        "matched_parameter_counts": parameter_counts,
        "evaluation_runtime_gates": "pending predictions",
        "production_runs_started": 0,
    }
    write_json(run / "preparation_gates.json", gates)
    implementation_hashes = {}
    for source in mvp_implementation_files(repo):
        relative = source.relative_to(repo)
        implementation_hashes[str(relative)] = sha256_file(source)
        destination = run / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    for source in study.glob("*.yaml"):
        target = run / "source" / source.relative_to(repo)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    artifacts = {
        str(p.relative_to(run)): sha256_file(p) for p in sorted(run.rglob("*")) if p.is_file()
    }
    seal = {
        **amplitude_binding,
        "study_id": config["study_id"],
        "candidate_id": config["candidate_id"],
        "mask_id": config["mask_id"],
        "mask_seed": config["mask_seed"],
        "git_commit": commit,
        "dirty_worktree": dirty,
        "status": "prepared",
        "start_time": started.isoformat(),
        "end_time": datetime.now(timezone.utc).isoformat(),
        "input_hashes": hashes,
        "mapping_hash": sha256_file(run / "input/m1_trace_manifest.parquet"),
        "mask_hash": mask_hash,
        "global_rms": rms,
        "observed_trace_count": len(values),
        "test_trace_count": int(mask.split.eq("test").sum()),
        "clean_test_trace_count": int((mask.split.eq("test") & mask.clean_target).sum()),
        "time_sample_count": values.shape[1],
        **({"alignment": alignment, "canvas_sample_count": canvas_length} if alignment else {}),
        **({"parameter_counts": parameter_counts} if parameter_counts else {}),
        "implementation_hashes": implementation_hashes,
        "artifacts": artifacts,
    }
    write_json(run / "preparation_manifest.json", seal)
    for path in run.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    return run
