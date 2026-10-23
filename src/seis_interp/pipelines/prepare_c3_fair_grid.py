"""Prepare an immutable C3 fair-grid run in the layout of a FORGE preparation.

The study's ``config.yaml`` is the contract (``dataset_id: seg_c3_na``),
``inputs.yaml`` binds the C3 volume index, its ``volume.json`` and the interim
amplitude and time arrays by repo-relative path and sha256, and
``<method>.yaml`` holds each of the five method configs. The preparation is read
by the FORGE fair-grid stage planner, matched tuner and final evaluator; the
scored rows are read later through ``input/c3_truth.json``.

With ``validation_holdout`` the same outer mask is built and a hash-selected
fraction of its observed traces becomes the validation ``test`` split; the outer
test traces become ``excluded``. Only the observed rows of the preparation are
read: no test (or validation) waveform is read here.
"""

import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from seis_interp.c3_fair_grid_config import (
    C3_FAIR_GRID_METHODS,
    C3_NEURAL_METHODS,
    validate_c3_fair_grid_configs,
)
from seis_interp.data.c3_fair_grid_truth import (
    C3_TRACE_MANIFEST,
    C3_TRUTH,
    c3_truth_record,
    read_c3_amplitude_rows,
    validate_c3_time_axis,
)
from seis_interp.data.c3_volume_index_store import (
    VOLUME_INDEX_FILE_NAME,
    VOLUME_METADATA_FILE_NAME,
    load_c3_volume_index,
)
from seis_interp.data.forge_headers import sha256_file
from seis_interp.data.forge_mvp_artifacts import observed_global_rms, verify_hashes, write_json
from seis_interp.forge_mvp_config import validate_parameter_counts
from seis_interp.processing.c3_fair_grid_contract import (
    C3_DATASET_ID,
    c3_geometry_table,
    c3_trace_manifest,
    make_c3_outer_mask,
)
from seis_interp.processing.forge_mvp_contract import make_validation_holdout, validation_contract
from seis_interp.training.forge_mvp_methods import model_parameter_count

MASK_PATH = "masks/m1_random80_mvp_v1.parquet"
SCRIPT = "scripts/c3_fair_grid_prepare.py"
INPUT_NAMES = ("volume_index", "volume_metadata", "amplitudes", "time_s")
# Optional provenance file (e.g. the C3 v3 conditions lock), verified and recorded only.
PROVENANCE_NAME = "conditions_lock"


def prepare_c3_fair_grid(repo: Path, study: Path, *, validation_holdout: bool = False) -> Path:
    config = yaml.safe_load((study / "config.yaml").read_text())
    inputs = yaml.safe_load((study / "inputs.yaml").read_text())
    configs = {
        method: yaml.safe_load((study / f"{method}.yaml").read_text())
        for method in C3_FAIR_GRID_METHODS
    }
    # The validation contract carries the holdout counts; the mask is still built from config.
    contract = validation_contract(config) if validation_holdout else config
    validate_c3_fair_grid_configs(configs, contract)
    hashes = c3_input_hashes(inputs)
    volume_index = _load_volume_index(repo, inputs, config)
    manifest = c3_trace_manifest(volume_index, config["spatial_shape"], config["candidate_id"])
    geometry = c3_geometry_table(volume_index, config["spatial_shape"])
    mask = make_c3_outer_mask(manifest, config)
    reordered = make_c3_outer_mask(manifest.sample(frac=1, random_state=17), config)
    pd.testing.assert_frame_equal(mask, reordered)
    parent_mask = mask
    if validation_holdout:
        mask = make_validation_holdout(parent_mask, config)
        pd.testing.assert_frame_equal(mask, make_validation_holdout(reordered, config))
    canvas_length = config["time_sample_count"]
    parameter_counts = {
        name: model_parameter_count(configs[name], canvas_length) for name in C3_NEURAL_METHODS
    }
    validate_parameter_counts(parameter_counts, config["matched_training"])
    print(f"Verifying {len(hashes)} C3 input identities; reading observed rows only", flush=True)
    verify_hashes(repo, hashes)
    validate_c3_time_axis(
        np.load(repo / inputs["time_s"], allow_pickle=False),
        config["time_sample_count"],
        config["sample_interval_us"],
    )
    if not np.array_equal(manifest.cell_id, mask.cell_id):
        raise ValueError("trace manifest and mask must list every cell in cell order")
    observed_rows = manifest.loc[mask.split.eq("observed").to_numpy()].reset_index(drop=True)
    values = read_c3_amplitude_rows(
        repo / inputs["amplitudes"], observed_rows.array_row, config["time_sample_count"]
    )
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
    if validation_holdout:
        run_id += "_validation"
    run = repo / "runs" / config["study_id"] / run_id
    for directory in ("input", "masks", "features", "configs", "tables"):
        (run / directory).mkdir(parents=True, exist_ok=False)
    write_json(run / "input/input_contract.json", contract)
    write_json(run / "input/input_hashes.json", hashes)
    write_json(run / C3_TRUTH, c3_truth_record(inputs, config))
    manifest.to_parquet(run / C3_TRACE_MANIFEST, index=False)
    observed_rows[["trace_id", "cell_id"]].to_parquet(
        run / "input/observed_index.parquet", index=False
    )
    np.save(run / "input/observed_waveforms.npy", values, allow_pickle=False)
    mask_path = run / MASK_PATH
    # Consumers read the validation mask under the fixed name; the outer mask is kept alongside.
    parent_path = run / "masks/parent_mask.parquet" if validation_holdout else mask_path
    parent_mask.to_parquet(parent_path, index=False)
    if validation_holdout:
        mask.to_parquet(mask_path, index=False)
    mask_hash = sha256_file(mask_path)
    parent_mask_hash = sha256_file(parent_path)
    write_json(
        run / "masks/mask_manifest.json",
        _mask_record(config, contract, mask, parent_mask, mask_hash, parent_mask_hash),
    )
    # One exact-lattice table serves both geometry modes (``c3_geometry_table``).
    for mode in ("real", "grid"):
        geometry.to_parquet(run / "features" / f"{mode}_geometry.parquet", index=False)
    write_json(
        run / "features/feature_hashes.json",
        {
            mode: sha256_file(run / "features" / f"{mode}_geometry.parquet")
            for mode in ("real", "grid")
        },
    )
    for method, resolved in configs.items():
        resolved.update(
            amplitude_scaling="observed_global_rms",
            mask_hash=mask_hash,
            mask_id=contract["mask_id"],
            mask_seed=contract["mask_seed"],
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
    validation_gates = {}
    if validation_holdout:
        # Geometry and feature scaling see every eligible cell, as in the final run.
        if not np.array_equal(geometry.cell_id, mask.cell_id):
            raise ValueError("validation geometry must keep every eligible cell")
        validation_gates = {
            "selection_domain": "validation_holdout",
            "validation_waveforms_read": 0,
            "excluded_waveforms_read": 0,
            "geometry_covers_every_eligible_cell": True,
        }
    write_json(
        run / "preparation_gates.json",
        {
            "volume_index_c_order": True,
            "mask": True,
            "row_order_invariant": True,
            "real_geometry_equals_grid_geometry": True,
            "test_waveforms_read": 0,
            "observed_waveforms_stored": len(values),
            "shared_alignment": None,
            "canvas_sample_count": canvas_length,
            "matched_parameter_counts": parameter_counts,
            "production_runs_started": 0,
            **validation_gates,
        },
    )
    implementation_hashes = _copy_sources(repo, study, run)
    artifacts = {
        str(p.relative_to(run)): sha256_file(p) for p in sorted(run.rglob("*")) if p.is_file()
    }
    seal = {
        "amplitude_scaling": "observed_global_rms",
        "dataset_id": C3_DATASET_ID,
        "study_id": config["study_id"],
        "candidate_id": config["candidate_id"],
        "mask_id": contract["mask_id"],
        "mask_seed": contract["mask_seed"],
        "git_commit": commit,
        "dirty_worktree": dirty,
        "status": "prepared",
        "start_time": started.isoformat(),
        "end_time": datetime.now(timezone.utc).isoformat(),
        "input_hashes": hashes,
        "mapping_hash": sha256_file(run / C3_TRACE_MANIFEST),
        "mask_hash": mask_hash,
        "global_rms": rms,
        "observed_trace_count": len(values),
        "test_trace_count": int(mask.split.eq("test").sum()),
        "clean_test_trace_count": int((mask.split.eq("test") & mask.clean_target).sum()),
        "time_sample_count": values.shape[1],
        "test_waveforms_read": 0,
        **(
            {
                "selection_domain": "validation_holdout",
                "validation_fraction": config["validation_holdout"]["fraction"],
                "validation_mask_hash": mask_hash,
                "parent_mask_hash": parent_mask_hash,
                "parent_mask_id": config["mask_id"],
                "parent_mask_seed": config["mask_seed"],
                "parent_mask_study_id": config.get("mask_study_id", config["study_id"]),
                "parent_observed_trace_count": int(parent_mask.split.eq("observed").sum()),
                "parent_test_trace_count": int(parent_mask.split.eq("test").sum()),
                "excluded_trace_count": int(mask.split.eq("excluded").sum()),
                "excluded_waveforms_read": 0,
            }
            if validation_holdout
            else {}
        ),
        # No moveout alignment: the stage planner compares both keys across preparations.
        "alignment": None,
        "canvas_sample_count": canvas_length,
        "parameter_counts": parameter_counts,
        "implementation_hashes": implementation_hashes,
        "artifacts": artifacts,
    }
    write_json(run / "preparation_manifest.json", seal)
    for path in run.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    return run


def c3_input_hashes(inputs: dict) -> dict:
    """``{repo-relative path: sha256}`` of the declared C3 inputs."""
    names = [*INPUT_NAMES, *([PROVENANCE_NAME] if PROVENANCE_NAME in inputs else [])]
    if set(inputs) != {key for name in names for key in (name, f"{name}_sha256")}:
        raise ValueError(f"inputs.yaml declares {INPUT_NAMES} (+ {PROVENANCE_NAME}) and sha256")
    return {inputs[name]: inputs[f"{name}_sha256"] for name in names}


def _load_volume_index(repo: Path, inputs: dict, config: dict) -> pd.DataFrame:
    """Verified volume index whose metadata matches the contract crop (time from sample 0)."""
    index_path, metadata_path = Path(inputs["volume_index"]), Path(inputs["volume_metadata"])
    if (
        index_path.name != VOLUME_INDEX_FILE_NAME
        or metadata_path.name != VOLUME_METADATA_FILE_NAME
        or index_path.parent != metadata_path.parent
    ):
        raise ValueError("volume_index and volume_metadata must be one volume artifact")
    verify_hashes(
        repo,
        {
            inputs["volume_index"]: inputs["volume_index_sha256"],
            inputs["volume_metadata"]: inputs["volume_metadata_sha256"],
        },
    )
    volume_index, metadata = load_c3_volume_index(repo / index_path.parent)
    count = config["time_sample_count"]
    if (
        metadata["dataset_id"] != C3_DATASET_ID
        or metadata["shape"] != [count, *config["spatial_shape"]]
        or metadata["selection"]["time"] != [0, count]
    ):
        raise ValueError("volume metadata does not match the C3 contract crop")
    return volume_index


def _mask_record(config, contract, mask, parent_mask, mask_hash, parent_mask_hash) -> dict:
    algorithm = (
        "SHA256(UTF-8 canonical compact JSON list "
        "[mask_study_id,candidate_id,mask_id,integer mask_seed,integer cell_id]); "
        "digest ascending, then cell_id ascending; hash_rank is 1-based"
    )
    record = {
        "mask_hash": parent_mask_hash,
        "mask_id": config["mask_id"],
        "mask_seed": config["mask_seed"],
        "study_id": config.get("mask_study_id", config["study_id"]),
        "candidate_id": config["candidate_id"],
        "algorithm": algorithm,
        "counts": parent_mask.split.value_counts().to_dict(),
        "row_order_invariant": True,
    }
    if contract.get("selection_domain") != "validation_holdout":
        return record
    return {
        "mask_hash": mask_hash,
        "mask_id": contract["mask_id"],
        "mask_seed": contract["mask_seed"],
        "study_id": config["study_id"],
        "candidate_id": config["candidate_id"],
        "selection_domain": "validation_holdout",
        "fraction": config["validation_holdout"]["fraction"],
        "algorithm": (
            "SHA256(UTF-8 canonical compact JSON list "
            "[study_id,candidate_id,mask_id,integer mask_seed,integer cell_id]) over "
            "parent-observed cells only (validation_hash_rank), digest then cell_id "
            "ascending; the first round(fraction x parent observed count) become split test, "
            "the other parent-observed cells stay observed and parent test cells become excluded"
        ),
        "counts": mask.split.value_counts().to_dict(),
        "row_order_invariant": True,
        "parent_mask": record,
    }


def _copy_sources(repo: Path, study: Path, run: Path) -> dict:
    """Copy and hash the implementation and the study yamls into ``run/source``."""
    hashes = {}
    for source in [*sorted((repo / "src/seis_interp").rglob("*.py")), repo / SCRIPT]:
        relative = source.relative_to(repo)
        hashes[str(relative)] = sha256_file(source)
        destination = run / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    for source in sorted(study.glob("*.yaml")):
        target = run / "source" / source.relative_to(repo)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return hashes
