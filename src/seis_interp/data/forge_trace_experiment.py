"""Prepare and load a shared FORGE trace assignment for fixed-config comparisons."""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from seis_interp.data.forge_aligned_inputs import shared_alignment
from seis_interp.data.forge_headers import sha256_file
from seis_interp.data.forge_mvp_artifacts import (
    ForgeModelInputs,
    load_candidate_inputs,
    observed_global_rms,
    read_selected_traces,
    verify_hashes,
    write_json,
)
from seis_interp.data.forge_mvp_run_records import execution_identity, utc_now
from seis_interp.processing.forge_mvp_contract import (
    make_geometry,
    make_outer_mask,
    make_validation_holdout,
    validate_mapping,
)
from seis_interp.processing.forge_time_sampling import native_time_contract, resample_forge_traces
from seis_interp.processing.forge_trace_scaling import prepare_oracle_trace_scales
from seis_interp.training.forge_mvp_methods import model_parameter_count


def prepare_forge_trace_experiment(repo: Path, study: Path) -> Path:
    """Freeze inputs once; target waveforms are read only for the authorized oracle RMS."""
    contract = yaml.safe_load((study / "config.yaml").read_text())
    sources = yaml.safe_load((study / "inputs.yaml").read_text())
    if contract["preprocessing"]["amplitude_scaling"] != "trace_oracle":
        raise ValueError("this comparison requires the existing FORGE trace_oracle contract")
    methods = contract["methods"]
    configs = {name: yaml.safe_load((study / f"{name}.yaml").read_text()) for name in methods}
    if any(config["method_id"] != name for name, config in configs.items()):
        raise ValueError("method configuration identity mismatch")
    native = native_time_contract(contract)
    tables, grids, hashes = load_candidate_inputs(repo, sources, contract["candidate_id"])
    references = {
        entry["selection"]: entry["sha256"] for entry in sources.get("c3_selections", {}).values()
    }
    verify_hashes(repo, references)
    hashes.update(references)
    manifest = (
        validate_mapping(
            tables["trace_mapping"],
            {role: tables[f"{role}_stations"] for role in ("source", "receiver")},
            tables["grid_cells"],
            grids,
            native,
        )
        .sort_values("cell_id")
        .reset_index(drop=True)
    )
    outer = make_outer_mask(
        manifest, {**contract, "study_id": contract.get("mask_study_id", contract["study_id"])}
    )
    partition = make_validation_holdout(
        outer,
        {**contract, "study_id": contract.get("validation_study_id", contract["study_id"])},
    )
    partition["role"] = partition.split.map(
        {"observed": "train", "test": "validation", "excluded": "test"}
    )
    geometries = {mode: make_geometry(manifest, grids, mode) for mode in ("real", "grid")}
    alignment = shared_alignment(
        contract["preprocessing"]["alignment"]["velocity_m_per_s"],
        list(geometries.values()),
        contract["sample_interval_us"] / 1e6,
    )
    canvas = contract["time_sample_count"] + alignment["max_delay_samples"]
    counts = {
        name: model_parameter_count(config, canvas)
        for name, config in configs.items()
        if "training" in config
    }
    identity = execution_identity(repo)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    experiment = repo / "runs" / contract["study_id"] / f"{stamp}_{identity['git_commit'][:12]}"
    experiment.mkdir(parents=True, exist_ok=False)
    write_json(experiment / "preparation_status.json", {"status": "verifying_raw_inputs"})
    lock = repo / sources["raw_input_lock"]
    verify_hashes(repo, {sources["raw_input_lock"]: sources["raw_input_lock_sha256"]})
    inventory = {entry["path"]: entry["sha256"] for entry in json.loads(lock.read_text())}
    raw_hashes = {name: inventory[name] for name in sorted(manifest.source_file.unique())}
    print(f"{experiment}: verifying {len(raw_hashes)} SEG-Y identities", flush=True)
    verify_hashes(repo / sources["dataset_root"], raw_hashes)
    for path in sorted(study.iterdir()):
        if path.is_file():
            destination = experiment / "study" / path.name
            destination.parent.mkdir(exist_ok=True)
            shutil.copyfile(path, destination)
    for path in sorted((repo / "src/seis_interp").rglob("*.py")):
        destination = experiment / "source" / path.relative_to(repo)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    write_json(experiment / "input/raw_input_hashes.json", raw_hashes)
    manifest.to_parquet(experiment / "input/manifest.parquet", index=False)
    partition.to_parquet(experiment / "input/partition.parquet", index=False)
    for mode, table in geometries.items():
        table.to_parquet(experiment / "input" / f"{mode}_geometry.parquet", index=False)
    values = np.lib.format.open_memmap(
        experiment / "input/waveforms.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(manifest), contract["time_sample_count"]),
    )
    write_json(experiment / "preparation_status.json", {"status": "reading_and_resampling"})
    for start in range(0, len(manifest), 8192):
        rows = manifest.iloc[start : start + 8192]
        native_values = read_selected_traces(
            repo / sources["dataset_root"],
            rows,
            native["time_sample_count"],
            native["sample_interval_us"],
        )
        values[start : start + len(rows)] = resample_forge_traces(native_values, contract)
        print(f"FORGE preparation: {start + len(rows)}/{len(manifest)} traces", flush=True)
    values.flush()
    training = partition.role.eq("train").to_numpy()
    cells = manifest.cell_id.to_numpy()
    observed_values = np.ascontiguousarray(values[training])
    np.save(experiment / "input/observed_waveforms.npy", observed_values, allow_pickle=False)
    scales = prepare_oracle_trace_scales(
        observed_values, cells[training], values[~training], cells[~training]
    )
    scales.to_parquet(experiment / "input/trace_scales.parquet", index=False)
    seal = {
        **identity,
        "status": "prepared",
        "prepared_at": utc_now(),
        "study_id": contract["study_id"],
        "methods": methods,
        "total_runs": len(methods),
        "counts": partition.role.value_counts().to_dict(),
        "alignment": alignment,
        "canvas_sample_count": canvas,
        "parameter_counts": counts,
        "global_rms": observed_global_rms(observed_values),
        "input_hashes": hashes,
        "oracle_target_waveforms_read": int((~training).sum()),
        "target_waveform_usage": "RMS preparation and post-finalization scoring only",
        "artifacts": {
            str(path.relative_to(experiment)): sha256_file(path)
            for path in sorted(experiment.rglob("*"))
            if path.is_file() and path.name != "preparation_status.json"
        },
    }
    write_json(experiment / "experiment.json", seal)
    write_json(experiment / "preparation_status.json", {"status": "prepared"})
    for relative in seal["artifacts"]:
        (experiment / relative).chmod(0o444)
    return experiment


def load_forge_trace_inputs(experiment: Path, method: str, *, verify_target_waveforms=True):
    """Return compact training amplitudes and restoration scalars.

    The default verifies every sealed artifact. Validation-only probes exclude
    the held-out waveform file from checksum reads as well as model inputs.
    Its saved digest remains bound by the experiment seal.
    """
    seal = json.loads((experiment / "experiment.json").read_text())
    if seal["status"] != "prepared" or method not in seal["methods"]:
        raise ValueError("unprepared experiment or inactive method")
    verify_hashes(
        experiment,
        {
            name: digest
            for name, digest in seal["artifacts"].items()
            if verify_target_waveforms or name != "input/waveforms.npy"
        },
    )
    contract = yaml.safe_load((experiment / "study/config.yaml").read_text())
    config = yaml.safe_load((experiment / "study" / f"{method}.yaml").read_text())
    partition = pd.read_parquet(experiment / "input/partition.parquet")
    scales = pd.read_parquet(experiment / "input/trace_scales.parquet").set_index("cell_id")
    observed = partition.cell_id[partition.role.eq("train")].to_numpy()
    targets = partition.cell_id[~partition.role.eq("train")].to_numpy()
    config["alignment"] = seal["alignment"]
    binding = {
        "study_id": seal["study_id"],
        "experiment_hash": sha256_file(experiment / "experiment.json"),
        "partition_hash": seal["artifacts"]["input/partition.parquet"],
        "trace_scale_hash": seal["artifacts"]["input/trace_scales.parquet"],
        "amplitude_scaling": "trace_oracle",
    }
    inputs = ForgeModelInputs(
        geometry=pd.read_parquet(
            experiment / "input" / f"{config['geometry_mode']}_geometry.parquet"
        ),
        observed_cells=observed,
        test_cells=targets,
        observed_values=np.load(experiment / "input/observed_waveforms.npy", mmap_mode="r"),
        spatial_shape=tuple(contract["spatial_shape"]),
        time_s=np.arange(contract["time_sample_count"], dtype=np.float64)
        * contract["sample_interval_us"]
        / 1e6,
        global_rms=seal["global_rms"],
        binding=binding,
        observed_rms=scales.loc[observed, "rms"].to_numpy(),
        test_rms=scales.loc[targets, "rms"].to_numpy(),
    )
    return inputs, config, partition


def read_forge_trace_targets(experiment: Path, domain: str):
    """Read one scoring domain after its model or reconstruction has been finalized."""
    if domain not in ("validation", "test"):
        raise ValueError("unknown scoring domain")
    partition = pd.read_parquet(experiment / "input/partition.parquet")
    selected = partition.role.eq(domain).to_numpy()
    manifest = pd.read_parquet(experiment / "input/manifest.parquet")
    if not np.array_equal(manifest.cell_id, partition.cell_id):
        raise ValueError("scoring waveform and partition indices differ")
    values = np.load(experiment / "input/waveforms.npy", mmap_mode="r")
    return manifest.loc[selected].reset_index(drop=True), np.ascontiguousarray(values[selected])
