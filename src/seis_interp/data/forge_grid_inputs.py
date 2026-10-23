"""Frozen four-partition FORGE grid inputs derived from verified study_066 data."""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from seis_interp.data.forge_headers import sha256_file
from seis_interp.data.forge_mvp_artifacts import (
    ForgeModelInputs,
    observed_global_rms,
    verify_hashes,
    write_json,
)
from seis_interp.data.forge_mvp_run_records import execution_identity
from seis_interp.processing.c3_region_search import c3_search_candidates
from seis_interp.processing.experiment_commonality import load_experiment_grid
from seis_interp.processing.forge_grid_partition import (
    partition_forge_traces,
    verify_forge_s1_partition,
)
from seis_interp.training.forge_mvp_methods import model_parameter_count
from seis_interp.training.gpu_admission import prepare_admission_capacity
from seis_interp.training.rehearsal import load_rehearsal_plan, rehearsal_candidates, rehearsal_seal


def prepare_forge_grid_experiment(repo: Path, study: Path, *, destination=None):
    """Reuse the sealed anti-aliased physical waveforms, never a fitted model input.

    The reference is mandatory: it both avoids repeating SEG-Y preprocessing and
    makes equality with the historical S1 assignment an executable condition.
    """
    contract = yaml.safe_load((study / "config.yaml").read_text())
    plan = load_rehearsal_plan(study, contract)
    grid = load_experiment_grid(study, contract)
    prepare_admission_capacity(
        contract.get("cuda_execution"),
        [
            candidate
            for method in contract.get("methods", grid["methods"])
            for candidate in rehearsal_candidates(
                c3_search_candidates(
                    yaml.safe_load((study / f"{method}.yaml").read_text()), grid["methods"][method]
                ),
                contract,
                method,
                plan,
            )
        ],
    )
    specification = contract["trace_split"]
    seeds = [specification["search_seed"], *specification["evaluation_seeds"]]
    if len(seeds) != 4 or len(set(seeds)) != 4:
        raise ValueError("FORGE requires one S1 seed and three distinct evaluation seeds")
    methods = contract["methods"]
    if set(methods) != set(grid["methods"]):
        raise ValueError("FORGE method and grid identities differ")
    reference = repo / contract["reference_experiment"]
    seal = json.loads((reference / "experiment.json").read_text())
    original = yaml.safe_load((reference / "study/config.yaml").read_text())
    for key in (
        "candidate_id",
        "spatial_shape",
        "time_sample_count",
        "sample_interval_us",
        "preprocessing",
        "time_sampling",
    ):
        if contract[key] != original[key]:
            raise ValueError(f"FORGE reference preprocessing/input differs: {key}")
    names = [
        "manifest.parquet",
        "partition.parquet",
        "waveforms.npy",
        "trace_scales.parquet",
        "real_geometry.parquet",
        "grid_geometry.parquet",
    ]
    verify_hashes(
        reference, {f"input/{name}": seal["artifacts"][f"input/{name}"] for name in names}
    )
    manifest = pd.read_parquet(reference / "input/manifest.parquet")
    partitions = {seed: partition_forge_traces(manifest, contract, seed) for seed in seeds}
    verify_forge_s1_partition(
        partitions[seeds[0]], pd.read_parquet(reference / "input/partition.parquet")
    )
    parameter_counts = {}
    canvas = contract["time_sample_count"] + seal["alignment"]["max_delay_samples"]
    for method in methods:
        base = yaml.safe_load((study / f"{method}.yaml").read_text())
        candidates = c3_search_candidates(base, grid["methods"][method])
        if method in ("pocs_grid", "drr_grid"):
            parameter_counts[method] = 0
        else:
            counts = {model_parameter_count(item["config"], canvas) for item in candidates}
            if len(counts) != 1:
                raise ValueError(f"{method}: search must preserve the parameter count")
            parameter_counts[method] = counts.pop()
    selected_counts = {
        method: len(
            rehearsal_candidates(
                c3_search_candidates(
                    yaml.safe_load((study / f"{method}.yaml").read_text()), grid["methods"][method]
                ),
                contract,
                method,
                plan,
            )
        )
        for method in methods
    }
    rehearsal_flags = rehearsal_seal(study, plan, selected_counts)
    identity = execution_identity(repo)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    experiment = (
        Path(destination)
        if destination is not None
        else repo / "runs" / contract["study_id"] / f"{stamp}_{identity['git_commit'][:12]}_forge"
    )
    experiment.mkdir(parents=True, exist_ok=False)
    shutil.copytree(study, experiment / "study")
    (experiment / "study/grid.yaml").write_text(yaml.safe_dump(grid, sort_keys=False))
    for source in [
        *sorted((repo / "src/seis_interp").rglob("*.py")),
        repo / "scripts/run_forge_grid_experiment.py",
    ]:
        target = experiment / "source" / source.relative_to(repo)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    frozen = experiment / "input"
    frozen.mkdir()
    for name in names:
        if name not in ("partition.parquet", "trace_scales.parquet"):
            shutil.copyfile(reference / "input" / name, frozen / name)
    values = np.load(frozen / "waveforms.npy", mmap_mode="r", allow_pickle=False)
    counts = {}
    trace_scales = pd.read_parquet(reference / "input/trace_scales.parquet")
    for seed, partition in partitions.items():
        directory = frozen / f"seed_{seed}"
        directory.mkdir()
        partition.to_parquet(directory / "partition.parquet", index=False)
        positions = pd.Index(manifest.cell_id).get_indexer(
            partition.cell_id[partition.role.eq("train")]
        )
        observed = np.ascontiguousarray(values[positions])
        np.save(directory / "observed_waveforms.npy", observed, allow_pickle=False)
        scales = trace_scales.copy()
        scales["scale_source"] = np.where(
            scales.cell_id.isin(partition.cell_id[partition.role.eq("train")]),
            "observed",
            "oracle_test",
        )
        scales.to_parquet(directory / "trace_scales.parquet", index=False)
        counts[str(seed)] = {
            **partition.role.value_counts().to_dict(),
            "global_rms": observed_global_rms(observed),
        }
    artifacts = {
        str(path.relative_to(experiment)): sha256_file(path)
        for path in sorted(experiment.rglob("*"))
        if path.is_file()
    }
    write_json(
        experiment / "experiment.json",
        {
            **identity,
            "study_id": contract["study_id"],
            "dataset_id": "forge_2017",
            "status": "prepared",
            "methods": methods,
            "partitions": counts,
            "alignment": seal["alignment"],
            "canvas_sample_count": canvas,
            "parameter_counts": parameter_counts,
            "input_hashes": seal["input_hashes"],
            "artifacts": artifacts,
            "s1_runs_per_method": 25,
            "additional_runs_per_method": 3,
            "total_runs": 28 * len(methods),
            **rehearsal_flags,
            "reference_experiment": str(reference),
            "reference_experiment_sha256": sha256_file(reference / "experiment.json"),
            "target_waveform_usage": (
                "sealed oracle RMS at normalization/restoration and post-finalization scoring only"
            ),
            "partition_namespaces": {
                "outer": contract["mask_study_id"],
                "validation": contract["validation_study_id"],
            },
        },
    )
    for path in experiment.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    return experiment


def load_forge_grid_inputs(experiment, method, seed, domains):
    seal = json.loads((experiment / "experiment.json").read_text())
    contract = yaml.safe_load((experiment / "study/config.yaml").read_text())
    config = yaml.safe_load((experiment / "study" / f"{method}.yaml").read_text())
    config["alignment"] = seal["alignment"]
    prefix = f"input/seed_{seed}"
    names = [
        f"{prefix}/{name}"
        for name in ("partition.parquet", "observed_waveforms.npy", "trace_scales.parquet")
    ]
    names.append(f"input/{config['geometry_mode']}_geometry.parquet")
    verify_hashes(experiment, {name: seal["artifacts"][name] for name in names})
    partition = pd.read_parquet(experiment / prefix / "partition.parquet")
    scales = pd.read_parquet(experiment / prefix / "trace_scales.parquet").set_index("cell_id")
    observed = partition.cell_id[partition.role.eq("train")].to_numpy()
    query = partition.cell_id[partition.role.isin(domains)].to_numpy()
    excluded = partition.cell_id[~partition.role.isin(["train", *domains])].to_numpy()
    inputs = ForgeModelInputs(
        geometry=pd.read_parquet(experiment / names[-1]),
        observed_cells=observed,
        test_cells=query,
        excluded_cells=excluded,
        observed_values=np.load(
            experiment / prefix / "observed_waveforms.npy", mmap_mode="r", allow_pickle=False
        ),
        spatial_shape=tuple(contract["spatial_shape"]),
        time_s=np.arange(contract["time_sample_count"], dtype=np.float64)
        * contract["sample_interval_us"]
        / 1e6,
        global_rms=seal["partitions"][str(seed)]["global_rms"],
        binding={
            "study_id": seal["study_id"],
            "trace_seed": seed,
            "partition_hash": seal["artifacts"][f"{prefix}/partition.parquet"],
            "trace_scale_hash": seal["artifacts"][f"{prefix}/trace_scales.parquet"],
            "amplitude_scaling": "trace_oracle",
        },
        observed_rms=scales.loc[observed, "rms"].to_numpy(),
        test_rms=scales.loc[query, "rms"].to_numpy(),
    )
    return inputs, config, partition


def read_forge_grid_targets(experiment, partition, domain):
    if domain not in ("validation", "test"):
        raise ValueError("unknown scoring domain")
    manifest = pd.read_parquet(experiment / "input/manifest.parquet")
    if not np.array_equal(manifest.cell_id, partition.cell_id):
        raise ValueError("scoring waveform and partition indices differ")
    selected = partition.role.eq(domain).to_numpy()
    values = np.load(experiment / "input/waveforms.npy", mmap_mode="r", allow_pickle=False)
    return manifest.loc[selected].reset_index(drop=True), np.ascontiguousarray(values[selected])
