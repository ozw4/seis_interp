"""Frozen C3 region or trace assignments and observed-only model inputs."""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from seis_interp.c3_fair_grid_config import c3_experiment_methods, validate_c3_fair_grid_configs
from seis_interp.data.c3_experiment_sources import bind_c3_continuing_experiment
from seis_interp.data.c3_fair_grid_truth import read_c3_amplitude_rows, validate_c3_time_axis
from seis_interp.data.c3_volume_index_store import load_c3_volume_index
from seis_interp.data.forge_headers import sha256_file
from seis_interp.data.forge_mvp_artifacts import (
    ForgeModelInputs,
    observed_global_rms,
    verify_hashes,
    write_json,
)
from seis_interp.data.forge_mvp_run_records import execution_identity
from seis_interp.processing.c3_fair_grid_contract import (
    c3_geometry_table,
    c3_trace_manifest,
    make_c3_outer_mask,
)
from seis_interp.processing.c3_region_partition import partition_c3_regions, region_scoring_mask
from seis_interp.processing.c3_region_search import (
    c3_reference_configuration,
    c3_search_candidates,
    c3_split_settings,
    resolve_c3_candidate,
    validate_c3_experiment_plan,
)
from seis_interp.processing.c3_trace_partition import partition_c3_traces
from seis_interp.processing.experiment_commonality import load_experiment_grid
from seis_interp.training.forge_mvp_methods import model_parameter_count
from seis_interp.training.gpu_admission import prepare_admission_capacity
from seis_interp.training.rehearsal import load_rehearsal_plan, rehearsal_candidates, rehearsal_seal

INPUT_NAMES = ("volume_index", "volume_metadata", "amplitudes", "time_s")


def prepare_c3_region_experiment(repo: Path, study: Path, *, destination=None) -> Path:
    """Freeze code/configs and four partitions, without reading any target waveform."""
    contract = yaml.safe_load((study / "config.yaml").read_text())
    inputs = yaml.safe_load((study / "inputs.yaml").read_text())
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
    validate_c3_experiment_plan(contract, grid)
    methods = c3_experiment_methods(contract)
    hashes = {inputs[name]: inputs[f"{name}_sha256"] for name in INPUT_NAMES}
    verify_hashes(repo, hashes)
    index, metadata = load_c3_volume_index(repo / Path(inputs["volume_index"]).parent)
    if metadata["shape"] != [contract["time_sample_count"], *contract["spatial_shape"]]:
        raise ValueError("C3 crop shape does not match the experiment")
    time = np.load(repo / inputs["time_s"], allow_pickle=False)
    validate_c3_time_axis(time, contract["time_sample_count"], contract["sample_interval_us"])
    manifest = c3_trace_manifest(index, contract["spatial_shape"], contract["candidate_id"])
    geometry = c3_geometry_table(index, contract["spatial_shape"])
    parent = make_c3_outer_mask(manifest, contract)
    kind, specification = c3_split_settings(contract)
    seeds = [specification["search_seed"], *specification["evaluation_seeds"]]
    partitions = {}
    for seed in seeds:
        if kind == "region":
            partition = partition_c3_regions(parent, contract["spatial_shape"], specification, seed)
        else:
            outer = (
                parent
                if seed == specification["search_seed"]
                else make_c3_outer_mask(manifest, {**contract, "mask_seed": seed})
            )
            partition = partition_c3_traces(outer, contract, seed)
        partitions[seed] = partition
    bases = {method: yaml.safe_load((study / f"{method}.yaml").read_text()) for method in methods}
    candidates = {
        method: c3_search_candidates(base, grid["methods"][method])
        for method, base in bases.items()
    }
    selected_counts = {
        method: len(rehearsal_candidates(options, contract, method, plan))
        for method, options in candidates.items()
    }
    references = {
        method: c3_reference_configuration(candidates[method], settings)
        for method, settings in contract.get("reference_settings", {}).items()
    }
    counts = {}
    for seed, partition in partitions.items():
        observed_count = int(partition.input_observed.sum())
        resolved = {
            method: resolve_c3_candidate(
                base, observed_count, contract["matched_training"]["loss_traces_per_update"]
            )
            for method, base in bases.items()
        }
        validate_c3_fair_grid_configs(
            resolved, {**contract, "observed_trace_count": observed_count}
        )
        counts[str(seed)] = {
            "observed": observed_count,
            **{
                role: int(partition[f"{kind}_role"].eq(role).sum())
                for role in ("train", "validation", "test")
            },
        }
    parameter_counts = {}
    for method, options in candidates.items():
        if method in ("pocs_grid", "drr_grid"):
            parameter_counts[method] = 0
            continue
        values = {
            model_parameter_count(option["config"], contract["time_sample_count"])
            for option in options
        }
        if len(values) != 1:
            raise ValueError(f"{method}: search must preserve the parameter count")
        parameter_counts[method] = values.pop()
    rehearsal_flags = rehearsal_seal(study, plan, selected_counts)
    identity = execution_identity(repo)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    experiment = (
        Path(destination)
        if destination is not None
        else (
            repo / "runs" / contract["study_id"] / f"{stamp}_{identity['git_commit'][:12]}_{kind}s"
        )
    )
    experiment.mkdir(parents=True, exist_ok=False)
    source_hashes = {}
    for source in [
        *sorted((repo / "src/seis_interp").rglob("*.py")),
        repo / "scripts/run_c3_region_experiment.py",
    ]:
        target = experiment / "source" / source.relative_to(repo)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        source_hashes[str(target.relative_to(experiment))] = sha256_file(target)
    frozen_study = experiment / "study"
    shutil.copytree(study, frozen_study)
    if "common_grid" in contract:
        (frozen_study / "grid.yaml").write_text(yaml.safe_dump(grid, sort_keys=False))
    frozen_inputs = experiment / "input"
    frozen_inputs.mkdir()
    geometry.to_parquet(frozen_inputs / "geometry.parquet", index=False)
    manifest.to_parquet(frozen_inputs / "trace_manifest.parquet", index=False)
    parent.to_parquet(frozen_inputs / "outer_mask.parquet", index=False)
    np.save(frozen_inputs / "time_s.npy", time[: contract["time_sample_count"]], allow_pickle=False)
    for seed, partition in partitions.items():
        destination = frozen_inputs / f"seed_{seed}"
        destination.mkdir()
        partition.to_parquet(destination / "partition.parquet", index=False)
        cells = partition.loc[partition.input_observed, "cell_id"].to_numpy()
        rows = manifest.set_index("cell_id").loc[cells].array_row.to_numpy()
        values = read_c3_amplitude_rows(
            repo / inputs["amplitudes"], rows, contract["time_sample_count"]
        )
        np.save(destination / "observed_waveforms.npy", values, allow_pickle=False)
        counts[str(seed)]["global_rms"] = observed_global_rms(values)
    continuing = (
        bind_c3_continuing_experiment(repo, experiment, contract["continuing_experiment"], methods)
        if "continuing_experiment" in contract
        else None
    )
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
            "status": "prepared",
            "input_hashes": hashes,
            "implementation_hashes": source_hashes,
            "artifacts": artifacts,
            "partitions": counts,
            "parameter_counts": parameter_counts,
            "s1_runs_per_method": 25,
            "additional_runs_per_method": 3,
            "total_runs": 28 * len(methods),
            **rehearsal_flags,
            "target_waveforms_read": 0,
            **({"partition_unit": "trace"} if kind == "trace" else {}),
            **({"methods": list(methods)} if "methods" in contract else {}),
            **({"reference_configurations": references} if references else {}),
            **({"continuing_experiment": continuing} if continuing is not None else {}),
        },
    )
    for path in experiment.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    return experiment


def load_c3_region_inputs(experiment: Path, seed: int, domains):
    """Return visible observations; heldout amplitudes stay outside the object."""
    seal = json.loads((experiment / "experiment.json").read_text())
    prefix = f"input/seed_{seed}"
    names = [
        "input/geometry.parquet",
        "input/time_s.npy",
        f"{prefix}/partition.parquet",
        f"{prefix}/observed_waveforms.npy",
    ]
    verify_hashes(experiment, {name: seal["artifacts"][name] for name in names})
    partition = pd.read_parquet(experiment / prefix / "partition.parquet")
    contract = yaml.safe_load((experiment / "study/config.yaml").read_text())
    kind, _ = c3_split_settings(contract)
    mask = region_scoring_mask(partition, domains, f"{kind}_role")
    return ForgeModelInputs(
        geometry=pd.read_parquet(experiment / "input/geometry.parquet"),
        observed_cells=mask.loc[mask.split.eq("observed"), "cell_id"].to_numpy(),
        test_cells=mask.loc[mask.split.eq("test"), "cell_id"].to_numpy(),
        excluded_cells=mask.loc[mask.split.eq("excluded"), "cell_id"].to_numpy(),
        observed_values=np.load(
            experiment / prefix / "observed_waveforms.npy", mmap_mode="r", allow_pickle=False
        ),
        spatial_shape=tuple(contract["spatial_shape"]),
        time_s=np.load(experiment / "input/time_s.npy", allow_pickle=False),
        global_rms=seal["partitions"][str(seed)]["global_rms"],
        binding={
            "study_id": seal["study_id"],
            f"{kind}_seed": seed,
            "partition_hash": seal["artifacts"][f"{prefix}/partition.parquet"],
            "observed_waveforms_hash": seal["artifacts"][f"{prefix}/observed_waveforms.npy"],
            "global_rms": seal["partitions"][str(seed)]["global_rms"],
        },
    ), partition


def read_c3_region_targets(repo: Path, experiment: Path, partition, domain: str):
    """Read one finalized scoring domain once, after predictions have been saved."""
    if domain not in ("validation", "test"):
        raise ValueError("only validation/test amplitudes may be scored")
    seal = json.loads((experiment / "experiment.json").read_text())
    name = "input/trace_manifest.parquet"
    verify_hashes(experiment, {name: seal["artifacts"][name]})
    inputs = yaml.safe_load((experiment / "study/inputs.yaml").read_text())
    contract = yaml.safe_load((experiment / "study/config.yaml").read_text())
    verify_hashes(
        repo,
        {
            inputs[name]: inputs[f"{name}_sha256"]
            for name in (("amplitudes", "time_s") if domain == "test" else ("time_s",))
        },
    )
    kind, _ = c3_split_settings(contract)
    selected = partition.loc[partition[f"{kind}_role"].eq(domain)].copy()
    manifest = pd.read_parquet(experiment / name).set_index("cell_id", verify_integrity=True)
    rows = manifest.loc[selected.cell_id].array_row.to_numpy()
    targets = read_c3_amplitude_rows(
        repo / inputs["amplitudes"], rows, contract["time_sample_count"]
    )
    return selected, targets
