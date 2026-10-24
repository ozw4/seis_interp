"""Read-only access to common-grid manifests, identities and saved arrays."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from seis_interp.data.forge_headers import sha256_file
from seis_interp.processing.forge_log_rms_idw import LOG_RMS_IDW_SETTINGS, log_rms_idw_scales


def read_json(path):
    return json.loads(Path(path).read_text())


def read_yaml(path):
    return yaml.safe_load(Path(path).read_text())


def manifest_records(experiment, method, stage):
    """Only the current logical runs; attempts are accounted for separately."""
    return [
        (p.parent, read_json(p))
        for p in sorted((experiment / "method_runs" / method / stage).glob("*/manifest.json"))
    ]


def partition(experiment, seed):
    return pd.read_parquet(experiment / "input" / f"seed_{seed}" / "partition.parquet")


def scoring_rows(assignment, domain):
    role = "trace_role" if "trace_role" in assignment else "role"
    return assignment.loc[assignment[role].eq(domain)].copy()


def saved_traces(directory, domain):
    table = pd.read_parquet(directory / f"{domain}_per_trace.parquet")
    if table.cell_id.duplicated().any() or table.trace_id.duplicated().any():
        raise ValueError(f"duplicate scoring identity: {directory}")
    with np.errstate(divide="ignore", invalid="ignore"):
        table["snr_db"] = 10 * (np.log10(table.target_energy) - np.log10(table.error_energy))
    return table


def target_source(repo, experiment, dataset, contract):
    """Memory-map physical targets; copy only the requested trace chunk later."""
    if dataset == "c3":
        inputs = read_yaml(experiment / "study/inputs.yaml")
        manifest = pd.read_parquet(experiment / "input/trace_manifest.parquet")
        positions = manifest.set_index("cell_id", verify_integrity=True).array_row
        path = repo / inputs["amplitudes"]
    else:
        manifest = pd.read_parquet(experiment / "input/manifest.parquet")
        positions = pd.Series(np.arange(len(manifest)), index=manifest.cell_id)
        path = experiment / "input/waveforms.npy"
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    samples = contract["time_sample_count"]
    if values.ndim != 2 or values.dtype != np.float32 or values.shape[1] < samples:
        raise ValueError("physical targets must be float32 [trace, sample]")
    return values, positions, samples


def grid_spacings(geometry):
    """Physical spacings from the saved projected grid (not the real stations)."""
    grids = {}
    for role in ("source", "receiver"):
        columns = [f"{role}_grid_line", f"{role}_grid_point"]
        points = geometry.drop_duplicates(columns).sort_values(columns)
        coords = points[[f"{role}_x_m", f"{role}_y_m"]].to_numpy()
        grids[role] = {}
        for axis, name in enumerate(("across_spacing_m", "along_spacing_m")):
            other = columns[1 - axis]
            along = points.loc[points[other].eq(points[other].min())]
            xy = along.sort_values(columns[axis])[[f"{role}_x_m", f"{role}_y_m"]].to_numpy()
            steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            if len(coords) < 2 or not len(steps) or not np.allclose(steps, steps[0], rtol=1e-8):
                raise ValueError("projected grid does not have a fixed physical spacing")
            grids[role][name] = float(np.median(steps))
    return grids


def forge_idw_scales(experiment, seed, contract):
    assignment = partition(experiment, seed)
    observed = assignment.loc[assignment.role.eq("train"), "cell_id"].to_numpy()
    query = assignment.loc[assignment.role.isin(["validation", "test"]), "cell_id"].to_numpy()
    prefix = experiment / "input" / f"seed_{seed}"
    values = np.load(prefix / "observed_waveforms.npy", mmap_mode="r", allow_pickle=False)
    geometry = pd.read_parquet(experiment / "input/grid_geometry.parquet")
    grids = grid_spacings(geometry)
    scales, spacing = log_rms_idw_scales(
        values, observed, query, tuple(contract["spatial_shape"]), grids, LOG_RMS_IDW_SETTINGS
    )
    oracle = (
        pd.read_parquet(prefix / "trace_scales.parquet")
        .set_index("cell_id", verify_integrity=True)
        .rms.loc[query]
        .to_numpy()
    )
    if not np.isfinite(oracle).all() or np.any(oracle <= 0):
        raise ValueError("oracle RMS must be finite and positive")
    return pd.DataFrame(
        {
            "cell_id": query,
            "oracle_rms": oracle,
            "idw_rms": scales,
            "idw_scale_k_i": scales / oracle,
        }
    ), spacing


def verify_run_rows(table, assignment, domain):
    expected = (
        scoring_rows(assignment, domain).set_index("cell_id", verify_integrity=True).sort_index()
    )
    actual = table.set_index("cell_id", verify_integrity=True).sort_index()
    if not actual.index.equals(expected.index) or not actual.trace_id.equals(expected.trace_id):
        raise ValueError("saved scoring rows differ from the seed's partition")


def analysis_hashes(repo, evaluation, experiments):
    paths = [
        evaluation,
        repo / "studies/study_067_forge_c3_common_grid/grid.yaml",
        repo / "scripts/evaluate_study067.py",
    ]
    paths += [
        p
        for folder in ("data", "evaluation", "pipelines", "visualization")
        for p in (repo / "src/seis_interp" / folder).glob("*study067*.py")
    ]
    paths += [
        repo / f"src/seis_interp/{name}"
        for name in (
            "evaluation/forge_fair_grid_statistics.py",
            "evaluation/trace_snr.py",
            "processing/forge_log_rms_idw.py",
            "processing/trace_rms_idw.py",
        )
    ]
    for experiment in experiments:
        paths += list((experiment / "study").glob("*.yaml"))
        paths += list((experiment / "method_runs").glob("*/*/*/config.resolved.yaml"))
        paths += list((experiment / "method_runs").glob("*/*/*/manifest.json"))
        paths.append(experiment / "experiment.json")
    return {
        str(p.relative_to(repo)) if p.is_relative_to(repo) else str(p): sha256_file(p)
        for p in sorted(set(paths))
    }


def require_output_directory(repo, output):
    output = output.resolve()
    if output == repo / "runs" or (repo / "runs").resolve() in output.parents:
        raise ValueError("analysis output must be outside runs/")
    output.mkdir(parents=True, exist_ok=False)
    return output
