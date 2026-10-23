"""Conservative prediction reuse from native grids, historical runs and pilots."""

import hashlib
import json
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from seis_interp.data.forge_headers import sha256_file
from seis_interp.data.forge_mvp_artifacts import verify_hashes
from seis_interp.forge_mvp_config import canvas_time_stages
from seis_interp.processing.drr_grid import resolve_drr_grid_config
from seis_interp.processing.experiment_commonality import flatten_settings

# Declared execution/diagnostic switches. cuDNN autotuning may change bitwise
# results; the study explicitly excludes it from scientific identity. Precision,
# patch/core shape, RNG, losses, model settings and training batches stay significant.
IMPORT_EXECUTION_ONLY_KEYS = (
    "device",
    "execution.cpu_threads",
    "execution.workers",
    "execution.batch_matrix_limit",
    "execution.saved_tensors_on_cpu",
    "execution.model_memory_options.receptive_field_pruning",
    "execution.model_memory_options.fused_edge_aggregation",
    "execution.model_memory_options.activation_checkpointing",
    "training.execution.cpu_threads",
    "prediction.query_batch_size",
    "prediction.profile_batch_size",
    "execution.cudnn_benchmark",
    "execution.expected_report_seconds",
    "execution.watchdog_floor_seconds",
    "execution.wall_clock_limit_seconds",
    "training.ema_eval_decays",
)


def method_config_hash(config, *, time_sample_count=None):
    config = deepcopy(config)
    if config.get("method_id") == "drr_grid":
        # Preserve a saved run's declared computation even if the current grid
        # adapter no longer permits that backend/boundary at this overlap.
        declared = {
            key: config["algorithm"].pop(key)
            for key in ("backend", "boundary_rule")
            if key in config["algorithm"]
        }
        config = resolve_drr_grid_config(config)
        config["algorithm"].update(declared)
    training = config.get("training", {})
    training.pop("ema_eval_decays", None)
    if config.get("method_id") == "nersi_profile":
        if training.get("coordinate_jitter_cells") is None:
            training.pop("coordinate_jitter_cells", None)
        alignment = config.get("alignment") or {}
        canvas_length = (
            time_sample_count + alignment.get("max_delay_samples", 0)
            if time_sample_count is not None and (not alignment or "max_delay_samples" in alignment)
            else None
        )
        # A YAML profile length is overwritten at runtime. Without the input
        # time count (or a fixed alignment maximum), inertness is unproven.
        if not config.get("canvas_time_upsampling") or (
            canvas_length is not None and not canvas_time_stages(canvas_length)
        ):
            config.pop("canvas_time_upsampling", None)
        if not training.get("loss_support_mask") or not config.get("alignment"):
            training.pop("loss_support_mask", None)
    settings = flatten_settings(config)
    if config.get("method_id") == "nersi_profile":
        # The input time axis and alignment fix the canvas; the runtime always
        # derives this decoder length again in canvas_method_config.
        settings.pop("model.profile_shape.0", None)
    for key in IMPORT_EXECUTION_ONLY_KEYS:
        settings.pop(key, None)
    return hashlib.sha256(
        json.dumps(settings, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _array_digest(digest, name, values):
    array = np.asarray(values)
    digest.update(json.dumps([name, str(array.dtype), array.shape]).encode())
    for start in range(0, len(array), 1024):
        digest.update(np.ascontiguousarray(array[start : start + 1024]).tobytes())


def saved_input_fingerprint(experiment: Path, seed: int, method: str, domains: tuple):
    """Hash saved assignment, geometry, O waveforms, time and boundary RMS scalars.

    No held-out waveform is loaded. A single-config FORGE prediction may cover
    more queries; only the current domains' restoration scales enter this hash.
    """
    experiment = Path(experiment)
    paths = [experiment / "study/config.yaml", experiment / "experiment.json"]
    paths.extend(
        path
        for path in (experiment / "input").rglob("*")
        if path.is_file() and path.name != "waveforms.npy"
    )
    revision = tuple(
        (str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in sorted(paths)
    )
    return _saved_input_fingerprint(experiment, seed, method, domains, revision)


@lru_cache(maxsize=256)
def _saved_input_fingerprint(experiment, seed, method, domains, revision):
    contract = yaml.safe_load((experiment / "study/config.yaml").read_text())
    base = experiment / "input"
    directory = base / f"seed_{seed}"
    if not directory.exists():
        directory = base
    partition = pd.read_parquet(directory / "partition.parquet")
    column = next(name for name in ("role", "trace_role", "region_role") if name in partition)
    roles = partition[column]
    observed = partition.cell_id[roles.eq("train")].to_numpy(dtype=np.int64)
    queries = partition.cell_id[roles.isin(domains)].to_numpy(dtype=np.int64)
    digest = hashlib.sha256()
    for role in ("train", "validation", "test"):
        _array_digest(
            digest, role, np.sort(partition.cell_id[roles.eq(role)].to_numpy(dtype=np.int64))
        )
    _array_digest(digest, "observed_order", observed)
    waveforms = np.load(directory / "observed_waveforms.npy", mmap_mode="r", allow_pickle=False)
    _array_digest(digest, "observed_waveforms", waveforms)
    mode = "real" if method in ("regsi_real", "nersi_profile") else "grid"
    geometry_path = base / f"{mode}_geometry.parquet"
    if not geometry_path.exists():
        geometry_path = base / "geometry.parquet"
    geometry = pd.read_parquet(geometry_path)
    digest.update(json.dumps(list(geometry.columns)).encode())
    _array_digest(digest, "geometry", pd.util.hash_pandas_object(geometry, index=False).to_numpy())
    time_path = base / "time_s.npy"
    times = (
        np.load(time_path, allow_pickle=False)
        if time_path.exists()
        else np.arange(contract["time_sample_count"], dtype=np.float64)
        * contract["sample_interval_us"]
        / 1e6
    )
    _array_digest(digest, "time_s", times)
    digest.update(json.dumps(contract["spatial_shape"]).encode())
    scales_path = directory / "trace_scales.parquet"
    if scales_path.exists():
        scales = pd.read_parquet(scales_path).set_index("cell_id")
        _array_digest(digest, "observed_rms", scales.loc[observed, "rms"].to_numpy())
        _array_digest(digest, "restoration_rms", scales.loc[queries, "rms"].to_numpy())
    seal = json.loads((experiment / "experiment.json").read_text())
    rms = (
        seal["partitions"][str(seed)]["global_rms"] if "partitions" in seal else seal["global_rms"]
    )
    digest.update(json.dumps(rms).encode())
    return digest.hexdigest()


@lru_cache(maxsize=64)
def source_snapshot_hash(experiment):
    digest = hashlib.sha256()
    files = sorted((experiment / "source").rglob("*"))
    files = [path for path in files if path.is_file() and "__pycache__" not in path.parts]
    if not files:
        raise ValueError("import source has no source snapshot")
    for path in files:
        digest.update(str(path.relative_to(experiment / "source")).encode())
        digest.update(sha256_file(path).encode())
    return digest.hexdigest()


def import_run_paths(source: Path, method: str, stage: str):
    """Find native runs or declared pilot/historical candidates, never summary rows."""
    if (source / "manifest.json").is_file():
        return [source]
    directory = source / "method_runs" / method
    candidates_path = source / "candidates.json"
    if candidates_path.exists() and stage == "S1":
        # P1/P1b grids and one-candidate P2/P3 pilots declare their own IDs.
        # Their summaries and worker state are not evidence of completion.
        candidates = json.loads(candidates_path.read_text()).get(method, [])
        paths = sorted({directory / stage / item["id"] / "manifest.json" for item in candidates})
        paths = [path for path in paths if path.is_file()]
    else:
        paths = sorted((directory / stage).glob("*/manifest.json"))
    # FORGE P3 uses the single-config executor rather than a candidate subdir.
    single = directory / "manifest.json"
    if stage == "S1" and single.exists():
        paths.append(single)
    return [path.parent for path in paths]


def source_experiment(run):
    for path in (run, *run.parents):
        if (path / "experiment.json").is_file():
            return path
    raise ValueError(f"import run has no experiment: {run}")


def _covers_source_domains(run, experiment, seed, domains):
    directory = experiment / "input" / f"seed_{seed}"
    if not directory.exists():
        directory = experiment / "input"
    partition = pd.read_parquet(directory / "partition.parquet")
    role = next(name for name in ("role", "trace_role", "region_role") if name in partition)
    expected = partition.cell_id[partition[role].isin(domains)].to_numpy()
    cells = np.load(run / "cell_ids.npy", allow_pickle=False)
    prediction = np.load(run / "prediction.npy", mmap_mode="r", allow_pickle=False)
    return (
        len(np.unique(cells)) == len(cells)
        and pd.Index(cells).get_indexer(expected).min() >= 0
        and prediction.ndim == 2
        and len(prediction) == len(cells)
        and prediction.dtype == np.float32
    )


def find_prediction_import(repo, sources, config, input_hash, seed, stage):
    """Return only a completed, hash-verified prediction with identical computation."""
    method = config["method_id"]
    domains = ("validation",) if stage == "S1" else ("validation", "test")
    for source in sources:
        root = (repo / source).resolve()
        for run in import_run_paths(root, method, stage):
            try:
                record = json.loads((run / "manifest.json").read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                # A continuing historical worker publishes its manifest in place.
                continue
            if record.get("status") != "completed" or not all(
                (run / name).is_file()
                for name in ("prediction.npy", "cell_ids.npy", "config.resolved.yaml")
            ):
                continue
            experiment = source_experiment(run)
            seal = json.loads((experiment / "experiment.json").read_text())
            if seal.get("validation_only") and stage != "S1":
                continue
            contract = yaml.safe_load((experiment / "study/config.yaml").read_text())
            # The fingerprint below requires the same input time axis. Use its
            # runtime length for both configs, never either YAML profile length.
            time_count = contract["time_sample_count"]
            wanted = method_config_hash(config, time_sample_count=time_count)
            original = yaml.safe_load((run / "config.resolved.yaml").read_text())
            if method_config_hash(original, time_sample_count=time_count) != wanted:
                continue
            source_seed = record.get("trace_seed", record.get("region_seed"))
            if source_seed is None:
                source_seed = contract.get("trace_split", {}).get(
                    "search_seed", contract["validation_holdout"]["mask_seed"]
                )
            if saved_input_fingerprint(experiment, source_seed, method, domains) != input_hash:
                continue
            names = ("prediction.npy", "cell_ids.npy", "config.resolved.yaml")
            if not all(name in record.get("artifacts", {}) for name in names):
                continue
            verify_hashes(run, {name: record["artifacts"][name] for name in names})
            if not _covers_source_domains(run, experiment, source_seed, domains):
                continue
            return {
                "source_path": str(run),
                "source_config_sha256": sha256_file(run / "config.resolved.yaml"),
                "source_method_config_hash": wanted,
                "source_input_hash": input_hash,
                "source_snapshot_sha256": source_snapshot_hash(experiment),
                "execution_only_keys": list(IMPORT_EXECUTION_ONLY_KEYS),
            }
    return None


def imported_prediction(provenance, cells):
    """Subset a completed full FORGE prediction without exposing test amplitudes."""
    root = Path(provenance["source_path"])
    ids = np.load(root / "cell_ids.npy", allow_pickle=False)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("import prediction has duplicate cell IDs")
    positions = pd.Index(ids).get_indexer(cells)
    if np.any(positions < 0):
        raise ValueError("import prediction does not cover current scoring cells")
    values = np.load(root / "prediction.npy", mmap_mode="r", allow_pickle=False)
    return np.ascontiguousarray(values[positions])
