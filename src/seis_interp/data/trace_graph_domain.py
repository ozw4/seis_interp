"""Verified trace-list geometry, visibility, and row/time mappings without labels."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from seis_interp import run_records
from seis_interp.data.benchmark_case_inputs import (
    validate_benchmark_preparation,
    validated_benchmark_mask_summary,
    verify_benchmark_case_inputs,
)
from seis_interp.data.benchmark_case_store import BENCHMARK_CASE_FILE_NAME, load_benchmark_case
from seis_interp.data.c3_volume_index_inputs import load_bound_benchmark_case
from seis_interp.data.c3_volume_index_store import OUTPUT_FILE_NAMES as VOLUME_FILE_NAMES
from seis_interp.data.c3_volume_index_store import load_c3_volume_index
from seis_interp.data.file_checksums import file_sha256
from seis_interp.data.interim_trace_dataset import load_interim_trace_dataset
from seis_interp.data.interpolation_mask_store import load_interpolation_mask
from seis_interp.data.prepared_partition import (
    NORMALIZATION_FILE_NAME,
    PREPARATION_FILE_NAME,
    TRACE_SPLIT_FILE_NAME,
)
from seis_interp.data.prepared_partition import OUTPUT_FILE_NAMES as PREPARED_FILE_NAMES
from seis_interp.data.prepared_partition_inputs import validated_prepared_split_rows
from seis_interp.data.trace_store import AMPLITUDES_FILE_NAME
from seis_interp.data.trace_store import OUTPUT_FILE_NAMES as INTERIM_FILE_NAMES
from seis_interp.processing.c3_volume_index import validated_index_range
from seis_interp.processing.interpolation_masks import (
    EVALUATION_TARGET_ROLE,
    OBSERVATION_ROLE_COLUMN,
    OBSERVED_ROLE,
    validate_interpolation_mask,
)
from seis_interp.processing.normalization import read_normalization_parameters
from seis_interp.processing.trace_canonicalization import (
    PHYSICAL_COORDINATE_COLUMNS,
    canonicalize_eligible_physical_coordinates,
)
from seis_interp.processing.trace_graph_geometry import compute_trace_graph_geometry
from seis_interp.processing.trace_splits import (
    SPLIT_COLUMN,
    TRAIN_SPLIT,
    validate_prepared_split_assignments,
)

TRAINING_POOLS = ("all_train_traces", "mask_observed")


@dataclass(frozen=True)
class TraceGraphDomain:
    """Metadata in stable trace order; no waveform or evaluation labels.

    Construct with :func:`build_trace_graph_domain` or the verified loaders.
    Native IDs are canonical ``array_row`` values. Arbitrary queries use unique
    IDs supplied by the caller; their array mapping may be absent or ``-1``.
    ``time_samples`` refers to the source array's half-open sample range.
    """

    trace_ids: np.ndarray
    source_xy_m: np.ndarray
    receiver_xy_m: np.ndarray
    ffids: np.ndarray
    array_rows: np.ndarray | None
    observed_mask: np.ndarray
    query_mask: np.ndarray
    time_s: np.ndarray
    time_samples: tuple[int, int]
    inputs_lock: dict[str, object]
    pool: str | None = None
    amplitudes_path: Path | None = None

    @property
    def query_indices(self) -> np.ndarray:
        """Local output positions, retaining the domain's query order."""
        return np.flatnonzero(self.query_mask)


def build_trace_graph_domain(
    *,
    trace_ids: np.ndarray,
    source_xy_m: np.ndarray,
    receiver_xy_m: np.ndarray,
    ffids: np.ndarray,
    observed_mask: np.ndarray,
    time_s: np.ndarray,
    array_rows: np.ndarray | None = None,
    query_mask: np.ndarray | None = None,
    time_samples: tuple[int, int] | None = None,
    inputs_lock: Mapping[str, object] | None = None,
    pool: str | None = None,
    amplitudes_path: Path | None = None,
) -> TraceGraphDomain:
    """Build a domain, including arbitrary coordinates without disk row IDs.

    Physical pairs are compared exactly, using the existing canonicalization
    key. A repeated physical pair under distinct IDs is rejected, so observed
    truth cannot remain visible under an alias of a missing query.
    """
    ids = _integer_vector(trace_ids, "trace_ids")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("trace_ids must be unique")
    geometry = compute_trace_graph_geometry(source_xy_m, receiver_xy_m, azimuth_min_offset_m=0.0)
    if len(geometry.source_xy_m) != len(ids):
        raise ValueError("geometry must contain one source/receiver pair per trace_id")
    physical_pairs = np.column_stack((geometry.source_xy_m, geometry.receiver_xy_m))
    if len(np.unique(physical_pairs, axis=0)) != len(ids):
        raise ValueError("duplicate physical source/receiver pair under different trace_ids")
    shots = _integer_vector(ffids, "ffids", size=len(ids))
    observed = _boolean_vector(observed_mask, "observed_mask", len(ids))
    query = ~observed if query_mask is None else _boolean_vector(query_mask, "query_mask", len(ids))
    if np.any(observed & query):
        raise ValueError("query nodes must be unobserved")
    rows = None
    if array_rows is not None:
        rows = _integer_vector(array_rows, "array_rows", size=len(ids))
        if np.any(rows < -1) or np.any(rows[observed] < 0):
            raise ValueError(
                "array_rows must be nonnegative for observed nodes; -1 denotes a query"
            )
        mapped = rows[rows >= 0]
        if len(np.unique(mapped)) != len(mapped):
            raise ValueError("mapped array_rows must be unique")
    raw_times = np.asarray(time_s)
    if raw_times.dtype.kind not in "fiu":
        raise ValueError("time_s must be a nonempty, finite, strictly increasing real vector")
    times = np.asarray(raw_times, dtype=np.float64)
    if (
        times.ndim != 1
        or not len(times)
        or not np.all(np.isfinite(times))
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("time_s must be a nonempty, finite, strictly increasing real vector")
    selection = validated_index_range(
        (0, len(times)) if time_samples is None else time_samples,
        name="time_samples",
    )
    if selection[1] - selection[0] != len(times):
        raise ValueError("time_samples length must match time_s")
    if pool is not None and pool not in TRAINING_POOLS:
        raise ValueError(f"pool must be one of {TRAINING_POOLS}")
    if pool is not None and (not np.all(observed) or np.any(query)):
        raise ValueError("a training pool must contain only allowed observed training traces")
    return TraceGraphDomain(
        trace_ids=ids,
        source_xy_m=geometry.source_xy_m,
        receiver_xy_m=geometry.receiver_xy_m,
        ffids=shots,
        array_rows=rows,
        observed_mask=observed,
        query_mask=query.copy(),
        time_s=np.array(times, dtype=np.float64, copy=True),
        time_samples=selection,
        inputs_lock=deepcopy(dict(inputs_lock or {})),
        pool=pool,
        amplitudes_path=None if amplitudes_path is None else Path(amplitudes_path),
    )


def load_benchmark_trace_graph_domain(
    *,
    interim_dir: Path,
    processed_dir: Path,
    mask_dir: Path,
    case_dir: Path,
    volume_dir: Path | None = None,
    time_samples: tuple[int, int] | None = None,
) -> TraceGraphDomain:
    """Verify a native case and optionally restrict both roles to a volume crop."""
    case = load_benchmark_case(case_dir)
    verify_benchmark_case_inputs(
        case, interim_dir=interim_dir, processed_dir=processed_dir, mask_dir=mask_dir
    )
    input_files = case["input_files"]
    canonical, times, dataset_id, duplicate_audit = _load_partition(
        interim_dir, processed_dir, input_files
    )
    mask, mask_metadata = load_interpolation_mask(mask_dir)
    summary = validated_benchmark_mask_summary(
        mask_metadata,
        mask_row_count=len(mask),
        trace_count=int(duplicate_audit["source_trace_count"]),
        mask_array_rows=mask["array_row"].to_numpy(dtype=np.int64),
        dataset_id=dataset_id,
        input_hashes=input_files,
    )
    if case["dataset_id"] != dataset_id or case["partition"] != mask_metadata["partition"]:
        raise ValueError("benchmark case dataset_id and partition must match its mask")
    if case["mask"] != summary:
        raise ValueError("benchmark case mask summary does not match interpolation_mask.json")
    selected = canonical.loc[canonical[SPLIT_COLUMN].eq(case["partition"])].copy()
    validate_interpolation_mask(
        mask, expected_array_rows=selected["array_row"].to_numpy(dtype=np.int64)
    )
    duplicates = summary["duplicate_physical_coordinates"]
    if duplicates != {key: duplicate_audit[key] for key in ("policy", "removed_trace_count")}:
        raise ValueError(
            "mask duplicate physical coordinate summary does not match canonicalization"
        )
    if summary["candidate_ffid_count"] != selected["ffid"].nunique():
        raise ValueError("mask candidate_ffid_count does not match the canonical partition")
    selected = selected.merge(mask, on="array_row", validate="one_to_one", sort=False)
    lock: dict[str, object] = {
        "partition": case["partition"],
        "benchmark_case": {
            "case_id": case["case_id"],
            "file": BENCHMARK_CASE_FILE_NAME,
            "sha256": file_sha256(Path(case_dir) / BENCHMARK_CASE_FILE_NAME),
            "input_files": deepcopy(input_files),
        },
    }
    if volume_dir is not None:
        selected, time_samples, volume_lock = _select_volume(
            selected, case_dir, volume_dir, time_samples
        )
        lock["benchmark_volume"] = volume_lock
    return _domain_from_table(selected, times, time_samples, lock, interim_dir)


def load_training_trace_graph_domain(
    *,
    interim_dir: Path,
    processed_dir: Path,
    pool: str,
    mask_dir: Path | None = None,
    case_dir: Path | None = None,
    volume_dir: Path | None = None,
    time_samples: tuple[int, int] | None = None,
) -> TraceGraphDomain:
    """Select O0 once from canonical train rows or a train case's observed rows."""
    if pool not in TRAINING_POOLS:
        raise ValueError(f"pool must be one of {TRAINING_POOLS}")
    if (mask_dir is None) != (case_dir is None):
        raise ValueError("mask_dir and case_dir must be supplied together")
    if pool == "mask_observed" or volume_dir is not None or case_dir is not None:
        if mask_dir is None or case_dir is None:
            raise ValueError("mask_observed and optional volume require a training case and mask")
        domain = load_benchmark_trace_graph_domain(
            interim_dir=interim_dir,
            processed_dir=processed_dir,
            mask_dir=mask_dir,
            case_dir=case_dir,
            volume_dir=volume_dir,
            time_samples=time_samples,
        )
        if domain.inputs_lock["partition"] != TRAIN_SPLIT:
            raise ValueError("training domain requires a train partition case")
        selected = (
            domain.observed_mask
            if pool == "mask_observed"
            else np.ones(len(domain.trace_ids), dtype=np.bool_)
        )
        lock = deepcopy(domain.inputs_lock)
        lock["training_data"] = {"pool": pool, "time_samples": list(domain.time_samples)}
        return replace(
            domain,
            trace_ids=domain.trace_ids[selected].copy(),
            source_xy_m=domain.source_xy_m[selected].copy(),
            receiver_xy_m=domain.receiver_xy_m[selected].copy(),
            ffids=domain.ffids[selected].copy(),
            array_rows=domain.array_rows[selected].copy(),
            observed_mask=np.ones(int(selected.sum()), dtype=np.bool_),
            query_mask=np.zeros(int(selected.sum()), dtype=np.bool_),
            inputs_lock=lock,
            pool=pool,
        )
    input_files = {
        "interim": run_records.file_hashes(Path(interim_dir), INTERIM_FILE_NAMES),
        "processed": run_records.file_hashes(Path(processed_dir), PREPARED_FILE_NAMES),
    }
    canonical, times, _, _ = _load_partition(interim_dir, processed_dir, input_files)
    selected = canonical.loc[canonical[SPLIT_COLUMN].eq(TRAIN_SPLIT)].copy()
    selected[OBSERVATION_ROLE_COLUMN] = OBSERVED_ROLE
    lock = {"partition": TRAIN_SPLIT, "input_files": input_files}
    domain = _domain_from_table(selected, times, time_samples, lock, interim_dir)
    lock = deepcopy(domain.inputs_lock)
    lock["training_data"] = {"pool": pool, "time_samples": list(domain.time_samples)}
    return replace(domain, inputs_lock=lock, pool=pool)


def _load_partition(
    interim_dir: Path,
    processed_dir: Path,
    input_files: Mapping[str, object],
) -> tuple[pd.DataFrame, np.ndarray, str, dict[str, object]]:
    # mmap validates the file header; the empty row selection reads no amplitudes.
    dataset = load_interim_trace_dataset(
        interim_dir,
        memory_map_amplitudes=True,
        amplitude_validation_rows=np.empty(0, dtype=np.int64),
    )
    preparation = json.loads((Path(processed_dir) / PREPARATION_FILE_NAME).read_text("utf-8"))
    if not isinstance(preparation, dict):
        raise ValueError("preparation.json must contain a JSON object")
    dataset_id = validate_benchmark_preparation(
        dataset.metadata, preparation, interim_hashes=input_files["interim"]
    )
    read_normalization_parameters(Path(processed_dir) / NORMALIZATION_FILE_NAME)
    split = pd.read_parquet(Path(processed_dir) / TRACE_SPLIT_FILE_NAME)
    validated_prepared_split_rows(
        split, expected_array_rows=dataset.trace_table["array_row"].to_numpy(dtype=np.int64)
    )
    columns = ["array_row", "ffid", *PHYSICAL_COORDINATE_COLUMNS]
    joined = dataset.trace_table[columns].merge(
        split[["array_row", SPLIT_COLUMN]], on="array_row", validate="one_to_one", sort=False
    )
    validate_prepared_split_assignments(joined, preparation)
    canonical, audit = canonicalize_eligible_physical_coordinates(joined)
    audit["source_trace_count"] = len(dataset.trace_table)
    return canonical, dataset.time_s, dataset_id, audit


def _select_volume(
    table: pd.DataFrame,
    case_dir: Path,
    volume_dir: Path,
    time_samples: tuple[int, int] | None,
) -> tuple[pd.DataFrame, tuple[int, int], dict[str, object]]:
    index, metadata = load_c3_volume_index(volume_dir)
    load_bound_benchmark_case(metadata, case_dir=case_dir)
    rows = index["array_row"].to_numpy(dtype=np.int64)
    if not np.all(np.isin(rows, table["array_row"].to_numpy(dtype=np.int64))):
        raise ValueError("volume index rows must belong to the canonical case partition")
    selected = table.set_index("array_row").loc[rows].reset_index()
    for column in ("ffid", "source_x_m", "source_y_m"):
        if not np.array_equal(selected[column].to_numpy(), index[column].to_numpy()):
            raise ValueError(f"volume index {column} does not match the canonical case rows")
    for axis in ("x", "y"):
        relative = selected[f"receiver_{axis}_m"] - selected[f"source_{axis}_m"]
        if not np.array_equal(relative.to_numpy(), index[f"relative_receiver_{axis}_m"].to_numpy()):
            raise ValueError(
                f"volume index receiver_{axis}_m does not match the canonical case rows"
            )
    counts = {
        role: int(selected[OBSERVATION_ROLE_COLUMN].eq(role).sum())
        for role in (OBSERVED_ROLE, EVALUATION_TARGET_ROLE)
    }
    if counts != metadata["role_counts"]:
        raise ValueError("selected volume role counts do not match volume metadata")
    volume_time = tuple(metadata["selection"]["time"])
    if (
        time_samples is not None
        and validated_index_range(time_samples, name="time_samples") != volume_time
    ):
        raise ValueError("time_samples must match the optional volume time selection")
    lock = {
        "volume_id": metadata["volume_id"],
        "files": run_records.file_hashes(Path(volume_dir), VOLUME_FILE_NAMES),
        "array_rows": rows.tolist(),
        "selection": deepcopy(metadata["selection"]),
    }
    return selected, volume_time, lock


def _domain_from_table(
    table: pd.DataFrame,
    times: np.ndarray,
    time_samples: tuple[int, int] | None,
    lock: dict[str, object],
    interim_dir: Path,
) -> TraceGraphDomain:
    if table.empty:
        raise ValueError("selected trace domain is empty")
    selection = validated_index_range(
        (0, len(times)) if time_samples is None else time_samples, name="time_samples"
    )
    if selection[1] > len(times):
        raise ValueError("time_samples is outside the source time grid")
    table = table.sort_values("array_row", kind="stable")
    rows = table["array_row"].to_numpy(dtype=np.int64)
    lock["time_samples"] = list(selection)
    return build_trace_graph_domain(
        trace_ids=rows,
        source_xy_m=table[["source_x_m", "source_y_m"]].to_numpy(dtype=np.float64),
        receiver_xy_m=table[["receiver_x_m", "receiver_y_m"]].to_numpy(dtype=np.float64),
        ffids=table["ffid"].to_numpy(),
        array_rows=rows,
        observed_mask=table[OBSERVATION_ROLE_COLUMN].eq(OBSERVED_ROLE).to_numpy(),
        query_mask=table[OBSERVATION_ROLE_COLUMN].eq(EVALUATION_TARGET_ROLE).to_numpy(),
        time_s=times[slice(*selection)],
        time_samples=selection,
        inputs_lock=lock,
        amplitudes_path=Path(interim_dir) / AMPLITUDES_FILE_NAME,
    )


def _integer_vector(values: np.ndarray, name: str, *, size: int | None = None) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or result.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if size is not None and len(result) != size:
        raise ValueError(f"{name} must contain one value per trace")
    if len(result) and int(result.max()) > np.iinfo(np.int64).max:
        raise ValueError(f"{name} values must fit in int64")
    return np.array(result, dtype=np.int64, copy=True)


def _boolean_vector(values: np.ndarray, name: str, size: int) -> np.ndarray:
    result = np.asarray(values)
    if result.shape != (size,) or result.dtype != np.bool_:
        raise ValueError(f"{name} must be a boolean vector with one value per trace")
    return result.copy()
