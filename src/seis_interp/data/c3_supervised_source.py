"""Read complete labels only from disjoint canonical C3 training regions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import pandas as pd

from seis_interp import run_records
from seis_interp.data.interim_trace_dataset import load_interim_trace_dataset
from seis_interp.data.prepared_partition import OUTPUT_FILE_NAMES as PREPARED_FILE_NAMES
from seis_interp.data.prepared_partition import PREPARATION_FILE_NAME, TRACE_SPLIT_FILE_NAME
from seis_interp.data.trace_store import OUTPUT_FILE_NAMES as INTERIM_FILE_NAMES
from seis_interp.data.trace_table import validated_array_rows
from seis_interp.processing.c3_volume_index import (
    VOLUME_AXIS_ORDER,
    build_c3_row_id_grid,
    build_c3_volume_index,
    validated_index_range,
)
from seis_interp.processing.trace_canonicalization import (
    DUPLICATE_PHYSICAL_COORDINATE_POLICY,
    canonicalize_eligible_physical_coordinates,
)
from seis_interp.processing.trace_splits import (
    C3_SOURCE_LINE_BLOCKS_SPLIT_SCOPE,
    EXCLUDED_SPLIT,
    SPLIT_COLUMN,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    validate_prepared_split_assignments,
    validated_random_seed,
)

ARRAY_ROWS_HASH_RULE = "sha256_shape_le_int64_then_c_order_le_int64"
_RMS_TRACE_CHUNK_SIZE = 4096


@dataclass(frozen=True)
class C3SupervisedRegion:
    """A canonical row map with global ranges and region-local shape."""

    array_rows: np.ndarray
    time_range: tuple[int, int]
    shape: tuple[int, int, int, int, int]
    selection: dict[str, list[int]]


@dataclass(frozen=True)
class C3SupervisedSource:
    """Bound train-only row maps and their fixed RMS, without a dense amplitude cube."""

    fit: C3SupervisedRegion
    selection: C3SupervisedRegion
    amplitude_rms: float
    inputs_lock: dict[str, object]
    _amplitudes: np.ndarray = field(repr=False, compare=False)

    def read_patch(
        self,
        region: str,
        start: tuple[int, int, int, int, int],
        shape: tuple[int, int, int, int, int],
    ) -> np.ndarray:
        """Copy one bounded label patch as contiguous ``(T,Sx,Sy,Rx,Ry)`` float32."""
        if region not in ("fit", "selection"):
            raise ValueError("region must be fit or selection")
        rows = self.patch_array_rows(region, start, shape)
        selected = self.fit if region == "fit" else self.selection
        starts = _five_integers(start, name="start", minimum=0)
        sizes = _five_integers(shape, name="shape", minimum=1)
        time_start = selected.time_range[0] + starts[0]
        values = self._amplitudes[rows.reshape(-1), time_start : time_start + sizes[0]]
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{region} label patch contains non-finite amplitudes")
        return np.array(values.T.reshape(sizes), dtype=np.float32, order="C", copy=True)

    def patch_array_rows(
        self,
        region: str,
        start: tuple[int, int, int, int, int],
        shape: tuple[int, int, int, int, int],
    ) -> np.ndarray:
        """Return one patch's row map, rejecting unauthorized sentinel cells."""
        if region not in ("fit", "selection"):
            raise ValueError("region must be fit or selection")
        selected = self.fit if region == "fit" else self.selection
        starts = _five_integers(start, name="start", minimum=0)
        sizes = _five_integers(shape, name="shape", minimum=1)
        for axis, offset, length, available in zip(
            VOLUME_AXIS_ORDER, starts, sizes, selected.shape, strict=True
        ):
            if offset + length > available:
                raise ValueError(f"patch {axis} range is outside the {region} region")
        spatial_slices = tuple(
            slice(offset, offset + length)
            for offset, length in zip(starts[1:], sizes[1:], strict=True)
        )
        rows = selected.array_rows[spatial_slices]
        if np.any(rows < 0):
            raise ValueError(f"{region} patch contains unauthorized or absent trace cells")
        return rows


def load_c3_supervised_source(
    *,
    interim_dir: Path,
    processed_dir: Path,
    fit_region: Mapping[str, object],
    selection_region: Mapping[str, object],
) -> C3SupervisedSource:
    """Verify prepared source-line splits and fit RMS using complete fit labels only."""
    loaded = _load_prepared_training_inputs(interim_dir, processed_dir)
    dataset, preparation, train_rows = loaded.dataset, loaded.preparation, loaded.train_rows
    regions = {
        name: _build_region(
            selection,
            name=name,
            trace_table=dataset.trace_table,
            train_rows=train_rows,
            time_count=len(dataset.time_s),
            train_line_range=preparation["source_line_ranges"][TRAIN_SPLIT],
        )
        for name, selection in (("fit", fit_region), ("selection", selection_region))
    }
    fit, selection = regions["fit"], regions["selection"]
    if np.intersect1d(fit.array_rows, selection.array_rows).size:
        raise ValueError("fit and selection regions must have disjoint array_row sets")
    amplitude_rms = _fit_amplitude_rms(dataset.amplitudes, fit)
    inputs_lock = _source_inputs_lock(loaded, regions)
    return C3SupervisedSource(fit, selection, amplitude_rms, inputs_lock, dataset.amplitudes)


def load_c3_training_dataset_source(
    *,
    interim_dir: Path,
    processed_dir: Path,
    authorized_train_rows: np.ndarray,
    time_range: Sequence[int],
    selection_region: Mapping[str, object],
    amplitude_rms: float,
    normalization_source: Mapping[str, object],
) -> C3SupervisedSource:
    """Bind the complete QC training dataset to a sparse row grid and fixed RMS."""
    loaded = _load_prepared_training_inputs(interim_dir, processed_dir)
    dataset, preparation, train_rows = loaded.dataset, loaded.preparation, loaded.train_rows
    authorized = np.asarray(authorized_train_rows)
    if (
        authorized.ndim != 1
        or authorized.dtype.kind not in "iu"
        or authorized.dtype.kind == "b"
        or len(authorized) == 0
        or len(np.unique(authorized)) != len(authorized)
    ):
        raise ValueError("authorized_train_rows must be a non-empty unique integer vector")
    authorized = authorized.astype(np.int64, copy=False)
    if not np.array_equal(np.sort(authorized), np.sort(train_rows)):
        raise ValueError("authorized_train_rows differ from canonical QC train rows")
    selected_time = validated_index_range(time_range, name="training_dataset.time")
    if selected_time[1] > len(dataset.time_s):
        raise ValueError("training_dataset.time is outside the interim time axis")
    grid, spatial_selection = build_c3_row_id_grid(
        dataset.trace_table,
        authorized,
        source_line_range=tuple(preparation["source_line_ranges"][TRAIN_SPLIT]),
    )
    fit_selection = {"time": list(selected_time), **spatial_selection}
    fit = C3SupervisedRegion(
        grid,
        selected_time,
        (selected_time[1] - selected_time[0], *grid.shape),
        fit_selection,
    )
    selection = _build_region(
        selection_region,
        name="selection",
        trace_table=dataset.trace_table,
        train_rows=train_rows,
        time_count=len(dataset.time_s),
        train_line_range=preparation["source_line_ranges"][TRAIN_SPLIT],
    )
    rms = _positive_finite(amplitude_rms, name="amplitude_rms")
    if not isinstance(normalization_source, Mapping) or not normalization_source:
        raise ValueError("normalization_source must be a non-empty mapping")
    regions = {"fit": fit, "selection": selection}
    inputs_lock = _source_inputs_lock(loaded, regions)
    inputs_lock["training_dataset"] = {
        "authorized_trace_count": len(authorized),
        "time_samples": list(selected_time),
        "invalid_row_sentinel": -1,
    }
    inputs_lock["normalization_source"] = dict(normalization_source)
    return C3SupervisedSource(fit, selection, rms, inputs_lock, dataset.amplitudes)


@dataclass(frozen=True)
class _PreparedTrainingInputs:
    dataset: object
    preparation: Mapping[str, object]
    train_rows: np.ndarray
    dataset_id: str
    partition_seed: int
    interim_hashes: dict[str, object]
    processed_hashes: dict[str, object]


def _load_prepared_training_inputs(
    interim_dir: Path, processed_dir: Path
) -> _PreparedTrainingInputs:
    interim, processed = Path(interim_dir), Path(processed_dir)
    interim_hashes = run_records.file_hashes(interim, INTERIM_FILE_NAMES)
    processed_hashes = run_records.file_hashes(processed, PREPARED_FILE_NAMES)
    preparation = json.loads((processed / PREPARATION_FILE_NAME).read_text(encoding="utf-8"))
    if not isinstance(preparation, Mapping):
        raise ValueError("preparation.json must contain an object")
    if preparation.get("input_files") != interim_hashes:
        raise ValueError("preparation.json input_files do not match the current interim files")
    if preparation.get("split_scope") != C3_SOURCE_LINE_BLOCKS_SPLIT_SCOPE:
        raise ValueError("supervised C3 source requires c3_source_line_blocks preparation")
    partition_seed = validated_random_seed(preparation.get("random_seed"))
    dataset = load_interim_trace_dataset(
        interim,
        memory_map_amplitudes=True,
        amplitude_validation_rows=np.empty(0, dtype=np.int64),
    )
    dataset_id = dataset.metadata.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError("dataset_id must be a non-empty string")
    for key in ("dataset_id", "trace_count", "sample_count"):
        if preparation.get(key) != dataset.metadata[key]:
            raise ValueError(f"preparation.json {key} does not match the interim dataset")
    split_table = pd.read_parquet(processed / TRACE_SPLIT_FILE_NAME)
    split_rows = validated_array_rows(split_table, require_contiguous=True)
    if len(split_rows) != len(dataset.trace_table):
        raise ValueError("trace_split.parquet array_row values must match the interim table")
    if (
        SPLIT_COLUMN not in split_table
        or not split_table[SPLIT_COLUMN]
        .isin([TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT, EXCLUDED_SPLIT])
        .all()
    ):
        raise ValueError("trace_split.parquet must contain known split values")
    joined = dataset.trace_table.merge(
        split_table[["array_row", SPLIT_COLUMN]], on="array_row", sort=False, validate="one_to_one"
    )
    validate_prepared_split_assignments(joined, preparation)
    canonical, _ = canonicalize_eligible_physical_coordinates(joined)
    train_rows = canonical.loc[canonical[SPLIT_COLUMN].eq(TRAIN_SPLIT), "array_row"].to_numpy(
        dtype=np.int64
    )
    return _PreparedTrainingInputs(
        dataset,
        preparation,
        train_rows,
        dataset_id,
        partition_seed,
        interim_hashes,
        processed_hashes,
    )


def _source_inputs_lock(
    loaded: _PreparedTrainingInputs, regions: Mapping[str, C3SupervisedRegion]
) -> dict[str, object]:
    return {
        "dataset_id": loaded.dataset_id,
        "partition": TRAIN_SPLIT,
        "partition_random_seed": loaded.partition_seed,
        "interim": loaded.interim_hashes,
        "processed": loaded.processed_hashes,
        "canonical_policy": DUPLICATE_PHYSICAL_COORDINATE_POLICY,
        "array_rows_hash_rule": ARRAY_ROWS_HASH_RULE,
        "regions": {
            name: {
                "selection": {axis: list(bounds) for axis, bounds in region.selection.items()},
                "shape": list(region.shape),
                "array_rows_sha256": _mapping_sha256(region.array_rows),
            }
            for name, region in regions.items()
        },
    }


def _build_region(
    value: Mapping[str, object],
    *,
    name: str,
    trace_table: pd.DataFrame,
    train_rows: np.ndarray,
    time_count: int,
    train_line_range: Sequence[int],
) -> C3SupervisedRegion:
    if not isinstance(value, Mapping) or set(value) != set(VOLUME_AXIS_ORDER):
        raise ValueError(f"{name} region must contain exactly {list(VOLUME_AXIS_ORDER)}")
    ranges = {
        axis: validated_index_range(value[axis], name=f"{name}.{axis}")
        for axis in VOLUME_AXIS_ORDER
    }
    if ranges["time"][1] > time_count:
        raise ValueError(f"{name}.time is outside the interim time axis")
    if (
        not train_line_range[0]
        <= ranges["source_line"][0]
        < ranges["source_line"][1]
        <= train_line_range[1]
    ):
        raise ValueError(f"{name}.source_line must select only train partition lines")
    try:
        index = build_c3_volume_index(
            trace_table,
            train_rows,
            source_line_range=ranges["source_line"],
            shot_in_line_range=ranges["shot_in_line"],
            relative_receiver_x_range=ranges["relative_receiver_x"],
            relative_receiver_y_range=ranges["relative_receiver_y"],
        )
    except ValueError as error:
        raise ValueError(f"{name} region: {error}") from error
    shape = tuple(stop - start for start, stop in ranges.values())
    rows = np.ascontiguousarray(index["array_row"].to_numpy(dtype=np.int64).reshape(shape[1:]))
    if not np.all(np.isin(rows, train_rows)):
        raise ValueError(f"{name} region contains non-train array rows")
    rows.flags.writeable = False
    return C3SupervisedRegion(
        rows, ranges["time"], shape, {axis: list(bounds) for axis, bounds in ranges.items()}
    )


def _fit_amplitude_rms(amplitudes: np.ndarray, region: C3SupervisedRegion) -> float:
    rows = region.array_rows.reshape(-1)
    energy = 0.0
    for start in range(0, len(rows), _RMS_TRACE_CHUNK_SIZE):
        chunk = amplitudes[rows[start : start + _RMS_TRACE_CHUNK_SIZE], slice(*region.time_range)]
        if not np.all(np.isfinite(chunk)):
            raise ValueError("fit labels contain non-finite amplitudes")
        with np.errstate(over="ignore", invalid="ignore"):
            energy += float(np.sum(np.square(chunk, dtype=np.float64), dtype=np.float64))
    rms = float(np.sqrt(energy / (len(rows) * region.shape[0])))
    if not np.isfinite(rms) or rms <= 0:
        raise ValueError("fit amplitude RMS must be positive and finite")
    return rms


def _mapping_sha256(rows: np.ndarray) -> str:
    """Hash four little-endian int64 dimensions, then C-order little-endian row IDs."""
    digest = hashlib.sha256(np.asarray(rows.shape, dtype="<i8").tobytes())
    digest.update(np.asarray(rows, dtype="<i8").tobytes(order="C"))
    return digest.hexdigest()


def _five_integers(value: object, *, name: str, minimum: int) -> tuple[int, int, int, int, int]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, (Sequence, np.ndarray))
        or len(value) != 5
    ):
        raise ValueError(f"{name} must contain five integers >= {minimum}")
    if any(
        isinstance(item, bool) or not isinstance(item, Integral) or item < minimum for item in value
    ):
        raise ValueError(f"{name} must contain five integers >= {minimum}")
    return tuple(int(item) for item in value)


def _positive_finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite number")
    converted = float(value)
    if not np.isfinite(converted) or converted <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return converted
