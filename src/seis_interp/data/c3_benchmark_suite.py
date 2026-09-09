"""Freeze and independently verify a C3 suite of existing data artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from seis_interp.data.benchmark_case_inputs import (
    validate_benchmark_preparation,
    validated_benchmark_mask_summary,
)
from seis_interp.data.benchmark_case_store import load_benchmark_case
from seis_interp.data.c3_benchmark_artifacts import (
    benchmark_file_record,
    load_c3_geometry_inputs,
    read_benchmark_json,
    write_benchmark_json,
)
from seis_interp.data.c3_volume_index_inputs import load_bound_benchmark_case
from seis_interp.data.c3_volume_index_store import load_c3_volume_index
from seis_interp.data.file_checksums import file_sha256
from seis_interp.data.interpolation_mask_store import load_interpolation_mask
from seis_interp.data.prepared_partition import OUTPUT_FILE_NAMES as PARTITION_FILES
from seis_interp.data.trace_store import OUTPUT_FILE_NAMES as INTERIM_FILES
from seis_interp.evaluation.c3_volume_metrics import validate_c3_volume_evaluation_config
from seis_interp.processing.c3_benchmark_contract import MAIN_C3_DIMENSIONS, C3BenchmarkDimensions
from seis_interp.processing.c3_benchmark_masks import (
    effective_c3_crop_mask,
    validate_partition_mask_ffids,
    validated_benchmark_cases,
)
from seis_interp.processing.c3_benchmark_partition import (
    audit_c3_benchmark_partition,
    c3_benchmark_partition_ranges,
    validated_c3_benchmark_amplitude_filter,
)
from seis_interp.processing.c3_crop_signal_qc import summarize_c3_crop_signal
from seis_interp.processing.c3_geometry_qc import summarize_c3_geometry
from seis_interp.processing.c3_volume_index import (
    VOLUME_AXIS_ORDER,
    build_c3_volume_index,
    validated_index_range,
)
from seis_interp.processing.interpolation_masks import (
    make_random_trace_mask,
    make_random_whole_ffid_mask,
)
from seis_interp.processing.normalization import read_normalization_parameters
from seis_interp.processing.trace_canonicalization import canonicalize_eligible_physical_coordinates

SUITE_FILE_NAME = "benchmark_suite.json"
INFERENCE_CONTRACT = {
    "amplitude_domain": "same_crop_observed_traces",
    "outside_crop_context": False,
    "target_coordinates": "allowed",
    "target_amplitudes": "scoring_only",
}


@dataclass(frozen=True)
class VerifiedC3BenchmarkSuite:
    """Fully verify once and pin the manifest for explicit reuse within one run.

    Input readers still check their required files and existing case bindings.
    This object is not a persistent cache or permission to use changed inputs.
    """

    directory: Path
    dimensions: C3BenchmarkDimensions = MAIN_C3_DIMENSIONS
    _manifest: dict = field(init=False, repr=False, compare=False)
    _manifest_sha256: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        directory = Path(self.directory).resolve()
        object.__setattr__(self, "directory", directory)
        object.__setattr__(
            self, "_manifest", verify_c3_benchmark_suite(directory, dimensions=self.dimensions)
        )
        object.__setattr__(self, "_manifest_sha256", file_sha256(directory / SUITE_FILE_NAME))


def suite_path(directory: Path, reference: object) -> Path:
    """Resolve one portable reference against the manifest's declared directory."""
    if (
        not isinstance(reference, str)
        or not reference
        or Path(reference).is_absolute()
        or "\\" in reference
    ):
        raise ValueError("suite path references must be nonempty relative POSIX paths")
    return (Path(directory) / reference).resolve()


def benchmark_suite_files(directory: Path, suite: Mapping) -> list[Path]:
    """Enumerate required artifacts explicitly, including both mask and volume files."""
    paths = _input_suite_files(directory, suite, None)
    paths.extend(
        suite_path(directory, suite["geometry"]) / name
        for name in ("geometry.json", "sail_line_mapping.parquet")
    )
    for reference in suite["crops"].values():
        crop = suite_path(directory, reference)
        paths.extend(crop / name for name in ("crop.json", "crop_index.parquet"))
        if (crop / "crop_qc.png").is_file():
            paths.append(crop / "crop_qc.png")
    for entry in suite["cases"]:
        paths.extend(_case_suite_files(directory, entry))
    for source in suite["source_records"]:
        paths.append(suite_path(directory, source["file"]))
    return sorted(set(path.resolve() for path in paths))


def write_c3_benchmark_suite(
    directory: Path, suite: dict, *, dimensions: C3BenchmarkDimensions = MAIN_C3_DIMENSIONS
) -> dict:
    """Save only after every required file, binding, crop, role, and metric validates."""
    destination = directory / SUITE_FILE_NAME
    if destination.exists():
        raise FileExistsError(f"suite manifest already exists: {destination}")
    candidate = {
        **suite,
        "files": [
            benchmark_file_record(path, directory)
            for path in benchmark_suite_files(directory, suite)
        ],
    }
    verify_c3_benchmark_suite(directory, candidate=candidate, dimensions=dimensions)
    write_benchmark_json(destination, candidate)
    return candidate


def verify_c3_benchmark_suite(
    directory: Path,
    *,
    candidate: dict | None = None,
    dimensions: C3BenchmarkDimensions = MAIN_C3_DIMENSIONS,
) -> dict:
    """Recompute hashes and semantic relationships read-only, without a repair mode."""
    directory = Path(directory).resolve()
    suite = read_benchmark_json(directory / SUITE_FILE_NAME) if candidate is None else candidate
    _validate_suite_header(suite)
    hashes = _verified_suite_hashes(directory, suite)
    config = _load_suite_configuration(directory, suite, dimensions)
    for source in suite["source_records"]:
        if hashes[suite_path(directory, source["file"])]["sha256"] != source["sha256"]:
            raise ValueError("configuration source snapshot hash mismatch")
    interim = suite_path(directory, suite["interim"])
    processed = suite_path(directory, suite["processed"])
    input_hashes = {
        "interim": {name: hashes[(interim / name).resolve()] for name in INTERIM_FILES},
        "processed": {name: hashes[(processed / name).resolve()] for name in PARTITION_FILES},
    }
    preparation = read_benchmark_json(processed / "preparation.json")
    amplitude_filter = validated_c3_benchmark_amplitude_filter(config["sampling"], preparation)
    dataset = read_benchmark_json(interim / "dataset.json")
    validate_benchmark_preparation(dataset, preparation, interim_hashes=input_hashes["interim"])
    if dataset["dataset_id"] != config["data"]["dataset_id"]:
        raise ValueError("configured dataset differs from suite input")
    read_normalization_parameters(processed / "normalization.json")
    table, times = load_c3_geometry_inputs(interim)
    geometry_dir = suite_path(directory, suite["geometry"])
    geometry = read_benchmark_json(geometry_dir / "geometry.json")
    mapping, measured = summarize_c3_geometry(
        table,
        times,
        requested_inclusive=dimensions.sail_line_numbers,
        time_range=dimensions.time_range,
        original_number_column=geometry["sail_lines"]["original_number_column"],
    )
    if any(geometry[key] != value for key, value in measured.items()):
        raise ValueError("geometry QC summary differs from measured geometry")
    pd.testing.assert_frame_equal(
        pd.read_parquet(geometry_dir / "sail_line_mapping.parquet"), mapping
    )
    if config["c3_benchmark"]["sail_lines"] != measured["sail_lines"]:
        raise ValueError("configured sail-line mapping differs from the current trace table")
    ranges = c3_benchmark_partition_ranges(
        tuple(measured["sail_lines"]["index_range"]), measured["sail_lines"]["source_line_count"]
    )
    if preparation["source_line_ranges"] != {key: list(value) for key, value in ranges.items()}:
        raise ValueError("prepared ranges do not preserve the fixed test and adjacent partitions")
    crops = {
        partition: pd.read_parquet(suite_path(directory, path) / "crop_index.parquet")
        for partition, path in suite["crops"].items()
    }
    if set(crops) != {"test", "validation"}:
        raise ValueError("suite requires test and validation crops")
    split = pd.read_parquet(processed / "trace_split.parquet")
    pool, summary = audit_c3_benchmark_partition(
        table,
        split,
        preparation,
        crops,
        time_range=dimensions.time_range,
        sampling=config["sampling"],
        amplitudes=(
            np.load(interim / "amplitudes.npy", mmap_mode="r", allow_pickle=False)
            if amplitude_filter is not None
            else None
        ),
    )
    if (
        summary != read_benchmark_json(directory / "partition_summary.json")
        or summary != suite["partition_summary"]
    ):
        raise ValueError("partition summary differs from actual assignments")
    if not np.array_equal(pool, np.load(directory / "train_pool.npy", allow_pickle=False)):
        raise ValueError("authorized train rows do not match canonical partition rows")
    if suite["train_pool"] != {
        "file": "train_pool.npy",
        "trace_count": len(pool),
        "time_samples": list(dimensions.time_range),
    }:
        raise ValueError("authorized training pool/time contract mismatch")
    joined = table.merge(
        split[["array_row", "split"]], on="array_row", validate="one_to_one", sort=False
    )
    canonical, duplicate_audit = canonicalize_eligible_physical_coordinates(joined)
    crop_metadata = _verify_crop_artifacts(
        directory, suite, dimensions, config, interim, table, times, canonical, crops
    )
    for entry in suite["cases"]:
        _verify_case(
            directory,
            entry,
            input_hashes,
            hashes,
            canonical,
            duplicate_audit,
            dataset,
            crops,
            crop_metadata,
        )
    return suite


def load_c3_benchmark_input_manifest(
    directory: Path,
    *,
    case_id: str | None = None,
    dimensions: C3BenchmarkDimensions = MAIN_C3_DIMENSIONS,
    verified_suite: VerifiedC3BenchmarkSuite | None = None,
) -> dict:
    """Check common inputs and an optional case without rerunning suite-wide QC.

    Preparation and standalone verification establish geometry, QC and mask RNG
    conformance. Here hashes bind only the files needed by the requested reader;
    its existing lower-level validation remains responsible for actual inputs.
    """
    directory = Path(directory).resolve()
    if verified_suite is None:
        suite = read_benchmark_json(directory / SUITE_FILE_NAME)
    else:
        if verified_suite.directory != directory or verified_suite.dimensions != dimensions:
            raise ValueError("verified suite directory/dimensions do not match requested inputs")
        if file_sha256(directory / SUITE_FILE_NAME) != verified_suite._manifest_sha256:
            raise ValueError("suite manifest changed after complete verification")
        suite = deepcopy(verified_suite._manifest)
    _validate_suite_header(suite)
    entry = None if case_id is None else c3_suite_case(suite, case_id)
    _verified_suite_hashes(
        directory, suite, required_paths=_input_suite_files(directory, suite, entry)
    )
    config = _load_suite_configuration(directory, suite, dimensions)
    _validate_input_partition(directory, suite, config, dimensions)
    if entry is not None:
        _validate_input_case(directory, suite, entry, config, dimensions)
    return suite


def _validate_suite_header(suite: dict) -> None:
    if suite.get("path_base") != "manifest_directory" or suite.get("status") != "locked":
        raise ValueError("suite must declare a locked manifest_directory contract")
    if suite.get("inference") != INFERENCE_CONTRACT:
        raise ValueError("suite inference must restrict amplitudes to same-crop observed traces")
    validate_c3_volume_evaluation_config({"evaluation": suite.get("evaluation")})


def _load_suite_configuration(directory: Path, suite: dict, dimensions: C3BenchmarkDimensions):
    config = yaml.safe_load((directory / "config.resolved.yaml").read_text())
    inputs = yaml.safe_load((directory / "inputs.resolved.yaml").read_text())
    dimensions.validate(config)
    if suite["contract"] != config["c3_benchmark"]:
        raise ValueError("suite contract does not match its resolved configuration")
    if (
        config["c3_benchmark"]["inference"] != suite["inference"]
        or config["evaluation"] != suite["evaluation"]
    ):
        raise ValueError("resolved inference/evaluation contract differs from suite")
    recipes = validated_benchmark_cases(inputs["cases"])
    if [{key: entry[key] for key in recipes[0]} for entry in suite["cases"]] != recipes:
        raise ValueError("suite does not contain the complete configured case list in order")
    return config


def _input_suite_files(directory: Path, suite: dict, entry: dict | None) -> list[Path]:
    paths = [
        directory / name
        for name in (
            "config.resolved.yaml",
            "inputs.resolved.yaml",
            "partition_summary.json",
            "train_pool.npy",
        )
    ]
    for group, names in (("interim", INTERIM_FILES), ("processed", PARTITION_FILES)):
        paths.extend(suite_path(directory, suite[group]) / name for name in names)
    if entry is not None:
        paths.extend(_case_suite_files(directory, entry))
        crop_dir = suite_path(directory, suite["crops"][entry["partition"]])
        paths.extend(crop_dir / name for name in ("crop.json", "crop_index.parquet"))
    return [path.resolve() for path in paths]


def _case_suite_files(directory: Path, entry: dict) -> list[Path]:
    paths = [suite_path(directory, entry["config_file"])]
    for key, names in (
        ("mask_dir", ("observation_mask.parquet", "interpolation_mask.json")),
        ("case_dir", ("benchmark_case.json",)),
        ("volume_dir", ("volume.json", "volume_index.parquet")),
    ):
        paths.extend(suite_path(directory, entry[key]) / name for name in names)
    return paths


def _validate_input_partition(directory, suite, config, dimensions):
    summary = read_benchmark_json(directory / "partition_summary.json")
    if summary != suite["partition_summary"]:
        raise ValueError("partition summary differs from the suite")
    lines = config["c3_benchmark"]["sail_lines"]
    ranges = c3_benchmark_partition_ranges(tuple(lines["index_range"]), lines["source_line_count"])
    expected = {key: list(value) for key, value in ranges.items()}
    preparation = read_benchmark_json(
        suite_path(directory, suite["processed"]) / "preparation.json"
    )
    amplitude_filter = validated_c3_benchmark_amplitude_filter(config["sampling"], preparation)
    if amplitude_filter is not None:
        quality = summary.get("amplitude_qc", {})
        if quality.get("policy") != amplitude_filter.to_dict() or any(
            quality.get(key) != value for key, value in preparation.get("trace_quality", {}).items()
        ):
            raise ValueError("input amplitude QC summary differs from the prepared filter")
    elif "amplitude_qc" in summary:
        raise ValueError("unfiltered benchmark must not declare amplitude QC exclusions")
    if any(
        value != expected
        for value in (
            summary["source_line_ranges"],
            config["sampling"]["source_line_ranges"],
            preparation["source_line_ranges"],
        )
    ):
        raise ValueError("input partition ranges differ from the fixed suite contract")
    dataset = read_benchmark_json(suite_path(directory, suite["interim"]) / "dataset.json")
    if dataset["dataset_id"] != config["data"]["dataset_id"]:
        raise ValueError("configured dataset differs from suite input")
    if suite["train_pool"] != {
        "file": "train_pool.npy",
        "trace_count": summary["canonical_trace_counts"]["train"],
        "time_samples": list(dimensions.time_range),
    } or summary["training_time_samples"] != list(dimensions.time_range):
        raise ValueError("authorized training pool/time contract mismatch")


def _validate_input_case(directory, suite, entry, config, dimensions):
    partition = entry["partition"]
    crop_dir = suite_path(directory, suite["crops"][partition])
    crop = read_benchmark_json(crop_dir / "crop.json")
    selection = _validated_crop_selection(crop, partition, config, dimensions)
    index, volume = load_c3_volume_index(suite_path(directory, entry["volume_dir"]))
    if volume["selection"] != selection or not index.equals(
        pd.read_parquet(crop_dir / "crop_index.parquet")
    ):
        raise ValueError("case volume differs semantically from the fixed premask crop")
    case = load_bound_benchmark_case(volume, case_dir=suite_path(directory, entry["case_dir"]))
    if (
        case["case_id"] != entry["case_id"]
        or case["partition"] != partition
        or any(
            case["mask"][key] != entry[key] for key in ("kind", "missing_fraction", "random_seed")
        )
    ):
        raise ValueError("case does not match its configured recipe")
    _validate_case_configuration(directory, entry, case, volume)


def _verify_case(
    directory,
    entry,
    input_hashes,
    hashes,
    canonical,
    duplicate_audit,
    dataset,
    crops,
    crop_metadata,
):
    mask_dir = suite_path(directory, entry["mask_dir"])
    case_dir = suite_path(directory, entry["case_dir"])
    volume_dir = suite_path(directory, entry["volume_dir"])
    mask, metadata = load_interpolation_mask(mask_dir)
    case = load_benchmark_case(case_dir)
    index, volume = load_c3_volume_index(volume_dir)
    load_bound_benchmark_case(volume, case_dir=case_dir)
    current = {
        **input_hashes,
        "mask": {
            name: hashes[(mask_dir / name).resolve()]
            for name in ("observation_mask.parquet", "interpolation_mask.json")
        },
    }
    if case["input_files"] != current or case["case_id"] != entry["case_id"]:
        raise ValueError("case input binding or ID mismatch")
    if any(
        metadata[key] != entry[key]
        for key in ("partition", "kind", "missing_fraction", "random_seed")
    ):
        raise ValueError("mask does not match its configured recipe")
    summary = validated_benchmark_mask_summary(
        metadata,
        mask_row_count=len(mask),
        trace_count=dataset["trace_count"],
        mask_array_rows=mask["array_row"].to_numpy(),
        dataset_id=dataset["dataset_id"],
        input_hashes=current,
    )
    if case["mask"] != summary or case["partition"] != entry["partition"]:
        raise ValueError("case mask summary mismatch")
    partition = entry["partition"]
    candidate = canonical.loc[canonical["split"].eq(partition)]
    validate_partition_mask_ffids(candidate, mask, kind=entry["kind"])
    maker = (
        make_random_trace_mask if entry["kind"] == "random_trace" else make_random_whole_ffid_mask
    )
    expected_mask = maker(
        candidate, missing_fraction=entry["missing_fraction"], random_seed=entry["random_seed"]
    )
    if not mask.equals(expected_mask):
        raise ValueError("mask roles do not match the configured seed and canonical candidates")
    if metadata["candidate_ffid_count"] != candidate["ffid"].nunique() or metadata[
        "duplicate_physical_coordinates"
    ] != {key: duplicate_audit[key] for key in ("policy", "removed_trace_count")}:
        raise ValueError("mask canonical candidate summary mismatch")
    if volume["selection"] != crop_metadata[partition]["selection"] or not index.equals(
        crops[partition]
    ):
        raise ValueError("case volume differs semantically from the fixed premask crop")
    effective = effective_c3_crop_mask(index, mask, kind=entry["kind"])
    if effective != entry["effective_mask"] or entry["partition_counts"] != metadata["counts"]:
        raise ValueError("effective mask counts differ from actual roles")
    if volume["role_counts"] != {
        "observed": effective["observed_trace_count"],
        "evaluation_target": effective["target_trace_count"],
    }:
        raise ValueError("volume role counts differ from mask")
    _validate_case_configuration(directory, entry, case, volume)


def _validate_case_configuration(directory, entry, case, volume):
    config = yaml.safe_load(suite_path(directory, entry["config_file"]).read_text())
    if (
        config["benchmark_case"]["id"] != case["case_id"]
        or config["benchmark_volume"]["id"] != volume["volume_id"]
        or config["benchmark_volume"]["selection"] != volume["selection"]
        or config["project"]["random_seed"] != entry["random_seed"]
    ):
        raise ValueError("case configuration does not match its verified artifacts")


def c3_suite_case(suite: Mapping, case_id: str) -> dict:
    """Return exactly one configured case; arbitrary uncropped domains are not inferred."""
    matches = [entry for entry in suite["cases"] if entry["case_id"] == case_id]
    if len(matches) != 1:
        raise ValueError(f"suite has no unique case {case_id!r}")
    return matches[0]


def _verify_crop_artifacts(
    directory, suite, dimensions, config, interim, table, times, canonical, crops
):
    """Check premask grid and signal records against the actual selected samples."""
    amplitudes = np.load(interim / "amplitudes.npy", mmap_mode="r", allow_pickle=False)
    crop_metadata = {}
    for partition, index in crops.items():
        crop = read_benchmark_json(suite_path(directory, suite["crops"][partition]) / "crop.json")
        selection = _validated_crop_selection(crop, partition, config, dimensions)
        candidates = canonical.loc[canonical["split"].eq(partition), "array_row"].to_numpy(
            dtype=np.int64
        )
        actual = build_c3_volume_index(
            table,
            candidates,
            **{f"{axis}_range": tuple(selection[axis]) for axis in VOLUME_AXIS_ORDER[1:]},
        )
        if not actual.equals(index):
            raise ValueError("QC trace-to-cell table differs from the physical grid")
        signal = summarize_c3_crop_signal(
            amplitudes, index["array_row"].to_numpy(), times, time_range=dimensions.time_range
        )
        if not signal["ok"] or signal != crop["signal"] or crop["time"] != signal["time"]:
            raise ValueError("crop signal QC is inconsistent or contains nonfinite samples")
        crop_metadata[partition] = crop
    return crop_metadata


def _validated_crop_selection(crop, partition, config, dimensions):
    selection = crop["selection"]
    selected_ranges = [
        validated_index_range(selection[axis], name=f"selection.{axis}")
        for axis in VOLUME_AXIS_ORDER
    ]
    shape = [stop - start for start, stop in selected_ranges]
    if crop["shape"] != shape or selection["time"] != list(dimensions.time_range):
        raise ValueError("crop shape/time contract mismatch")
    if partition == "test" and (
        shape != list(dimensions.shape) or selection != config["benchmark_volume"]["selection"]
    ):
        raise ValueError("main test crop differs from fixed dimensions/selection")
    if partition == "validation" and (
        shape[0] != dimensions.shape[0] or shape[2:] != list(dimensions.shape[2:])
    ):
        raise ValueError("validation crop must retain the fixed time/shot/receiver lengths")
    return selection


def _verified_suite_hashes(
    directory: Path, suite: dict, *, required_paths: list[Path] | None = None
) -> dict:
    """Hash required input files, or the complete inventory for full verification."""
    hashes = {}
    required = None if required_paths is None else set(required_paths)
    for record in suite["files"]:
        path = suite_path(directory, record["path"])
        if required is not None and path not in required:
            continue
        if path in hashes:
            raise ValueError("duplicate suite file reference")
        digest = file_sha256(path)
        if digest != record["sha256"]:
            raise ValueError(f"suite SHA-256 mismatch: {record['path']}")
        hashes[path] = {"sha256": digest}
    expected = set(benchmark_suite_files(directory, suite)) if required is None else required
    if set(hashes) != expected:
        raise ValueError("suite file inventory does not match the required artifacts")
    return hashes
