"""Verified native trace domains and arbitrary-query identity contracts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seis_interp.data.benchmark_case_inputs import collect_benchmark_input_hashes
from seis_interp.data.benchmark_case_store import load_benchmark_case, write_benchmark_case
from seis_interp.data.interpolation_mask_store import (
    load_interpolation_mask,
    write_interpolation_mask,
)
from seis_interp.data.trace_graph_domain import (
    build_trace_graph_domain,
    load_benchmark_trace_graph_domain,
    load_training_trace_graph_domain,
)
from seis_interp.pipelines.prepare_baseline import prepare_baseline_dataset
from seis_interp.pipelines.prepare_benchmark_case import prepare_benchmark_case
from seis_interp.pipelines.prepare_interpolation_mask import prepare_interpolation_mask
from seis_interp.processing.interpolation_masks import OBSERVATION_ROLE_COLUMN, OBSERVED_ROLE
from seis_interp.processing.trace_amplitude_filter import TraceAmplitudeFilterConfig
from tests.fixtures.benchmark_case_artifacts import prepare_benchmark_case_artifacts
from tests.fixtures.c3_volume_run_artifacts import prepare_c3_volume_run_artifacts


def _native_paths(tmp_path: Path, partition: str = "test") -> dict[str, Path]:
    interim, processed, mask = prepare_benchmark_case_artifacts(tmp_path)
    if partition != "test":
        mask = processed / "masks" / f"{partition}-trace"
        prepare_interpolation_mask(
            interim,
            processed,
            mask,
            partition=partition,
            kind="random_trace",
            missing_fraction=0.5,
            random_seed=42,
        )
    case = processed / "cases" / partition
    prepare_benchmark_case(interim, processed, mask, case, case_id=partition)
    return {
        "interim_dir": interim,
        "processed_dir": processed,
        "mask_dir": mask,
        "case_dir": case,
    }


def test_native_case_needs_no_dense_index_and_reads_no_amplitude_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _native_paths(tmp_path)
    mask, _ = load_interpolation_mask(paths["mask_dir"])

    def reject_row_read(*args: object, **kwargs: object) -> None:
        raise AssertionError("domain validation must not read waveform rows")

    monkeypatch.setattr(np.memmap, "__getitem__", reject_row_read)
    domain = load_benchmark_trace_graph_domain(**paths, time_samples=(1, 3))
    expected = mask.sort_values("array_row")
    np.testing.assert_array_equal(domain.trace_ids, expected["array_row"])
    np.testing.assert_array_equal(domain.array_rows, expected["array_row"])
    np.testing.assert_array_equal(
        domain.observed_mask, expected[OBSERVATION_ROLE_COLUMN].eq(OBSERVED_ROLE)
    )
    np.testing.assert_array_equal(domain.query_indices, np.flatnonzero(~domain.observed_mask))
    np.testing.assert_array_equal(domain.time_s, np.array([0.008, 0.016]))
    assert domain.source_xy_m.dtype == np.float64
    assert domain.time_samples == (1, 3)
    assert domain.amplitudes_path == paths["interim_dir"] / "amplitudes.npy"
    assert domain.inputs_lock["partition"] == "test"
    assert "benchmark_volume" not in domain.inputs_lock
    assert not hasattr(domain, "amplitudes")


@pytest.mark.parametrize("mutation", ["hash", "role", "partition"])
def test_case_validation_rejects_unbound_or_inconsistent_inputs(
    tmp_path: Path, mutation: str
) -> None:
    paths = _native_paths(tmp_path)
    case_path = paths["case_dir"] / "benchmark_case.json"
    if mutation == "hash":
        with (paths["interim_dir"] / "amplitudes.npy").open("ab") as file:
            file.write(b"changed")
        message = "input_files do not match"
    else:
        case = json.loads(case_path.read_text())
        if mutation == "role":
            del case["role_contract"]["observed_role"]
            message = "role_contract"
        else:
            case["partition"] = "validation"
            message = "partition must match"
        case_path.write_text(json.dumps(case))
    with pytest.raises(ValueError, match=message):
        load_benchmark_trace_graph_domain(**paths)


def test_incomplete_mask_is_rejected_even_with_updated_hash_bindings(tmp_path: Path) -> None:
    paths = _native_paths(tmp_path)
    mask, metadata = load_interpolation_mask(paths["mask_dir"])
    metadata = {key: value for key, value in metadata.items() if key not in {"counts", "files"}}
    removed = mask.index[mask[OBSERVATION_ROLE_COLUMN].eq(OBSERVED_ROLE)][0]
    smaller = mask.drop(index=removed)
    metadata["candidate_trace_count"] = len(smaller)
    write_interpolation_mask(paths["mask_dir"], smaller, metadata, overwrite=True)
    prepare_benchmark_case(
        paths["interim_dir"],
        paths["processed_dir"],
        paths["mask_dir"],
        paths["case_dir"],
        case_id="test",
        overwrite=True,
    )
    with pytest.raises(ValueError, match="do not match the expected set"):
        load_benchmark_trace_graph_domain(**paths)


def test_changed_prepared_partition_is_checked_beyond_case_hashes(tmp_path: Path) -> None:
    paths = _native_paths(tmp_path)
    split_path = paths["processed_dir"] / "trace_split.parquet"
    split = pd.read_parquet(split_path)
    split.loc[split["split"].eq("test").idxmax(), "split"] = "train"
    split.to_parquet(split_path, index=False)
    case = load_benchmark_case(paths["case_dir"])
    case["input_files"] = collect_benchmark_input_hashes(
        paths["interim_dir"], paths["processed_dir"], paths["mask_dir"]
    )
    write_benchmark_case(paths["case_dir"], case, overwrite=True)
    with pytest.raises(ValueError, match="split_counts"):
        load_benchmark_trace_graph_domain(**paths)


def test_volume_crop_limits_both_context_and_queries_and_preserves_output_mapping(
    tmp_path: Path,
) -> None:
    artifacts = prepare_c3_volume_run_artifacts(tmp_path, time_sample_count=3)
    paths = {
        "interim_dir": artifacts.interim,
        "processed_dir": artifacts.processed,
        "mask_dir": artifacts.mask,
        "case_dir": artifacts.case,
    }
    native = load_benchmark_trace_graph_domain(**paths)
    cropped = load_benchmark_trace_graph_domain(**paths, volume_dir=artifacts.volume)
    index = pd.read_parquet(artifacts.volume / "volume_index.parquet")
    np.testing.assert_array_equal(cropped.trace_ids, np.sort(index["array_row"]))
    assert len(cropped.trace_ids) < len(native.trace_ids)
    assert cropped.observed_mask.sum() < native.observed_mask.sum()
    assert cropped.query_mask.sum() < native.query_mask.sum()
    assert cropped.time_samples == (0, 3)
    assert cropped.inputs_lock["benchmark_volume"]["array_rows"] == index["array_row"].tolist()
    with pytest.raises(ValueError, match="match the optional volume time selection"):
        load_benchmark_trace_graph_domain(**paths, volume_dir=artifacts.volume, time_samples=(1, 3))


def test_volume_case_binding_is_exact(tmp_path: Path) -> None:
    artifacts = prepare_c3_volume_run_artifacts(tmp_path)
    case = load_benchmark_case(artifacts.case)
    case["case_id"] = "another-case"
    write_benchmark_case(artifacts.case, case, overwrite=True)
    with pytest.raises(ValueError, match="SHA-256 does not match the volume binding"):
        load_benchmark_trace_graph_domain(
            interim_dir=artifacts.interim,
            processed_dir=artifacts.processed,
            mask_dir=artifacts.mask,
            case_dir=artifacts.case,
            volume_dir=artifacts.volume,
        )


def test_training_pool_distinguishes_all_train_traces_and_mask_observed(tmp_path: Path) -> None:
    paths = _native_paths(tmp_path, partition="train")
    all_train = load_training_trace_graph_domain(
        interim_dir=paths["interim_dir"],
        processed_dir=paths["processed_dir"],
        pool="all_train_traces",
    )
    observed = load_training_trace_graph_domain(**paths, pool="mask_observed")
    split = pd.read_parquet(paths["processed_dir"] / "trace_split.parquet")
    mask, _ = load_interpolation_mask(paths["mask_dir"])
    np.testing.assert_array_equal(
        all_train.trace_ids, np.sort(split.loc[split["split"].eq("train"), "array_row"])
    )
    np.testing.assert_array_equal(
        observed.trace_ids,
        np.sort(mask.loc[mask[OBSERVATION_ROLE_COLUMN].eq(OBSERVED_ROLE), "array_row"]),
    )
    assert len(observed.trace_ids) < len(all_train.trace_ids)
    for domain, pool in ((all_train, "all_train_traces"), (observed, "mask_observed")):
        assert domain.observed_mask.all()
        assert not domain.query_mask.any()
        assert domain.pool == pool
        assert domain.inputs_lock["partition"] == "train"
        assert domain.inputs_lock["training_data"]["pool"] == pool
    with_case = load_training_trace_graph_domain(**paths, pool="all_train_traces")
    np.testing.assert_array_equal(with_case.trace_ids, all_train.trace_ids)


@pytest.mark.parametrize("partition", ["validation", "test"])
def test_training_rejects_evaluation_partition_cases(tmp_path: Path, partition: str) -> None:
    paths = _native_paths(tmp_path, partition=partition)
    with pytest.raises(ValueError, match="train partition"):
        load_training_trace_graph_domain(**paths, pool="mask_observed")


def test_canonical_training_ids_remove_exact_duplicates_and_excluded_rows(tmp_path: Path) -> None:
    paths = _native_paths(tmp_path, partition="train")
    split = pd.read_parquet(paths["processed_dir"] / "trace_split.parquet")
    train_rows = split.loc[split["split"].eq("train"), "array_row"].to_numpy()
    kept, removed, excluded = train_rows[0], train_rows[-1], train_rows[-2]
    table_path = paths["interim_dir"] / "traces.parquet"
    table = pd.read_parquet(table_path)
    coordinates = ["source_x_m", "source_y_m", "receiver_x_m", "receiver_y_m"]
    table.loc[removed, coordinates] = table.loc[kept, coordinates].to_numpy()
    table.to_parquet(table_path, index=False)
    amplitude_path = paths["interim_dir"] / "amplitudes.npy"
    amplitudes = np.load(amplitude_path)
    amplitudes[excluded] = 0.0
    np.save(amplitude_path, amplitudes)
    prepare_baseline_dataset(
        paths["interim_dir"],
        paths["processed_dir"],
        holdout_fraction=0.5,
        validation_fraction_of_holdout=0.5,
        random_seed=42,
        split_scope="whole_ffid",
        trace_amplitude_filter=TraceAmplitudeFilterConfig(True, 1000.0),
        overwrite=True,
    )
    domain = load_training_trace_graph_domain(
        interim_dir=paths["interim_dir"],
        processed_dir=paths["processed_dir"],
        pool="all_train_traces",
    )
    assert kept in domain.trace_ids
    assert removed not in domain.trace_ids
    assert excluded not in domain.trace_ids
    np.testing.assert_array_equal(domain.trace_ids, np.sort(domain.trace_ids))
    np.testing.assert_array_equal(domain.array_rows, domain.trace_ids)


def _arbitrary_arguments() -> dict[str, np.ndarray]:
    return {
        "trace_ids": np.array([3, 20, -1]),
        "source_xy_m": np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.5]]),
        "receiver_xy_m": np.array([[0.0, 3.0], [1.0, 3.0], [2.0, 3.5]]),
        "ffids": np.array([8, 13, 29]),
        "observed_mask": np.array([True, True, False]),
        "time_s": np.arange(4, dtype=np.float64) * 0.008,
    }


def test_arbitrary_queries_need_no_array_row_and_retain_caller_order() -> None:
    arguments = _arbitrary_arguments()
    domain = build_trace_graph_domain(**arguments)
    assert domain.array_rows is None
    np.testing.assert_array_equal(domain.trace_ids, [3, 20, -1])
    np.testing.assert_array_equal(domain.query_indices, [2])
    arguments["source_xy_m"][2] = 100.0
    np.testing.assert_array_equal(domain.source_xy_m[2], [2.0, 0.5])
    mapped = build_trace_graph_domain(**arguments, array_rows=np.array([3, 20, -1]))
    np.testing.assert_array_equal(mapped.array_rows, [3, 20, -1])


def test_arbitrary_query_rejects_observed_physical_alias_without_coordinate_tolerance() -> None:
    arguments = _arbitrary_arguments()
    arguments["source_xy_m"][2] = arguments["source_xy_m"][0]
    arguments["receiver_xy_m"][2] = arguments["receiver_xy_m"][0]
    with pytest.raises(ValueError, match="duplicate physical"):
        build_trace_graph_domain(**arguments)
    arguments["source_xy_m"][2, 0] += 1.0e-12
    assert len(build_trace_graph_domain(**arguments).trace_ids) == 3


@pytest.mark.parametrize("time_samples", [(0, 5), (-1, 2), (2, 2), (1.0, 3.0)])
def test_native_time_selection_is_validated(tmp_path: Path, time_samples: tuple[int, int]) -> None:
    paths = _native_paths(tmp_path)
    with pytest.raises(ValueError, match="time_samples"):
        load_benchmark_trace_graph_domain(**paths, time_samples=time_samples)
