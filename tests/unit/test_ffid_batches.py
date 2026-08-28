from __future__ import annotations

import gc
import weakref
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seis_interp.processing.trace_splits import EXCLUDED_SPLIT
from seis_interp.training.ffid_batches import (
    FullFfidBatch,
    FullFfidBatchSampler,
    array_rows_by_ffid_for_split,
    build_global_rms_trace_points,
    build_per_trace_rms_trace_points,
    validate_all_ffids_have_split_rows,
)
from seis_interp.training.point_sampler import build_trace_points


def _trace_and_split_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    ffid_by_row = np.asarray([10, 10, 10, 10, 20, 20, 20, 30, 30, 30], dtype=np.int64)
    table_order = np.asarray([7, 1, 9, 0, 5, 3, 8, 2, 6, 4])
    trace_table = pd.DataFrame(
        {
            "array_row": table_order,
            "ffid": ffid_by_row[table_order],
        },
        index=np.arange(100, 110) * 3,
    )
    split_by_row = np.asarray(
        [
            "train",
            "train",
            "validation",
            "test",
            "train",
            "validation",
            "test",
            "train",
            "validation",
            "test",
        ]
    )
    split_order = np.arange(9, -1, -1)
    split_table = pd.DataFrame(
        {
            "array_row": split_order,
            "split": split_by_row[split_order],
        },
        index=np.arange(10) + 500,
    )
    return trace_table, split_table


def _point_arrays(trace_count: int = 9) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time = np.linspace(-1.0, 1.0, 4, dtype=np.float64)
    spatial = np.arange(trace_count * 5, dtype=np.float64).reshape(trace_count, 5) / 10.0
    amplitudes = np.asarray(
        [[100 * row + sample for sample in range(len(time))] for row in range(trace_count)],
        dtype=np.float32,
    )
    return time, spatial, amplitudes


def test_groups_requested_split_by_array_row_not_dataframe_order() -> None:
    trace_table, split_table = _trace_and_split_tables()

    grouped = array_rows_by_ffid_for_split(trace_table, split_table, split="train")

    assert list(grouped) == [10, 20, 30]
    np.testing.assert_array_equal(grouped[10], [0, 1])
    np.testing.assert_array_equal(grouped[20], [4])
    np.testing.assert_array_equal(grouped[30], [7])
    assert all(not rows.flags.writeable for rows in grouped.values())
    validate_all_ffids_have_split_rows(trace_table, grouped, split="train")


def test_grouping_omits_ffid_without_requested_split_and_coverage_rejects_it() -> None:
    trace_table, split_table = _trace_and_split_tables()
    split_table.loc[split_table["array_row"] == 7, "split"] = "validation"

    grouped = array_rows_by_ffid_for_split(trace_table, split_table, split="train")

    assert list(grouped) == [10, 20]
    with pytest.raises(ValueError, match=r"missing_ffids=\[30\]"):
        validate_all_ffids_have_split_rows(trace_table, grouped, split="train")


def test_grouping_accepts_explicit_exclusions_and_checks_only_eligible_ffids() -> None:
    trace_table, split_table = _trace_and_split_tables()
    excluded_rows = np.asarray([7, 8, 9], dtype=np.int64)
    split_table.loc[split_table["array_row"].isin(excluded_rows), "split"] = EXCLUDED_SPLIT

    grouped = array_rows_by_ffid_for_split(trace_table, split_table, split="train")

    assert list(grouped) == [10, 20]
    eligible_rows = split_table.loc[split_table["split"] != EXCLUDED_SPLIT, "array_row"].to_numpy(
        dtype=np.int64
    )
    validate_all_ffids_have_split_rows(
        trace_table,
        grouped,
        split="train",
        eligible_array_rows=eligible_rows,
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda table: table.drop(index=table.index[0]), "missing_rows"),
        (
            lambda table: table.assign(
                array_row=[table["array_row"].iloc[0], *table["array_row"].iloc[:-1]]
            ),
            "duplicate array_row",
        ),
        (
            lambda table: table.assign(array_row=[100, *table["array_row"].iloc[1:]]),
            "out_of_range_rows",
        ),
    ],
)
def test_grouping_rejects_missing_duplicate_and_out_of_range_split_rows(
    mutation,
    message: str,
) -> None:
    trace_table, split_table = _trace_and_split_tables()

    with pytest.raises(ValueError, match=message):
        array_rows_by_ffid_for_split(trace_table, mutation(split_table), split="train")


def test_grouping_rejects_invalid_split_values_and_non_integer_ffids() -> None:
    trace_table, split_table = _trace_and_split_tables()
    invalid_split = split_table.copy()
    invalid_split.loc[invalid_split.index[0], "split"] = "unknown"
    with pytest.raises(ValueError, match="invalid split"):
        array_rows_by_ffid_for_split(trace_table, invalid_split, split="train")

    invalid_ffid = trace_table.copy()
    invalid_ffid["ffid"] = invalid_ffid["ffid"].astype(np.float64)
    with pytest.raises(ValueError, match="ffid must have an integer dtype"):
        array_rows_by_ffid_for_split(invalid_ffid, split_table, split="train")


def test_point_builder_matches_materialized_reference_and_global_rms_only() -> None:
    time, spatial, amplitudes = _point_arrays()
    rows = np.asarray([6, 1, 4])
    original = amplitudes.copy()

    coordinates, targets = build_global_rms_trace_points(
        time,
        spatial,
        amplitudes,
        rows,
        amplitude_rms=2.5,
    )
    expected_coordinates, expected_targets = build_trace_points(
        time,
        spatial,
        amplitudes / 2.5,
        rows,
    )

    np.testing.assert_array_equal(coordinates, expected_coordinates)
    np.testing.assert_array_equal(targets, expected_targets)
    np.testing.assert_array_equal(amplitudes, original)
    assert coordinates.shape == (12, 6)
    assert coordinates.dtype == np.float64
    assert targets.shape == (12,)
    assert targets.dtype == np.float32


def test_per_trace_point_builder_scales_each_selected_trace_to_unit_rms() -> None:
    time, spatial, amplitudes = _point_arrays()
    rows = np.asarray([6, 1, 4])
    original = amplitudes.copy()

    coordinates, targets = build_per_trace_rms_trace_points(
        time,
        spatial,
        amplitudes,
        rows,
    )

    expected_coordinates, _ = build_trace_points(time, spatial, amplitudes, rows)
    target_traces = targets.reshape(len(rows), len(time)).astype(np.float64)
    np.testing.assert_array_equal(coordinates, expected_coordinates)
    np.testing.assert_allclose(
        np.sqrt(np.mean(np.square(target_traces), axis=1)),
        np.ones(len(rows)),
        rtol=1.0e-7,
    )
    np.testing.assert_array_equal(amplitudes, original)
    assert targets.dtype == np.float32


def test_per_trace_point_builder_identifies_a_zero_rms_array_row() -> None:
    time, spatial, amplitudes = _point_arrays()
    amplitudes[4] = 0.0

    with pytest.raises(ValueError, match="array_row 4"):
        build_per_trace_rms_trace_points(
            time,
            spatial,
            amplitudes,
            np.asarray([6, 4]),
        )


def test_point_builder_uses_broadcast_assignment_without_full_point_temporaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    time, spatial, amplitudes = _point_arrays()

    def reject_full_point_temporary(*_args, **_kwargs):
        raise AssertionError("point construction must not call tile or repeat")

    monkeypatch.setattr(np, "tile", reject_full_point_temporary)
    monkeypatch.setattr(np, "repeat", reject_full_point_temporary)

    coordinates, targets = build_global_rms_trace_points(
        time,
        spatial,
        amplitudes,
        np.asarray([6, 1, 4]),
        amplitude_rms=2.5,
    )

    assert coordinates.shape == (12, 6)
    assert targets.shape == (12,)


def test_point_builder_reads_a_read_only_memmap(tmp_path: Path) -> None:
    time, spatial, amplitudes = _point_arrays()
    amplitude_path = tmp_path / "amplitudes.npy"
    np.save(amplitude_path, amplitudes)
    memory_mapped = np.load(amplitude_path, mmap_mode="r", allow_pickle=False)
    assert isinstance(memory_mapped, np.memmap)
    assert not memory_mapped.flags.writeable

    coordinates, targets = build_global_rms_trace_points(
        time,
        spatial,
        memory_mapped,
        np.asarray([8, 2]),
        amplitude_rms=4.0,
    )

    expected_coordinates, expected_targets = build_trace_points(
        time,
        spatial,
        amplitudes / 4.0,
        np.asarray([8, 2]),
    )
    np.testing.assert_array_equal(coordinates, expected_coordinates)
    np.testing.assert_array_equal(targets, expected_targets)

    sampler = FullFfidBatchSampler(
        time,
        spatial,
        memory_mapped,
        {10: np.asarray([8, 2])},
        amplitude_rms=4.0,
        random_seed=1,
    )
    batch = next(sampler.iter_epoch())
    sampler_coordinates, sampler_targets = build_trace_points(
        time,
        spatial,
        amplitudes / 4.0,
        np.asarray([2, 8]),
    )
    np.testing.assert_array_equal(batch.coordinates, sampler_coordinates)
    np.testing.assert_array_equal(batch.targets, sampler_targets)

    per_trace_coordinates, per_trace_targets = build_per_trace_rms_trace_points(
        time,
        spatial,
        memory_mapped,
        np.asarray([8, 2]),
    )
    np.testing.assert_array_equal(per_trace_coordinates, expected_coordinates)
    per_trace_targets = per_trace_targets.reshape(2, len(time)).astype(np.float64)
    np.testing.assert_allclose(
        np.sqrt(np.mean(np.square(per_trace_targets), axis=1)),
        np.ones(2),
        rtol=1.0e-7,
    )


def _sampler(
    *,
    seed: int,
    rows_by_ffid: dict[int, np.ndarray] | None = None,
    amplitude_scaling: str = "train_global_rms",
) -> FullFfidBatchSampler:
    time, spatial, amplitudes = _point_arrays()
    groups = rows_by_ffid or {
        30: np.asarray([8]),
        10: np.asarray([2, 0]),
        20: np.asarray([6, 5, 4]),
    }
    return FullFfidBatchSampler(
        time,
        spatial,
        amplitudes,
        groups,
        amplitude_rms=2.0,
        random_seed=seed,
        amplitude_scaling=amplitude_scaling,
    )


def test_epoch_visits_every_ffid_once_with_variable_point_counts() -> None:
    sampler = _sampler(seed=12)

    batches = list(sampler.iter_epoch())

    assert sampler.ffid_count == 3
    assert sampler.ffids == (10, 20, 30)
    assert sorted(batch.ffid for batch in batches) == [10, 20, 30]
    by_ffid = {batch.ffid: batch for batch in batches}
    assert (by_ffid[10].trace_count, by_ffid[10].point_count) == (2, 8)
    assert (by_ffid[20].trace_count, by_ffid[20].point_count) == (3, 12)
    assert (by_ffid[30].trace_count, by_ffid[30].point_count) == (1, 4)
    assert all(batch.coordinates.shape == (batch.point_count, 6) for batch in batches)
    assert all(batch.targets.shape == (batch.point_count,) for batch in batches)


@pytest.mark.parametrize("amplitude_scaling", ["train_global_rms", "per_trace_rms"])
def test_sampler_does_not_retain_a_yielded_ffid_batch(amplitude_scaling: str) -> None:
    epoch = _sampler(seed=12, amplitude_scaling=amplitude_scaling).iter_epoch()
    batch = next(epoch)
    coordinate_reference = weakref.ref(batch.coordinates)
    target_reference = weakref.ref(batch.targets)

    del batch
    gc.collect()

    assert coordinate_reference() is None
    assert target_reference() is None
    next(epoch)


def test_sampler_batches_contain_every_selected_trace_and_sample_once() -> None:
    time, spatial, amplitudes = _point_arrays()
    groups = {
        10: np.asarray([2, 0]),
        20: np.asarray([6, 5, 4]),
        30: np.asarray([8]),
    }
    sampler = _sampler(seed=3, rows_by_ffid=groups)

    for batch in sampler.iter_epoch():
        rows = np.sort(groups[batch.ffid])
        expected_coordinates, expected_targets = build_trace_points(
            time,
            spatial,
            amplitudes / 2.0,
            rows,
        )
        np.testing.assert_array_equal(batch.coordinates, expected_coordinates)
        np.testing.assert_array_equal(batch.targets, expected_targets)


def test_sampler_supports_per_trace_rms_scaling() -> None:
    time, spatial, amplitudes = _point_arrays()
    groups = {
        10: np.asarray([2, 0]),
        20: np.asarray([6, 5, 4]),
        30: np.asarray([8]),
    }
    sampler = FullFfidBatchSampler(
        time,
        spatial,
        amplitudes,
        groups,
        amplitude_rms=2.0,
        random_seed=3,
        amplitude_scaling="per_trace_rms",
    )

    assert sampler.amplitude_scaling == "per_trace_rms"
    for batch in sampler.iter_epoch():
        traces = batch.targets.reshape(batch.trace_count, len(time)).astype(np.float64)
        np.testing.assert_allclose(
            np.sqrt(np.mean(np.square(traces), axis=1)),
            np.ones(batch.trace_count),
            rtol=1.0e-7,
        )


def test_same_seed_reproduces_multiple_epochs_independent_of_mapping_order() -> None:
    first = _sampler(seed=7)
    second = _sampler(
        seed=7,
        rows_by_ffid={
            10: np.asarray([0, 2]),
            20: np.asarray([4, 6, 5]),
            30: np.asarray([8]),
        },
    )

    first_orders = [[batch.ffid for batch in first.iter_epoch()] for _ in range(5)]
    second_orders = [[batch.ffid for batch in second.iter_epoch()] for _ in range(5)]

    assert first_orders == second_orders
    assert len({tuple(order) for order in first_orders}) > 1


def test_different_seed_changes_at_least_one_epoch_order() -> None:
    first = _sampler(seed=1)
    second = _sampler(seed=2)

    first_orders = [[batch.ffid for batch in first.iter_epoch()] for _ in range(5)]
    second_orders = [[batch.ffid for batch in second.iter_epoch()] for _ in range(5)]

    assert any(left != right for left, right in zip(first_orders, second_orders, strict=True))


def test_grouped_training_sampler_excludes_validation_and_test_rows() -> None:
    trace_table, split_table = _trace_and_split_tables()
    training_groups = array_rows_by_ffid_for_split(trace_table, split_table, split="train")
    time, spatial, amplitudes = _point_arrays(trace_count=10)
    sampler = FullFfidBatchSampler(
        time,
        spatial,
        amplitudes,
        training_groups,
        amplitude_rms=1.0,
        random_seed=4,
    )

    observed_first_samples = {
        int(trace[0])
        for batch in sampler.iter_epoch()
        for trace in batch.targets.reshape(batch.trace_count, len(time))
    }

    assert observed_first_samples == {0, 100, 400, 700}


def test_sampler_rejects_rows_shared_by_ffids_and_invalid_seed() -> None:
    time, spatial, amplitudes = _point_arrays()
    with pytest.raises(ValueError, match="more than one FFID"):
        FullFfidBatchSampler(
            time,
            spatial,
            amplitudes,
            {10: np.asarray([0, 1]), 20: np.asarray([1, 2])},
            amplitude_rms=1.0,
            random_seed=1,
        )
    with pytest.raises(ValueError, match="random_seed"):
        FullFfidBatchSampler(
            time,
            spatial,
            amplitudes,
            {10: np.asarray([0])},
            amplitude_rms=1.0,
            random_seed=-1,
        )


@pytest.mark.parametrize("amplitude_rms", [0.0, -1.0, np.inf, np.nan, True])
def test_point_builder_rejects_invalid_global_rms(amplitude_rms: float) -> None:
    time, spatial, amplitudes = _point_arrays()
    with pytest.raises(ValueError, match="amplitude_rms"):
        build_global_rms_trace_points(
            time,
            spatial,
            amplitudes,
            np.asarray([0]),
            amplitude_rms=amplitude_rms,
        )


def test_point_builder_rejects_duplicate_out_of_range_and_non_finite_selected_rows() -> None:
    time, spatial, amplitudes = _point_arrays()
    with pytest.raises(ValueError, match="unique"):
        build_global_rms_trace_points(
            time, spatial, amplitudes, np.asarray([1, 1]), amplitude_rms=1.0
        )
    with pytest.raises(ValueError, match=r"within \[0, 9\)"):
        build_global_rms_trace_points(time, spatial, amplitudes, np.asarray([9]), amplitude_rms=1.0)

    amplitudes[4, 2] = np.nan
    with pytest.raises(ValueError, match="selected amplitudes"):
        build_global_rms_trace_points(time, spatial, amplitudes, np.asarray([4]), amplitude_rms=1.0)

    spatial[3, 0] = np.inf
    with pytest.raises(ValueError, match="selected normalized spatial"):
        build_global_rms_trace_points(time, spatial, amplitudes, np.asarray([3]), amplitude_rms=1.0)


def test_batch_record_is_frozen() -> None:
    batch = next(_sampler(seed=1).iter_epoch())

    with pytest.raises(FrozenInstanceError):
        batch.ffid = 99  # type: ignore[misc]
    assert isinstance(batch, FullFfidBatch)
