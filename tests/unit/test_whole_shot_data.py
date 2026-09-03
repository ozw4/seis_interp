from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from seis_interp.data.whole_shot import (
    NEIGHBORHOOD_TYPE,
    SOURCE_DISTANCE,
    TARGET_COORDINATE_SCALING,
    TARGET_COORDINATES,
    WholeShotTargets,
    WholeShotTensorSource,
    build_gather_tensors,
    nearest_train_source_indices,
)
from seis_interp.processing.c3_receiver_grid import RECEIVER_X_COUNT, RECEIVER_Y_COUNT

DEVICE = torch.device("cpu")
TIME_SAMPLES = 3
X_OFFSETS = np.arange(RECEIVER_X_COUNT, dtype=np.float64) * 12.5
Y_OFFSETS = np.arange(RECEIVER_Y_COUNT, dtype=np.float64) * 6.25


def _table_and_amplitudes(
    entries: list[tuple[int, tuple[float, float], int, int, float]],
) -> tuple[pd.DataFrame, np.ndarray]:
    rows = []
    amplitudes = []
    for ffid, (source_x, source_y), x_index, y_index, amplitude in entries:
        rows.append(
            {
                "ffid": ffid,
                "source_x_m": source_x,
                "source_y_m": source_y,
                "receiver_x_m": source_x + X_OFFSETS[x_index],
                "receiver_y_m": source_y + Y_OFFSETS[y_index],
            }
        )
        amplitudes.append(np.full(TIME_SAMPLES, amplitude, dtype=np.float32))
    return pd.DataFrame(rows), np.asarray(amplitudes, dtype=np.float32)


def _tensor_source(
    train_ffids: list[int],
    train_sources: list[tuple[float, float]],
    *,
    source_gather_count: int = 2,
    train_availability: torch.Tensor | None = None,
) -> WholeShotTensorSource:
    count = len(train_ffids)
    gathers = torch.zeros(count, RECEIVER_X_COUNT, RECEIVER_Y_COUNT, TIME_SAMPLES)
    for index in range(count):
        gathers[index] = float(index + 1)
    availability = (
        torch.ones(count, RECEIVER_X_COUNT, RECEIVER_Y_COUNT, dtype=torch.bool)
        if train_availability is None
        else train_availability
    )
    return WholeShotTensorSource(
        train_ffids=np.asarray(train_ffids, dtype=np.int64),
        train_source_coordinates_m=np.asarray(train_sources, dtype=np.float64),
        train_gathers=gathers,
        train_availability=availability,
        source_gather_count=source_gather_count,
        device=DEVICE,
    )


def test_contract_constants_are_fixed() -> None:
    assert NEIGHBORHOOD_TYPE == "nearest_train_source_gathers"
    assert SOURCE_DISTANCE == "euclidean_source_xy_m"
    assert TARGET_COORDINATES == ("source_x_m", "source_y_m")
    assert TARGET_COORDINATE_SCALING == "train_minmax"


def test_build_gather_tensors_orders_ffids_and_receiver_cells() -> None:
    table, amplitudes = _table_and_amplitudes(
        [
            (7, (300.0, 400.0), 1, 2, 5.0),
            (3, (100.0, 200.0), 0, 0, 2.0),
            (7, (300.0, 400.0), 4, 60, 9.0),
        ]
    )
    ffids, sources, gathers, availability = build_gather_tensors(
        table,
        amplitudes,
        receiver_x_offsets=X_OFFSETS,
        receiver_y_offsets=Y_OFFSETS,
        device=DEVICE,
    )

    np.testing.assert_array_equal(ffids, [3, 7])
    assert ffids.dtype == np.int64
    assert sources.dtype == np.float64
    assert gathers.dtype == torch.float32
    assert availability.dtype == torch.bool
    assert gathers.shape == (2, RECEIVER_X_COUNT, RECEIVER_Y_COUNT, TIME_SAMPLES)
    assert availability.shape == (2, RECEIVER_X_COUNT, RECEIVER_Y_COUNT)
    assert torch.all(gathers[0, 0, 0] == 2.0)
    assert torch.all(gathers[1, 1, 2] == 5.0)
    assert torch.all(gathers[1, 4, 60] == 9.0)


def test_build_gather_tensors_zero_fills_missing_cells() -> None:
    table, amplitudes = _table_and_amplitudes([(3, (100.0, 200.0), 0, 0, 2.0)])
    _ffids, _sources, gathers, availability = build_gather_tensors(
        table,
        amplitudes,
        receiver_x_offsets=X_OFFSETS,
        receiver_y_offsets=Y_OFFSETS,
        device=DEVICE,
    )

    assert bool(availability[0, 0, 0])
    assert not bool(availability[0, 5, 30])
    assert torch.all(gathers[0, 5, 30] == 0.0)
    assert int(torch.count_nonzero(availability)) == 1


def test_build_gather_tensors_maps_source_coordinates_to_ffids() -> None:
    table, amplitudes = _table_and_amplitudes(
        [
            (7, (300.0, 400.0), 1, 2, 5.0),
            (3, (100.0, 200.0), 0, 0, 2.0),
        ]
    )
    _ffids, sources, _gathers, _availability = build_gather_tensors(
        table,
        amplitudes,
        receiver_x_offsets=X_OFFSETS,
        receiver_y_offsets=Y_OFFSETS,
        device=DEVICE,
    )

    np.testing.assert_array_equal(sources[0], [100.0, 200.0])
    np.testing.assert_array_equal(sources[1], [300.0, 400.0])


def test_build_gather_tensors_rejects_row_count_mismatch() -> None:
    table, amplitudes = _table_and_amplitudes([(3, (100.0, 200.0), 0, 0, 2.0)])

    with pytest.raises(ValueError, match="equal length"):
        build_gather_tensors(
            table,
            np.concatenate([amplitudes, amplitudes]),
            receiver_x_offsets=X_OFFSETS,
            receiver_y_offsets=Y_OFFSETS,
            device=DEVICE,
        )


def test_build_gather_tensors_rejects_conflicting_source_coordinates() -> None:
    table, amplitudes = _table_and_amplitudes(
        [
            (3, (100.0, 200.0), 0, 0, 2.0),
            (3, (101.0, 200.0), 1, 0, 4.0),
        ]
    )

    with pytest.raises(ValueError, match="multiple source coordinates"):
        build_gather_tensors(
            table,
            amplitudes,
            receiver_x_offsets=X_OFFSETS,
            receiver_y_offsets=Y_OFFSETS,
            device=DEVICE,
        )


def test_build_gather_tensors_rejects_trace_outside_grid() -> None:
    table, amplitudes = _table_and_amplitudes([(3, (100.0, 200.0), 0, 0, 2.0)])
    table.loc[0, "receiver_x_m"] += 0.5

    with pytest.raises(ValueError, match="outside the validated receiver grid"):
        build_gather_tensors(
            table,
            amplitudes,
            receiver_x_offsets=X_OFFSETS,
            receiver_y_offsets=Y_OFFSETS,
            device=DEVICE,
        )


def test_build_gather_tensors_rejects_duplicate_receiver_cell() -> None:
    table, amplitudes = _table_and_amplitudes(
        [
            (3, (100.0, 200.0), 0, 0, 2.0),
            (3, (100.0, 200.0), 0, 0, 4.0),
        ]
    )

    with pytest.raises(ValueError, match="duplicate receiver cell"):
        build_gather_tensors(
            table,
            amplitudes,
            receiver_x_offsets=X_OFFSETS,
            receiver_y_offsets=Y_OFFSETS,
            device=DEVICE,
        )


def test_nearest_sources_exclude_target_ffid_and_zero_distance() -> None:
    train_ffids = np.asarray([1, 2, 3, 4], dtype=np.int64)
    train_sources = np.asarray([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0], [15.0, 0.0]])

    zero_distance = nearest_train_source_indices(
        train_ffids,
        train_sources,
        np.asarray([99], dtype=np.int64),
        np.asarray([[5.0, 0.0]]),
        source_gather_count=2,
    )
    same_ffid = nearest_train_source_indices(
        train_ffids,
        train_sources,
        np.asarray([3], dtype=np.int64),
        np.asarray([[10.1, 0.0]]),
        source_gather_count=2,
    )

    np.testing.assert_array_equal(zero_distance[0], [0, 2])
    np.testing.assert_array_equal(same_ffid[0], [3, 1])


def test_nearest_sources_break_distance_ties_by_ffid() -> None:
    train_ffids = np.asarray([9, 4], dtype=np.int64)
    train_sources = np.asarray([[-5.0, 0.0], [5.0, 0.0]])

    result = nearest_train_source_indices(
        train_ffids,
        train_sources,
        np.asarray([99], dtype=np.int64),
        np.asarray([[0.0, 0.0]]),
        source_gather_count=2,
    )

    np.testing.assert_array_equal(result[0], [1, 0])


def test_nearest_sources_reject_insufficient_candidates() -> None:
    train_ffids = np.asarray([1, 2, 3], dtype=np.int64)
    train_sources = np.asarray([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])

    with pytest.raises(ValueError, match="FFID 2 has only 2 non-colliding TRAIN sources; 3"):
        nearest_train_source_indices(
            train_ffids,
            train_sources,
            np.asarray([2], dtype=np.int64),
            np.asarray([[5.0, 0.0]]),
            source_gather_count=3,
        )


def test_whole_shot_targets_counts() -> None:
    availability = torch.zeros(2, RECEIVER_X_COUNT, RECEIVER_Y_COUNT, dtype=torch.bool)
    availability[0, 0, 0] = True
    availability[1, 3, 7] = True
    availability[1, 3, 8] = True
    targets = WholeShotTargets(
        ffids=np.asarray([3, 7], dtype=np.int64),
        source_coordinates_m=np.zeros((2, 2)),
        gathers=torch.zeros(2, RECEIVER_X_COUNT, RECEIVER_Y_COUNT, TIME_SAMPLES),
        availability=availability,
        neighbor_train_indices=np.zeros((2, 2), dtype=np.int64),
    )

    assert targets.ffid_count == 2
    assert targets.trace_count == 3


def test_source_inputs_preserve_neighbor_order_deltas_and_scaling() -> None:
    source = _tensor_source(
        [10, 11, 12, 13],
        [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0)],
    )
    targets = source.build_targets(
        ffids=np.asarray([20], dtype=np.int64),
        source_coordinates_m=np.asarray([[1.0, 0.0]]),
        gathers=torch.zeros(1, RECEIVER_X_COUNT, RECEIVER_Y_COUNT, TIME_SAMPLES),
        availability=torch.ones(1, RECEIVER_X_COUNT, RECEIVER_Y_COUNT, dtype=torch.bool),
    )

    np.testing.assert_array_equal(targets.neighbor_train_indices[0], [0, 1])
    neighbors, availability, source_deltas, target_coordinates = source.inputs(
        targets,
        np.asarray([0], dtype=np.int64),
    )

    assert torch.all(neighbors[0, 0] == 1.0)
    assert torch.all(neighbors[0, 1] == 2.0)
    assert bool(availability.all())
    torch.testing.assert_close(
        source_deltas[0],
        torch.tensor([[-1.0, 0.0], [9.0, 0.0]]),
    )
    torch.testing.assert_close(
        target_coordinates[0],
        torch.tensor([1.0 / 30.0, 0.0]),
    )


def test_constant_target_coordinate_axis_stays_finite_zero() -> None:
    source = _tensor_source(
        [10, 11, 12, 13],
        [(0.0, 5.0), (10.0, 5.0), (20.0, 5.0), (30.0, 5.0)],
    )
    targets = source.build_targets(
        ffids=np.asarray([20], dtype=np.int64),
        source_coordinates_m=np.asarray([[15.0, 5.0]]),
        gathers=torch.zeros(1, RECEIVER_X_COUNT, RECEIVER_Y_COUNT, TIME_SAMPLES),
        availability=torch.ones(1, RECEIVER_X_COUNT, RECEIVER_Y_COUNT, dtype=torch.bool),
    )

    _neighbors, _availability, _deltas, target_coordinates = source.inputs(
        targets,
        np.asarray([0], dtype=np.int64),
    )

    assert torch.isfinite(target_coordinates).all()
    assert float(target_coordinates[0, 1]) == 0.0


def test_neighbor_dropout_masks_whole_source_gathers() -> None:
    source = _tensor_source(
        [10, 11, 12, 13],
        [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0)],
        source_gather_count=3,
    )
    targets = source.build_targets(
        ffids=np.asarray([20, 21], dtype=np.int64),
        source_coordinates_m=np.asarray([[1.0, 0.0], [29.0, 0.0]]),
        gathers=torch.zeros(2, RECEIVER_X_COUNT, RECEIVER_Y_COUNT, TIME_SAMPLES),
        availability=torch.ones(2, RECEIVER_X_COUNT, RECEIVER_Y_COUNT, dtype=torch.bool),
    )

    generator = torch.Generator(device=DEVICE).manual_seed(11)
    neighbors, availability, _deltas, _coordinates = source.inputs(
        targets,
        np.asarray([0, 1], dtype=np.int64),
        generator=generator,
        neighbor_dropout=0.5,
    )
    expected_keep = (
        torch.rand(
            (2, 3, 1, 1),
            generator=torch.Generator(device=DEVICE).manual_seed(11),
            device=DEVICE,
        )
        >= 0.5
    )

    assert not bool(expected_keep.all()), "seed must exercise at least one dropped gather"
    for batch_index in range(2):
        for neighbor_index in range(3):
            kept = bool(expected_keep[batch_index, neighbor_index, 0, 0])
            cell_mask = availability[batch_index, neighbor_index]
            assert bool(cell_mask.all()) == kept
            assert bool(cell_mask.any()) == kept
            if not kept:
                assert torch.all(neighbors[batch_index, neighbor_index] == 0.0)


def test_neighbor_dropout_requires_generator() -> None:
    source = _tensor_source(
        [10, 11, 12, 13],
        [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0)],
    )
    targets = source.build_targets(
        ffids=np.asarray([20], dtype=np.int64),
        source_coordinates_m=np.asarray([[1.0, 0.0]]),
        gathers=torch.zeros(1, RECEIVER_X_COUNT, RECEIVER_Y_COUNT, TIME_SAMPLES),
        availability=torch.ones(1, RECEIVER_X_COUNT, RECEIVER_Y_COUNT, dtype=torch.bool),
    )

    with pytest.raises(ValueError, match="generator is required"):
        source.inputs(targets, np.asarray([0], dtype=np.int64), neighbor_dropout=0.5)


def test_audit_reports_exact_keys_and_coverage() -> None:
    train_availability = torch.ones(4, RECEIVER_X_COUNT, RECEIVER_Y_COUNT, dtype=torch.bool)
    train_availability[:, 0, 0] = False
    source = _tensor_source(
        [10, 11, 12, 13],
        [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0)],
        train_availability=train_availability,
    )
    targets = source.build_targets(
        ffids=np.asarray([20, 21], dtype=np.int64),
        source_coordinates_m=np.asarray([[1.0, 0.0], [29.0, 0.0]]),
        gathers=torch.zeros(2, RECEIVER_X_COUNT, RECEIVER_Y_COUNT, TIME_SAMPLES),
        availability=torch.ones(2, RECEIVER_X_COUNT, RECEIVER_Y_COUNT, dtype=torch.bool),
    )

    audit = source.audit(targets)

    receiver_cell_count = RECEIVER_X_COUNT * RECEIVER_Y_COUNT
    assert list(audit) == [
        "target_ffid_count",
        "source_gather_count",
        "neighbor_source_entries",
        "target_ffid_neighbor_entries",
        "non_train_neighbor_entries",
        "target_trace_count",
        "uncovered_target_receiver_cells",
        "receiver_cells_with_any_neighbor",
    ]
    assert audit["target_ffid_count"] == 2
    assert audit["source_gather_count"] == 2
    assert audit["neighbor_source_entries"] == 4
    assert audit["target_ffid_neighbor_entries"] == 0
    assert audit["non_train_neighbor_entries"] == 0
    assert audit["target_trace_count"] == 2 * receiver_cell_count
    assert audit["uncovered_target_receiver_cells"] == 2
    assert audit["receiver_cells_with_any_neighbor"] == {
        "min": receiver_cell_count - 1,
        "mean": float(receiver_cell_count - 1),
        "max": receiver_cell_count - 1,
    }
