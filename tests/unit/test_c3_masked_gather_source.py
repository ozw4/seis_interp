from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import seis_interp.data.c3_masked_gather_source as source_module
from seis_interp.data.c3_masked_gather_source import (
    CONTEXT_SELECTION,
    SOURCE_DISTANCE,
    TARGET_COORDINATE_SCALING,
    MaskedC3GatherSource,
    load_masked_c3_gather_source,
)
from seis_interp.data.c3_volume_adapter import ObservedC3Volume
from seis_interp.processing.c3_receiver_grid import RECEIVER_X_COUNT, RECEIVER_Y_COUNT
from seis_interp.processing.c3_volume_index import (
    INDEX_CONTRACT,
    VOLUME_AXIS_ORDER,
    VOLUME_INDEX_COLUMNS,
)

TIME_COUNT = 3
LINE_COUNT = 2
SHOT_COUNT = 2
SOURCE_COUNT = LINE_COUNT * SHOT_COUNT
SPATIAL_SHAPE = (LINE_COUNT, SHOT_COUNT, RECEIVER_X_COUNT, RECEIVER_Y_COUNT)
DEFAULT_COORDINATES = ((0.0, 0.0), (0.0, 2.0), (2.0, 0.0), (2.0, 2.0))
DEFAULT_FFIDS = (91, 12, 77, 43)
DEFAULT_AMPLITUDE_DTYPE = np.dtype(np.float32)


def _default_observed_mask() -> np.ndarray:
    observed = np.ones((SOURCE_COUNT, RECEIVER_X_COUNT, RECEIVER_Y_COUNT), dtype=bool)
    observed[0, 0, 0] = False
    observed[1, 0, 1] = False
    observed[3] = False
    return observed


def _volume_case(
    *,
    coordinates: tuple[tuple[float, float], ...] = DEFAULT_COORDINATES,
    ffids: tuple[int, ...] = DEFAULT_FFIDS,
    observed_flat: np.ndarray | None = None,
    amplitude_dtype: np.dtype = DEFAULT_AMPLITUDE_DTYPE,
) -> tuple[ObservedC3Volume, pd.DataFrame, dict[str, object]]:
    if observed_flat is None:
        observed_flat = _default_observed_mask()
    observed_flat = np.asarray(observed_flat, dtype=bool)
    target_flat = ~observed_flat

    records: list[dict[str, int | float]] = []
    for flat_source, (source_x, source_y) in enumerate(coordinates):
        source_line, shot_in_line = divmod(flat_source, SHOT_COUNT)
        for receiver_x in range(RECEIVER_X_COUNT):
            for receiver_y in range(RECEIVER_Y_COUNT):
                spatial_flat = (
                    flat_source * RECEIVER_X_COUNT + receiver_x
                ) * RECEIVER_Y_COUNT + receiver_y
                records.append(
                    {
                        "array_row": 1000 + spatial_flat,
                        "ffid": ffids[flat_source],
                        "source_line_index": source_line,
                        "shot_in_line_index": shot_in_line,
                        "relative_receiver_x_index": receiver_x,
                        "relative_receiver_y_index": receiver_y,
                        "source_x_m": source_x,
                        "source_y_m": source_y,
                        "relative_receiver_x_m": -140.0 + 40.0 * receiver_x,
                        "relative_receiver_y_m": -2680.0 + 40.0 * receiver_y,
                    }
                )
    index_table = pd.DataFrame.from_records(records, columns=VOLUME_INDEX_COLUMNS)
    for column in VOLUME_INDEX_COLUMNS[:6]:
        index_table[column] = index_table[column].astype(np.int64)
    for column in VOLUME_INDEX_COLUMNS[6:]:
        index_table[column] = index_table[column].astype(np.float64)

    trace_count = int(np.prod(SPATIAL_SHAPE, dtype=np.int64))
    metadata: dict[str, object] = {
        "volume_id": "masked_gather_test",
        "dataset_id": "synthetic",
        "partition": "test",
        "config_source": None,
        "axis_order": list(VOLUME_AXIS_ORDER),
        "selection": {
            "time": [0, TIME_COUNT],
            "source_line": [0, LINE_COUNT],
            "shot_in_line": [0, SHOT_COUNT],
            "relative_receiver_x": [0, RECEIVER_X_COUNT],
            "relative_receiver_y": [0, RECEIVER_Y_COUNT],
        },
        "shape": [TIME_COUNT, *SPATIAL_SHAPE],
        "trace_count": trace_count,
        "role_counts": {
            "observed": int(np.count_nonzero(observed_flat)),
            "evaluation_target": int(np.count_nonzero(target_flat)),
        },
        "index_contract": dict(INDEX_CONTRACT),
        "benchmark_case": {
            "case_id": "masked_gather_case",
            "file": "benchmark_case.json",
            "sha256": "a" * 64,
        },
        "files": {
            "volume_index.parquet": {
                "sha256": "b" * 64,
                "row_count": trace_count,
            }
        },
    }

    shot_values = np.arange(
        SOURCE_COUNT * RECEIVER_X_COUNT * RECEIVER_Y_COUNT * TIME_COUNT,
        dtype=amplitude_dtype,
    ).reshape(SOURCE_COUNT, RECEIVER_X_COUNT, RECEIVER_Y_COUNT, TIME_COUNT)
    shot_values += np.asarray(1.0, dtype=amplitude_dtype)
    shot_values[~observed_flat] = 0.0
    values = np.ascontiguousarray(
        shot_values.reshape(*SPATIAL_SHAPE, TIME_COUNT).transpose(4, 0, 1, 2, 3)
    )
    volume = ObservedC3Volume(
        values=values,
        time_s=np.arange(TIME_COUNT, dtype=np.float64) * 0.008,
        array_rows=index_table["array_row"].to_numpy(dtype=np.int64).reshape(SPATIAL_SHAPE),
        observed_trace_mask=observed_flat.reshape(SPATIAL_SHAPE).copy(),
        evaluation_target_trace_mask=target_flat.reshape(SPATIAL_SHAPE).copy(),
    )
    return volume, index_table, metadata


def _source(
    *,
    context_gather_count: int = 2,
    coordinates: tuple[tuple[float, float], ...] = DEFAULT_COORDINATES,
    ffids: tuple[int, ...] = DEFAULT_FFIDS,
    observed_flat: np.ndarray | None = None,
    amplitude_dtype: np.dtype = DEFAULT_AMPLITUDE_DTYPE,
) -> tuple[MaskedC3GatherSource, ObservedC3Volume]:
    volume, index_table, metadata = _volume_case(
        coordinates=coordinates,
        ffids=ffids,
        observed_flat=observed_flat,
        amplitude_dtype=amplitude_dtype,
    )
    return (
        MaskedC3GatherSource(
            volume,
            index_table,
            metadata,
            context_gather_count=context_gather_count,
            device="cpu",
        ),
        volume,
    )


def test_public_selection_contract_constants() -> None:
    assert CONTEXT_SELECTION == "nearest_observed_source_positions"
    assert SOURCE_DISTANCE == "euclidean_source_xy_m"
    assert TARGET_COORDINATE_SCALING == "volume_source_minmax"


def test_source_layout_targets_and_copying_properties_follow_flat_volume_order() -> None:
    source, volume = _source()

    assert source.source_count == 4
    assert source.target_count == 3
    assert source.context_gather_count == 2
    np.testing.assert_array_equal(source.target_source_indices, [[0, 0], [0, 1], [1, 1]])
    np.testing.assert_array_equal(source.target_ffids, [91, 12, 43])
    expected_rows = volume.array_rows.reshape(SOURCE_COUNT, RECEIVER_X_COUNT, RECEIVER_Y_COUNT)[
        [0, 1, 3]
    ]
    np.testing.assert_array_equal(source.target_array_rows, expected_rows)
    np.testing.assert_array_equal(
        source.target_evaluation_mask,
        volume.evaluation_target_trace_mask.reshape(
            SOURCE_COUNT, RECEIVER_X_COUNT, RECEIVER_Y_COUNT
        )[[0, 1, 3]],
    )

    returned = source.target_array_rows
    returned[:] = -1
    assert np.all(source.target_array_rows >= 0)
    target_sources = source.target_source_indices
    target_sources[:] = -1
    np.testing.assert_array_equal(source.target_source_indices, [[0, 0], [0, 1], [1, 1]])
    target_ffids = source.target_ffids
    target_ffids[:] = -1
    np.testing.assert_array_equal(source.target_ffids, [91, 12, 43])
    evaluation_mask = source.target_evaluation_mask
    evaluation_mask[:] = False
    assert source.target_evaluation_mask.any()
    contexts = source.context_source_indices
    contexts[:] = -1
    assert np.all(source.context_source_indices >= 0)


def test_contexts_are_observed_exclude_target_and_use_distance_then_flat_tie_break() -> None:
    source, volume = _source()

    expected = np.array([[1, 2], [0, 2], [1, 2]], dtype=np.int64)
    np.testing.assert_array_equal(source.context_source_indices, expected)
    target_flat = np.ravel_multi_index(source.target_source_indices.T, (LINE_COUNT, SHOT_COUNT))
    assert not np.any(source.context_source_indices == target_flat[:, None])
    observed_sources = volume.observed_trace_mask.reshape(SOURCE_COUNT, -1).any(axis=1)
    assert np.all(observed_sources[source.context_source_indices])


def test_irregular_ffid_values_do_not_change_context_order() -> None:
    first, _ = _source(ffids=(91, 12, 77, 43))
    second, _ = _source(ffids=(9001, -4, 22, 5))

    np.testing.assert_array_equal(first.context_source_indices, second.context_source_indices)


def test_target_coordinates_use_whole_volume_minmax_and_ignore_mask_roles() -> None:
    coordinates = ((0.0, 0.0), (0.0, 2.0), (10.0, 8.0), (10.0, 10.0))
    source, _ = _source(context_gather_count=1, coordinates=coordinates)
    first_inputs = source.inputs(np.arange(source.target_count, dtype=np.int64))
    np.testing.assert_allclose(
        first_inputs.target_coordinates.numpy(),
        [[0.0, 0.0], [0.0, 0.2], [1.0, 1.0]],
    )

    changed_roles = _default_observed_mask()
    changed_roles[2, 1, 1] = False
    changed_source, _ = _source(
        context_gather_count=1,
        coordinates=coordinates,
        observed_flat=changed_roles,
    )
    changed_target = int(
        np.flatnonzero(np.all(changed_source.target_source_indices == [0, 1], axis=1))[0]
    )
    changed_inputs = changed_source.inputs(np.array([changed_target], dtype=np.int64))
    np.testing.assert_allclose(changed_inputs.target_coordinates.numpy(), [[0.0, 0.2]])


def test_inputs_preserve_partial_target_and_whole_shot_missing_contract() -> None:
    source, volume = _source()

    inputs = source.inputs(np.array([0, 2], dtype=np.int64))

    assert inputs.target_observed.shape == (2, 8, 68, TIME_COUNT)
    assert inputs.target_observation_mask.shape == (2, 8, 68)
    assert inputs.context_gathers.shape == (2, 2, 8, 68, TIME_COUNT)
    assert inputs.context_availability.shape == (2, 2, 8, 68)
    assert inputs.source_deltas_m.shape == (2, 2, 2)
    assert inputs.target_coordinates.shape == (2, 2)
    assert inputs.target_observed.dtype == torch.float32
    assert inputs.context_gathers.dtype == torch.float32
    assert inputs.source_deltas_m.dtype == torch.float32
    assert inputs.target_coordinates.dtype == torch.float32
    assert inputs.target_observation_mask.dtype == torch.bool
    assert inputs.context_availability.dtype == torch.bool

    shot_major = volume.values.transpose(1, 2, 3, 4, 0).reshape(
        SOURCE_COUNT, RECEIVER_X_COUNT, RECEIVER_Y_COUNT, TIME_COUNT
    )
    np.testing.assert_array_equal(inputs.target_observed[0].numpy(), shot_major[0])
    assert inputs.target_observation_mask[0].any()
    assert not inputs.target_observation_mask[1].any()
    assert not inputs.target_observed[1].any()
    assert not inputs.context_availability[1, 0, 0, 1]
    assert not inputs.context_gathers[1, 0, 0, 1].any()


def test_inputs_follow_amplitude_dtype_and_compute_context_minus_target_deltas() -> None:
    source, _ = _source(amplitude_dtype=np.dtype(np.float64))

    inputs = source.inputs(np.array([0], dtype=np.int64))

    assert inputs.target_observed.dtype == torch.float64
    assert inputs.context_gathers.dtype == torch.float64
    assert inputs.source_deltas_m.dtype == torch.float64
    assert inputs.target_coordinates.dtype == torch.float64
    np.testing.assert_array_equal(inputs.source_deltas_m.numpy(), [[[0.0, 2.0], [2.0, 0.0]]])


def test_repeated_target_indices_keep_requested_order() -> None:
    source, _ = _source()

    inputs = source.inputs(np.array([2, 0, 2], dtype=np.int64))

    assert torch.equal(inputs.target_observed[0], inputs.target_observed[2])
    assert torch.equal(inputs.target_observation_mask[0], inputs.target_observation_mask[2])
    assert not torch.equal(inputs.target_observed[0], inputs.target_observed[1])
    np.testing.assert_array_equal(inputs.target_coordinates.numpy(), [[1, 1], [0, 0], [1, 1]])


@pytest.mark.parametrize(
    ("indices", "error", "message"),
    [
        ([0], TypeError, "NumPy array"),
        (np.array([[0]], dtype=np.int64), ValueError, "one-dimensional"),
        (np.array([0.0]), TypeError, "integer dtype"),
        (np.array([True]), TypeError, "integer dtype"),
        (np.array([], dtype=np.int64), ValueError, "must not be empty"),
        (np.array([-1], dtype=np.int64), IndexError, "within"),
        (np.array([3], dtype=np.int64), IndexError, "within"),
    ],
)
def test_inputs_reject_invalid_target_indices(
    indices: object,
    error: type[Exception],
    message: str,
) -> None:
    source, _ = _source()

    with pytest.raises(error, match=message):
        source.inputs(indices)  # type: ignore[arg-type]


def test_constructor_rejects_insufficient_observed_context_sources() -> None:
    volume, index_table, metadata = _volume_case()

    with pytest.raises(
        ValueError,
        match=r"target source indices \(0, 0\) has only 2 observed context sources; 3 required",
    ):
        MaskedC3GatherSource(
            volume,
            index_table,
            metadata,
            context_gather_count=3,
            device="cpu",
        )


def test_constructor_rejects_context_shortage_before_result_allocation() -> None:
    volume, index_table, metadata = _volume_case()
    impossible_count = int(np.iinfo(np.intp).max)

    with pytest.raises(
        ValueError,
        match=rf"has only 2 observed context sources; {impossible_count} required",
    ):
        MaskedC3GatherSource(
            volume,
            index_table,
            metadata,
            context_gather_count=impossible_count,
            device="cpu",
        )


@pytest.mark.parametrize("count", [0, -1, 1.0, True])
def test_constructor_rejects_invalid_context_count(count: object) -> None:
    volume, index_table, metadata = _volume_case()

    with pytest.raises(ValueError, match="context_gather_count must be a positive integer"):
        MaskedC3GatherSource(
            volume,
            index_table,
            metadata,
            context_gather_count=count,  # type: ignore[arg-type]
            device="cpu",
        )


def test_constructor_rejects_volume_index_row_misalignment() -> None:
    volume, index_table, metadata = _volume_case()
    wrong_rows = volume.array_rows.copy()
    wrong_rows.reshape(-1)[:2] = wrong_rows.reshape(-1)[1::-1]

    with pytest.raises(ValueError, match="array_rows do not match volume index row order"):
        MaskedC3GatherSource(
            replace(volume, array_rows=wrong_rows),
            index_table,
            metadata,
            context_gather_count=2,
            device="cpu",
        )


def test_constructor_rejects_non_c3_receiver_shape_explicitly() -> None:
    volume, index_table, metadata = _volume_case()
    receiver_x_count = RECEIVER_X_COUNT - 1
    keep = index_table["relative_receiver_x_index"] < receiver_x_count
    cropped_index = index_table.loc[keep].reset_index(drop=True)
    cropped_observed = volume.observed_trace_mask[:, :, :receiver_x_count, :]
    cropped_target = volume.evaluation_target_trace_mask[:, :, :receiver_x_count, :]
    cropped_metadata = deepcopy(metadata)
    cropped_metadata["selection"]["relative_receiver_x"] = [0, receiver_x_count]  # type: ignore[index]
    cropped_metadata["shape"] = [
        TIME_COUNT,
        LINE_COUNT,
        SHOT_COUNT,
        receiver_x_count,
        RECEIVER_Y_COUNT,
    ]
    cropped_metadata["trace_count"] = len(cropped_index)
    cropped_metadata["role_counts"] = {
        "observed": int(np.count_nonzero(cropped_observed)),
        "evaluation_target": int(np.count_nonzero(cropped_target)),
    }
    cropped_metadata["files"]["volume_index.parquet"]["row_count"] = len(cropped_index)  # type: ignore[index]
    cropped_volume = replace(
        volume,
        values=volume.values[:, :, :, :receiver_x_count, :],
        array_rows=volume.array_rows[:, :, :receiver_x_count, :],
        observed_trace_mask=cropped_observed,
        evaluation_target_trace_mask=cropped_target,
    )

    with pytest.raises(ValueError, match="fixed 8 x 68 receiver grid"):
        MaskedC3GatherSource(
            cropped_volume,
            cropped_index,
            cropped_metadata,
            context_gather_count=2,
            device="cpu",
        )


def test_constructor_rejects_time_axis_too_short_for_masked_input_contract() -> None:
    volume, index_table, metadata = _volume_case()
    short_metadata = deepcopy(metadata)
    short_metadata["selection"]["time"] = [0, 1]  # type: ignore[index]
    short_metadata["shape"] = [1, *SPATIAL_SHAPE]

    with pytest.raises(ValueError, match="time dimension must contain at least two samples"):
        MaskedC3GatherSource(
            replace(volume, values=volume.values[:1], time_s=volume.time_s[:1]),
            index_table,
            short_metadata,
            context_gather_count=2,
            device="cpu",
        )


@pytest.mark.parametrize(
    ("field", "value", "error", "message"),
    [
        ("values", np.zeros((2, *SPATIAL_SHAPE), dtype=np.float32), ValueError, "values shape"),
        ("time_s", np.zeros(TIME_COUNT - 1), ValueError, "time_s shape"),
        ("array_rows", np.zeros((1, 2, 8, 68), dtype=np.int64), ValueError, "array_rows shape"),
        (
            "observed_trace_mask",
            np.zeros((1, 2, 8, 68), dtype=bool),
            ValueError,
            "observed_trace_mask shape",
        ),
        ("values", np.zeros((TIME_COUNT, *SPATIAL_SHAPE), dtype=np.int64), TypeError, "values"),
        ("time_s", np.arange(TIME_COUNT), TypeError, "time_s"),
        ("array_rows", np.zeros(SPATIAL_SHAPE, dtype=np.float32), TypeError, "array_rows"),
        (
            "evaluation_target_trace_mask",
            np.zeros(SPATIAL_SHAPE, dtype=np.int8),
            TypeError,
            "evaluation_target_trace_mask",
        ),
    ],
)
def test_constructor_validates_observed_volume_array_shapes_and_dtypes(
    field: str,
    value: np.ndarray,
    error: type[Exception],
    message: str,
) -> None:
    volume, index_table, metadata = _volume_case()

    with pytest.raises(error, match=message):
        MaskedC3GatherSource(
            replace(volume, **{field: value}),
            index_table,
            metadata,
            context_gather_count=2,
            device="cpu",
        )


def test_constructor_rejects_overlapping_or_incomplete_masks_and_role_count_mismatch() -> None:
    volume, index_table, metadata = _volume_case()
    overlap = volume.evaluation_target_trace_mask.copy()
    overlap[0, 0, 0, 1] = True
    with pytest.raises(ValueError, match="disjointly cover"):
        MaskedC3GatherSource(
            replace(volume, evaluation_target_trace_mask=overlap),
            index_table,
            metadata,
            context_gather_count=2,
            device="cpu",
        )

    incomplete_observed = volume.observed_trace_mask.copy()
    incomplete_observed[0, 0, 0, 1] = False
    with pytest.raises(ValueError, match="disjointly cover"):
        MaskedC3GatherSource(
            replace(volume, observed_trace_mask=incomplete_observed),
            index_table,
            metadata,
            context_gather_count=2,
            device="cpu",
        )

    changed_observed = volume.observed_trace_mask.copy()
    changed_target = volume.evaluation_target_trace_mask.copy()
    changed_observed[0, 0, 0, 1] = False
    changed_target[0, 0, 0, 1] = True
    with pytest.raises(ValueError, match="role counts"):
        MaskedC3GatherSource(
            replace(
                volume,
                observed_trace_mask=changed_observed,
                evaluation_target_trace_mask=changed_target,
            ),
            index_table,
            metadata,
            context_gather_count=2,
            device="cpu",
        )


def test_constructor_does_not_make_a_full_contiguous_shot_major_values_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    volume, index_table, metadata = _volume_case()
    original_ascontiguousarray = np.ascontiguousarray
    copied_full_volume = False

    def tracking_ascontiguousarray(array: object, *args: object, **kwargs: object) -> np.ndarray:
        nonlocal copied_full_volume
        candidate = np.asanyarray(array)
        if candidate.size == volume.values.size and np.shares_memory(candidate, volume.values):
            copied_full_volume = True
        return original_ascontiguousarray(array, *args, **kwargs)

    monkeypatch.setattr(source_module.np, "ascontiguousarray", tracking_ascontiguousarray)

    MaskedC3GatherSource(
        volume,
        index_table,
        metadata,
        context_gather_count=2,
        device="cpu",
    )

    assert not copied_full_volume


def test_inputs_do_not_modify_source_volume() -> None:
    source, volume = _source()
    originals = (
        volume.values.copy(),
        volume.time_s.copy(),
        volume.array_rows.copy(),
        volume.observed_trace_mask.copy(),
        volume.evaluation_target_trace_mask.copy(),
    )

    source.inputs(np.array([2, 0, 2], dtype=np.int64))

    for actual, original in zip(
        (
            volume.values,
            volume.time_s,
            volume.array_rows,
            volume.observed_trace_mask,
            volume.evaluation_target_trace_mask,
        ),
        originals,
        strict=True,
    ):
        np.testing.assert_array_equal(actual, original)


def test_loader_forwards_artifact_paths_and_builds_source_without_raw_amplitude_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    volume, index_table, metadata = _volume_case()
    paths = {
        "interim_dir": tmp_path / "interim",
        "processed_dir": tmp_path / "processed",
        "mask_dir": tmp_path / "mask",
        "case_dir": tmp_path / "case",
        "volume_dir": tmp_path / "volume",
    }
    calls: list[tuple[str, object]] = []

    def fake_load_observed_c3_volume(**kwargs: Path) -> ObservedC3Volume:
        calls.append(("observed", kwargs))
        return volume

    def fake_load_c3_volume_index(directory: Path) -> tuple[pd.DataFrame, dict[str, object]]:
        calls.append(("index", directory))
        return index_table, metadata

    def forbidden_np_load(*args: object, **kwargs: object) -> None:
        raise AssertionError("masked gather loader must not read raw NumPy artifacts")

    monkeypatch.setattr(source_module, "load_observed_c3_volume", fake_load_observed_c3_volume)
    monkeypatch.setattr(source_module, "load_c3_volume_index", fake_load_c3_volume_index)
    monkeypatch.setattr(source_module.np, "load", forbidden_np_load)

    result = load_masked_c3_gather_source(
        **paths,
        context_gather_count=2,
        device=torch.device("cpu"),
    )

    assert isinstance(result, MaskedC3GatherSource)
    assert result.context_gather_count == 2
    assert calls == [
        ("observed", paths),
        ("index", paths["volume_dir"]),
    ]
    assert result.inputs(np.array([0], dtype=np.int64)).target_observed.device.type == "cpu"


def test_loader_propagates_upstream_binding_errors_without_loading_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = ValueError("benchmark case input_files do not match")

    def fail_observed(**kwargs: Path) -> ObservedC3Volume:
        raise expected

    def unexpected_index(directory: Path) -> tuple[pd.DataFrame, dict[str, object]]:
        raise AssertionError("index load must follow observed artifact verification")

    monkeypatch.setattr(source_module, "load_observed_c3_volume", fail_observed)
    monkeypatch.setattr(source_module, "load_c3_volume_index", unexpected_index)

    with pytest.raises(ValueError, match="benchmark case input_files") as caught:
        load_masked_c3_gather_source(
            interim_dir=tmp_path / "interim",
            processed_dir=tmp_path / "processed",
            mask_dir=tmp_path / "mask",
            case_dir=tmp_path / "case",
            volume_dir=tmp_path / "volume",
            context_gather_count=2,
            device="cpu",
        )

    assert caught.value is expected
