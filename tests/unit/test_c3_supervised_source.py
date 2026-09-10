from __future__ import annotations

import hashlib
import json
import struct
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seis_interp.data import c3_supervised_source as source_module
from seis_interp.data.c3_supervised_source import (
    C3SupervisedSource,
    load_c3_supervised_source,
    load_c3_training_dataset_source,
)
from seis_interp.data.file_checksums import file_sha256
from seis_interp.pipelines.prepare_baseline import prepare_baseline_dataset
from seis_interp.training.ccnet5d_patches import (
    load_ccnet_patch,
    make_ccnet_training_dataset_patch_plan,
)
from tests.fixtures.ccnet5d_artifacts import PreparedCCNet5DArtifacts, prepare_ccnet5d_artifacts


def _load(artifacts: PreparedCCNet5DArtifacts, **changes) -> C3SupervisedSource:
    return load_c3_supervised_source(
        interim_dir=artifacts.interim,
        processed_dir=artifacts.processed,
        fit_region=changes.get("fit_region", artifacts.fit_region),
        selection_region=changes.get("selection_region", artifacts.selection_region),
    )


def _fit_rows() -> np.ndarray:
    return np.array(
        [
            line * 3 * 8 * 68 + receiver_x * 68 + receiver_y
            for line in range(2)
            for receiver_x in range(2)
            for receiver_y in range(3)
        ],
        dtype=np.int64,
    ).reshape(2, 1, 2, 3)


def _rebind_amplitudes(artifacts: PreparedCCNet5DArtifacts, values: np.ndarray) -> None:
    path = artifacts.interim / "amplitudes.npy"
    np.save(path, values, allow_pickle=False)
    preparation_path = artifacts.processed / "preparation.json"
    preparation = json.loads(preparation_path.read_text())
    preparation["input_files"]["amplitudes.npy"]["sha256"] = file_sha256(path)
    preparation_path.write_text(json.dumps(preparation), encoding="utf-8")


def _training_dataset_source(
    artifacts: PreparedCCNet5DArtifacts, *, authorized_rows: np.ndarray | None = None
) -> C3SupervisedSource:
    split = pd.read_parquet(artifacts.processed / "trace_split.parquet")
    train_rows = split.loc[split["split"].eq("train"), "array_row"].to_numpy(dtype=np.int64)
    return load_c3_training_dataset_source(
        interim_dir=artifacts.interim,
        processed_dir=artifacts.processed,
        authorized_train_rows=train_rows if authorized_rows is None else authorized_rows,
        time_range=(1, 4),
        selection_region=artifacts.selection_region,
        amplitude_rms=7.5,
        normalization_source={"sha256": "a" * 64, "amplitude_rms": 7.5},
    )


@pytest.mark.parametrize("shuffled", [False, True])
def test_global_mapping_nonzero_time_patch_and_fit_rms(tmp_path: Path, shuffled: bool) -> None:
    artifacts = prepare_ccnet5d_artifacts(tmp_path, shuffle_tables=shuffled)

    source = _load(artifacts)

    np.testing.assert_array_equal(source.fit.array_rows, _fit_rows())
    assert source.fit.time_range == (1, 4)
    assert source.fit.shape == (3, 2, 1, 2, 3)
    assert source.selection.shape == (3, 2, 2, 2, 3)
    assert np.intersect1d(source.fit.array_rows, source.selection.array_rows).size == 0
    assert not source.fit.array_rows.flags.writeable
    expected_fit = _fit_rows().reshape(-1, 1) * 10 + np.array([-1, 0, 1])
    expected_rms = float(np.sqrt(np.mean(np.square(expected_fit, dtype=np.float64))))
    assert source.amplitude_rms == pytest.approx(expected_rms)
    patch = source.read_patch("fit", (1, 1, 0, 1, 1), (2, 1, 1, 1, 2))
    expected = np.array([17010, 17020, 17011, 17021], dtype=np.float32).reshape(2, 1, 1, 1, 2)
    np.testing.assert_array_equal(patch, expected)
    assert patch.dtype == np.float32
    assert patch.flags.c_contiguous
    patch[:] = -999
    np.testing.assert_array_equal(
        source.read_patch("fit", (1, 1, 0, 1, 1), (2, 1, 1, 1, 2)), expected
    )


def test_provenance_binds_portable_hashes_regions_and_mapping_rule(tmp_path: Path) -> None:
    artifacts = prepare_ccnet5d_artifacts(tmp_path)
    source = _load(artifacts)
    lock = source.inputs_lock

    assert lock["dataset_id"] == "synthetic_c3_ccnet5d"
    assert lock["partition"] == "train"
    assert lock["partition_random_seed"] == 42
    assert lock["canonical_policy"] == "keep_lowest_array_row"
    assert lock["array_rows_hash_rule"] == "sha256_shape_le_int64_then_c_order_le_int64"
    for group, directory in (("interim", artifacts.interim), ("processed", artifacts.processed)):
        assert lock[group]
        for filename, record in lock[group].items():
            assert record == {"sha256": file_sha256(directory / filename)}
    for name, region in (("fit", source.fit), ("selection", source.selection)):
        digest = hashlib.sha256(struct.pack("<4q", *region.array_rows.shape))
        digest.update(region.array_rows.astype("<i8").tobytes(order="C"))
        assert lock["regions"][name] == {
            "selection": region.selection,
            "shape": list(region.shape),
            "array_rows_sha256": digest.hexdigest(),
        }
    serialized = json.dumps(lock, allow_nan=False)
    assert str(tmp_path) not in serialized
    assert 'array_rows"' not in serialized


def test_duplicate_physical_cell_keeps_lowest_array_row(tmp_path: Path) -> None:
    clean = _load(prepare_ccnet5d_artifacts(tmp_path / "clean"))
    duplicated = _load(
        prepare_ccnet5d_artifacts(
            tmp_path / "duplicated", duplicate_fit_trace=True, shuffle_tables=True
        )
    )

    np.testing.assert_array_equal(duplicated.fit.array_rows, clean.fit.array_rows)
    assert duplicated.amplitude_rms == clean.amplitude_rms
    np.testing.assert_array_equal(
        duplicated.read_patch("fit", (0, 0, 0, 0, 0), clean.fit.shape),
        clean.read_patch("fit", (0, 0, 0, 0, 0), clean.fit.shape),
    )


def test_nonfit_and_nontrain_changes_do_not_change_fit_labels_or_rms(tmp_path: Path) -> None:
    clean = _load(prepare_ccnet5d_artifacts(tmp_path / "clean"))
    changed = _load(prepare_ccnet5d_artifacts(tmp_path / "nonfit", nonfit_value=1e8))
    nontrain = _load(prepare_ccnet5d_artifacts(tmp_path / "nontrain", nontrain_value=-1e8))

    for source in (changed, nontrain):
        assert source.amplitude_rms == clean.amplitude_rms
        np.testing.assert_array_equal(
            source.read_patch("fit", (0, 0, 0, 0, 0), source.fit.shape),
            clean.read_patch("fit", (0, 0, 0, 0, 0), clean.fit.shape),
        )
    np.testing.assert_array_equal(
        nontrain.read_patch("selection", (0, 0, 0, 0, 0), nontrain.selection.shape),
        clean.read_patch("selection", (0, 0, 0, 0, 0), clean.selection.shape),
    )


def test_source_does_not_scan_nonfit_rows_or_outside_fit_time(tmp_path: Path) -> None:
    artifacts = prepare_ccnet5d_artifacts(tmp_path)
    values = np.load(artifacts.interim / "amplitudes.npy", allow_pickle=False)
    fit_rows = _fit_rows().reshape(-1)
    values[~np.isin(np.arange(len(values)), fit_rows)] = np.nan
    values[fit_rows[:, None], np.array([0, 4])[None, :]] = np.nan
    _rebind_amplitudes(artifacts, values)

    source = _load(artifacts)

    assert np.isfinite(source.amplitude_rms)
    assert np.all(np.isfinite(source.read_patch("fit", (0, 0, 0, 0, 0), source.fit.shape)))
    with pytest.raises(ValueError, match="selection label patch.*non-finite"):
        source.read_patch("selection", (0, 0, 0, 0, 0), (1, 1, 1, 1, 1))


def test_rms_reads_fit_rows_and_time_in_bounded_trace_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = prepare_ccnet5d_artifacts(tmp_path)
    original_load = source_module.load_interim_trace_dataset
    reads = []

    class ReadSpy:
        def __init__(self, values):
            self.values = values

        def __getitem__(self, key):
            rows, time_slice = key
            reads.append((np.array(rows, copy=True), time_slice))
            assert len(rows) <= 5
            assert np.isin(rows, _fit_rows()).all()
            assert time_slice == slice(1, 4)
            return self.values[key]

    def load(directory, *, memory_map_amplitudes, amplitude_validation_rows):
        assert memory_map_amplitudes is True
        np.testing.assert_array_equal(amplitude_validation_rows, np.empty(0, dtype=np.int64))
        dataset = original_load(
            directory,
            memory_map_amplitudes=memory_map_amplitudes,
            amplitude_validation_rows=amplitude_validation_rows,
        )
        assert isinstance(dataset.amplitudes, np.memmap)
        return replace(dataset, amplitudes=ReadSpy(dataset.amplitudes))

    monkeypatch.setattr(source_module, "load_interim_trace_dataset", load)
    monkeypatch.setattr(source_module, "_RMS_TRACE_CHUNK_SIZE", 5)

    _load(artifacts)

    assert [len(rows) for rows, _ in reads] == [5, 5, 2]
    np.testing.assert_array_equal(
        np.concatenate([rows for rows, _ in reads]), _fit_rows().reshape(-1)
    )


def test_global_source_indices_are_not_reranked_after_train_filtering(tmp_path: Path) -> None:
    artifacts = prepare_ccnet5d_artifacts(tmp_path)
    prepare_baseline_dataset(
        artifacts.interim,
        artifacts.processed,
        holdout_fraction=None,
        validation_fraction_of_holdout=None,
        random_seed=42,
        split_scope="c3_source_line_blocks",
        source_line_ranges={"validation": [0, 1], "train": [1, 3], "test": [3, 4]},
        overwrite=True,
    )
    fit = {**artifacts.fit_region, "source_line": [1, 3]}
    selection = {**artifacts.selection_region, "source_line": [1, 3]}

    source = _load(artifacts, fit_region=fit, selection_region=selection)

    np.testing.assert_array_equal(source.fit.array_rows, _fit_rows() + 3 * 8 * 68)


def test_regions_cannot_share_spatial_rows_even_with_disjoint_time(tmp_path: Path) -> None:
    artifacts = prepare_ccnet5d_artifacts(tmp_path)
    fit = {**artifacts.fit_region, "time": [0, 2]}
    selection = {**artifacts.fit_region, "time": [2, 5]}

    with pytest.raises(ValueError, match="disjoint array_row"):
        _load(artifacts, fit_region=fit, selection_region=selection)


@pytest.mark.parametrize(
    ("axis", "bounds", "message"),
    [
        ("source_line", [2, 3], "train partition"),
        ("source_line", [3, 4], "train partition"),
        ("time", [0, 6], "time.*outside"),
        ("shot_in_line", [2, 4], "shot_in_line_range.*outside"),
        ("relative_receiver_x", [7, 9], "receiver grid"),
        ("relative_receiver_y", [67, 69], "receiver grid"),
    ],
)
def test_regions_reject_nontrain_or_out_of_bounds_ranges(
    tmp_path: Path, axis: str, bounds: list[int], message: str
) -> None:
    artifacts = prepare_ccnet5d_artifacts(tmp_path)
    fit = {**artifacts.fit_region, axis: bounds}
    with pytest.raises(ValueError, match=message):
        _load(artifacts, fit_region=fit)


def test_missing_canonical_train_cell_is_not_zero_filled(tmp_path: Path) -> None:
    artifacts = prepare_ccnet5d_artifacts(tmp_path)
    split_path = artifacts.processed / "trace_split.parquet"
    splits = pd.read_parquet(split_path)
    splits.loc[splits["array_row"].eq(0), "split"] = "excluded"
    splits.to_parquet(split_path, index=False)
    path = artifacts.processed / "preparation.json"
    preparation = json.loads(path.read_text())
    preparation["split_counts"]["train"] -= 1
    path.write_text(json.dumps(preparation), encoding="utf-8")

    with pytest.raises(ValueError, match="fit region.*not dense"):
        _load(artifacts)

    training_dataset = _training_dataset_source(artifacts)
    assert training_dataset.fit.array_rows[0, 0, 0, 0] == -1
    with pytest.raises(ValueError, match="unauthorized or absent"):
        training_dataset.patch_array_rows("fit", (0, 0, 0, 0, 0), (1, 1, 1, 1, 1))


def test_training_dataset_grid_is_canonical_under_row_table_order(tmp_path: Path) -> None:
    clean = _training_dataset_source(prepare_ccnet5d_artifacts(tmp_path / "clean"))
    shuffled = _training_dataset_source(
        prepare_ccnet5d_artifacts(tmp_path / "shuffled", shuffle_tables=True)
    )

    np.testing.assert_array_equal(clean.fit.array_rows, shuffled.fit.array_rows)
    assert clean.fit.selection == shuffled.fit.selection
    assert clean.fit.shape == shuffled.fit.shape == (3, 2, 3, 8, 68)
    assert clean.amplitude_rms == shuffled.amplitude_rms == 7.5


def test_training_dataset_rejects_nontrain_rows_and_accepts_zero_labels(tmp_path: Path) -> None:
    artifacts = prepare_ccnet5d_artifacts(tmp_path)
    split = pd.read_parquet(artifacts.processed / "trace_split.parquet")
    train_rows = split.loc[split["split"].eq("train"), "array_row"].to_numpy(dtype=np.int64)
    validation_row = int(split.loc[split["split"].eq("validation"), "array_row"].iloc[0])
    with pytest.raises(ValueError, match="canonical QC train rows"):
        _training_dataset_source(
            artifacts, authorized_rows=np.append(train_rows[:-1], validation_row)
        )

    amplitudes = np.load(artifacts.interim / "amplitudes.npy", allow_pickle=False)
    amplitudes[train_rows] = 0.0
    _rebind_amplitudes(artifacts, amplitudes)
    source = _training_dataset_source(artifacts)
    patch = source.read_patch("fit", (0, 0, 0, 0, 0), (1, 1, 1, 1, 1))
    assert not patch.any()


def test_validation_amplitudes_do_not_change_fit_plan_or_training_inputs(tmp_path: Path) -> None:
    original_artifacts = prepare_ccnet5d_artifacts(tmp_path / "original")
    changed_artifacts = prepare_ccnet5d_artifacts(tmp_path / "changed")
    split = pd.read_parquet(changed_artifacts.processed / "trace_split.parquet")
    validation_rows = split.loc[split["split"].eq("validation"), "array_row"].to_numpy(
        dtype=np.int64
    )
    amplitudes = np.load(changed_artifacts.interim / "amplitudes.npy", allow_pickle=False)
    amplitudes[validation_rows] += 1234.0
    _rebind_amplitudes(changed_artifacts, amplitudes)

    original = _training_dataset_source(original_artifacts)
    changed = _training_dataset_source(changed_artifacts)
    plan_options = {
        "patch_shape": (2, 1, 2, 2, 3),
        "fit_count": 64,
        "selection_count": 3,
        "missing_fraction": 0.5,
        "random_seed": 19,
    }
    original_plan, _ = make_ccnet_training_dataset_patch_plan(original, **plan_options)
    changed_plan, _ = make_ccnet_training_dataset_patch_plan(changed, **plan_options)

    assert original_plan == changed_plan
    for index in range(len(original_plan.fit)):
        original_input, original_label, _ = load_ccnet_patch(
            original, original_plan, region="fit", index=index
        )
        changed_input, changed_label, _ = load_ccnet_patch(
            changed, changed_plan, region="fit", index=index
        )
        np.testing.assert_array_equal(original_input, changed_input)
        np.testing.assert_array_equal(original_label, changed_label)


def test_split_assignment_and_preparation_binding_are_validated(tmp_path: Path) -> None:
    artifacts = prepare_ccnet5d_artifacts(tmp_path)
    split_path = artifacts.processed / "trace_split.parquet"
    splits = pd.read_parquet(split_path)
    splits.loc[splits["array_row"].eq(0), "split"] = "validation"
    splits.to_parquet(split_path, index=False)
    with pytest.raises(ValueError, match="split_counts"):
        _load(artifacts)
    values = np.load(artifacts.interim / "amplitudes.npy", allow_pickle=False)
    values[0, 0] += 1
    np.save(artifacts.interim / "amplitudes.npy", values, allow_pickle=False)
    with pytest.raises(ValueError, match="input_files"):
        _load(artifacts)


@pytest.mark.parametrize("value", [0.0, float("nan")])
def test_fit_labels_require_positive_finite_rms(tmp_path: Path, value: float) -> None:
    artifacts = prepare_ccnet5d_artifacts(tmp_path)
    values = np.load(artifacts.interim / "amplitudes.npy", allow_pickle=False)
    values[_fit_rows().reshape(-1), 1:4] = value
    _rebind_amplitudes(artifacts, values)
    with pytest.raises(ValueError, match="fit.*(RMS|non-finite)"):
        _load(artifacts)


def test_read_patch_rejects_regions_and_invalid_ranges(tmp_path: Path) -> None:
    source = _load(prepare_ccnet5d_artifacts(tmp_path))
    with pytest.raises(ValueError, match="region must be"):
        source.read_patch("validation", (0, 0, 0, 0, 0), (1, 1, 1, 1, 1))
    for value in ((0, 0, 0, 0), (False, 0, 0, 0, 0), (-1, 0, 0, 0, 0)):
        with pytest.raises(ValueError, match="start"):
            source.read_patch("fit", value, (1, 1, 1, 1, 1))
    for value in ((1, 1, 1, 1), (True, 1, 1, 1, 1), (0, 1, 1, 1, 1)):
        with pytest.raises(ValueError, match="shape"):
            source.read_patch("fit", (0, 0, 0, 0, 0), value)
    for axis in range(5):
        start = [0] * 5
        start[axis] = source.fit.shape[axis]
        with pytest.raises(ValueError, match="outside"):
            source.read_patch("fit", tuple(start), (1, 1, 1, 1, 1))


def test_missing_region_axis_is_rejected(tmp_path: Path) -> None:
    artifacts = prepare_ccnet5d_artifacts(tmp_path)
    fit = deepcopy(artifacts.fit_region)
    del fit["time"]
    with pytest.raises(ValueError, match="contain exactly"):
        _load(artifacts, fit_region=fit)
