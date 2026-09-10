from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from seis_interp.data.c3_supervised_source import (
    C3SupervisedSource,
    load_c3_supervised_source,
    load_c3_training_dataset_source,
)
from seis_interp.training.ccnet5d_patches import (
    CCNetPatchPlan,
    PatchSpec,
    load_ccnet_patch,
    make_ccnet_patch_plan,
    make_ccnet_training_dataset_patch_plan,
)
from tests.fixtures.ccnet5d_artifacts import prepare_ccnet5d_artifacts


@pytest.fixture
def source(tmp_path: Path) -> C3SupervisedSource:
    artifacts = prepare_ccnet5d_artifacts(tmp_path)
    return load_c3_supervised_source(
        interim_dir=artifacts.interim,
        processed_dir=artifacts.processed,
        fit_region=artifacts.fit_region,
        selection_region=artifacts.selection_region,
    )


def _shape_only_source():
    return SimpleNamespace(
        fit=SimpleNamespace(shape=(5, 4, 3, 4, 5)),
        selection=SimpleNamespace(shape=(6, 5, 4, 5, 6)),
    )


def _plan(source, **changes) -> CCNetPatchPlan:
    settings = {
        "patch_shape": (2, 1, 1, 2, 2),
        "fit_count": 5,
        "selection_count": 3,
        "missing_fraction": 0.5,
        "random_seed": 19,
    }
    settings.update(changes)
    return make_ccnet_patch_plan(source, **settings)


def _full_training_source(tmp_path: Path) -> C3SupervisedSource:
    artifacts = prepare_ccnet5d_artifacts(tmp_path)
    split = pd.read_parquet(artifacts.processed / "trace_split.parquet")
    train_rows = split.loc[split["split"].eq("train"), "array_row"].to_numpy(dtype=np.int64)
    return load_c3_training_dataset_source(
        interim_dir=artifacts.interim,
        processed_dir=artifacts.processed,
        authorized_train_rows=train_rows,
        time_range=(1, 4),
        selection_region=artifacts.selection_region,
        amplitude_rms=2.0,
        normalization_source={"sha256": "a" * 64, "amplitude_rms": 2.0},
    )


def _full_plan(source: C3SupervisedSource, *, seed: int = 19, count: int = 64):
    return make_ccnet_training_dataset_patch_plan(
        source,
        patch_shape=(2, 1, 2, 2, 3),
        fit_count=count,
        selection_count=3,
        missing_fraction=0.5,
        random_seed=seed,
    )


def test_seeded_plan_and_json_roundtrip_are_identical() -> None:
    source = _shape_only_source()
    plan = _plan(source)
    assert plan == _plan(source)
    payload = plan.to_dict()
    serialized = json.dumps(payload, sort_keys=True, allow_nan=False)
    restored = CCNetPatchPlan.from_dict(json.loads(serialized))

    assert restored == plan
    assert set(payload) == {"patch_shape", "missing_fraction", "random_seed", "fit", "selection"}
    assert len(plan.fit) == 5
    assert len(plan.selection) == 3
    for name in ("fit", "selection"):
        for record in payload[name]:
            assert set(record) == {"start", "mask_seed"}
            assert len(record["start"]) == 5
            assert isinstance(record["mask_seed"], int)
    assert plan != _plan(source, random_seed=20)


def test_fit_and_selection_streams_do_not_depend_on_other_region_count() -> None:
    source = _shape_only_source()
    plan = _plan(source)

    assert plan.selection == _plan(source, fit_count=17).selection
    assert plan.fit == _plan(source, selection_count=17).fit


def test_training_dataset_plan_is_seeded_unique_valid_and_amplitude_independent(
    tmp_path: Path,
) -> None:
    source = _full_training_source(tmp_path)
    first, diagnostics = _full_plan(source)
    repeated, _ = _full_plan(source)
    changed_seed, _ = _full_plan(source, seed=20)
    changed_amplitudes = replace(source, _amplitudes=np.full_like(source._amplitudes, 1234.0))
    amplitude_changed, _ = _full_plan(changed_amplitudes)

    assert first == repeated == amplitude_changed
    assert first != changed_seed
    assert len(first.fit) == len({spec.start for spec in first.fit}) == 64
    assert first.patch_shape == (2, 1, 2, 2, 3)
    assert diagnostics["fit_patch_count"] == diagnostics["unique_start_count"] == 64
    for spec in first.fit:
        assert np.all(source.patch_array_rows("fit", spec.start, first.patch_shape) >= 0)
    legacy = make_ccnet_patch_plan(
        source,
        patch_shape=first.patch_shape,
        fit_count=64,
        selection_count=3,
        missing_fraction=0.5,
        random_seed=19,
    )
    assert first.selection == legacy.selection


def test_training_dataset_sampler_rejects_patch_with_one_unauthorized_row(
    tmp_path: Path,
) -> None:
    source = _full_training_source(tmp_path)
    rows = source.fit.array_rows.copy()
    rows[:, 0, :, :] = -1
    rows.flags.writeable = False
    source = replace(source, fit=replace(source.fit, array_rows=rows))

    with pytest.raises(ValueError, match="unauthorized or absent"):
        source.patch_array_rows("fit", (0, 0, 0, 0, 0), (1, 1, 1, 1, 1))
    plan, diagnostics = _full_plan(source, count=128)
    assert diagnostics["rejection_reason_counts"]["unauthorized_or_absent_trace_cell"] > 0
    for spec in plan.fit:
        assert np.all(source.patch_array_rows("fit", spec.start, plan.patch_shape) >= 0)


def test_patch_starts_stay_inside_every_axis_and_include_boundary_positions() -> None:
    source = _shape_only_source()
    plan = _plan(source, patch_shape=(2, 2, 2, 2, 2), fit_count=100, selection_count=100)

    for name in ("fit", "selection"):
        starts = np.array([spec.start for spec in getattr(plan, name)])
        maxima = np.array(getattr(source, name).shape) - plan.patch_shape
        assert np.all(starts >= 0)
        assert np.all(starts <= maxima)
        np.testing.assert_array_equal(starts.min(axis=0), np.zeros(5, dtype=np.int64))
        np.testing.assert_array_equal(starts.max(axis=0), maxima)


def test_plan_generation_never_reads_or_filters_amplitudes(
    source: C3SupervisedSource, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_read(*args, **kwargs):
        pytest.fail("plan generation must not read or filter label patches")

    monkeypatch.setattr(C3SupervisedSource, "read_patch", unexpected_read)
    plan = _plan(source, fit_count=8, selection_count=4)

    assert len(plan.fit) == 8
    assert len(plan.selection) == 4


@pytest.mark.parametrize("region", ["fit", "selection"])
def test_patch_mask_is_trace_wise_and_retains_complete_supervision(
    source: C3SupervisedSource, region: str
) -> None:
    plan = _plan(source)
    descriptor = getattr(plan, region)[0]
    raw = source.read_patch(region, descriptor.start, plan.patch_shape)
    original_raw = raw.copy()

    corrupted, label, mask = load_ccnet_patch(source, plan, region=region, index=0)

    assert corrupted.shape == label.shape == plan.patch_shape
    assert corrupted.dtype == label.dtype == np.float32
    assert corrupted.flags.c_contiguous and label.flags.c_contiguous
    assert mask.shape == plan.patch_shape[1:]
    assert mask.dtype == np.bool_
    assert np.count_nonzero(~mask) == 2
    np.testing.assert_array_equal(label, raw / source.amplitude_rms)
    np.testing.assert_array_equal(corrupted[:, mask], label[:, mask])
    np.testing.assert_array_equal(
        corrupted[:, ~mask], np.zeros((plan.patch_shape[0], 2), dtype=np.float32)
    )
    assert np.any(label[:, ~mask] != 0)
    expected_missing = np.random.default_rng(descriptor.mask_seed).choice(4, size=2, replace=False)
    expected_mask = np.ones(4, dtype=np.bool_)
    expected_mask[expected_missing] = False
    np.testing.assert_array_equal(mask.reshape(-1), expected_mask)
    np.testing.assert_array_equal(raw, original_raw)
    assert not np.shares_memory(corrupted, label)


def test_descriptor_result_is_independent_of_read_order_and_prior_array_changes(
    source: C3SupervisedSource,
) -> None:
    plan = _plan(source)
    expected = load_ccnet_patch(source, plan, region="fit", index=1)
    for index in (4, 0, 2, 1):
        loaded = load_ccnet_patch(source, plan, region="fit", index=index)
        if index == 1:
            for actual, reference in zip(loaded, expected, strict=True):
                np.testing.assert_array_equal(actual, reference)
        loaded[0][:] = 999
        loaded[1][:] = -999
        loaded[2][:] = False
    repeated = load_ccnet_patch(source, plan, region="fit", index=1)
    for actual, reference in zip(repeated, expected, strict=True):
        np.testing.assert_array_equal(actual, reference)


def test_both_regions_use_only_the_source_fit_rms(source: C3SupervisedSource) -> None:
    plan = _plan(source)
    changed_scale = replace(source, amplitude_rms=source.amplitude_rms * 2)

    for region in ("fit", "selection"):
        original_input, original_label, original_mask = load_ccnet_patch(
            source, plan, region=region, index=0
        )
        scaled_input, scaled_label, scaled_mask = load_ccnet_patch(
            changed_scale, plan, region=region, index=0
        )
        np.testing.assert_array_equal(scaled_mask, original_mask)
        np.testing.assert_array_equal(scaled_label, original_label / 2)
        np.testing.assert_array_equal(scaled_input, original_input / 2)


def test_zero_amplitudes_do_not_determine_observation_mask_or_filter_patch(
    source: C3SupervisedSource, monkeypatch: pytest.MonkeyPatch
) -> None:
    def zero_patch(self, region, start, shape):
        return np.zeros(shape, dtype=np.float32)

    monkeypatch.setattr(C3SupervisedSource, "read_patch", zero_patch)
    plan = _plan(source)
    corrupted, label, mask = load_ccnet_patch(source, plan, region="fit", index=0)

    assert not corrupted.any()
    assert not label.any()
    assert mask.sum() == 2
    assert (~mask).sum() == 2


@pytest.mark.parametrize(("fraction", "expected_missing"), [(0.25, 2), (0.75, 4)])
def test_missing_count_uses_round_to_nearest_even(
    source: C3SupervisedSource, fraction: float, expected_missing: int
) -> None:
    # Six spatial traces: round(1.5) == 2 and round(4.5) == 4.
    plan = _plan(source, patch_shape=(2, 1, 1, 2, 3), missing_fraction=fraction)
    _, _, mask = load_ccnet_patch(source, plan, region="fit", index=0)

    assert int((~mask).sum()) == expected_missing


@pytest.mark.parametrize("fraction", [0.01, 0.99])
def test_rounded_missing_count_must_leave_observed_and_missing_traces(fraction: float) -> None:
    with pytest.raises(ValueError, match="rounded missing trace count"):
        _plan(_shape_only_source(), missing_fraction=fraction)


@pytest.mark.parametrize("axis", range(5))
def test_shape_must_fit_both_regions_on_every_axis(axis: int) -> None:
    source = _shape_only_source()
    shape = [2, 2, 2, 2, 2]
    shape[axis] = source.fit.shape[axis] + 1
    with pytest.raises(ValueError, match="fit region"):
        _plan(source, patch_shape=tuple(shape))
    source.fit.shape = (20, 20, 20, 20, 20)
    shape[axis] = source.selection.shape[axis] + 1
    with pytest.raises(ValueError, match="selection region"):
        _plan(source, patch_shape=tuple(shape))


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("fit_count", 0),
        ("selection_count", True),
        ("random_seed", -1),
        ("random_seed", False),
        ("patch_shape", (2, 2, 2, 2)),
        ("patch_shape", (2, 2, True, 2, 2)),
        ("patch_shape", (2, 2, 0, 2, 2)),
        ("missing_fraction", 0.0),
        ("missing_fraction", 1.0),
        ("missing_fraction", True),
        ("missing_fraction", float("nan")),
    ],
)
def test_invalid_plan_parameters_are_rejected(key: str, value: object) -> None:
    with pytest.raises(ValueError, match=key):
        _plan(_shape_only_source(), **{key: value})


def test_invalid_descriptors_and_serialized_contract_are_rejected() -> None:
    with pytest.raises(ValueError, match="start"):
        PatchSpec((-1, 0, 0, 0, 0), 4)
    with pytest.raises(ValueError, match="mask_seed"):
        PatchSpec((0, 0, 0, 0, 0), True)
    base = _plan(_shape_only_source()).to_dict()
    variants = []
    missing = deepcopy(base)
    del missing["patch_shape"]
    variants.append(missing)
    extra = deepcopy(base)
    extra["mask_array"] = []
    variants.append(extra)
    empty = deepcopy(base)
    empty["fit"] = []
    variants.append(empty)
    malformed = deepcopy(base)
    malformed["fit"][0]["other"] = 1
    variants.append(malformed)
    invalid_start = deepcopy(base)
    invalid_start["fit"][0]["start"][0] = -1
    variants.append(invalid_start)
    for payload in variants:
        with pytest.raises(ValueError):
            CCNetPatchPlan.from_dict(payload)


def test_invalid_patch_access_is_rejected_before_reading(
    source: C3SupervisedSource, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(source)

    def unexpected_read(*args, **kwargs):
        pytest.fail("invalid patch access must fail before reading labels")

    monkeypatch.setattr(C3SupervisedSource, "read_patch", unexpected_read)
    with pytest.raises(ValueError, match="region"):
        load_ccnet_patch(source, plan, region="validation", index=0)
    for index in (-1, True, len(plan.fit)):
        with pytest.raises(ValueError, match="index"):
            load_ccnet_patch(source, plan, region="fit", index=index)


def test_loaded_descriptor_must_still_fit_the_source_region(source: C3SupervisedSource) -> None:
    plan = _plan(source)
    invalid = replace(plan, fit=(PatchSpec((100, 0, 0, 0, 0), 4),))
    with pytest.raises(ValueError, match="outside the fit region"):
        load_ccnet_patch(source, invalid, region="fit", index=0)
