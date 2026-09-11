from dataclasses import replace

import numpy as np
import pytest

from seis_interp.data.c3_volume_adapter import ObservedC3Volume
from seis_interp.training.ccnet5d_observed_patches import CCNet5DObservedPatchSource


def _observed_volume() -> ObservedC3Volume:
    shape = (384, 2, 2, 1, 2)
    observed = np.array(
        [
            [[[True, True]], [[True, False]]],
            [[[True, False]], [[True, True]]],
        ],
        dtype=np.bool_,
    )
    target = np.array(
        [
            [[[False, False]], [[False, True]]],
            [[[False, True]], [[False, False]]],
        ],
        dtype=np.bool_,
    )
    values = np.full(shape, 777.0, dtype=np.float32)
    observed_positions = np.flatnonzero(observed)
    for index, position in enumerate(observed_positions, start=1):
        values.reshape(shape[0], -1)[:, position] = index + np.arange(shape[0]) / 1000
    values[:, target] = 999.0
    return ObservedC3Volume(
        values=values,
        time_s=np.arange(shape[0], dtype=np.float64) * 0.004,
        array_rows=np.arange(np.prod(shape[1:]), dtype=np.int64).reshape(shape[1:]),
        observed_trace_mask=observed,
        evaluation_target_trace_mask=target,
    )


def test_patch_contains_only_normalized_observed_pseudo_targets_and_visible_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    volume = _observed_volume()
    before = volume.values.copy()

    def forbid_rms_recomputation(*_args: object, **_kwargs: object) -> float:
        pytest.fail("patch source must use the supplied observed global RMS")

    monkeypatch.setattr(
        "seis_interp.training.amplitude_scaling.compute_observed_global_rms",
        forbid_rms_recomputation,
    )
    source = CCNet5DObservedPatchSource(
        volume,
        amplitude_scale=2.0,
        patch_shape=volume.values.shape,
        inner_mask_fraction=0.5,
        placement_seed=7,
        inner_mask_seed=7,
    )
    batch = source.sample()

    hidden = batch.pseudo_target_mask
    visible = batch.visible_observed_mask
    assert np.any(hidden) and np.any(visible)
    assert np.all(hidden <= volume.observed_trace_mask)
    assert not np.any(hidden & volume.evaluation_target_trace_mask)
    assert not np.any(visible & hidden)
    np.testing.assert_array_equal(visible | hidden, volume.observed_trace_mask)
    np.testing.assert_array_equal(batch.model_input[:, hidden], 0)
    np.testing.assert_allclose(batch.pseudo_target[:, hidden], volume.values[:, hidden] / 2.0)
    np.testing.assert_allclose(batch.model_input[:, visible], volume.values[:, visible] / 2.0)
    np.testing.assert_array_equal(batch.pseudo_target[:, ~hidden], 0)
    np.testing.assert_array_equal(batch.model_input[:, ~visible], 0)
    assert 999.0 not in batch.model_input and 999.0 not in batch.pseudo_target
    assert 777.0 not in batch.model_input and 777.0 not in batch.pseudo_target
    np.testing.assert_array_equal(volume.values, before)


def test_seed_replays_pseudo_mask_without_using_global_numpy_rng() -> None:
    volume = _observed_volume()
    arguments = {
        "amplitude_scale": 1.0,
        "patch_shape": volume.values.shape,
        "inner_mask_fraction": 0.5,
    }
    np.random.seed(123)
    first = CCNet5DObservedPatchSource(
        volume, placement_seed=11, inner_mask_seed=11, **arguments
    ).sample()
    np.random.seed(999)
    replay = CCNet5DObservedPatchSource(
        volume, placement_seed=11, inner_mask_seed=11, **arguments
    ).sample()
    other = CCNet5DObservedPatchSource(
        volume, placement_seed=12, inner_mask_seed=12, **arguments
    ).sample()

    np.testing.assert_array_equal(first.pseudo_target_mask, replay.pseudo_target_mask)
    assert not np.array_equal(first.pseudo_target_mask, other.pseudo_target_mask)


def test_patch_source_rejects_domain_without_target_and_context() -> None:
    volume = _observed_volume()
    observed = np.zeros_like(volume.observed_trace_mask)
    observed.reshape(-1)[0] = True
    target = np.zeros_like(observed)
    target.reshape(-1)[1] = True

    with pytest.raises(ValueError, match="pseudo-target and visible observed context"):
        CCNet5DObservedPatchSource(
            replace(
                volume,
                observed_trace_mask=observed,
                evaluation_target_trace_mask=target,
            ),
            amplitude_scale=1.0,
            patch_shape=volume.values.shape,
            inner_mask_fraction=0.5,
            placement_seed=1,
            inner_mask_seed=1,
        )


def test_sampled_patch_uses_bounded_spatial_placement_and_complete_trace_mask() -> None:
    volume = _observed_volume()
    source = CCNet5DObservedPatchSource(
        volume,
        amplitude_scale=2.0,
        patch_shape=(384, 1, 2, 1, 2),
        inner_mask_fraction=0.5,
        placement_seed=5,
        inner_mask_seed=5,
    )

    batch = source.sample()

    assert batch.model_input.shape == (384, 1, 2, 1, 2)
    assert batch.pseudo_target_mask.shape == (1, 2, 1, 2)
    assert all(item.start is not None and item.stop is not None for item in batch.patch_slices)
    np.testing.assert_array_equal(batch.model_input[:, batch.pseudo_target_mask], 0)


def test_patch_must_span_the_complete_trace_time_axis() -> None:
    volume = _observed_volume()

    with pytest.raises(ValueError, match="complete trace time axis"):
        CCNet5DObservedPatchSource(
            volume,
            amplitude_scale=2.0,
            patch_shape=(128, 1, 2, 1, 2),
            inner_mask_fraction=0.5,
            placement_seed=5,
            inner_mask_seed=5,
        )


def test_placement_and_inner_mask_streams_have_independent_random_consumption():
    volume = _observed_volume()
    arguments = dict(
        amplitude_scale=1.0,
        patch_shape=(384, 1, 2, 1, 2),
        inner_mask_fraction=0.5,
        placement_seed=301,
        inner_mask_seed=401,
    )
    baseline = CCNet5DObservedPatchSource(volume, **arguments)
    changed_mask = CCNet5DObservedPatchSource(volume, **{**arguments, "inner_mask_seed": 402})
    for _ in range(12):
        changed_mask._mask_generator.random(37)
        assert baseline.sample().patch_slices == changed_mask.sample().patch_slices

    # Equal candidate observed counts isolate mask draws from placement differences.
    observed = np.ones_like(volume.observed_trace_mask)
    volume = replace(
        volume, observed_trace_mask=observed, evaluation_target_trace_mask=np.zeros_like(observed)
    )
    baseline = CCNet5DObservedPatchSource(volume, **arguments)
    changed_placement = CCNet5DObservedPatchSource(volume, **{**arguments, "placement_seed": 302})
    for _ in range(12):
        changed_placement._placement_generator.random(37)
        np.testing.assert_array_equal(
            baseline.sample().pseudo_target_mask, changed_placement.sample().pseudo_target_mask
        )
