"""Observed-only pseudo-target patches for per-volume CCNet-5D fitting."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from numbers import Integral, Real

import numpy as np

from seis_interp.data.c3_volume_adapter import ObservedC3Volume
from seis_interp.processing.ccnet5d_tiles import validate_ccnet5d_shape
from seis_interp.training.amplitude_scaling import normalize_by_global_rms


@dataclass(frozen=True)
class CCNet5DObservedPatch:
    """One normalized patch whose targets are hidden observed traces only."""

    model_input: np.ndarray
    pseudo_target: np.ndarray
    pseudo_target_mask: np.ndarray
    visible_observed_mask: np.ndarray
    patch_slices: tuple[slice, ...]


class CCNet5DObservedPatchSource:
    """Draw finite, reproducible pseudo-target patches from one observed volume."""

    def __init__(
        self,
        observed_volume: ObservedC3Volume,
        *,
        amplitude_scale: float,
        patch_shape: tuple[int, ...],
        inner_mask_fraction: float,
        random_seed: int | None = None,
        generator: np.random.Generator | None = None,
    ) -> None:
        volume, observed_mask = _validated_observed_volume(observed_volume)
        self.patch_shape = validate_ccnet5d_shape(patch_shape, "patch_shape")
        if any(
            patch > extent for patch, extent in zip(self.patch_shape, volume.shape, strict=True)
        ):
            raise ValueError("patch_shape must fit inside the observed volume on every axis")
        self.inner_mask_fraction = _inner_mask_fraction(inner_mask_fraction)
        self.amplitude_scale = _positive_finite_scale(amplitude_scale)
        self._generator = _random_generator(random_seed=random_seed, generator=generator)
        self._observed_mask = np.array(observed_mask, dtype=np.bool_, copy=True, order="C")

        observed_values = np.where(self._observed_mask[None, ...], volume, 0)
        self._normalized_observed = np.ascontiguousarray(
            normalize_by_global_rms(observed_values, self.amplitude_scale)
        )
        self._eligible_spatial_starts = _eligible_spatial_starts(
            self._observed_mask,
            patch_shape=self.patch_shape[1:],
            inner_mask_fraction=self.inner_mask_fraction,
        )
        if not self._eligible_spatial_starts:
            raise ValueError(
                "no patch can provide both a pseudo-target and visible observed context"
            )

    @property
    def volume_shape(self) -> tuple[int, ...]:
        """Return the time-first source shape without exposing source amplitudes."""
        return self._normalized_observed.shape

    def sample(self) -> CCNet5DObservedPatch:
        """Draw one patch and a whole-trace pseudo-mask from the source RNG."""
        time_limit = self.volume_shape[0] - self.patch_shape[0] + 1
        time_start = int(self._generator.integers(time_limit))
        placement_index = int(self._generator.integers(len(self._eligible_spatial_starts)))
        starts = (time_start, *self._eligible_spatial_starts[placement_index])
        patch_slices = tuple(
            slice(start, start + extent)
            for start, extent in zip(starts, self.patch_shape, strict=True)
        )
        spatial_slices = patch_slices[1:]
        available = self._observed_mask[spatial_slices]
        hidden = _draw_pseudo_target_mask(
            available,
            fraction=self.inner_mask_fraction,
            generator=self._generator,
        )
        visible = available & ~hidden

        normalized = self._normalized_observed[patch_slices]
        model_input = np.where(visible[None, ...], normalized, 0)
        pseudo_target = np.where(hidden[None, ...], normalized, 0)
        return CCNet5DObservedPatch(
            model_input=np.ascontiguousarray(model_input),
            pseudo_target=np.ascontiguousarray(pseudo_target),
            pseudo_target_mask=np.ascontiguousarray(hidden, dtype=np.bool_),
            visible_observed_mask=np.ascontiguousarray(visible, dtype=np.bool_),
            patch_slices=patch_slices,
        )


def _validated_observed_volume(
    observed_volume: ObservedC3Volume,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(observed_volume, ObservedC3Volume):
        raise TypeError("observed_volume must be an ObservedC3Volume")
    values = observed_volume.values
    if (
        not isinstance(values, np.ndarray)
        or values.ndim != 5
        or values.size == 0
        or values.dtype.kind not in "fiu"
        or values.dtype.kind == "b"
    ):
        raise ValueError("observed volume values must be a non-empty numeric 5-D array")
    observed = observed_volume.observed_trace_mask
    target = observed_volume.evaluation_target_trace_mask
    for name, mask in (("observed", observed), ("evaluation target", target)):
        if (
            not isinstance(mask, np.ndarray)
            or mask.dtype != np.bool_
            or mask.shape != values.shape[1:]
        ):
            raise ValueError(f"{name} trace mask must be boolean and match the spatial shape")
    if np.any(observed & target):
        raise ValueError("observed and evaluation target trace masks must not overlap")
    if not np.all(np.isfinite(values[:, observed])):
        raise ValueError("observed amplitudes must contain only finite values")
    return values, observed


def _eligible_spatial_starts(
    observed_mask: np.ndarray,
    *,
    patch_shape: tuple[int, ...],
    inner_mask_fraction: float,
) -> tuple[tuple[int, ...], ...]:
    starts: list[tuple[int, ...]] = []
    ranges = (
        range(length - extent + 1)
        for length, extent in zip(observed_mask.shape, patch_shape, strict=True)
    )
    for candidate in product(*ranges):
        slices = tuple(
            slice(start, start + extent)
            for start, extent in zip(candidate, patch_shape, strict=True)
        )
        available_count = int(np.count_nonzero(observed_mask[slices]))
        hidden_count = _pseudo_target_count(available_count, inner_mask_fraction)
        if 0 < hidden_count < available_count:
            starts.append(candidate)
    return tuple(starts)


def _draw_pseudo_target_mask(
    available_mask: np.ndarray,
    *,
    fraction: float,
    generator: np.random.Generator,
) -> np.ndarray:
    available_positions = np.flatnonzero(available_mask)
    hidden_count = _pseudo_target_count(len(available_positions), fraction)
    if hidden_count < 1 or hidden_count >= len(available_positions):
        raise ValueError("patch must contain at least one pseudo-target and one visible context")
    selected = generator.choice(available_positions, size=hidden_count, replace=False)
    hidden = np.zeros(available_mask.shape, dtype=np.bool_)
    hidden.reshape(-1)[selected] = True
    return hidden


def _pseudo_target_count(available_count: int, fraction: float) -> int:
    return int(round(available_count * fraction))


def _inner_mask_fraction(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("inner_mask_fraction must be finite and strictly between 0 and 1")
    fraction = float(value)
    if not np.isfinite(fraction) or fraction <= 0.0 or fraction >= 1.0:
        raise ValueError("inner_mask_fraction must be finite and strictly between 0 and 1")
    return fraction


def _positive_finite_scale(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("amplitude_scale must be positive and finite")
    scale = float(value)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("amplitude_scale must be positive and finite")
    return scale


def _random_generator(
    *,
    random_seed: int | None,
    generator: np.random.Generator | None,
) -> np.random.Generator:
    if random_seed is not None and generator is not None:
        raise ValueError("pass either random_seed or generator, not both")
    if generator is not None:
        if not isinstance(generator, np.random.Generator):
            raise TypeError("generator must be a NumPy Generator")
        return generator
    if isinstance(random_seed, bool) or not isinstance(random_seed, Integral) or random_seed < 0:
        raise ValueError("random_seed must be a non-negative integer")
    return np.random.default_rng(int(random_seed))
