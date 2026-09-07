"""Nonoverlapping five-axis cores with clipped receptive-field halos."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import product
from numbers import Integral


@dataclass(frozen=True)
class CCNet5DTile:
    """Global output/input slices and the core inside one model output."""

    core_slices: tuple[slice, ...]
    input_slices: tuple[slice, ...]
    local_core_slices: tuple[slice, ...]


def iter_ccnet5d_tiles(
    volume_shape: tuple[int, ...],
    core_shape: tuple[int, ...],
    *,
    halo_radius: int,
) -> Iterator[CCNet5DTile]:
    """Cover each real sample once, without padding beyond the actual volume."""
    volume = validate_ccnet5d_shape(volume_shape, "volume_shape")
    core = validate_ccnet5d_shape(core_shape, "core_shape")
    if isinstance(halo_radius, bool) or not isinstance(halo_radius, Integral) or halo_radius < 0:
        raise ValueError("halo_radius must be a nonnegative integer")
    for starts in product(
        *(range(0, length, step) for length, step in zip(volume, core, strict=True))
    ):
        stops = tuple(
            min(start + step, length)
            for start, step, length in zip(starts, core, volume, strict=True)
        )
        input_starts = tuple(max(0, start - halo_radius) for start in starts)
        input_stops = tuple(
            min(length, stop + halo_radius) for stop, length in zip(stops, volume, strict=True)
        )
        yield CCNet5DTile(
            core_slices=tuple(
                slice(start, stop) for start, stop in zip(starts, stops, strict=True)
            ),
            input_slices=tuple(
                slice(start, stop) for start, stop in zip(input_starts, input_stops, strict=True)
            ),
            local_core_slices=tuple(
                slice(start - offset, stop - offset)
                for start, stop, offset in zip(starts, stops, input_starts, strict=True)
            ),
        )


def validate_ccnet5d_shape(value: Sequence[int], name: str) -> tuple[int, ...]:
    """Validate a concrete five-axis volume, patch, or core extent."""
    if not isinstance(value, (tuple, list)) or len(value) != 5:
        raise ValueError(f"{name} must contain five positive integers")
    if any(isinstance(item, bool) or not isinstance(item, Integral) or item <= 0 for item in value):
        raise ValueError(f"{name} must contain five positive integers")
    return tuple(int(item) for item in value)
