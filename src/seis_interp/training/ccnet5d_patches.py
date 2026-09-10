"""Fixed seeded patch descriptors and artificial trace masks for CCNet labels."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from seis_interp.data.c3_supervised_source import C3SupervisedSource
from seis_interp.processing.c3_volume_index import VOLUME_AXIS_ORDER


@dataclass(frozen=True)
class PatchSpec:
    """One region-local start and a fixed, independently replayable mask seed."""

    start: tuple[int, int, int, int, int]
    mask_seed: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", _five_integers(self.start, name="start", minimum=0))
        object.__setattr__(self, "mask_seed", _integer(self.mask_seed, name="mask_seed", minimum=0))


@dataclass(frozen=True)
class CCNetPatchPlan:
    """Small immutable descriptors; amplitudes and complete masks are never stored."""

    patch_shape: tuple[int, int, int, int, int]
    missing_fraction: float
    random_seed: int
    fit: tuple[PatchSpec, ...]
    selection: tuple[PatchSpec, ...]

    def __post_init__(self) -> None:
        shape = _five_integers(self.patch_shape, name="patch_shape", minimum=1)
        fraction = _missing_fraction(self.missing_fraction)
        _missing_count(shape, fraction)
        object.__setattr__(self, "patch_shape", shape)
        object.__setattr__(self, "missing_fraction", fraction)
        object.__setattr__(
            self, "random_seed", _integer(self.random_seed, name="random_seed", minimum=0)
        )
        for name in ("fit", "selection"):
            specs = getattr(self, name)
            if (
                not isinstance(specs, (tuple, list))
                or not specs
                or not all(isinstance(spec, PatchSpec) for spec in specs)
            ):
                raise ValueError(f"{name} must contain a non-empty sequence of PatchSpec values")
            object.__setattr__(self, name, tuple(specs))

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready scalar descriptors with no paths, labels, or mask arrays."""
        return {
            "patch_shape": list(self.patch_shape),
            "missing_fraction": self.missing_fraction,
            "random_seed": self.random_seed,
            "fit": [{"start": list(spec.start), "mask_seed": spec.mask_seed} for spec in self.fit],
            "selection": [
                {"start": list(spec.start), "mask_seed": spec.mask_seed} for spec in self.selection
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> CCNetPatchPlan:
        """Restore and validate the exact descriptor-only JSON representation."""
        keys = {"patch_shape", "missing_fraction", "random_seed", "fit", "selection"}
        if not isinstance(payload, Mapping) or set(payload) != keys:
            raise ValueError(f"patch plan must contain exactly {sorted(keys)}")
        regions = {}
        for name in ("fit", "selection"):
            records = payload[name]
            if not isinstance(records, list) or not records:
                raise ValueError(f"patch plan {name} must be a non-empty list")
            specs = []
            for record in records:
                if not isinstance(record, Mapping) or set(record) != {"start", "mask_seed"}:
                    raise ValueError("patch descriptor must contain exactly start and mask_seed")
                specs.append(PatchSpec(record["start"], record["mask_seed"]))
            regions[name] = tuple(specs)
        return cls(
            payload["patch_shape"],
            payload["missing_fraction"],
            payload["random_seed"],
            regions["fit"],
            regions["selection"],
        )


def make_ccnet_patch_plan(
    source: C3SupervisedSource,
    *,
    patch_shape: tuple[int, int, int, int, int],
    fit_count: int,
    selection_count: int,
    missing_fraction: float,
    random_seed: int,
) -> CCNetPatchPlan:
    """Sample bounded starts with separate fit/selection streams, without reading labels."""
    shape = _five_integers(patch_shape, name="patch_shape", minimum=1)
    fraction = _missing_fraction(missing_fraction)
    _missing_count(shape, fraction)
    seed = _integer(random_seed, name="random_seed", minimum=0)
    counts = (
        _integer(fit_count, name="fit_count", minimum=1),
        _integer(selection_count, name="selection_count", minimum=1),
    )
    streams = np.random.SeedSequence(seed).spawn(2)
    plans = []
    for name, count, stream in zip(("fit", "selection"), counts, streams, strict=True):
        region_shape = getattr(source, name).shape
        if any(length > available for length, available in zip(shape, region_shape, strict=True)):
            raise ValueError(f"patch_shape must fit inside the {name} region on every axis")
        plans.append(_sample_bounded_specs(shape, region_shape, count, stream))
    return CCNetPatchPlan(shape, fraction, seed, plans[0], plans[1])


def make_ccnet_training_dataset_patch_plan(
    source: C3SupervisedSource,
    *,
    patch_shape: tuple[int, int, int, int, int],
    fit_count: int,
    selection_count: int,
    missing_fraction: float,
    random_seed: int,
) -> tuple[CCNetPatchPlan, dict[str, object]]:
    """Rejection-sample unique all-valid fit patches from the QC training dataset."""
    shape = _five_integers(patch_shape, name="patch_shape", minimum=1)
    fraction = _missing_fraction(missing_fraction)
    _missing_count(shape, fraction)
    seed = _integer(random_seed, name="random_seed", minimum=0)
    fit_total = _integer(fit_count, name="fit_count", minimum=1)
    selection_total = _integer(selection_count, name="selection_count", minimum=1)
    for name in ("fit", "selection"):
        region_shape = getattr(source, name).shape
        if any(length > available for length, available in zip(shape, region_shape, strict=True)):
            raise ValueError(f"patch_shape must fit inside the {name} region on every axis")

    fit_stream, selection_stream = np.random.SeedSequence(seed).spawn(2)
    rng = np.random.default_rng(fit_stream)
    accepted: list[PatchSpec] = []
    starts: set[tuple[int, int, int, int, int]] = set()
    rejection_reasons = {"duplicate_start": 0, "unauthorized_or_absent_trace_cell": 0}
    maximum_attempts = max(10_000, fit_total * 1_000)
    attempts = 0
    while len(accepted) < fit_total and attempts < maximum_attempts:
        attempts += 1
        start = tuple(
            int(rng.integers(0, available - length + 1))
            for length, available in zip(shape, source.fit.shape, strict=True)
        )
        if start in starts:
            rejection_reasons["duplicate_start"] += 1
            continue
        try:
            source.patch_array_rows("fit", start, shape)
        except ValueError as error:
            if "unauthorized or absent trace cells" not in str(error):
                raise
            rejection_reasons["unauthorized_or_absent_trace_cell"] += 1
            continue
        starts.add(start)
        accepted.append(PatchSpec(start, int(rng.integers(0, np.iinfo(np.int64).max))))
    if len(accepted) != fit_total:
        raise ValueError(
            "could not sample the requested valid unique fit patches: "
            f"accepted={len(accepted)}, attempts={attempts}, "
            f"fit_shape={source.fit.shape}, patch_shape={shape}, "
            f"rejections={rejection_reasons}"
        )
    selection = _sample_bounded_specs(
        shape, source.selection.shape, selection_total, selection_stream
    )
    plan = CCNetPatchPlan(shape, fraction, seed, tuple(accepted), selection)
    diagnostics = ccnet_training_dataset_patch_diagnostics(
        source,
        plan,
        candidate_count=attempts,
        rejection_reasons=rejection_reasons,
    )
    return plan, diagnostics


def ccnet_training_dataset_patch_diagnostics(
    source: C3SupervisedSource,
    plan: CCNetPatchPlan,
    *,
    candidate_count: int,
    rejection_reasons: Mapping[str, int],
) -> dict[str, object]:
    """Summarize descriptor distribution, trace coverage, overlap, and fixed scale."""
    fit_starts = np.asarray([spec.start for spec in plan.fit], dtype=np.int64)
    fit_rows = source.fit.array_rows[source.fit.array_rows >= 0]
    row_limit = int(fit_rows.max()) + 1
    covered = np.zeros(row_limit, dtype=np.bool_)
    for spec in plan.fit:
        covered[source.patch_array_rows("fit", spec.start, plan.patch_shape)] = True
    selection_covered = np.zeros(row_limit, dtype=np.bool_)
    for spec in plan.selection:
        rows = source.patch_array_rows("selection", spec.start, plan.patch_shape)
        selection_covered[rows[rows < row_limit]] = True
    unique_covered = int(np.count_nonzero(covered))
    training_trace_count = int(
        source.inputs_lock.get("training_dataset", {}).get(
            "authorized_trace_count", len(np.unique(fit_rows))
        )
    )
    offsets = [source.fit.selection[axis][0] for axis in VOLUME_AXIS_ORDER]
    global_starts = fit_starts + np.asarray(offsets, dtype=np.int64)
    line_start_histogram = _integer_histogram(global_starts[:, 1])
    line_bounds = source.fit.selection["source_line"]
    line_inclusion = {
        str(line): int(
            np.count_nonzero(
                (global_starts[:, 1] <= line) & (line < global_starts[:, 1] + plan.patch_shape[1])
            )
        )
        for line in range(*line_bounds)
    }
    normalization = source.inputs_lock.get("normalization_source")
    return {
        "fit_patch_count": len(plan.fit),
        "unique_start_count": len({spec.start for spec in plan.fit}),
        "candidate_count": candidate_count,
        "rejected_candidate_count": candidate_count - len(plan.fit),
        "rejection_reason_counts": dict(rejection_reasons),
        "source_line_start": {
            "min": int(global_starts[:, 1].min()),
            "max": int(global_starts[:, 1].max()),
            "histogram": line_start_histogram,
        },
        "shot_start": {
            "min": int(global_starts[:, 2].min()),
            "max": int(global_starts[:, 2].max()),
            "histogram": _integer_histogram(global_starts[:, 2]),
        },
        "receiver_y_start": {
            "min": int(global_starts[:, 4].min()),
            "max": int(global_starts[:, 4].max()),
        },
        "source_line_patch_inclusion_histogram": line_inclusion,
        "all_source_lines_covered": all(count > 0 for count in line_inclusion.values()),
        "unique_canonical_trace_rows_covered": unique_covered,
        "training_dataset_trace_count": training_trace_count,
        "unique_trace_coverage_fraction": unique_covered / training_trace_count,
        "selection_trace_overlap_count": int(np.count_nonzero(covered & selection_covered)),
        "selection_metric_role": (
            "diagnostic_only_not_independent_holdout"
            if np.any(covered & selection_covered)
            else "diagnostic_only"
        ),
        "normalization_source": normalization,
        "normalization_value": source.amplitude_rms,
    }


def _sample_bounded_specs(
    shape: tuple[int, int, int, int, int],
    region_shape: tuple[int, int, int, int, int],
    count: int,
    stream: np.random.SeedSequence,
) -> tuple[PatchSpec, ...]:
    rng = np.random.default_rng(stream)
    descriptors = []
    for _ in range(count):
        start = tuple(
            int(rng.integers(0, available - length + 1))
            for length, available in zip(shape, region_shape, strict=True)
        )
        mask_seed = int(rng.integers(0, np.iinfo(np.int64).max))
        descriptors.append(PatchSpec(start, mask_seed))
    return tuple(descriptors)


def _integer_histogram(values: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(values, return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(unique, counts, strict=True)}


def load_ccnet_patch(
    source: C3SupervisedSource,
    plan: CCNetPatchPlan,
    *,
    region: str,
    index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one complete label, apply the fit RMS, and replay its whole-trace mask."""
    if region not in ("fit", "selection"):
        raise ValueError("region must be fit or selection")
    selected_index = _integer(index, name="index", minimum=0)
    specs = plan.fit if region == "fit" else plan.selection
    if selected_index >= len(specs):
        raise ValueError(f"index is outside the {region} patch plan")
    spec = specs[selected_index]
    raw = source.read_patch(region, spec.start, plan.patch_shape)
    label = np.ascontiguousarray(raw / source.amplitude_rms, dtype=np.float32)
    if not np.all(np.isfinite(label)):
        raise ValueError("normalized complete label must contain only finite values")
    spatial_shape = plan.patch_shape[1:]
    trace_count = math.prod(spatial_shape)
    missing = np.random.default_rng(spec.mask_seed).choice(
        trace_count, size=_missing_count(plan.patch_shape, plan.missing_fraction), replace=False
    )
    mask = np.ones(trace_count, dtype=np.bool_)
    mask[missing] = False
    mask = mask.reshape(spatial_shape)
    corrupted = np.where(mask[np.newaxis, ...], label, np.float32(0.0))
    return np.ascontiguousarray(corrupted), label, mask


def _integer(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _five_integers(value: object, *, name: str, minimum: int) -> tuple[int, int, int, int, int]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, (Sequence, np.ndarray))
        or len(value) != 5
    ):
        raise ValueError(f"{name} must contain five integers >= {minimum}")
    return tuple(_integer(item, name=name, minimum=minimum) for item in value)


def _missing_fraction(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or not 0.0 < value < 1.0
    ):
        raise ValueError("missing_fraction must be finite and strictly between 0 and 1")
    return float(value)


def _missing_count(shape: tuple[int, int, int, int, int], fraction: float) -> int:
    trace_count = math.prod(shape[1:])
    count = round(fraction * trace_count)
    if not 1 <= count < trace_count:
        raise ValueError("rounded missing trace count must be at least 1 and less than trace count")
    return count
