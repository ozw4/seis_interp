"""Pure selection rules of the pre-registered fair grid search (no I/O).

A run is scored at every EMA shadow; one (run, EMA) pair is a configuration.
Ties have no special case: candidates are visited in stage order, then run name,
then declared EMA order, and the first minimum wins.

A diverged run (its tuning failed on a non-finite loss, gradient norm or
prediction) is terminal: it keeps its place in the budget with the score +inf
at every shadow, so it ranks after every completed run and is never selected.
"""

import re
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import product
from math import inf, isfinite

STAGES = ("S1", "S2", "S3", "E", "R")
SHADOW_PATH = "training.ema_eval_decays"


@dataclass(frozen=True)
class RunRecord:
    """Validation scores of one finished run, keyed by EMA shadow in declared order.

    A ``diverged`` run scores +inf at every shadow and has no training loss.
    """

    name: str
    stage: str
    scores: dict[str, float]
    train_loss: float | None = None
    diverged: bool = False

    def __post_init__(self):
        if self.stage not in STAGES or not self.scores:
            raise ValueError(f"{self.name}: unknown stage or no scores")
        if self.diverged:
            if any(value != inf for value in self.scores.values()) or self.train_loss is not None:
                raise ValueError(f"{self.name}: a diverged run scores +inf and has no loss")
        elif not all(isfinite(value) for value in self.scores.values()):
            raise ValueError(f"{self.name}: non-finite validation score")

    @classmethod
    def diverged_run(cls, name: str, stage: str, keys: list[str]) -> "RunRecord":
        return cls(name, stage, dict.fromkeys(keys, inf), None, True)

    @property
    def best_score(self) -> float:
        return min(self.scores.values())


@dataclass(frozen=True)
class AxisPosition:
    """Where the E base sits on one ordered axis; ``index`` None means not applicable."""

    name: str
    size: int
    index: int | None
    neighbour_scores: dict[int, float] = field(default_factory=dict)
    extensions: frozenset = frozenset()
    midpoints: frozenset = frozenset()


@dataclass(frozen=True)
class PlannedRun:
    """One generated run: its overrides of the method config and, in S1, its grid cell."""

    name: str
    method_id: str
    overrides: dict
    axes: dict | None = None


@dataclass(frozen=True)
class EDecision:
    """The single E run: branch, axis and target (``lower``/``upper`` or a midpoint pair)."""

    branch: str
    axis: str
    target: str | tuple[int, int]


def apply_overrides(config: dict, overrides: dict) -> dict:
    """Copy of ``config`` with each dotted path replaced; integer segments index lists."""
    resolved = deepcopy(config)
    for path, value in overrides.items():
        *parents, leaf = path.split(".")
        node = resolved
        for part in parents:
            node = node[int(part)] if isinstance(node, list) else node[part]
        if isinstance(node, list):
            index = int(leaf)
            if not 0 <= index < len(node):
                raise ValueError(f"{path}: list index outside the configured list")
            node[index] = deepcopy(value)
        else:
            node[leaf] = deepcopy(value)
    return resolved


def merge_overrides(*layers: dict) -> dict:
    """Later layers win per path; a path may not nest inside another path."""
    merged = {}
    for layer in layers:
        merged.update(deepcopy(layer))
    paths = sorted(merged)
    for path in paths:
        if any(other.startswith(f"{path}.") for other in paths):
            raise ValueError(f"override path {path} nests another override")
    return merged


def pair_order(runs: list[RunRecord]) -> list[tuple[RunRecord, str]]:
    """Every (run, EMA) pair in the stable tie order."""
    ordered = sorted(runs, key=lambda run: (STAGES.index(run.stage), run.name))
    return [(run, key) for run in ordered for key in run.scores]


def best_pair(runs: list[RunRecord]) -> tuple[RunRecord, str]:
    """The (run, EMA) pair with the lowest validation score."""
    if not runs:
        raise ValueError("no completed runs to select from")
    run, key = min(pair_order(runs), key=lambda pair: pair[0].scores[pair[1]])
    if run.diverged:
        raise ValueError("every run to select from diverged")
    return run, key


def top_runs(runs: list[RunRecord], count: int) -> list[RunRecord]:
    """The ``count`` runs with the lowest best-EMA score."""
    if len(runs) < count:
        raise ValueError("fewer completed runs than requested")
    ordered = sorted(runs, key=lambda run: (STAGES.index(run.stage), run.name))
    top = sorted(ordered, key=lambda run: run.best_score)[:count]
    if any(run.diverged for run in top):
        raise ValueError("fewer non-diverged runs than requested")
    return top


def seed_mean_choice(
    seed_scores: dict[tuple[str, str], list[float]],
) -> tuple[tuple[str, str], float]:
    """(run, EMA) with the lowest seed-mean score; keys are visited in insertion order.

    A diverged seed (+inf) makes its setting's mean +inf, so it is never chosen.
    """
    means = {key: _mean(values) for key, values in seed_scores.items()}
    key = min(means, key=means.__getitem__)
    if means[key] == inf:
        raise ValueError("every setting has a diverged seed")
    return key, means[key]


def paired_seed_mean_choice(
    real: dict[tuple[str, str], list[float]], grid: dict[tuple[str, str], list[float]]
) -> tuple[tuple[str, str], float]:
    """ReGSI: lowest mean of the real and the grid seed means of one shared setting."""
    if list(real) != list(grid):
        raise ValueError("ReGSI real and grid must score the same settings")
    return seed_mean_choice({key: [_mean(real[key]), _mean(grid[key])] for key in real})


def e_decision(
    axis1: AxisPosition,
    axis2: AxisPosition,
    weight_decay: AxisPosition,
    loss_at_min_wd: float | None,
    loss_at_max_wd: float | None,
    max_relative_increase: float,
) -> EDecision:
    """Apply the E rules in their pre-registered order; the first applicable one wins.

    1. wd calibration: the largest wd raises the final training loss of the smallest
       by less than ``max_relative_increase`` -> one wd step up. A loss of None
       (that wd level diverged) makes this rule inapplicable.
    2. edge extension: the winner lies at an end of axis 1, axis 2 or wd (in that
       priority) and the grid defines a value beyond that end.
    3. midpoint: between the winner and its better neighbour on axis 1, else axis 2,
       where the grid defines it.
    4. otherwise one wd step up.
    """
    if loss_at_min_wd is not None and loss_at_max_wd is not None:
        if not loss_at_min_wd > 0:
            raise ValueError("training loss at the smallest weight decay must be positive")
        if loss_at_max_wd / loss_at_min_wd - 1 < max_relative_increase:
            return EDecision("wd_calibration", weight_decay.name, _upper(weight_decay))
    for axis in (axis1, axis2, weight_decay):
        if axis.index is None:
            continue
        if axis.index == 0 and "lower" in axis.extensions:
            return EDecision("edge_extension", axis.name, "lower")
        if axis.index == axis.size - 1 and "upper" in axis.extensions:
            return EDecision("edge_extension", axis.name, "upper")
    for axis in (axis1, axis2):
        pair = _better_neighbour_pair(axis)
        if pair is not None and pair in axis.midpoints:
            return EDecision("midpoint", axis.name, pair)
    return EDecision("fallback_wd_extension", weight_decay.name, _upper(weight_decay))


def batch_aligned_fraction(nominal: float, observed_count: int, batch: int) -> float:
    """Fraction hiding the multiple of ``batch`` nearest ``nominal * observed_count``."""
    batches = max(1, round(nominal * observed_count / batch))
    return round(batch * batches / observed_count, 5)


def hidden_trace_count(observed_count: int, fraction: float) -> int:
    """Hidden traces per ReGSI episode, as the episode generator rounds them."""
    return round(observed_count * fraction)


def _better_neighbour_pair(axis: AxisPosition) -> tuple[int, int] | None:
    if axis.index is None:
        return None
    neighbours = [i for i in (axis.index - 1, axis.index + 1) if 0 <= i < axis.size]
    better = min(neighbours, key=lambda i: (axis.neighbour_scores[i], i))
    return (min(axis.index, better), max(axis.index, better))


def _upper(axis: AxisPosition) -> str:
    if "upper" not in axis.extensions:
        raise ValueError(f"{axis.name}: the grid defines no upper extension")
    return "upper"


def _mean(values: list[float]) -> float:
    if not values or not all(isfinite(value) or value == inf for value in values):
        raise ValueError("seed scores must be finite or +inf (diverged) and nonempty")
    return sum(values) / len(values)


def _reported(value: float) -> float | None:
    """JSON value of a score: a diverged +inf is recorded as null."""
    return None if value == inf else value


def is_divergence(failure_reason: str | None, pattern: str) -> bool:
    """Whether a failed tuning run diverged, by the pre-registered failure pattern."""
    return failure_reason is not None and re.search(pattern, failure_reason) is not None


def decay_key(decay: float) -> str:
    """Text key of one EMA shadow, as the tuner records it (e.g. ``0.9995``)."""
    return repr(float(decay))


def is_classical(spec: dict) -> bool:
    return "factors" in spec


def compose_stage(
    stage: str,
    grid: dict,
    spec: dict,
    planned: dict[str, PlannedRun],
    records: dict[str, RunRecord],
) -> tuple[list[PlannedRun], dict | None]:
    """Runs of one stage and the recorded rule decision, from the earlier stages only.

    ``planned`` and ``records`` hold every earlier run of this method; ``records``
    are the completed validation scores of the search method's runs.
    """
    if is_classical(spec):
        if stage != "S1":
            raise ValueError("POCS/DRR have a single stage S1")
        return _compose_classical(spec), None
    compose = {"S1": _compose_s1, "S2": _compose_s2, "S3": _compose_s3, "E": _compose_e}
    if stage == "R":
        return _compose_r(grid, spec, planned, records)
    return compose[stage](grid, spec, planned, records)


def select_final(
    grid: dict,
    spec: dict,
    planned: dict[str, PlannedRun],
    records: dict[str, RunRecord],
    reseed_decision: dict | None,
) -> tuple[dict, list[PlannedRun]]:
    """Selected (run, EMA) and its final retraining runs (search overrides, final seeds).

    Neural: lowest 3-seed mean over the reseeded top runs x EMA shadows; ReGSI takes
    the mean of its real and grid seed means. POCS/DRR: lowest validation score.
    """
    if is_classical(spec):
        run, key = best_pair(_stage_records(records, ("S1",)))
        choice = {"rule": "lowest_validation_score", "run": run.name, "score": run.scores[key]}
        return choice, [
            PlannedRun(f"final_{run.name}", spec["method_id"], planned[run.name].overrides)
        ]
    seeds = grid["seeds"]
    top = [row["run"] for row in reseed_decision["top_runs"]]
    keys = [decay_key(decay) for decay in grid["ema_eval_decays"]]
    real = {
        (name, key): [
            records[name].scores[key],
            *(records[f"r_{name}_seed{seed}"].scores[key] for seed in seeds["reseed"]),
        ]
        for name in top
        for key in keys
    }
    pair = spec.get("pair_method_id")
    if pair:
        paired = {
            (name, key): [
                records[f"r_{name}_grid_seed{seed}"].scores[key]
                for seed in (seeds["search"], *seeds["reseed"])
            ]
            for name, key in real
        }
        (name, key), score = paired_seed_mean_choice(real, paired)
        rule = "lowest_mean_of_real_and_grid_seed_means"
    else:
        paired = None
        (name, key), score = seed_mean_choice(real)
        rule = "lowest_seed_mean"
    choice = {
        "rule": rule,
        "run": name,
        "ema_decay": key,
        "score": score,
        "seed_scores": {
            f"{run}@{decay}": [_reported(v) for v in values]
            for (run, decay), values in real.items()
        },
    }
    if paired is not None:
        choice["grid_seed_scores"] = {
            f"{run}@{decay}": [_reported(v) for v in values]
            for (run, decay), values in paired.items()
        }
    methods = [(spec["method_id"], "")] + ([(pair, "_grid")] if pair else [])
    # Final runs predict the selected EMA only: no other shadow can be chosen after B-test.
    selected = {k: v for k, v in planned[name].overrides.items() if k != SHADOW_PATH}
    runs = [
        PlannedRun(
            f"final{infix}_{name}_ema{key}_seed{seed}",
            method_id,
            merge_overrides(selected, {"training.ema_decay": float(key), "model_seed": seed}),
        )
        for method_id, infix in methods
        for seed in seeds["final"]
    ]
    return choice, runs


def final_overrides(spec: dict, overrides: dict) -> dict:
    """Overrides for final retraining: ReGSI inner masks move to their paired final value."""
    pairs = spec.get("inner_mask_pairs")
    path = "training.inner_mask_fraction"
    if not pairs or path not in overrides:
        return overrides
    mapped = {pair["search"]: pair["final"] for pair in pairs}
    if overrides[path] not in mapped:
        raise ValueError(f"inner mask fraction {overrides[path]} has no pre-registered pair")
    return {**overrides, path: mapped[overrides[path]]}


def _compose_classical(spec):
    runs = []
    for values in product(*(factor["values"] for factor in spec["factors"])):
        overrides = merge_overrides(spec["fixed"], *(value["set"] for value in values))
        name = "_".join(value["id"] for value in values)
        runs.append(PlannedRun(name, spec["method_id"], overrides))
    return runs


def _neural_fixed(grid, spec):
    return merge_overrides(
        spec["fixed"],
        {
            SHADOW_PATH: list(grid["ema_eval_decays"]),
            "training.ema_snapshot_updates": list(grid["ema_snapshot_updates"]),
            "model_seed": grid["seeds"]["search"],
        },
    )


def _compose_s1(grid, spec, planned, records):
    fixed = _neural_fixed(grid, spec)
    runs = [
        PlannedRun(
            f"s1_{first['id']}_{second['id']}",
            spec["method_id"],
            merge_overrides(fixed, first["set"], second["set"]),
            {"axis1": i, "axis2": j},
        )
        for i, first in enumerate(spec["axis1"]["values"])
        for j, second in enumerate(spec["axis2"]["values"])
    ]
    return runs, None


def _compose_s2(grid, spec, planned, records):
    winner, key = best_pair(_stage_records(records, ("S1",)))
    base = planned[winner.name]
    axis1_id = spec["axis1"]["values"][base.axes["axis1"]]["id"]
    runs = [
        PlannedRun(
            f"s2_{option['id']}",
            spec["method_id"],
            merge_overrides(
                base.overrides, option["set"], option.get("by_axis1", {}).get(axis1_id, {})
            ),
        )
        for option in spec["s2"]
    ]
    return runs, {"rule": "s1_winner", "base": _pair_record(winner, key)}


def _compose_s3(grid, spec, planned, records):
    winner, key = best_pair(_stage_records(records, ("S1", "S2")))
    path = spec["weight_decay"]["path"]
    runs = [
        PlannedRun(
            f"s3_wd{value:g}",
            spec["method_id"],
            merge_overrides(planned[winner.name].overrides, {path: value}),
        )
        for value in spec["weight_decay"]["s3"]
    ]
    return runs, {"rule": "incumbent_after_s2", "base": _pair_record(winner, key)}


def _compose_e(grid, spec, planned, records):
    s1_winner, s1_key = best_pair(_stage_records(records, ("S1",)))
    s3_base, s3_key = best_pair(_stage_records(records, ("S1", "S2")))
    e_base, e_key = best_pair(_stage_records(records, ("S1", "S2", "S3")))
    base = planned[e_base.name].overrides
    axes = planned[s1_winner.name].axes
    positions = [
        _axis_position(spec, name, axes, base, planned, records) for name in ("axis1", "axis2")
    ]
    wd = spec["weight_decay"]
    path = wd["path"]
    levels = sorted({spec["fixed"][path], *wd["s3"]})
    extensions = frozenset(side for side, value in wd["extension"].items() if value is not None)
    wd_position = AxisPosition(
        "weight_decay", len(levels), levels.index(base[path]), {}, extensions
    )
    level_runs = {planned[s3_base.name].overrides[path]: s3_base.name} | {
        planned[run.name].overrides[path]: run.name for run in _stage_records(records, ("S3",))
    }
    ends = [records[level_runs[level]] for level in (levels[0], levels[-1])]
    if any(run.train_loss is None and not run.diverged for run in ends):
        raise ValueError("the wd calibration needs the training loss of every S3 level")
    losses = [run.train_loss for run in ends]
    decision = e_decision(
        *positions,
        wd_position,
        *losses,
        grid["selection"]["wd_calibration_max_relative_increase"],
    )
    if decision.axis == "weight_decay":
        value = wd["extension"][decision.target]
        value_id, overrides = f"wd{value:g}", {path: value}
    else:
        axis = spec[decision.axis]
        entry = (
            axis["extension"][decision.target]
            if decision.branch == "edge_extension"
            else next(m for m in axis["midpoints"] if tuple(m["between"]) == decision.target)
        )
        value_id, overrides = entry["id"], entry["set"]
    run = PlannedRun(f"e_{value_id}", spec["method_id"], merge_overrides(base, overrides))
    record = {
        "rule": "e_rule_order",
        "branch": decision.branch,
        "axis": decision.axis,
        "target": list(decision.target) if isinstance(decision.target, tuple) else decision.target,
        "s1_winner": _pair_record(s1_winner, s1_key),
        "s3_base": _pair_record(s3_base, s3_key),
        "e_base": _pair_record(e_base, e_key),
        "axis_positions": [_position_record(p) for p in (*positions, wd_position)],
        "weight_decay_levels": levels,
        "train_loss_at_min_wd": losses[0],
        "train_loss_at_max_wd": losses[1],
        "wd_level_runs": {f"{level:g}": name for level, name in sorted(level_runs.items())},
    }
    return [run], record


def _compose_r(grid, spec, planned, records):
    searched = _stage_records(records, ("S1", "S2", "S3", "E"))
    top = top_runs(searched, grid["selection"]["reseed_top_runs"])
    seeds = grid["seeds"]
    runs = [
        PlannedRun(
            f"r_{run.name}_seed{seed}",
            spec["method_id"],
            merge_overrides(planned[run.name].overrides, {"model_seed": seed}),
        )
        for run in top
        for seed in seeds["reseed"]
    ]
    pair = spec.get("pair_method_id")
    if pair:
        runs += [
            PlannedRun(
                f"r_{run.name}_grid_seed{seed}",
                pair,
                merge_overrides(planned[run.name].overrides, {"model_seed": seed}),
            )
            for run in top
            for seed in (seeds["search"], *seeds["reseed"])
        ]
    decision = {
        "rule": "top_runs_by_best_ema",
        "top_runs": [{"run": run.name, "best_score": run.best_score} for run in top],
    }
    return runs, decision


def _axis_position(spec, name, axes, base, planned, records):
    """E position of the S1 winner on one axis; None if unordered or changed after S1."""
    axis = spec[name]
    size = len(axis["values"])
    index = axes[name]
    value = axis["values"][index]["set"]
    if not axis["ordered"] or any(base.get(path) != v for path, v in value.items()):
        return AxisPosition(name, size, None)
    cells = {
        (run.axes["axis1"], run.axes["axis2"]): run.name for run in planned.values() if run.axes
    }
    neighbours = {}
    for other in (index - 1, index + 1):
        if 0 <= other < size:
            cell = (other, axes["axis2"]) if name == "axis1" else (axes["axis1"], other)
            neighbours[other] = records[cells[cell]].best_score
    extensions = frozenset(
        side for side, entry in (axis.get("extension") or {}).items() if entry is not None
    )
    midpoints = frozenset(tuple(entry["between"]) for entry in axis.get("midpoints", []))
    return AxisPosition(name, size, index, neighbours, extensions, midpoints)


def _position_record(position):
    return {
        "axis": position.name,
        "size": position.size,
        "index": position.index,
        "neighbour_scores": {str(k): _reported(v) for k, v in position.neighbour_scores.items()},
        "extensions": sorted(position.extensions),
        "midpoints": sorted(list(pair) for pair in position.midpoints),
    }


def _stage_records(records, stages):
    return [record for record in records.values() if record.stage in stages]


def _pair_record(run, key):
    return {"run": run.name, "ema_decay": key, "score": run.scores[key]}
