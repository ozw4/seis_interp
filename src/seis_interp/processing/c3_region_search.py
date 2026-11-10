"""The fixed 25-candidate search and its three additional evaluations."""

import json
import math
from copy import deepcopy
from itertools import product

from seis_interp.c3_fair_grid_config import c3_experiment_methods
from seis_interp.processing.drr_grid import resolve_drr_grid_config
from seis_interp.processing.fair_grid_rules import apply_overrides
from seis_interp.processing.forge_mvp_contract import validation_holdout_counts


def c3_search_candidates(base: dict, axes: dict) -> list[dict]:
    """Expand two explicit five-value axes, keeping stable configuration IDs."""
    for name in ("axis_a", "axis_b"):
        values = axes[name]
        encoded = [json.dumps(value["set"], sort_keys=True) for value in values]
        if len(values) != 5 or len(set(encoded)) != 5:
            raise ValueError(f"{name} must contain five distinct explicit candidates")
    candidates = []
    for (a, first), (b, second) in product(enumerate(axes["axis_a"]), enumerate(axes["axis_b"])):
        if set(first["set"]) & set(second["set"]):
            raise ValueError("search axes must override different parameters")
        config = apply_overrides(base, first["set"] | second["set"])
        candidates.append(
            {
                "id": f"a{a + 1:02d}_b{b + 1:02d}",
                "axis_a": first["id"],
                "axis_b": second["id"],
                "config": config,
            }
        )
    return candidates


def select_c3_candidate(candidates: list[dict], records: list[dict], *, expected=25) -> dict:
    """Select only after all 25 runs finish, using V then the configuration ID."""
    ids = {candidate["id"] for candidate in candidates}
    if (
        expected < 1
        or len(ids) != expected
        or len(candidates) != expected
        or len(records) != expected
        or {r["id"] for r in records} != ids
    ):
        raise ValueError(f"selection requires all {expected} distinct S1 outcomes")
    if any(record["status"] not in ("completed", "diverged") for record in records):
        raise ValueError("selection requires final completed or diverged S1 outcomes")
    winner = min(
        records,
        key=lambda record: (
            record["validation_score"]
            if record["status"] == "completed" and math.isfinite(record["validation_score"])
            else math.inf,
            record["id"],
        ),
    )
    return deepcopy(next(candidate for candidate in candidates if candidate["id"] == winner["id"]))


def resolve_c3_candidate(config: dict, observed_count: int, traces_per_update: int) -> dict:
    """Resolve ReGSI mask rounding and the DRR backend/boundary contract."""
    if config["method_id"] == "drr_grid":
        return resolve_drr_grid_config(config)
    resolved = deepcopy(config)
    if config["method_id"] == "regsi_real":
        nominal = config["training"]["inner_mask_fraction"]
        hidden = traces_per_update * math.floor(nominal * observed_count / traces_per_update)
        if hidden < traces_per_update or hidden >= observed_count:
            raise ValueError("ReGSI train observations cannot supply one complete query batch")
        # Whole-gather episodes quantize the hidden count by gather size, so rounding it to
        # a query batch would move the declared gather fraction instead of the batch fill.
        if config["training"].get("episode_kind", "random_trace") == "random_trace":
            resolved["training"]["inner_mask_fraction"] = hidden / observed_count
    return resolved


def c3_split_settings(contract: dict) -> tuple[str, dict]:
    """Resolve the explicit region or trace assignment used by this experiment."""
    kinds = [kind for kind in ("region", "trace") if f"{kind}_split" in contract]
    if len(kinds) != 1:
        raise ValueError("exactly one region_split or trace_split is required")
    kind = kinds[0]
    return kind, contract[f"{kind}_split"]


def c3_reference_configuration(candidates: list[dict], settings: dict) -> str:
    """Require one of the 25 candidates to contain every declared reference setting."""
    matches = [
        candidate["id"]
        for candidate in candidates
        if apply_overrides(candidate["config"], settings) == candidate["config"]
    ]
    if not matches:
        raise ValueError("the historical reference settings are absent from the 25 candidates")
    return min(matches)


def validate_c3_experiment_plan(contract: dict, grid: dict) -> None:
    """Require five methods, one S1 assignment and three additional seeds."""
    kind, specification = c3_split_settings(contract)
    if set(grid["methods"]) != set(c3_experiment_methods(contract)):
        raise ValueError(f"{kind} experiment requires exactly the five C3 methods")
    seeds = specification["evaluation_seeds"]
    if len(seeds) != 3 or len(set(seeds)) != 3 or specification["search_seed"] in seeds:
        raise ValueError(f"three distinct additional {kind} seeds are required")
    if kind == "trace":
        validation_holdout_counts(contract)
        if contract["validation_holdout"]["mask_seed"] != specification["search_seed"]:
            raise ValueError("S1 trace seed must match the validation holdout seed")
