"""Atomic source blocks over a fixed C3 observation mask."""

import numpy as np
import pandas as pd

from seis_interp.processing.forge_mvp_contract import stable_cell_hashes

REGION_ROLES = ("train", "validation", "test")


def partition_c3_regions(parent: pd.DataFrame, shape, specification: dict, seed: int):
    """Rank source blocks once; receivers and complete profiles stay in one role.

    The outer mask is unchanged. Only outer observations in train blocks are
    visible. Every cell of a validation/test block is a scoring target.
    """
    lines, shots, rx, ry = (int(n) for n in shape)
    block_lines, block_shots = specification["source_block_shape"]
    if block_lines < 1 or block_shots < 1 or lines % block_lines or shots % block_shots:
        raise ValueError("source block shape must divide the source grid")
    result = parent.sort_values("cell_id").reset_index(drop=True).copy()
    if not np.array_equal(result.cell_id, np.arange(lines * shots * rx * ry)):
        raise ValueError("region partition requires the complete C-ordered crop")
    source = result.cell_id.to_numpy() // (rx * ry)
    blocks_per_line = shots // block_shots
    blocks = (source // shots // block_lines) * blocks_per_line + source % shots // block_shots
    count = lines // block_lines * blocks_per_line
    counts = specification["block_counts"]
    if set(counts) != set(REGION_ROLES) or any(counts[role] < 1 for role in REGION_ROLES):
        raise ValueError("each region role needs a positive block count")
    if sum(counts.values()) != count:
        raise ValueError("region block counts must cover the source grid")
    hashes = stable_cell_hashes(["c3_region_split", int(seed)], range(count))
    order = sorted(range(count), key=lambda block: (hashes[block], block))
    roles = np.empty(count, dtype=object)
    offset = 0
    for role in REGION_ROLES:
        roles[order[offset : offset + counts[role]]] = role
        offset += counts[role]
    result["outer_split"] = result.split
    result["block_id"] = blocks
    result["region_role"] = roles[blocks]
    result["region_seed"] = int(seed)
    result["input_observed"] = result.outer_split.eq("observed") & result.region_role.eq("train")
    return result


def region_scoring_mask(
    partition: pd.DataFrame, domains, role_column="region_role"
) -> pd.DataFrame:
    """The existing observed/test/excluded runtime contract for selected domains."""
    if not domains or not set(domains) <= {"validation", "test"}:
        raise ValueError("scoring domains must be validation and/or test")
    result = partition.copy()
    result["split"] = "excluded"
    result.loc[result.input_observed, "split"] = "observed"
    result.loc[result[role_column].isin(domains), "split"] = "test"
    return result
