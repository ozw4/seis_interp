"""FORGE trace splits using the historical outer-mask and holdout generators."""

import numpy as np

from seis_interp.processing.forge_mvp_contract import make_outer_mask, make_validation_holdout


def partition_forge_traces(manifest, contract, seed):
    """Retain S1 namespaces; evaluation uses each seed for both independent hash rankings."""
    search = contract["trace_split"]["search_seed"]
    outer_seed = contract["mask_seed"] if seed == search else int(seed)
    outer = make_outer_mask(
        manifest,
        {**contract, "study_id": contract["mask_study_id"], "mask_seed": outer_seed},
    )
    result = make_validation_holdout(
        outer,
        {
            **contract,
            "study_id": contract["validation_study_id"],
            "validation_holdout": {**contract["validation_holdout"], "mask_seed": int(seed)},
        },
    )
    result["role"] = result.split.map(
        {"observed": "train", "test": "validation", "excluded": "test"}
    )
    result["trace_seed"] = int(seed)
    return result


def verify_forge_s1_partition(actual, reference):
    for role in ("train", "validation", "test"):
        if not np.array_equal(
            np.sort(actual.cell_id[actual.role.eq(role)]),
            np.sort(reference.cell_id[reference.role.eq(role)]),
        ):
            raise ValueError(f"FORGE S1 {role} cell IDs differ from study_066")
