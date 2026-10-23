"""Trace assignments within one shared C3 crop, without spatial holdouts."""

import numpy as np
import pandas as pd

from seis_interp.processing.forge_mvp_contract import make_validation_holdout


def partition_c3_traces(parent: pd.DataFrame, contract: dict, seed: int):
    """Use the FORGE holdout rule: validation from observations, original test intact."""
    ordered = parent.sort_values("cell_id").reset_index(drop=True)
    if not np.array_equal(ordered.cell_id, np.arange(len(ordered))):
        raise ValueError("trace partition requires the complete C-ordered crop")
    holdout = {**contract["validation_holdout"], "mask_seed": int(seed)}
    result = make_validation_holdout(
        ordered,
        {
            **contract,
            "study_id": contract.get("validation_study_id", contract["study_id"]),
            "validation_holdout": holdout,
        },
    )
    result["outer_split"] = result.parent_split
    result["input_observed"] = result.split.eq("observed")
    result["trace_role"] = np.select(
        [result.split.eq("observed"), result.split.eq("test")], ["train", "validation"], "test"
    )
    result["trace_seed"] = int(seed)
    return result
