from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from seis_interp.processing.c3_benchmark_masks import (
    benchmark_case_config,
    effective_c3_crop_mask,
    validate_partition_mask_ffids,
)
from seis_interp.processing.c3_benchmark_partition import (
    audit_c3_benchmark_partition,
    c3_benchmark_partition_ranges,
)
from seis_interp.processing.interpolation_masks import (
    make_random_trace_mask,
    make_random_whole_ffid_mask,
)
from seis_interp.processing.trace_splits import assign_c3_source_line_block_splits
from tests.fixtures.c3_benchmark import synthetic_benchmark_cases, synthetic_benchmark_config
from tests.fixtures.c3_volume_artifacts import make_c3_trace_table


def test_existing_mask_sequences_and_effective_crop_counts():
    candidates = pd.DataFrame({"array_row": np.arange(40), "ffid": np.repeat(np.arange(10), 4)})
    for kind, maker in (
        ("random_trace", make_random_trace_mask),
        ("random_whole_ffid", make_random_whole_ffid_mask),
    ):
        first = maker(candidates, missing_fraction=0.5, random_seed=42)
        before = first.copy(deep=True)
        crop = candidates.loc[candidates.array_row.lt(13)]
        summary = effective_c3_crop_mask(crop, first, kind=kind)
        assert summary["trace_missing_fraction"] != 0.5
        pd.testing.assert_frame_equal(first, before)
        pd.testing.assert_frame_equal(
            first, maker(candidates, missing_fraction=0.5, random_seed=42)
        )
        validate_partition_mask_ffids(candidates, first, kind=kind)
    damaged = before.copy()
    damaged.loc[0, "observation_role"] = (
        "observed"
        if damaged.loc[1, "observation_role"] == "evaluation_target"
        else "evaluation_target"
    )
    with pytest.raises(ValueError, match="splits a candidate FFID"):
        validate_partition_mask_ffids(candidates, damaged, kind="random_whole_ffid")


@pytest.mark.parametrize("missing_fraction,random_seed", [(0.5, 42), (0.9, 99)])
def test_case_changes_drive_mask_config_without_mutating_training_settings(
    missing_fraction, random_seed
):
    config = synthetic_benchmark_config()
    before = deepcopy(config)
    inputs = {"cases": synthetic_benchmark_cases()}
    recipe = inputs["cases"][0]
    recipe.update(missing_fraction=missing_fraction, random_seed=random_seed)
    result = benchmark_case_config(config, recipe, {"time": [0, 4]})
    assert result["project"]["random_seed"] == random_seed
    assert result["interpolation_mask"] == {
        "partition": "test",
        "kind": "random_trace",
        "missing_fraction": missing_fraction,
    }
    assert config == before
    assert config["project"]["random_seed"] == 42
    assert config["c3_benchmark"]["training"]["random_seeds"] == [71]


def test_partition_fixed_test_and_crop_membership():
    table = make_c3_trace_table(physical_source_line_indices=(0, 1, 2, 3, 4))
    ranges = c3_benchmark_partition_ranges((1, 3), 5)
    split = assign_c3_source_line_block_splits(table, source_line_ranges=ranges)[
        ["array_row", "split"]
    ]
    preparation = {
        "split_scope": "c3_source_line_blocks",
        "source_line_ranges": ranges,
        "split_counts": {"train": 1632, "test": 3264, "validation": 3264},
        "ffid_split_counts": {"train": 3, "test": 6, "validation": 6},
    }
    crops = {
        "test": table.loc[table.source_x_m.between(1160, 1320)],
        "validation": table.loc[table.source_x_m.ge(1480)],
    }
    pool, summary = audit_c3_benchmark_partition(
        table, split, preparation, crops, time_range=(0, 4)
    )
    assert len(pool) == 3 * 544
    assert summary["training_time_samples"] == [0, 4]
    crossed = {**crops, "test": pd.concat([crops["test"], table.iloc[[0]]])}
    with pytest.raises(ValueError, match="outside its canonical partition"):
        audit_c3_benchmark_partition(table, split, preparation, crossed, time_range=(0, 4))
    damaged = split.copy()
    damaged.loc[0, "split"] = "test"
    with pytest.raises(ValueError):
        audit_c3_benchmark_partition(table, damaged, preparation, crops, time_range=(0, 4))
    with pytest.raises(ValueError, match="empty"):
        c3_benchmark_partition_ranges((1, 5), 5)
