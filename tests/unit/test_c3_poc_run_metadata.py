"""Comparison fields use target-only counts and fixed keys for every method."""

import numpy as np
import pytest

from seis_interp.c3_poc_run_records import (
    poc_artifact_metadata,
    poc_compute_metadata,
    poc_coverage_metadata,
)


@pytest.mark.parametrize("missing", [None, (1, 1, 1, 1), (0, 1, 1, 1)])
def test_coverage_counts_target_traces_samples_and_boundary(missing):
    target = np.ones((3, 3, 3, 3), dtype=bool)
    target[2, 2, 2, 2] = False
    covered = np.ones_like(target)
    covered[2, 2, 2, 2] = False
    if missing is not None:
        covered[missing] = False
    count = int(missing is not None)
    assert poc_coverage_metadata(target, covered, time_sample_count=8) == {
        "target_trace_count": 80,
        "covered_target_trace_count": 80 - count,
        "target_coverage_fraction": (80 - count) / 80,
        "uncovered_trace_count": count,
        "uncovered_sample_count": count * 8,
        "complete": count == 0,
        "boundary_targets_included": missing != (0, 1, 1, 1),
    }


def test_compute_and_artifact_keys_do_not_depend_on_method():
    assert poc_compute_metadata() == {
        "parameter_count": None,
        "optimizer_updates": None,
        "supervised_trace_presentations": None,
    }
    assert poc_compute_metadata(
        parameter_count=15, optimizer_updates=2, supervised_trace_presentations=7
    ) == {
        "parameter_count": 15,
        "optimizer_updates": 2,
        "supervised_trace_presentations": 7,
    }
    prediction = {"artifact": "prediction.npy", "sha256": "a" * 64, "shape": [2, 3]}
    for checkpoint in (None, {"artifact": "final.pt", "sha256": "b" * 64}):
        assert poc_artifact_metadata(prediction=prediction, checkpoint=checkpoint) == {
            "prediction": {"path": "prediction.npy", "sha256": "a" * 64},
            "checkpoint": None if checkpoint is None else {"path": "final.pt", "sha256": "b" * 64},
        }
