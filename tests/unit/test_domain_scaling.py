from __future__ import annotations

import numpy as np
import pytest

from seis_interp.pipelines.domain_scaling import deterministic_nested_trace_subsets


def test_deterministic_nested_trace_subsets_are_stable_and_order_independent() -> None:
    training_rows = np.asarray([11, 3, 8, 1, 14, 6, 20, 2], dtype=np.int64)
    counts = [1, 3, 8]

    first = deterministic_nested_trace_subsets(training_rows, counts, random_seed=42)
    repeated = deterministic_nested_trace_subsets(training_rows, counts, random_seed=42)
    reordered = deterministic_nested_trace_subsets(training_rows[::-1], counts, random_seed=42)

    for count in counts:
        np.testing.assert_array_equal(first[count], repeated[count])
        np.testing.assert_array_equal(first[count], reordered[count])
    np.testing.assert_array_equal(first[1], first[3][:1])
    np.testing.assert_array_equal(first[3], first[8][:3])
    assert set(first[8]) == set(training_rows)


def test_deterministic_nested_trace_subsets_reject_too_many_rows() -> None:
    with pytest.raises(ValueError, match="exceeds 3 available"):
        deterministic_nested_trace_subsets(
            np.asarray([1, 2, 3], dtype=np.int64),
            [1, 4],
            random_seed=42,
        )
