from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seis_interp.processing.trace_splits import (
    SPLIT_COLUMN,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    assign_random_trace_splits,
)


def make_trace_table(trace_count: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "array_row": np.arange(100, 100 + trace_count, dtype=np.int64),
            "trace_index": np.arange(trace_count, dtype=np.int64),
        },
        index=np.arange(10, 10 + 2 * trace_count, 2),
    )


def assign(trace_table: pd.DataFrame, *, seed: int = 42) -> pd.DataFrame:
    return assign_random_trace_splits(
        trace_table,
        holdout_fraction=0.20,
        validation_fraction_of_holdout=0.25,
        random_seed=seed,
    )


def assignments_by_array_row(trace_table: pd.DataFrame) -> dict[int, str]:
    return dict(zip(trace_table["array_row"], trace_table[SPLIT_COLUMN], strict=True))


def test_assigns_expected_split_counts() -> None:
    result = assign(make_trace_table())

    assert result[SPLIT_COLUMN].value_counts().to_dict() == {
        TRAIN_SPLIT: 16,
        TEST_SPLIT: 3,
        VALIDATION_SPLIT: 1,
    }


def test_same_seed_produces_identical_assignments() -> None:
    trace_table = make_trace_table()

    pd.testing.assert_frame_equal(assign(trace_table), assign(trace_table))


def test_different_seeds_produce_different_assignments() -> None:
    trace_table = make_trace_table()

    assert assignments_by_array_row(assign(trace_table, seed=1)) != assignments_by_array_row(
        assign(trace_table, seed=2)
    )


def test_does_not_modify_input_and_preserves_its_rows_and_index() -> None:
    trace_table = make_trace_table()
    original = trace_table.copy(deep=True)

    result = assign(trace_table)

    pd.testing.assert_frame_equal(trace_table, original)
    pd.testing.assert_frame_equal(result.drop(columns=SPLIT_COLUMN), original)
    assert result.index.equals(original.index)


def test_membership_uses_array_row_not_dataframe_index_or_row_order() -> None:
    trace_table = make_trace_table().sample(frac=1.0, random_state=7)
    with_different_index = trace_table.copy()
    with_different_index.index = np.arange(1000, 1000 + len(trace_table))
    reversed_rows = with_different_index.iloc[::-1]

    first = assignments_by_array_row(assign(trace_table))
    second = assignments_by_array_row(assign(reversed_rows))

    assert first == second


def test_each_array_row_has_exactly_one_split() -> None:
    result = assign(make_trace_table())

    assert result["array_row"].is_unique
    assert not result[SPLIT_COLUMN].isna().any()
    assert set(result[SPLIT_COLUMN]) == {TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT}


@pytest.mark.parametrize("dtype", [str, float, object])
def test_non_integer_array_row_dtype_is_rejected(dtype: type[object]) -> None:
    trace_table = make_trace_table()
    trace_table["array_row"] = trace_table["array_row"].astype(dtype)

    with pytest.raises(ValueError, match="integer dtype"):
        assign(trace_table)


def test_missing_array_row_is_rejected() -> None:
    with pytest.raises(ValueError, match="array_row"):
        assign(make_trace_table().drop(columns="array_row"))


def test_empty_table_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        assign(pd.DataFrame({"array_row": pd.Series(dtype=np.int64)}))


@pytest.mark.parametrize(
    "name,value",
    [
        ("holdout_fraction", 0.0),
        ("holdout_fraction", 1.0),
        ("holdout_fraction", -0.1),
        ("holdout_fraction", np.nan),
        ("holdout_fraction", np.inf),
        ("holdout_fraction", True),
        ("validation_fraction_of_holdout", 0.0),
        ("validation_fraction_of_holdout", 1.0),
        ("validation_fraction_of_holdout", "0.25"),
    ],
)
def test_invalid_fractions_are_rejected(name: str, value: object) -> None:
    arguments: dict[str, object] = {
        "holdout_fraction": 0.20,
        "validation_fraction_of_holdout": 0.25,
        "random_seed": 42,
    }
    arguments[name] = value

    with pytest.raises(ValueError, match=name):
        assign_random_trace_splits(make_trace_table(), **arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize("random_seed", [True, 1.5, "42", -1])
def test_invalid_random_seed_is_rejected(random_seed: object) -> None:
    with pytest.raises(ValueError, match="random_seed"):
        assign_random_trace_splits(
            make_trace_table(),
            holdout_fraction=0.20,
            validation_fraction_of_holdout=0.25,
            random_seed=random_seed,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "trace_count,holdout_fraction,validation_fraction",
    [
        (2, 0.20, 0.25),
        (3, 0.34, 0.25),
        (4, 0.75, 0.10),
        (4, 0.25, 0.90),
    ],
)
def test_split_count_that_rounds_to_zero_is_rejected(
    trace_count: int,
    holdout_fraction: float,
    validation_fraction: float,
) -> None:
    with pytest.raises(ValueError, match="empty split"):
        assign_random_trace_splits(
            make_trace_table(trace_count),
            holdout_fraction=holdout_fraction,
            validation_fraction_of_holdout=validation_fraction,
            random_seed=42,
        )


def test_does_not_read_or_change_numpy_global_rng_state() -> None:
    np.random.seed(1234)
    expected = np.random.random(5)
    np.random.seed(1234)

    assign(make_trace_table())
    actual = np.random.random(5)

    np.testing.assert_array_equal(actual, expected)
