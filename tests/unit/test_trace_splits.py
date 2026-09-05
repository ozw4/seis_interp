from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seis_interp.processing.trace_splits import (
    C3_SOURCE_LINE_BLOCKS_SPLIT_SCOPE,
    SPLIT_COLUMN,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    assign_c3_source_line_block_splits,
    assign_random_trace_splits,
    assign_random_trace_splits_by_ffid,
    assign_random_whole_ffid_splits,
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


def make_multi_ffid_trace_table(trace_counts: tuple[int, ...] = (20, 20)) -> pd.DataFrame:
    ffids = tuple(100 + index for index in range(len(trace_counts)))
    return pd.DataFrame(
        {
            "array_row": np.arange(sum(trace_counts), dtype=np.int64),
            "trace_index": np.arange(sum(trace_counts), dtype=np.int64),
            "ffid": np.repeat(ffids, trace_counts).astype(np.int64),
        }
    )


def assign_by_ffid(trace_table: pd.DataFrame, *, seed: int = 42) -> pd.DataFrame:
    return assign_random_trace_splits_by_ffid(
        trace_table,
        holdout_fraction=0.20,
        validation_fraction_of_holdout=0.25,
        random_seed=seed,
    )


def assign_whole_ffids(trace_table: pd.DataFrame, *, seed: int = 42) -> pd.DataFrame:
    return assign_random_whole_ffid_splits(
        trace_table,
        holdout_fraction=0.75,
        validation_fraction_of_holdout=0.25,
        random_seed=seed,
    )


def make_c3_source_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    array_row = 0
    for source_line, source_x in enumerate((0.0, 160.0, 320.0)):
        source_y_origin = 40.0 if source_line % 2 else 0.0
        for shot in range(2):
            source_y = source_y_origin + 80.0 * shot
            ffid = 100 + source_line * 2 + shot
            for receiver_x_offset in (-140.0, -100.0):
                rows.append(
                    {
                        "array_row": array_row,
                        "ffid": ffid,
                        "source_x_m": source_x,
                        "source_y_m": source_y,
                        "receiver_x_m": source_x + receiver_x_offset,
                        "receiver_y_m": source_y - 2680.0,
                    }
                )
                array_row += 1
    return pd.DataFrame(rows)


def assign_source_line_blocks(trace_table: pd.DataFrame) -> pd.DataFrame:
    return assign_c3_source_line_block_splits(
        trace_table,
        source_line_ranges={
            TRAIN_SPLIT: (0, 1),
            VALIDATION_SPLIT: (1, 2),
            TEST_SPLIT: (2, 3),
        },
    )


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


def test_per_ffid_split_assigns_all_three_splits_within_every_ffid() -> None:
    result = assign_by_ffid(make_multi_ffid_trace_table())

    counts = result.groupby(["ffid", SPLIT_COLUMN]).size().unstack(fill_value=0)

    assert counts.to_dict(orient="index") == {
        100: {TEST_SPLIT: 3, TRAIN_SPLIT: 16, VALIDATION_SPLIT: 1},
        101: {TEST_SPLIT: 3, TRAIN_SPLIT: 16, VALIDATION_SPLIT: 1},
    }


@pytest.mark.parametrize(
    ("trace_count", "expected_counts"),
    [
        (544, {TRAIN_SPLIT: 435, VALIDATION_SPLIT: 27, TEST_SPLIT: 82}),
        (112, {TRAIN_SPLIT: 90, VALIDATION_SPLIT: 6, TEST_SPLIT: 16}),
    ],
)
def test_per_ffid_split_uses_the_established_count_rule(
    trace_count: int,
    expected_counts: dict[str, int],
) -> None:
    result = assign_by_ffid(make_multi_ffid_trace_table((trace_count,)))

    assert result[SPLIT_COLUMN].value_counts().to_dict() == expected_counts


def test_per_ffid_membership_is_independent_of_table_order_and_index() -> None:
    trace_table = make_multi_ffid_trace_table().sample(frac=1.0, random_state=7)
    reordered = trace_table.iloc[::-1].copy()
    reordered.index = np.arange(1000, 1000 + len(reordered))

    assert assignments_by_array_row(assign_by_ffid(trace_table)) == assignments_by_array_row(
        assign_by_ffid(reordered)
    )


def test_adding_an_ffid_does_not_change_existing_per_ffid_membership() -> None:
    original = make_multi_ffid_trace_table((20, 20))
    extended = make_multi_ffid_trace_table((20, 20, 20))

    original_assignments = assignments_by_array_row(assign_by_ffid(original))
    extended_assignments = assignments_by_array_row(assign_by_ffid(extended))

    assert {row: extended_assignments[row] for row in original_assignments} == original_assignments


def test_per_ffid_seed_changes_membership() -> None:
    trace_table = make_multi_ffid_trace_table()

    assert assignments_by_array_row(
        assign_by_ffid(trace_table, seed=1)
    ) != assignments_by_array_row(assign_by_ffid(trace_table, seed=2))


def test_per_ffid_split_error_identifies_too_small_ffid_and_counts() -> None:
    trace_table = make_multi_ffid_trace_table((20, 3))

    with pytest.raises(ValueError, match=r"FFID 101.*counts="):
        assign_by_ffid(trace_table)


@pytest.mark.parametrize(
    "ffids",
    [
        pd.Series([100.0] * 20 + [101.0] * 20, dtype=np.float64),
        pd.Series([100] * 39 + [None], dtype="Int64"),
        pd.Series([True] * 20 + [False] * 20, dtype=bool),
    ],
)
def test_per_ffid_split_rejects_invalid_ffid_values(ffids: pd.Series) -> None:
    trace_table = make_multi_ffid_trace_table()
    trace_table["ffid"] = ffids

    with pytest.raises(ValueError, match="ffid"):
        assign_by_ffid(trace_table)


def test_per_ffid_split_accepts_signed_integer_ffids_deterministically() -> None:
    trace_table = make_multi_ffid_trace_table()
    trace_table["ffid"] = np.repeat([-100, 101], 20).astype(np.int64)

    first = assignments_by_array_row(assign_by_ffid(trace_table))
    second = assignments_by_array_row(assign_by_ffid(trace_table))

    assert first == second


def test_whole_ffid_split_assigns_each_ffid_to_exactly_one_split() -> None:
    result = assign_whole_ffids(make_multi_ffid_trace_table((2,) * 8))

    splits_per_ffid = result.groupby("ffid")[SPLIT_COLUMN].nunique()
    ffid_counts = result.groupby(SPLIT_COLUMN)["ffid"].nunique().to_dict()

    assert splits_per_ffid.eq(1).all()
    assert ffid_counts == {
        TEST_SPLIT: 4,
        TRAIN_SPLIT: 2,
        VALIDATION_SPLIT: 2,
    }


def test_whole_ffid_membership_is_independent_of_trace_order_and_index() -> None:
    trace_table = make_multi_ffid_trace_table((2,) * 8).sample(frac=1.0, random_state=7)
    reordered = trace_table.iloc[::-1].copy()
    reordered.index = np.arange(1000, 1000 + len(reordered))

    assert assignments_by_array_row(assign_whole_ffids(trace_table)) == assignments_by_array_row(
        assign_whole_ffids(reordered)
    )


def test_whole_ffid_seed_changes_membership() -> None:
    trace_table = make_multi_ffid_trace_table((2,) * 8)

    first = assign_whole_ffids(trace_table, seed=1)
    second = assign_whole_ffids(trace_table, seed=2)

    assert assignments_by_array_row(first) != assignments_by_array_row(second)
    assert first.groupby("ffid")[SPLIT_COLUMN].nunique().eq(1).all()


def test_whole_ffid_split_error_identifies_too_few_eligible_ffids() -> None:
    with pytest.raises(ValueError, match=r"eligible FFIDs.*counts="):
        assign_whole_ffids(make_multi_ffid_trace_table((20, 20, 20)))


def test_global_exact_membership_remains_unchanged() -> None:
    result = assignments_by_array_row(assign(make_trace_table()))

    assert result == {
        100: TRAIN_SPLIT,
        101: TRAIN_SPLIT,
        102: TRAIN_SPLIT,
        103: TRAIN_SPLIT,
        104: TRAIN_SPLIT,
        105: TRAIN_SPLIT,
        106: TRAIN_SPLIT,
        107: TEST_SPLIT,
        108: TRAIN_SPLIT,
        109: TEST_SPLIT,
        110: TRAIN_SPLIT,
        111: TRAIN_SPLIT,
        112: TRAIN_SPLIT,
        113: TRAIN_SPLIT,
        114: TEST_SPLIT,
        115: VALIDATION_SPLIT,
        116: TRAIN_SPLIT,
        117: TRAIN_SPLIT,
        118: TRAIN_SPLIT,
        119: TRAIN_SPLIT,
    }


def test_c3_source_line_blocks_assign_complete_lines_and_ffids() -> None:
    result = assign_source_line_blocks(make_c3_source_table())

    split_by_x = result.groupby("source_x_m")[SPLIT_COLUMN].unique().map(list).to_dict()

    assert split_by_x == {
        0.0: [TRAIN_SPLIT],
        160.0: [VALIDATION_SPLIT],
        320.0: [TEST_SPLIT],
    }
    assert result.groupby("ffid")[SPLIT_COLUMN].nunique().eq(1).all()


def test_c3_source_line_block_membership_is_independent_of_row_order_and_index() -> None:
    trace_table = make_c3_source_table().sample(frac=1.0, random_state=7)
    reordered = trace_table.iloc[::-1].copy()
    reordered.index = np.arange(1000, 1000 + len(reordered))

    assert assignments_by_array_row(
        assign_source_line_blocks(trace_table)
    ) == assignments_by_array_row(assign_source_line_blocks(reordered))


def test_c3_source_line_blocks_reject_an_ffid_crossing_a_block_boundary() -> None:
    trace_table = make_c3_source_table()
    crossing_shot = trace_table["source_x_m"].eq(320.0) & trace_table["source_y_m"].eq(0.0)
    trace_table.loc[crossing_shot, "ffid"] = 100

    with pytest.raises(ValueError, match="assign each FFID wholly"):
        assign_source_line_blocks(trace_table)


@pytest.mark.parametrize(
    ("ranges", "message"),
    [
        ({TRAIN_SPLIT: (0, 1), VALIDATION_SPLIT: (1, 2)}, "exactly"),
        (
            {
                TRAIN_SPLIT: (0, 1),
                VALIDATION_SPLIT: (1, 2),
                TEST_SPLIT: (2, 3),
                "holdout": (3, 4),
            },
            "exactly",
        ),
        (
            {
                TRAIN_SPLIT: (0, 1),
                VALIDATION_SPLIT: (1, 2),
                TEST_SPLIT: (2, 2),
            },
            "less than stop",
        ),
        (
            {
                TRAIN_SPLIT: (0, 2),
                VALIDATION_SPLIT: (1, 2),
                TEST_SPLIT: (2, 3),
            },
            "overlap",
        ),
        (
            {
                TRAIN_SPLIT: (0, 1),
                VALIDATION_SPLIT: (2, 3),
                TEST_SPLIT: (3, 4),
            },
            "gaps",
        ),
        (
            {
                TRAIN_SPLIT: (1, 2),
                VALIDATION_SPLIT: (2, 3),
                TEST_SPLIT: (3, 4),
            },
            "start at source line 0",
        ),
        (
            {
                TRAIN_SPLIT: (0, 1),
                VALIDATION_SPLIT: (1, 2),
                TEST_SPLIT: (2, 4),
            },
            "cover every source line",
        ),
    ],
)
def test_c3_source_line_blocks_reject_invalid_ranges(
    ranges: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        assign_c3_source_line_block_splits(
            make_c3_source_table(),
            source_line_ranges=ranges,  # type: ignore[arg-type]
        )


def test_c3_source_line_blocks_scope_is_public() -> None:
    assert C3_SOURCE_LINE_BLOCKS_SPLIT_SCOPE == "c3_source_line_blocks"
