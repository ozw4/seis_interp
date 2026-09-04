from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from seis_interp.processing.interpolation_masks import (
    EVALUATION_TARGET_ROLE,
    MASK_KINDS,
    OBSERVATION_ROLE_COLUMN,
    OBSERVATION_ROLES,
    OBSERVED_ROLE,
    RANDOM_TRACE_MASK_KIND,
    RANDOM_WHOLE_FFID_MASK_KIND,
    make_random_trace_mask,
    make_random_whole_ffid_mask,
    validate_interpolation_mask,
)


def _trace_candidates(trace_count: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "array_row": np.arange(100, 100 + trace_count, dtype=np.int64),
            "trace_index": np.arange(trace_count, dtype=np.int64),
        },
        index=np.arange(10, 10 + 2 * trace_count, 2),
    )


def _whole_ffid_candidates(ffid_count: int = 6, traces_per_ffid: int = 3) -> pd.DataFrame:
    trace_count = ffid_count * traces_per_ffid
    return pd.DataFrame(
        {
            "array_row": np.arange(200, 200 + trace_count, dtype=np.int64),
            "trace_index": np.arange(trace_count, dtype=np.int64),
            "ffid": np.repeat(
                np.arange(1000, 1000 + ffid_count, dtype=np.int64),
                traces_per_ffid,
            ),
        }
    )


def _trace_mask(table: pd.DataFrame, *, seed: int = 42) -> pd.DataFrame:
    return make_random_trace_mask(table, missing_fraction=0.25, random_seed=seed)


def _whole_ffid_mask(table: pd.DataFrame, *, seed: int = 42) -> pd.DataFrame:
    return make_random_whole_ffid_mask(table, missing_fraction=0.5, random_seed=seed)


def _roles_by_array_row(mask: pd.DataFrame) -> dict[int, str]:
    return dict(
        zip(
            mask["array_row"].to_numpy(),
            mask[OBSERVATION_ROLE_COLUMN].to_numpy(),
            strict=True,
        )
    )


def test_public_constant_order_is_fixed() -> None:
    assert OBSERVATION_ROLES == (OBSERVED_ROLE, EVALUATION_TARGET_ROLE)
    assert MASK_KINDS == (RANDOM_TRACE_MASK_KIND, RANDOM_WHOLE_FFID_MASK_KIND)


def test_random_trace_mask_has_exact_columns_sorted_rows_and_expected_counts() -> None:
    result = _trace_mask(_trace_candidates())

    assert result.columns.tolist() == ["array_row", OBSERVATION_ROLE_COLUMN]
    assert result["array_row"].tolist() == list(range(100, 120))
    assert result[OBSERVATION_ROLE_COLUMN].value_counts().to_dict() == {
        OBSERVED_ROLE: 15,
        EVALUATION_TARGET_ROLE: 5,
    }


def test_random_whole_ffid_mask_never_splits_an_ffid() -> None:
    candidates = _whole_ffid_candidates()

    result = _whole_ffid_mask(candidates)
    joined = candidates[["array_row", "ffid"]].merge(
        result,
        on="array_row",
        validate="one_to_one",
    )

    assert joined.groupby("ffid")[OBSERVATION_ROLE_COLUMN].nunique().eq(1).all()
    assert joined.groupby(OBSERVATION_ROLE_COLUMN)["ffid"].nunique().to_dict() == {
        EVALUATION_TARGET_ROLE: 3,
        OBSERVED_ROLE: 3,
    }


@pytest.mark.parametrize(
    ("maker", "candidates"),
    [
        (_trace_mask, _trace_candidates()),
        (_whole_ffid_mask, _whole_ffid_candidates()),
    ],
)
def test_same_seed_produces_identical_masks(
    maker: Callable[..., pd.DataFrame],
    candidates: pd.DataFrame,
) -> None:
    pd.testing.assert_frame_equal(maker(candidates, seed=7), maker(candidates, seed=7))


@pytest.mark.parametrize(
    ("maker", "candidates"),
    [
        (_trace_mask, _trace_candidates()),
        (_whole_ffid_mask, _whole_ffid_candidates(ffid_count=10)),
    ],
)
def test_different_seeds_normally_produce_different_masks(
    maker: Callable[..., pd.DataFrame],
    candidates: pd.DataFrame,
) -> None:
    assert _roles_by_array_row(maker(candidates, seed=1)) != _roles_by_array_row(
        maker(candidates, seed=2)
    )


@pytest.mark.parametrize(
    ("maker", "candidates"),
    [
        (_trace_mask, _trace_candidates()),
        (_whole_ffid_mask, _whole_ffid_candidates()),
    ],
)
def test_mask_mapping_is_independent_of_input_order_and_index(
    maker: Callable[..., pd.DataFrame],
    candidates: pd.DataFrame,
) -> None:
    reordered = candidates.sample(frac=1.0, random_state=13).copy()
    reordered.index = np.arange(500, 500 + len(reordered))

    assert _roles_by_array_row(maker(candidates)) == _roles_by_array_row(maker(reordered))


@pytest.mark.parametrize(
    ("maker", "candidates"),
    [
        (_trace_mask, _trace_candidates()),
        (_whole_ffid_mask, _whole_ffid_candidates()),
    ],
)
def test_mask_generation_does_not_modify_input(
    maker: Callable[..., pd.DataFrame],
    candidates: pd.DataFrame,
) -> None:
    original = candidates.copy(deep=True)

    maker(candidates)

    pd.testing.assert_frame_equal(candidates, original)


@pytest.mark.parametrize("missing_fraction", [0.0, 1.0, -0.1, np.nan, np.inf, True, "0.5"])
def test_invalid_missing_fraction_is_rejected(missing_fraction: object) -> None:
    with pytest.raises(ValueError, match="missing_fraction"):
        make_random_trace_mask(
            _trace_candidates(),
            missing_fraction=missing_fraction,  # type: ignore[arg-type]
            random_seed=42,
        )


@pytest.mark.parametrize("random_seed", [-1, 1.5, True, "42"])
def test_invalid_random_seed_is_rejected(random_seed: object) -> None:
    with pytest.raises(ValueError, match="random_seed"):
        make_random_trace_mask(
            _trace_candidates(),
            missing_fraction=0.25,
            random_seed=random_seed,  # type: ignore[arg-type]
        )


def test_duplicate_array_row_is_rejected() -> None:
    candidates = _trace_candidates()
    candidates.loc[candidates.index[-1], "array_row"] = candidates.iloc[0]["array_row"]

    with pytest.raises(ValueError, match="duplicate array_row"):
        _trace_mask(candidates)


@pytest.mark.parametrize(
    "ffids",
    [
        pd.Series([1000.0] * 9 + [1001.0] * 9, dtype=np.float64),
        pd.Series([1000] * 17 + [None], dtype="Int64"),
        pd.Series([True] * 9 + [False] * 9, dtype=bool),
    ],
)
def test_invalid_ffids_are_rejected(ffids: pd.Series) -> None:
    candidates = _whole_ffid_candidates()
    candidates["ffid"] = ffids

    with pytest.raises(ValueError, match="ffid"):
        _whole_ffid_mask(candidates)


def test_whole_ffid_mask_requires_ffid_column() -> None:
    with pytest.raises(ValueError, match="ffid"):
        _whole_ffid_mask(_whole_ffid_candidates().drop(columns="ffid"))


@pytest.mark.parametrize(
    ("candidates", "fraction"),
    [
        (_trace_candidates(trace_count=1), 0.5),
        (_trace_candidates(trace_count=2), 0.99),
    ],
)
def test_trace_mask_rejects_counts_that_leave_a_role_empty(
    candidates: pd.DataFrame,
    fraction: float,
) -> None:
    with pytest.raises(ValueError, match="at least one observed and one evaluation target"):
        make_random_trace_mask(candidates, missing_fraction=fraction, random_seed=42)


def test_whole_ffid_mask_rejects_one_unique_ffid() -> None:
    with pytest.raises(ValueError, match="at least one observed and one evaluation target"):
        _whole_ffid_mask(_whole_ffid_candidates(ffid_count=1))


def test_generator_does_not_change_numpy_global_rng_state() -> None:
    np.random.seed(1234)
    expected = np.random.random(5)
    np.random.seed(1234)

    _trace_mask(_trace_candidates())
    actual = np.random.random(5)

    np.testing.assert_array_equal(actual, expected)


def _valid_mask() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "array_row": np.asarray([4, 8, 12], dtype=np.int64),
            OBSERVATION_ROLE_COLUMN: [OBSERVED_ROLE, EVALUATION_TARGET_ROLE, OBSERVED_ROLE],
        }
    )


def test_validator_accepts_expected_rows_in_a_different_order() -> None:
    validate_interpolation_mask(
        _valid_mask(),
        expected_array_rows=np.asarray([12, 4, 8], dtype=np.int64),
    )


@pytest.mark.parametrize(
    "invalid_mask",
    [
        _valid_mask().assign(extra=True),
        _valid_mask().assign(observation_role=[OBSERVED_ROLE, "unknown", OBSERVED_ROLE]),
        _valid_mask().assign(observation_role=OBSERVED_ROLE),
    ],
)
def test_validator_rejects_invalid_columns_or_roles(invalid_mask: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        validate_interpolation_mask(invalid_mask)


def test_validator_rejects_mismatched_expected_row_set() -> None:
    with pytest.raises(ValueError, match="expected set"):
        validate_interpolation_mask(
            _valid_mask(),
            expected_array_rows=np.asarray([4, 8, 13], dtype=np.int64),
        )


def test_validator_rejects_non_dataframe_input() -> None:
    with pytest.raises(TypeError, match="DataFrame"):
        validate_interpolation_mask([])  # type: ignore[arg-type]
