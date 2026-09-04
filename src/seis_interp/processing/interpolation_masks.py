"""Create and validate deterministic interpolation masks."""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype

from seis_interp.data.trace_table import validated_array_rows

OBSERVATION_ROLE_COLUMN = "observation_role"

OBSERVED_ROLE = "observed"
EVALUATION_TARGET_ROLE = "evaluation_target"

RANDOM_TRACE_MASK_KIND = "random_trace"
RANDOM_WHOLE_FFID_MASK_KIND = "random_whole_ffid"

OBSERVATION_ROLES = (OBSERVED_ROLE, EVALUATION_TARGET_ROLE)
MASK_KINDS = (RANDOM_TRACE_MASK_KIND, RANDOM_WHOLE_FFID_MASK_KIND)

_ARRAY_ROW_COLUMN = "array_row"
_MASK_COLUMNS = (_ARRAY_ROW_COLUMN, OBSERVATION_ROLE_COLUMN)


def make_random_trace_mask(
    candidate_table: pd.DataFrame,
    *,
    missing_fraction: float,
    random_seed: int,
) -> pd.DataFrame:
    """Assign deterministic observed and target roles to individual traces."""
    array_rows = validated_array_rows(candidate_table)
    fraction = _validated_missing_fraction(missing_fraction)
    seed = _validated_random_seed(random_seed)
    target_count = _target_count(len(array_rows), fraction, unit_name="traces")

    sorted_rows = np.sort(array_rows)
    permutation = np.random.default_rng(seed).permutation(sorted_rows)
    target_rows = permutation[:target_count]
    roles = np.where(
        np.isin(sorted_rows, target_rows),
        EVALUATION_TARGET_ROLE,
        OBSERVED_ROLE,
    )
    result = pd.DataFrame(
        {
            _ARRAY_ROW_COLUMN: sorted_rows,
            OBSERVATION_ROLE_COLUMN: roles,
        }
    )
    validate_interpolation_mask(result, expected_array_rows=array_rows)
    return result


def make_random_whole_ffid_mask(
    candidate_table: pd.DataFrame,
    *,
    missing_fraction: float,
    random_seed: int,
) -> pd.DataFrame:
    """Assign each candidate FFID wholly to one observation role."""
    array_rows = validated_array_rows(candidate_table)
    ffids = _validated_ffids(candidate_table)
    fraction = _validated_missing_fraction(missing_fraction)
    seed = _validated_random_seed(random_seed)

    unique_ffids = np.unique(ffids)
    target_count = _target_count(len(unique_ffids), fraction, unit_name="FFIDs")
    permutation = np.random.default_rng(seed).permutation(unique_ffids)
    target_ffids = permutation[:target_count]

    order = np.argsort(array_rows)
    sorted_rows = array_rows[order]
    sorted_ffids = ffids[order]
    roles = np.where(
        np.isin(sorted_ffids, target_ffids),
        EVALUATION_TARGET_ROLE,
        OBSERVED_ROLE,
    )
    result = pd.DataFrame(
        {
            _ARRAY_ROW_COLUMN: sorted_rows,
            OBSERVATION_ROLE_COLUMN: roles,
        }
    )
    validate_interpolation_mask(result, expected_array_rows=array_rows)
    return result


def validate_interpolation_mask(
    mask_table: pd.DataFrame,
    *,
    expected_array_rows: np.ndarray | None = None,
) -> None:
    """Validate the two-column interpolation-mask table contract."""
    if not isinstance(mask_table, pd.DataFrame):
        raise TypeError(f"mask_table must be a pandas DataFrame, got {type(mask_table).__name__}")
    if mask_table.columns.tolist() != list(_MASK_COLUMNS):
        raise ValueError(
            "interpolation mask columns must be exactly "
            f"{list(_MASK_COLUMNS)}, got {mask_table.columns.tolist()}"
        )

    array_rows = validated_array_rows(mask_table)
    roles = mask_table[OBSERVATION_ROLE_COLUMN]
    if roles.isna().any():
        raise ValueError(f"{OBSERVATION_ROLE_COLUMN} contains missing values")
    known_roles = roles.isin(OBSERVATION_ROLES)
    if not bool(known_roles.all()):
        unknown_roles = sorted({repr(role) for role in roles.loc[~known_roles].tolist()})
        raise ValueError(f"{OBSERVATION_ROLE_COLUMN} contains unknown roles: {unknown_roles}")
    missing_roles = [role for role in OBSERVATION_ROLES if not roles.eq(role).any()]
    if missing_roles:
        raise ValueError(
            f"interpolation mask must contain both observation roles; missing {missing_roles}"
        )

    if expected_array_rows is not None:
        expected = _validated_expected_array_rows(expected_array_rows)
        if set(array_rows.tolist()) != set(expected.tolist()):
            raise ValueError("interpolation mask array_row values do not match the expected set")


def _validated_ffids(candidate_table: pd.DataFrame) -> np.ndarray:
    if "ffid" not in candidate_table.columns:
        raise ValueError("candidate table is missing required column: ffid")
    values = candidate_table["ffid"]
    if values.isna().any():
        raise ValueError("ffid contains missing values")
    if is_bool_dtype(values.dtype) or not is_integer_dtype(values.dtype):
        raise ValueError(f"ffid must have an integer dtype, got {values.dtype}")

    int64_info = np.iinfo(np.int64)
    if int(values.min()) < int64_info.min or int(values.max()) > int64_info.max:
        raise ValueError("ffid values must fit in int64")
    return values.to_numpy(dtype=np.int64)


def _validated_missing_fraction(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("missing_fraction must be a real number strictly between 0 and 1")
    fraction = float(value)
    if not np.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError(f"missing_fraction must be strictly between 0 and 1, got {value!r}")
    return fraction


def _validated_random_seed(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"random_seed must be an integer, got {value!r}")
    seed = int(value)
    if seed < 0:
        raise ValueError(f"random_seed must be non-negative, got {seed}")
    return seed


def _target_count(candidate_count: int, missing_fraction: float, *, unit_name: str) -> int:
    target_count = int(round(candidate_count * missing_fraction))
    observed_count = candidate_count - target_count
    if target_count == 0 or observed_count == 0:
        raise ValueError(
            "missing_fraction must produce at least one observed and one evaluation target "
            f"among {candidate_count} {unit_name}; got observed={observed_count}, "
            f"evaluation_target={target_count}"
        )
    return target_count


def _validated_expected_array_rows(values: np.ndarray) -> np.ndarray:
    expected = np.asarray(values)
    if expected.ndim != 1:
        raise ValueError("expected_array_rows must be a one-dimensional integer array")
    if expected.dtype.kind not in "iu" or expected.dtype.kind == "b":
        raise ValueError("expected_array_rows must be a one-dimensional integer array")
    if len(np.unique(expected)) != len(expected):
        raise ValueError("expected_array_rows must not contain duplicates")
    if len(expected):
        int64_info = np.iinfo(np.int64)
        if int(expected.min()) < int64_info.min or int(expected.max()) > int64_info.max:
            raise ValueError("expected_array_rows values must fit in int64")
    return expected.astype(np.int64, copy=False)
