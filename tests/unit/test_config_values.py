from __future__ import annotations

import math
import re

import pytest

from seis_interp import config_values
from seis_interp.configuration import ConfigurationError


def test_positive_integer_accepts_positive_values() -> None:
    assert config_values.positive_integer(3, "training.batch_size") == 3


@pytest.mark.parametrize("value", [0, -1, True, False, 1.5, "3", None])
def test_positive_integer_rejects_non_positive_and_non_integral(value: object) -> None:
    expected = r"training\.batch_size must be a positive integer"
    with pytest.raises(ConfigurationError, match=expected):
        config_values.positive_integer(value, "training.batch_size")


def test_nonnegative_integer_accepts_zero_and_positive_values() -> None:
    assert config_values.nonnegative_integer(0, "training.audit_count") == 0
    assert config_values.nonnegative_integer(7, "training.audit_count") == 7


@pytest.mark.parametrize("value", [-1, True, False, 2.0, "0"])
def test_nonnegative_integer_rejects_negative_bool_and_float(value: object) -> None:
    expected = r"training\.audit_count must be a non-negative integer"
    with pytest.raises(ConfigurationError, match=expected):
        config_values.nonnegative_integer(value, "training.audit_count")


def test_odd_positive_integer_accepts_odd_values() -> None:
    assert config_values.odd_positive_integer(5, "model.kernel_size") == 5


def test_odd_positive_integer_rejects_even_values() -> None:
    with pytest.raises(ConfigurationError, match=r"model\.kernel_size must be odd"):
        config_values.odd_positive_integer(4, "model.kernel_size")


def test_odd_positive_integer_rejects_zero_as_non_positive() -> None:
    with pytest.raises(ConfigurationError, match=r"model\.kernel_size must be a positive integer"):
        config_values.odd_positive_integer(0, "model.kernel_size")


def test_finite_float_accepts_integral_and_real_values() -> None:
    assert config_values.finite_float(2, "training.weight") == 2.0
    assert config_values.finite_float(0.25, "training.weight") == 0.25


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, True, False, "1.0", None])
def test_finite_float_rejects_non_finite_and_non_real(value: object) -> None:
    with pytest.raises(ConfigurationError, match=r"training\.weight must be a finite number"):
        config_values.finite_float(value, "training.weight")


def test_positive_float_rejects_zero_and_negative() -> None:
    assert config_values.positive_float(0.5, "training.learning_rate") == 0.5
    with pytest.raises(ConfigurationError, match=r"training\.learning_rate must be positive"):
        config_values.positive_float(0.0, "training.learning_rate")
    with pytest.raises(ConfigurationError, match=r"training\.learning_rate must be positive"):
        config_values.positive_float(-1.0, "training.learning_rate")


def test_nonnegative_float_accepts_zero_and_rejects_negative() -> None:
    assert config_values.nonnegative_float(0.0, "training.weight_decay") == 0.0
    with pytest.raises(ConfigurationError, match=r"training\.weight_decay must be non-negative"):
        config_values.nonnegative_float(-0.1, "training.weight_decay")


def test_probability_accepts_zero_and_values_below_one() -> None:
    assert config_values.probability(0.0, "training.neighbor_dropout") == 0.0
    assert config_values.probability(0.999, "training.neighbor_dropout") == 0.999


@pytest.mark.parametrize("value", [1.0, 1.5, -0.1])
def test_probability_rejects_one_and_out_of_range_values(value: object) -> None:
    expected = re.escape("training.neighbor_dropout must be in [0, 1)")
    with pytest.raises(ConfigurationError, match=expected):
        config_values.probability(value, "training.neighbor_dropout")


def test_require_exact_accepts_matching_value() -> None:
    config_values.require_exact({"training": {"loss": "l2"}}, "training.loss", "l2")


def test_require_exact_rejects_mismatched_value() -> None:
    expected = re.escape("training.loss must be 'l2', got 'l1'")
    with pytest.raises(ConfigurationError, match=expected):
        config_values.require_exact({"training": {"loss": "l1"}}, "training.loss", "l2")


def test_require_exact_rejects_missing_path() -> None:
    expected = re.escape("missing required configuration value: training.loss")
    with pytest.raises(ConfigurationError, match=expected):
        config_values.require_exact({"training": {}}, "training.loss", "l2")


def test_optional_ffid_range_returns_none_when_absent() -> None:
    assert config_values.optional_ffid_range({}) is None
    assert config_values.optional_ffid_range({"training": {}}) is None


def test_optional_ffid_range_accepts_two_element_list() -> None:
    config = {"training": {"ffid_range": [10, 20]}}

    assert config_values.optional_ffid_range(config) == (10, 20)


@pytest.mark.parametrize(
    "value",
    [
        [20, 10],
        [-1, 5],
        [True, 5],
        [1, 2, 3],
        (10, 20),
        "10-20",
    ],
)
def test_optional_ffid_range_rejects_invalid_values(value: object) -> None:
    expected = re.escape("training.ffid_range must be [minimum, maximum] integers")
    with pytest.raises(ConfigurationError, match=expected):
        config_values.optional_ffid_range({"training": {"ffid_range": value}})


def test_validated_effective_split_counts_accepts_exact_keys() -> None:
    counts = config_values.validated_effective_split_counts(
        {"train": 8, "validation": 2, "test": 2}
    )

    assert counts == {"train": 8, "validation": 2, "test": 2}


@pytest.mark.parametrize(
    "value",
    [
        {"train": 8, "validation": 2},
        {"train": 8, "validation": 2, "test": 2, "excluded": 1},
        "not a mapping",
    ],
)
def test_validated_effective_split_counts_requires_exact_keys(value: object) -> None:
    expected = re.escape(
        "evaluation.required_effective_split_counts must contain exactly "
        "['train', 'validation', 'test']"
    )
    with pytest.raises(ConfigurationError, match=expected):
        config_values.validated_effective_split_counts(value)


@pytest.mark.parametrize("bad_count", [0, True])
def test_validated_effective_split_counts_rejects_non_positive_counts(bad_count: object) -> None:
    expected = re.escape("evaluation.required_effective_split_counts.train must be a positive")
    with pytest.raises(ConfigurationError, match=expected):
        config_values.validated_effective_split_counts(
            {"train": bad_count, "validation": 2, "test": 2}
        )


def test_validated_ffid_split_counts_accepts_exact_keys() -> None:
    counts = config_values.validated_ffid_split_counts({"train": 4, "validation": 1, "test": 1})

    assert counts == {"train": 4, "validation": 1, "test": 1}


@pytest.mark.parametrize(
    "value",
    [
        {"train": 4, "test": 1},
        {"train": 4, "validation": 1, "test": 1, "excluded": 1},
        "not a mapping",
    ],
)
def test_validated_ffid_split_counts_requires_exact_keys(value: object) -> None:
    expected = re.escape(
        "evaluation.required_ffid_split_counts must contain exactly ['train', 'validation', 'test']"
    )
    with pytest.raises(ConfigurationError, match=expected):
        config_values.validated_ffid_split_counts(value)


def test_validated_sorted_ffids_accepts_sorted_unique_list() -> None:
    assert config_values.validated_sorted_ffids([0, 3, 7], "evaluation.ffids") == (0, 3, 7)


@pytest.mark.parametrize("value", [[3, 3], [7, 3], [-1, 3], [True, 3], (0, 3), "037"])
def test_validated_sorted_ffids_rejects_invalid_lists(value: object) -> None:
    expected = re.escape("evaluation.ffids must be a sorted unique list of non-negative integers")
    with pytest.raises(ConfigurationError, match=expected):
        config_values.validated_sorted_ffids(value, "evaluation.ffids")


def test_validated_positive_integer_list_accepts_positive_values() -> None:
    assert config_values.validated_positive_integer_list([1, 2], "model.dilations") == (1, 2)


def test_validated_positive_integer_list_rejects_empty_list() -> None:
    with pytest.raises(ConfigurationError, match=r"model\.dilations must be a non-empty list"):
        config_values.validated_positive_integer_list([], "model.dilations")


def test_validated_positive_integer_list_rejects_invalid_elements() -> None:
    expected = re.escape("model.dilations[1] must be a positive integer")
    with pytest.raises(ConfigurationError, match=expected):
        config_values.validated_positive_integer_list([1, 0], "model.dilations")


def test_validated_target_coordinate_names_accepts_unique_strings() -> None:
    names = config_values.validated_target_coordinate_names(["source_x_m", "source_y_m"])

    assert names == ("source_x_m", "source_y_m")


def test_validated_target_coordinate_names_rejects_empty_list() -> None:
    expected = re.escape("model.target_coordinates must be a non-empty list")
    with pytest.raises(ConfigurationError, match=expected):
        config_values.validated_target_coordinate_names([])


def test_validated_target_coordinate_names_rejects_empty_strings() -> None:
    expected = re.escape("model.target_coordinates must contain non-empty strings")
    with pytest.raises(ConfigurationError, match=expected):
        config_values.validated_target_coordinate_names(["source_x_m", ""])


def test_validated_target_coordinate_names_rejects_duplicates() -> None:
    expected = re.escape("model.target_coordinates must not contain duplicates")
    with pytest.raises(ConfigurationError, match=expected):
        config_values.validated_target_coordinate_names(["source_x_m", "source_x_m"])


def test_validated_positive_config_integer_reports_rejected_value() -> None:
    assert config_values.validated_positive_config_integer(4, "training.epochs") == 4
    expected = re.escape("training.epochs must be a positive integer, got 0")
    with pytest.raises(ConfigurationError, match=expected):
        config_values.validated_positive_config_integer(0, "training.epochs")


@pytest.mark.parametrize("value", [True, False, -3, 1.5, "4"])
def test_validated_positive_config_integer_rejects_bool_and_non_integral(value: object) -> None:
    expected = re.escape(f"training.epochs must be a positive integer, got {value!r}")
    with pytest.raises(ConfigurationError, match=expected):
        config_values.validated_positive_config_integer(value, "training.epochs")


def test_validated_positive_config_float_reports_rejected_value() -> None:
    assert config_values.validated_positive_config_float(0.1, "training.learning_rate") == 0.1
    expected = re.escape("training.learning_rate must be a positive finite number, got 0.0")
    with pytest.raises(ConfigurationError, match=expected):
        config_values.validated_positive_config_float(0.0, "training.learning_rate")


@pytest.mark.parametrize("value", [True, False, math.nan, math.inf, -math.inf, "0.1", -1.0])
def test_validated_positive_config_float_rejects_bool_and_non_finite(value: object) -> None:
    expected = re.escape(f"training.learning_rate must be a positive finite number, got {value!r}")
    with pytest.raises(ConfigurationError, match=expected):
        config_values.validated_positive_config_float(value, "training.learning_rate")


def test_validated_nonnegative_config_float_reports_rejected_value() -> None:
    assert config_values.validated_nonnegative_config_float(0.0, "training.weight_decay") == 0.0
    expected = re.escape("training.weight_decay must be a non-negative finite number, got -0.5")
    with pytest.raises(ConfigurationError, match=expected):
        config_values.validated_nonnegative_config_float(-0.5, "training.weight_decay")


@pytest.mark.parametrize("value", [True, False, math.nan, math.inf, -math.inf, "0.0"])
def test_validated_nonnegative_config_float_rejects_bool_and_non_finite(value: object) -> None:
    expected = re.escape(
        f"training.weight_decay must be a non-negative finite number, got {value!r}"
    )
    with pytest.raises(ConfigurationError, match=expected):
        config_values.validated_nonnegative_config_float(value, "training.weight_decay")
