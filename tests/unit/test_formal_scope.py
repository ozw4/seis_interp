from __future__ import annotations

from copy import deepcopy

import pytest

from seis_interp.evaluation.formal_scope import (
    CHECKPOINT_REVALIDATION_ABSOLUTE_TOLERANCE,
    CHECKPOINT_REVALIDATION_RELATIVE_TOLERANCE,
    build_formal_scope_audit,
    complete_neighbor_formal_scope_audit,
    complete_whole_shot_formal_scope_audit,
)
from seis_interp.processing.trace_splits import (
    EXCLUDED_SPLIT,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
)


def _per_ffid_selection_contract() -> dict[str, object]:
    return {
        "selected_ffid_count": 2,
        "sample_count": 64,
        "split_counts": {
            TRAIN_SPLIT: 6,
            VALIDATION_SPLIT: 2,
            TEST_SPLIT: 2,
            EXCLUDED_SPLIT: 1,
        },
        "ffid_split_counts": {TRAIN_SPLIT: 2, VALIDATION_SPLIT: 2, TEST_SPLIT: 2},
        "ffid_split_overlap_count": 2,
        "maximum_splits_per_ffid": 3,
    }


def _whole_ffid_selection_contract() -> dict[str, object]:
    return {
        "selected_ffid_count": 3,
        "sample_count": 32,
        "split_counts": {
            TRAIN_SPLIT: 2,
            VALIDATION_SPLIT: 1,
            TEST_SPLIT: 1,
            EXCLUDED_SPLIT: 0,
        },
        "ffid_split_counts": {TRAIN_SPLIT: 1, VALIDATION_SPLIT: 1, TEST_SPLIT: 1},
        "ffid_split_overlap_count": 0,
        "maximum_splits_per_ffid": 1,
    }


def _preparation_contract(
    split_scope: str,
    fully_excluded_ffids: list[int] | None = None,
) -> dict[str, object]:
    return {
        "split_scope": split_scope,
        "trace_quality": {"fully_excluded_ffids": fully_excluded_ffids or []},
    }


def _per_ffid_audit(**overrides: object) -> dict[str, object]:
    arguments: dict[str, object] = {
        "ffid_range": None,
        "exclude_target_ffid_neighbors": False,
        "required_eligible_ffid_count": 2,
        "required_sample_count": 64,
        "required_effective_split_counts": {
            TRAIN_SPLIT: 6,
            VALIDATION_SPLIT: 2,
            TEST_SPLIT: 2,
        },
        "required_ffid_split_counts": None,
        "required_fully_excluded_ffids": (),
        "selection_contract": _per_ffid_selection_contract(),
        "preparation_contract": _preparation_contract("per_ffid"),
    }
    arguments.update(overrides)
    return build_formal_scope_audit(**arguments)


def test_per_ffid_configured_scope_success_has_exact_structure() -> None:
    audit = _per_ffid_audit()

    assert audit == {
        "requirements": {
            "ffid_range": None,
            "split_scope": "per_ffid",
            "eligible_ffid_count": 2,
            "sample_count": 64,
            "effective_split_counts": {TRAIN_SPLIT: 6, VALIDATION_SPLIT: 2, TEST_SPLIT: 2},
            "ffid_split_counts": None,
            "fully_excluded_ffids": [],
        },
        "actual": {
            "ffid_range": None,
            "split_scope": "per_ffid",
            "eligible_ffid_count": 2,
            "sample_count": 64,
            "effective_split_counts": {TRAIN_SPLIT: 6, VALIDATION_SPLIT: 2, TEST_SPLIT: 2},
            "ffid_split_counts": {TRAIN_SPLIT: 2, VALIDATION_SPLIT: 2, TEST_SPLIT: 2},
            "ffid_split_overlap_count": 2,
            "fully_excluded_ffids": [],
        },
        "checks": {
            "ffid_range_not_configured": True,
            "eligible_ffid_count_matches": True,
            "sample_count_matches": True,
            "effective_split_counts_match": True,
            "fully_excluded_ffids_match": True,
            "split_scope_structure_matches": True,
            "target_ffid_context_matches": True,
        },
        "scope_success": True,
    }


def test_whole_ffid_configured_scope_success_includes_ffid_split_counts_check() -> None:
    audit = build_formal_scope_audit(
        ffid_range=None,
        exclude_target_ffid_neighbors=True,
        required_eligible_ffid_count=3,
        required_sample_count=32,
        required_effective_split_counts={TRAIN_SPLIT: 2, VALIDATION_SPLIT: 1, TEST_SPLIT: 1},
        required_ffid_split_counts={TRAIN_SPLIT: 1, VALIDATION_SPLIT: 1, TEST_SPLIT: 1},
        required_fully_excluded_ffids=(13,),
        selection_contract=_whole_ffid_selection_contract(),
        preparation_contract=_preparation_contract("whole_ffid", [13]),
    )

    assert audit["checks"] == {
        "ffid_range_not_configured": True,
        "eligible_ffid_count_matches": True,
        "sample_count_matches": True,
        "effective_split_counts_match": True,
        "fully_excluded_ffids_match": True,
        "split_scope_structure_matches": True,
        "target_ffid_context_matches": True,
        "ffid_split_counts_match": True,
    }
    assert audit["scope_success"] is True
    assert audit["requirements"]["fully_excluded_ffids"] == [13]
    assert audit["actual"]["fully_excluded_ffids"] == [13]


def test_configured_ffid_range_fails_the_range_check_and_is_reported() -> None:
    audit = _per_ffid_audit(ffid_range=(100, 200))

    assert audit["checks"]["ffid_range_not_configured"] is False
    assert audit["actual"]["ffid_range"] == [100, 200]
    assert audit["requirements"]["ffid_range"] is None
    assert audit["scope_success"] is False


@pytest.mark.parametrize(
    ("overrides", "failed_check"),
    [
        ({"required_eligible_ffid_count": 5}, "eligible_ffid_count_matches"),
        ({"required_sample_count": 65}, "sample_count_matches"),
        (
            {
                "required_effective_split_counts": {
                    TRAIN_SPLIT: 7,
                    VALIDATION_SPLIT: 2,
                    TEST_SPLIT: 2,
                }
            },
            "effective_split_counts_match",
        ),
        ({"required_fully_excluded_ffids": (99,)}, "fully_excluded_ffids_match"),
    ],
)
def test_requirement_mismatches_fail_their_specific_check(
    overrides: dict[str, object],
    failed_check: str,
) -> None:
    audit = _per_ffid_audit(**overrides)

    checks = dict(audit["checks"])
    assert checks.pop(failed_check) is False
    assert all(checks.values())
    assert audit["scope_success"] is False


def test_optional_required_ffid_split_counts_adds_check_only_when_present() -> None:
    without = _per_ffid_audit()
    with_matching = _per_ffid_audit(
        required_ffid_split_counts={TRAIN_SPLIT: 2, VALIDATION_SPLIT: 2, TEST_SPLIT: 2}
    )
    with_mismatched = _per_ffid_audit(
        required_ffid_split_counts={TRAIN_SPLIT: 3, VALIDATION_SPLIT: 2, TEST_SPLIT: 2}
    )

    assert "ffid_split_counts_match" not in without["checks"]
    assert with_matching["checks"]["ffid_split_counts_match"] is True
    assert with_mismatched["checks"]["ffid_split_counts_match"] is False
    assert with_mismatched["scope_success"] is False


def test_invalid_or_missing_trace_quality_contract_raises_runtime_errors() -> None:
    with pytest.raises(RuntimeError, match="missing trace_quality"):
        _per_ffid_audit(preparation_contract={"split_scope": "per_ffid"})

    with pytest.raises(RuntimeError, match="invalid fully_excluded_ffids"):
        _per_ffid_audit(
            preparation_contract={
                "split_scope": "per_ffid",
                "trace_quality": {"fully_excluded_ffids": "13"},
            }
        )

    with pytest.raises(RuntimeError, match="unsupported split_scope"):
        _per_ffid_audit(preparation_contract=_preparation_contract("global"))


def _neighbor_configured_audit() -> dict[str, object]:
    return {
        "requirements": {"eligible_ffid_count": 2},
        "actual": {"eligible_ffid_count": 2},
        "checks": {"ffid_range_not_configured": True},
        "scope_success": True,
    }


def _neighbor_completion_arguments() -> dict[str, object]:
    return {
        "validation_metric_domain": "oracle_per_trace_unit_rms",
        "collision_audit": {
            "canonical_remaining_duplicate_physical_cells": 0,
            "train_coordinate_collision_cells": 0,
            "train_validation_coordinate_overlap_rows": 0,
        },
        "geometry_contract": {
            "target_ffid_neighbor_policy": "exclude_exact_ffid",
            "center_offset_count": 0,
        },
        "availability_contract": {
            TRAIN_SPLIT: {"target_ffid_neighbor_entries": 0},
            VALIDATION_SPLIT: {"target_ffid_neighbor_entries": 0},
        },
        "source_bracketing_contract": None,
        "amplitude_access": {
            "value_rows_materialized_by_split": {TEST_SPLIT: False, EXCLUDED_SPLIT: False}
        },
        "checkpoint_revalidation_matches": True,
        "selected_metric": 10.0,
        "recomputed_metric": 10.0,
    }


def test_neighbor_completion_adds_all_basic_checks_and_keeps_success() -> None:
    completed = complete_neighbor_formal_scope_audit(
        _neighbor_configured_audit(),
        **_neighbor_completion_arguments(),
    )

    assert completed["checks"] == {
        "ffid_range_not_configured": True,
        "validation_metric_domain_matches": True,
        "checkpoint_raw_metric_reproduced": True,
        "selected_metric_matches_recomputed_raw_metric": True,
        "canonical_duplicate_physical_cells_remaining_zero": True,
        "train_geometry_collision_cells_zero": True,
        "train_validation_coordinate_overlap_zero": True,
        "neighbor_center_offset_count_zero": True,
        "target_ffid_neighbor_entries_zero": True,
        "test_value_rows_not_materialized": True,
        "excluded_value_rows_not_materialized": True,
    }
    assert completed["scope_success"] is True
    assert completed["requirements"] == {"eligible_ffid_count": 2}


def test_neighbor_completion_does_not_modify_the_configured_audit() -> None:
    configured = _neighbor_configured_audit()
    original = deepcopy(configured)

    complete_neighbor_formal_scope_audit(configured, **_neighbor_completion_arguments())

    assert configured == original


def test_neighbor_completion_with_source_bracketing_adds_bracketing_checks() -> None:
    arguments = _neighbor_completion_arguments()
    bracket_audit = {
        "unresolved_rows": 0,
        "target_ffid_reference_entries": 0,
        "same_source_y_reference_entries": 0,
        "source_split_counts": {"non_train": 0},
    }
    arguments["source_bracketing_contract"] = {
        TRAIN_SPLIT: dict(bracket_audit),
        VALIDATION_SPLIT: dict(bracket_audit),
    }

    completed = complete_neighbor_formal_scope_audit(
        _neighbor_configured_audit(),
        **arguments,
    )

    assert completed["checks"]["source_bracketing_unresolved_rows_zero"] is True
    assert completed["checks"]["source_bracketing_target_ffid_entries_zero"] is True
    assert completed["checks"]["source_bracketing_same_source_y_entries_zero"] is True
    assert completed["checks"]["source_bracketing_sources_train_only"] is True
    assert completed["scope_success"] is True


def test_neighbor_completion_flags_validation_metric_domain_mismatch() -> None:
    arguments = _neighbor_completion_arguments()
    arguments["validation_metric_domain"] = "wrong_domain"

    completed = complete_neighbor_formal_scope_audit(
        _neighbor_configured_audit(),
        **arguments,
    )

    assert completed["checks"]["validation_metric_domain_matches"] is False
    assert completed["scope_success"] is False


def test_neighbor_completion_metric_tolerance_is_checkpoint_revalidation_tolerance() -> None:
    assert CHECKPOINT_REVALIDATION_RELATIVE_TOLERANCE == 1.0e-8
    assert CHECKPOINT_REVALIDATION_ABSOLUTE_TOLERANCE == 1.0e-8
    within = _neighbor_completion_arguments()
    within["recomputed_metric"] = 10.0 + 5.0e-8
    outside = _neighbor_completion_arguments()
    outside["recomputed_metric"] = 10.001

    completed_within = complete_neighbor_formal_scope_audit(_neighbor_configured_audit(), **within)
    completed_outside = complete_neighbor_formal_scope_audit(
        _neighbor_configured_audit(), **outside
    )

    assert completed_within["checks"]["selected_metric_matches_recomputed_raw_metric"] is True
    assert completed_outside["checks"]["selected_metric_matches_recomputed_raw_metric"] is False
    assert completed_outside["scope_success"] is False


def _whole_shot_completion_arguments() -> dict[str, object]:
    return {
        "validation_metric_domain": "oracle_per_trace_unit_rms",
        "availability_contract": {
            TRAIN_SPLIT: {
                "target_ffid_neighbor_entries": 0,
                "non_train_neighbor_entries": 0,
            },
            VALIDATION_SPLIT: {
                "target_ffid_neighbor_entries": 0,
                "non_train_neighbor_entries": 0,
            },
        },
        "collision_audit": {
            "canonical_remaining_duplicate_physical_cells": 0,
            "train_duplicate_source_coordinates": 0,
            "train_validation_source_coordinate_overlap": 0,
        },
        "amplitude_access": {
            "value_rows_materialized_by_split": {TEST_SPLIT: False, EXCLUDED_SPLIT: False}
        },
        "checkpoint_revalidation_matches": True,
        "selected_metric": 12.0,
        "recomputed_metric": 12.0,
    }


def test_whole_shot_completion_adds_all_checks_and_keeps_success() -> None:
    completed = complete_whole_shot_formal_scope_audit(
        _neighbor_configured_audit(),
        **_whole_shot_completion_arguments(),
    )

    assert completed["checks"] == {
        "ffid_range_not_configured": True,
        "validation_metric_domain_matches": True,
        "checkpoint_raw_metric_reproduced": True,
        "selected_metric_matches_recomputed_raw_metric": True,
        "canonical_duplicate_physical_cells_remaining_zero": True,
        "train_source_coordinate_collisions_zero": True,
        "train_validation_source_coordinate_overlap_zero": True,
        "target_ffid_neighbor_entries_zero": True,
        "neighbor_sources_train_only": True,
        "test_value_rows_not_materialized": True,
        "excluded_value_rows_not_materialized": True,
    }
    assert completed["scope_success"] is True


@pytest.mark.parametrize(
    ("mutate_key", "mutate_value", "failed_check"),
    [
        (
            ("validation_metric_domain",),
            "wrong_domain",
            "validation_metric_domain_matches",
        ),
        (
            ("collision_audit", "train_duplicate_source_coordinates"),
            1,
            "train_source_coordinate_collisions_zero",
        ),
        (
            ("availability_contract", TRAIN_SPLIT, "target_ffid_neighbor_entries"),
            1,
            "target_ffid_neighbor_entries_zero",
        ),
        (
            ("availability_contract", VALIDATION_SPLIT, "non_train_neighbor_entries"),
            2,
            "neighbor_sources_train_only",
        ),
        (
            ("amplitude_access", "value_rows_materialized_by_split", TEST_SPLIT),
            True,
            "test_value_rows_not_materialized",
        ),
    ],
)
def test_whole_shot_completion_flags_each_violation(
    mutate_key: tuple[str, ...],
    mutate_value: object,
    failed_check: str,
) -> None:
    arguments = _whole_shot_completion_arguments()
    target: dict = arguments
    for key in mutate_key[:-1]:
        target = target[key]
    target[mutate_key[-1]] = mutate_value

    completed = complete_whole_shot_formal_scope_audit(
        _neighbor_configured_audit(),
        **arguments,
    )

    checks = dict(completed["checks"])
    assert checks.pop(failed_check) is False
    assert all(checks.values())
    assert completed["scope_success"] is False


def test_scope_success_is_conjunction_of_every_check() -> None:
    arguments = _neighbor_completion_arguments()
    arguments["checkpoint_revalidation_matches"] = False

    completed = complete_neighbor_formal_scope_audit(
        _neighbor_configured_audit(),
        **arguments,
    )

    assert completed["checks"]["checkpoint_raw_metric_reproduced"] is False
    assert completed["scope_success"] is False
    assert completed["scope_success"] == all(completed["checks"].values())
