"""Audit whether a run matches its configured formal survey scope."""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy

from seis_interp.processing.trace_splits import (
    EXCLUDED_SPLIT,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
)
from seis_interp.training.amplitude_scaling import ORACLE_PER_TRACE_RMS_VALIDATION_DOMAIN

CHECKPOINT_REVALIDATION_RELATIVE_TOLERANCE = 1.0e-8
CHECKPOINT_REVALIDATION_ABSOLUTE_TOLERANCE = 1.0e-8

_EFFECTIVE_SPLITS = (TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT)


def build_formal_scope_audit(
    *,
    ffid_range: tuple[int, int] | None,
    exclude_target_ffid_neighbors: bool,
    required_eligible_ffid_count: int,
    required_sample_count: int,
    required_effective_split_counts: Mapping[str, int],
    required_ffid_split_counts: Mapping[str, int] | None,
    required_fully_excluded_ffids: tuple[int, ...],
    selection_contract: Mapping[str, object],
    preparation_contract: Mapping[str, object],
) -> dict[str, object]:
    """Compare the selected scope against the configured formal requirements."""
    trace_quality = preparation_contract.get("trace_quality")
    if not isinstance(trace_quality, Mapping):
        raise RuntimeError("validated preparation contract is missing trace_quality")
    fully_excluded_ffids = trace_quality.get("fully_excluded_ffids")
    if not isinstance(fully_excluded_ffids, list):
        raise RuntimeError("validated preparation trace_quality has invalid fully_excluded_ffids")
    split_counts = selection_contract["split_counts"]
    if not isinstance(split_counts, Mapping):
        raise RuntimeError("selection split_counts must be an object")

    required_split_counts = dict(required_effective_split_counts)
    actual_split_counts = {split: int(split_counts[split]) for split in _EFFECTIVE_SPLITS}
    raw_actual_ffid_split_counts = selection_contract.get("ffid_split_counts")
    if not isinstance(raw_actual_ffid_split_counts, Mapping):
        raise RuntimeError("selection ffid_split_counts must be an object")
    actual_ffid_split_counts = {
        split: int(raw_actual_ffid_split_counts[split]) for split in _EFFECTIVE_SPLITS
    }
    split_scope = preparation_contract.get("split_scope")
    if split_scope not in {"per_ffid", "whole_ffid"}:
        raise RuntimeError("validated preparation contract has unsupported split_scope")
    checks = {
        "ffid_range_not_configured": ffid_range is None,
        "eligible_ffid_count_matches": (
            selection_contract["selected_ffid_count"] == required_eligible_ffid_count
        ),
        "sample_count_matches": (selection_contract["sample_count"] == required_sample_count),
        "effective_split_counts_match": actual_split_counts == required_split_counts,
        "fully_excluded_ffids_match": (fully_excluded_ffids == list(required_fully_excluded_ffids)),
        "split_scope_structure_matches": (
            selection_contract["ffid_split_overlap_count"] == 0
            and selection_contract["maximum_splits_per_ffid"] == 1
            and sum(actual_ffid_split_counts.values()) == selection_contract["selected_ffid_count"]
            if split_scope == "whole_ffid"
            else all(
                count == selection_contract["selected_ffid_count"]
                for count in actual_ffid_split_counts.values()
            )
        ),
        "target_ffid_context_matches": (
            split_scope != "whole_ffid" or exclude_target_ffid_neighbors
        ),
    }
    if required_ffid_split_counts is not None:
        checks["ffid_split_counts_match"] = actual_ffid_split_counts == dict(
            required_ffid_split_counts
        )
    return {
        "requirements": {
            "ffid_range": None,
            "split_scope": split_scope,
            "eligible_ffid_count": required_eligible_ffid_count,
            "sample_count": required_sample_count,
            "effective_split_counts": required_split_counts,
            "ffid_split_counts": (
                dict(required_ffid_split_counts) if required_ffid_split_counts is not None else None
            ),
            "fully_excluded_ffids": list(required_fully_excluded_ffids),
        },
        "actual": {
            "ffid_range": (list(ffid_range) if ffid_range is not None else None),
            "split_scope": split_scope,
            "eligible_ffid_count": selection_contract["selected_ffid_count"],
            "sample_count": selection_contract["sample_count"],
            "effective_split_counts": actual_split_counts,
            "ffid_split_counts": actual_ffid_split_counts,
            "ffid_split_overlap_count": selection_contract["ffid_split_overlap_count"],
            "fully_excluded_ffids": list(fully_excluded_ffids),
        },
        "checks": checks,
        "scope_success": all(checks.values()),
    }


def complete_neighbor_formal_scope_audit(
    configured_scope_audit: Mapping[str, object],
    *,
    validation_metric_domain: str,
    collision_audit: Mapping[str, object],
    geometry_contract: Mapping[str, object],
    availability_contract: Mapping[str, object],
    source_bracketing_contract: Mapping[str, object] | None,
    amplitude_access: Mapping[str, object],
    checkpoint_revalidation_matches: bool,
    selected_metric: float,
    recomputed_metric: float,
) -> dict[str, object]:
    """Extend the configured audit with neighbor-trace completion checks."""
    completed = deepcopy(dict(configured_scope_audit))
    raw_checks = completed.get("checks")
    if not isinstance(raw_checks, Mapping):
        raise RuntimeError("formal scope audit checks must be an object")
    materialized = amplitude_access.get("value_rows_materialized_by_split")
    if not isinstance(materialized, Mapping):
        raise RuntimeError("amplitude materialization audit must be an object")
    checks = dict(raw_checks)
    excludes_target_ffid = geometry_contract.get("target_ffid_neighbor_policy") == (
        "exclude_exact_ffid"
    )
    target_ffid_neighbor_entries = [
        availability_contract[split].get("target_ffid_neighbor_entries")
        for split in (TRAIN_SPLIT, VALIDATION_SPLIT)
        if isinstance(availability_contract.get(split), Mapping)
    ]
    checks.update(
        {
            "validation_metric_domain_matches": (
                validation_metric_domain == ORACLE_PER_TRACE_RMS_VALIDATION_DOMAIN
            ),
            "checkpoint_raw_metric_reproduced": checkpoint_revalidation_matches,
            "selected_metric_matches_recomputed_raw_metric": math.isclose(
                selected_metric,
                recomputed_metric,
                rel_tol=CHECKPOINT_REVALIDATION_RELATIVE_TOLERANCE,
                abs_tol=CHECKPOINT_REVALIDATION_ABSOLUTE_TOLERANCE,
            ),
            "canonical_duplicate_physical_cells_remaining_zero": (
                collision_audit["canonical_remaining_duplicate_physical_cells"] == 0
            ),
            "train_geometry_collision_cells_zero": (
                collision_audit["train_coordinate_collision_cells"] == 0
            ),
            "train_validation_coordinate_overlap_zero": (
                collision_audit["train_validation_coordinate_overlap_rows"] == 0
            ),
            "neighbor_center_offset_count_zero": geometry_contract["center_offset_count"] == 0,
            "target_ffid_neighbor_entries_zero": (
                not excludes_target_ffid or target_ffid_neighbor_entries == [0, 0]
            ),
            "test_value_rows_not_materialized": materialized.get(TEST_SPLIT) is False,
            "excluded_value_rows_not_materialized": materialized.get(EXCLUDED_SPLIT) is False,
        }
    )
    if source_bracketing_contract is not None:
        bracket_audits = [
            source_bracketing_contract.get(split) for split in (TRAIN_SPLIT, VALIDATION_SPLIT)
        ]
        if not all(isinstance(audit, Mapping) for audit in bracket_audits):
            raise RuntimeError("source bracketing contract is missing split audits")
        checks.update(
            {
                "source_bracketing_unresolved_rows_zero": all(
                    audit.get("unresolved_rows") == 0 for audit in bracket_audits
                ),
                "source_bracketing_target_ffid_entries_zero": all(
                    audit.get("target_ffid_reference_entries") == 0 for audit in bracket_audits
                ),
                "source_bracketing_same_source_y_entries_zero": all(
                    audit.get("same_source_y_reference_entries") == 0 for audit in bracket_audits
                ),
                "source_bracketing_sources_train_only": all(
                    isinstance(audit.get("source_split_counts"), Mapping)
                    and audit["source_split_counts"].get("non_train") == 0
                    for audit in bracket_audits
                ),
            }
        )
    completed["checks"] = checks
    completed["scope_success"] = all(checks.values())
    return completed


def complete_whole_shot_formal_scope_audit(
    configured_scope_audit: Mapping[str, object],
    *,
    validation_metric_domain: str,
    availability_contract: Mapping[str, Mapping[str, object]],
    collision_audit: Mapping[str, int],
    amplitude_access: Mapping[str, object],
    checkpoint_revalidation_matches: bool,
    selected_metric: float,
    recomputed_metric: float,
) -> dict[str, object]:
    """Extend the configured audit with whole-shot completion checks."""
    completed = deepcopy(dict(configured_scope_audit))
    checks = dict(completed["checks"])
    materialized = amplitude_access["value_rows_materialized_by_split"]
    checks.update(
        {
            "validation_metric_domain_matches": (
                validation_metric_domain == ORACLE_PER_TRACE_RMS_VALIDATION_DOMAIN
            ),
            "checkpoint_raw_metric_reproduced": checkpoint_revalidation_matches,
            "selected_metric_matches_recomputed_raw_metric": math.isclose(
                selected_metric,
                recomputed_metric,
                rel_tol=CHECKPOINT_REVALIDATION_RELATIVE_TOLERANCE,
                abs_tol=CHECKPOINT_REVALIDATION_ABSOLUTE_TOLERANCE,
            ),
            "canonical_duplicate_physical_cells_remaining_zero": (
                collision_audit["canonical_remaining_duplicate_physical_cells"] == 0
            ),
            "train_source_coordinate_collisions_zero": (
                collision_audit["train_duplicate_source_coordinates"] == 0
            ),
            "train_validation_source_coordinate_overlap_zero": (
                collision_audit["train_validation_source_coordinate_overlap"] == 0
            ),
            "target_ffid_neighbor_entries_zero": all(
                availability_contract[split]["target_ffid_neighbor_entries"] == 0
                for split in (TRAIN_SPLIT, VALIDATION_SPLIT)
            ),
            "neighbor_sources_train_only": all(
                availability_contract[split]["non_train_neighbor_entries"] == 0
                for split in (TRAIN_SPLIT, VALIDATION_SPLIT)
            ),
            "test_value_rows_not_materialized": materialized[TEST_SPLIT] is False,
            "excluded_value_rows_not_materialized": materialized[EXCLUDED_SPLIT] is False,
        }
    )
    completed["checks"] = checks
    completed["scope_success"] = all(checks.values())
    return completed
