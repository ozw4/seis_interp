"""Study 034 fixes one leakage-safe input and a finite NeRSI candidate matrix."""

from __future__ import annotations

import csv
import json
from copy import deepcopy

from seis_interp.configuration import REPOSITORY_ROOT, load_resolved_config
from seis_interp.data.file_checksums import file_sha256

STUDY = REPOSITORY_ROOT / "studies/study_034_c3_nersi_baseline"
RESULT = (
    REPOSITORY_ROOT / "results/study_034_c3_nersi_baseline/"
    "20260910T032100000000Z_d8568ae_candidate_selection"
)
REFERENCE_INPUTS = REPOSITORY_ROOT / "studies/study_032_c3_proposed_gnn_10db/inputs.yaml"
CANDIDATES = ("a", "b", "c", "d")


def _candidate_fragment(candidate: str) -> dict[str, object]:
    return load_resolved_config(STUDY / f"methods/nersi_candidate_{candidate}.yaml")


def _without_candidate_dimensions(config: dict[str, object]) -> dict[str, object]:
    normalized = deepcopy(config)
    model = normalized["model"]
    training = normalized["training"]
    assert isinstance(model, dict) and isinstance(training, dict)
    for key in ("encoder_width", "latent_channels", "decoder_channels"):
        del model[key]
    del training["learning_rate"]
    return normalized


def _contains_key(value: object, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, key) for item in value)
    return False


def test_inputs_reuse_the_exact_study_032_qc_validation_contract() -> None:
    inputs = load_resolved_config(STUDY / "inputs.yaml")
    reference = load_resolved_config(REFERENCE_INPUTS)

    assert inputs["frozen_suite"] == reference["frozen_suite"]
    assert (
        inputs["required_cases"]
        == reference["required_cases"]
        == ["c3_benchmark_validation_random_trace_80_seed142"]
    )
    assert inputs["validation_contract"] == reference["validation_contract"]
    assert inputs["validation_contract"]["shape"] == [384, 9, 32, 8, 32]
    assert inputs["validation_contract"]["selection"] == {
        "time": [0, 384],
        "source_line": [41, 50],
        "shot_in_line": [32, 64],
        "relative_receiver_x": [0, 8],
        "relative_receiver_y": [18, 50],
    }
    assert inputs["outputs"]["existing_directory_policy"] == "reject"
    assert not _contains_key(inputs, "test")


def test_four_explicit_candidate_plans_bind_each_native_fragment_once() -> None:
    expected_plans = {f"config_candidate_{candidate}.yaml" for candidate in CANDIDATES}
    assert {path.name for path in STUDY.glob("config_candidate_*.yaml")} == expected_plans

    native_fragments = []
    for candidate in CANDIDATES:
        plan = load_resolved_config(STUDY / f"config_candidate_{candidate}.yaml")
        assert "extends" not in plan
        assert plan["study"] == {"id": "study_034_c3_nersi_baseline", "status": "planned"}
        assert plan["seeds"] == {"partition": 42, "training": 20260908}
        assert set(plan["methods"]) == {"nersi"}
        native_fragment = plan["methods"]["nersi"]["native_fragment"]
        assert native_fragment == f"methods/nersi_candidate_{candidate}.yaml"
        assert (STUDY / native_fragment).is_file()
        native_fragments.append(native_fragment)
        assert plan["completion"]["candidate_set"] == list(CANDIDATES)
        assert plan["completion"]["automatic_quality_retries"] is False
        assert plan["completion"]["test_execution"] is False
        assert plan["completion"]["snr_threshold"] is None
        assert plan["evaluation"]["primary_metric"] == ("physical_amplitude_global_snr_db")
        assert plan["evaluation"]["domain"] == "evaluation_target"

    assert len(native_fragments) == len(set(native_fragments)) == 4


def test_candidates_differ_only_in_declared_capacity_and_learning_rate() -> None:
    fragments = {candidate: _candidate_fragment(candidate) for candidate in CANDIDATES}
    expected_fragments = {f"nersi_candidate_{candidate}.yaml" for candidate in CANDIDATES}
    assert {
        path.name for path in (STUDY / "methods").glob("nersi_candidate_*.yaml")
    } == expected_fragments

    expected_capacity = {
        "a": (256, 64, [64, 32, 16], 0.001),
        "b": (256, 64, [64, 32, 16], 0.0003),
        "c": (384, 96, [96, 48, 24], 0.001),
        "d": (384, 96, [96, 48, 24], 0.0003),
    }
    common = _without_candidate_dimensions(fragments["a"])
    for candidate, fragment in fragments.items():
        assert set(fragment) == {"model", "training", "prediction", "evaluation"}
        model = fragment["model"]
        training = fragment["training"]
        assert isinstance(model, dict) and isinstance(training, dict)
        assert (
            model["encoder_width"],
            model["latent_channels"],
            model["decoder_channels"],
            training["learning_rate"],
        ) == expected_capacity[candidate]
        assert _without_candidate_dimensions(fragment) == common

        assert model["name"] == "nersi"
        assert model["coordinate_order"] == [
            "source_line",
            "shot_in_line",
            "relative_receiver_x",
        ]
        assert model["fourier_components"] == 40
        assert model["frequency_schedule"] == "exponential"
        assert model["frequency_base"] == 1.25
        assert model["upsample_scales"] == [2, 2, 2]
        assert model["kernel_size"] == 3
        assert model["activation"] == "gelu"
        assert model["output_activation"] == "linear"
        assert training == {
            "random_seed": 20260908,
            "optimizer": "adam",
            "loss": "observed_masked_mse",
            "amplitude_scaling": "observed_volume_global_rms",
            "learning_rate": expected_capacity[candidate][-1],
            "profiles_per_step": 16,
            "max_steps": 5000,
            "report_interval": 100,
            "device": "cuda:0",
        }
        assert fragment["prediction"] == {"batch_size": 64}
        assert fragment["evaluation"] == {
            "primary_metric": "physical_amplitude_global_snr_db",
            "domain": "evaluation_target",
        }
        assert not _contains_key(fragment, "nuclear_norm")


def test_preflight_uses_maximum_capacity_candidate_c_without_changing_full_budget() -> None:
    plan = load_resolved_config(STUDY / "config_preflight.yaml")
    fragment = load_resolved_config(STUDY / plan["methods"]["nersi"]["native_fragment"])
    candidate_c = _candidate_fragment("c")
    preflight = plan["methods"]["nersi"]["preflight"]

    assert "extends" not in plan
    assert fragment == candidate_c
    assert fragment["training"]["max_steps"] == 5000
    assert preflight == {
        "smoke_steps": 10,
        "prediction_scope": "one_configured_profile_batch_or_fewer",
        "target_scoring": False,
        "full_prediction": False,
        "state_reused_by_full_run": False,
    }
    assert plan["completion"]["test_execution"] is False


def test_contract_labels_out_of_scope_work_and_index_status_once() -> None:
    contract = (STUDY / "implementation_contract.md").read_text(encoding="utf-8")
    for heading in (
        "## Paper-specified",
        "## Repository reimplementation choice",
        "## Unspecified / unresolved",
        "## Explicitly out of scope",
    ):
        assert heading in contract
    assert "nuclear norm: out of scope for initial C3 baseline" in contract

    index = (REPOSITORY_ROOT / "studies/README.md").read_text(encoding="utf-8")
    matching_rows = [
        line for line in index.splitlines() if line.startswith("| [study_034_c3_nersi_baseline]")
    ]
    assert len(matching_rows) == 1
    assert "| `validation_complete_candidate_c_selected` " in matching_rows[0]
    assert "Candidate C 15.9189 dB" in matching_rows[0]


def test_validation_result_selects_highest_final_snr_and_fixes_artifact_hashes() -> None:
    summary = json.loads((RESULT / "summary.json").read_text(encoding="utf-8"))
    decision = json.loads((RESULT / "adoption_decision.json").read_text(encoding="utf-8"))
    manifest = json.loads((RESULT / "manifest.json").read_text(encoding="utf-8"))
    with (RESULT / "candidate_comparison.csv").open(newline="", encoding="utf-8") as stream:
        candidates = list(csv.DictReader(stream))

    assert summary["status"] == "validation_complete_candidate_c_selected"
    assert summary["selected_candidate"] == decision["candidate"] == "C"
    assert len(candidates) == 4
    assert [row["candidate"] for row in candidates] == [
        candidate.upper() for candidate in CANDIDATES
    ]
    selected = [row for row in candidates if row["selected"] == "true"]
    assert len(selected) == 1 and selected[0]["candidate"] == "C"
    assert float(selected[0]["snr_db"]) == max(float(row["snr_db"]) for row in candidates)
    assert float(selected[0]["snr_db"]) == decision["physical_target_metrics"]["snr_db"]
    assert decision["test_partition_used"] is False
    for name, digest in manifest["artifact_sha256"].items():
        assert file_sha256(RESULT / name) == digest
    assert manifest["audit_results"]["candidate_d_first"]["status"] == "failed_verification"
    assert manifest["audit_results"]["candidate_d_recorded_settings_repeat"]["status"] == "success"
