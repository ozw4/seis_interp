from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path, PureWindowsPath

import pytest

from seis_interp.data.benchmark_case_inputs import verify_benchmark_case_inputs
from seis_interp.data.benchmark_case_store import (
    BENCHMARK_CASE_FILE_NAME,
    load_benchmark_case,
)
from seis_interp.data.file_checksums import file_sha256
from seis_interp.data.interpolation_mask_store import (
    MASK_TABLE_FILE_NAME,
    load_interpolation_mask,
)
from seis_interp.data.interpolation_mask_store import (
    OUTPUT_FILE_NAMES as MASK_FILE_NAMES,
)
from seis_interp.data.prepared_partition import (
    NORMALIZATION_FILE_NAME,
)
from seis_interp.data.prepared_partition import (
    OUTPUT_FILE_NAMES as PREPARED_FILE_NAMES,
)
from seis_interp.data.trace_store import (
    AMPLITUDES_FILE_NAME,
)
from seis_interp.data.trace_store import (
    OUTPUT_FILE_NAMES as INTERIM_FILE_NAMES,
)
from seis_interp.pipelines.prepare_benchmark_case import prepare_benchmark_case
from tests.fixtures.benchmark_case_artifacts import (
    CONFIG_SOURCE,
    prepare_benchmark_case_artifacts,
)

_FORBIDDEN_KEYS = frozenset(
    (
        "model",
        "training",
        "metrics",
        "prediction",
        "checkpoint",
        "timestamp",
        "schema_version",
    )
)


def _artifact_bytes(directories: Sequence[tuple[Path, tuple[str, ...]]]) -> dict[Path, bytes]:
    return {
        directory / file_name: (directory / file_name).read_bytes()
        for directory, file_names in directories
        for file_name in file_names
    }


def _expected_input_hashes(
    interim: Path,
    processed: Path,
    mask: Path,
) -> dict[str, dict[str, dict[str, str]]]:
    return {
        "interim": {
            file_name: {"sha256": file_sha256(interim / file_name)}
            for file_name in INTERIM_FILE_NAMES
        },
        "processed": {
            file_name: {"sha256": file_sha256(processed / file_name)}
            for file_name in PREPARED_FILE_NAMES
        },
        "mask": {
            file_name: {"sha256": file_sha256(mask / file_name)} for file_name in MASK_FILE_NAMES
        },
    }


def _assert_portable_model_independent_content(value: object) -> None:
    if isinstance(value, Mapping):
        assert _FORBIDDEN_KEYS.isdisjoint(value)
        for nested in value.values():
            _assert_portable_model_independent_content(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_portable_model_independent_content(nested)
    elif isinstance(value, str):
        assert not Path(value).is_absolute()
        assert not PureWindowsPath(value).is_absolute()


def test_partition_mask_and_case_form_one_immutable_benchmark_contract(
    tmp_path: Path,
) -> None:
    interim, processed, mask = prepare_benchmark_case_artifacts(tmp_path)
    case_id = "synthetic_test_seed42"
    output = processed / "cases" / case_id
    protected_inputs = (
        (processed, PREPARED_FILE_NAMES),
        (mask, MASK_FILE_NAMES),
    )
    before = _artifact_bytes(protected_inputs)

    prepare_benchmark_case(
        interim,
        processed,
        mask,
        output,
        case_id=case_id,
        config_source=CONFIG_SOURCE,
    )
    case = load_benchmark_case(output)
    verify_benchmark_case_inputs(
        case,
        interim_dir=interim,
        processed_dir=processed,
        mask_dir=mask,
    )
    _, mask_metadata = load_interpolation_mask(mask)

    assert len({processed, mask, output}) == 3
    assert _artifact_bytes(protected_inputs) == before
    assert case["input_files"] == _expected_input_hashes(interim, processed, mask)
    assert case["partition"] == mask_metadata["partition"]
    case_mask = case["mask"]
    assert isinstance(case_mask, dict)
    for key in (
        "kind",
        "missing_fraction",
        "random_seed",
        "candidate_trace_count",
        "candidate_ffid_count",
        "counts",
        "duplicate_physical_coordinates",
    ):
        assert case_mask[key] == mask_metadata[key]
    assert case["role_contract"] == {
        "domain": "canonical_present_traces",
        "observed_role": "observed",
        "evaluation_target_role": "evaluation_target",
        "evaluation_target_amplitude_use": "scoring_only",
    }
    _assert_portable_model_independent_content(case)
    assert [path.name for path in output.iterdir()] == [BENCHMARK_CASE_FILE_NAME]
    assert not (output / MASK_TABLE_FILE_NAME).exists()

    second_output = processed / "cases" / "same-inputs-second-case"
    second_case = prepare_benchmark_case(
        interim,
        processed,
        mask,
        second_output,
        case_id="same-inputs-second-case",
        config_source=CONFIG_SOURCE,
    )

    assert second_case["input_files"] == case["input_files"]
    assert _artifact_bytes(protected_inputs) == before


@pytest.mark.parametrize(
    ("group", "file_name"),
    [
        ("interim", AMPLITUDES_FILE_NAME),
        ("processed", NORMALIZATION_FILE_NAME),
        ("mask", MASK_TABLE_FILE_NAME),
    ],
)
def test_benchmark_case_verification_detects_tampered_artifact(
    tmp_path: Path,
    group: str,
    file_name: str,
) -> None:
    interim, processed, mask = prepare_benchmark_case_artifacts(tmp_path)
    case = prepare_benchmark_case(
        interim,
        processed,
        mask,
        processed / "case",
        case_id="tamper_detection",
        config_source=CONFIG_SOURCE,
    )
    directories = {"interim": interim, "processed": processed, "mask": mask}
    changed_path = directories[group] / file_name
    changed_path.write_bytes(changed_path.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="input_files"):
        verify_benchmark_case_inputs(
            case,
            interim_dir=interim,
            processed_dir=processed,
            mask_dir=mask,
        )
