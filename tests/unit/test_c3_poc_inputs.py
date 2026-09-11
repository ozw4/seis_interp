from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seis_interp.configuration import ConfigurationError
from seis_interp.data import c3_poc_inputs
from seis_interp.data.c3_poc_inputs import (
    C3_RANDOM80_POC_BENCHMARK_ID,
    C3_RANDOM80_POC_DATASET_ID,
    C3_RANDOM80_POC_SELECTION,
    load_c3_random80_poc_inputs,
)
from seis_interp.data.c3_volume_adapter import ObservedC3Volume
from seis_interp.data.c3_volume_run_inputs import C3VolumeRunInputs, load_c3_volume_run_inputs
from seis_interp.data.interpolation_mask_store import load_interpolation_mask
from seis_interp.processing.c3_benchmark_contract import MAIN_C3_DIMENSIONS, C3BenchmarkDimensions
from seis_interp.processing.c3_volume_index import VOLUME_AXIS_ORDER
from seis_interp.processing.interpolation_masks import (
    EVALUATION_TARGET_ROLE,
    OBSERVATION_ROLE_COLUMN,
    OBSERVED_ROLE,
    RANDOM_WHOLE_FFID_MASK_KIND,
)
from tests.fixtures.c3_volume_run_artifacts import prepare_c3_volume_run_artifacts


def _fixture_dimensions(shape: tuple[int, ...]) -> C3BenchmarkDimensions:
    return C3BenchmarkDimensions(
        time_range=(0, shape[0]),
        sail_line_numbers=(2, 3),
        shape=shape,
    )


def _artifact_arguments(artifacts: object) -> dict[str, Path]:
    return {
        "interim_dir": artifacts.interim,
        "processed_dir": artifacts.processed,
        "mask_dir": artifacts.mask,
        "case_dir": artifacts.case,
        "volume_dir": artifacts.volume,
    }


def _poc_config(metadata: Mapping[str, object]) -> dict[str, object]:
    return {
        "benchmark_volume": {
            "selection": deepcopy(metadata["selection"]),
        }
    }


def _memory_inputs() -> tuple[C3VolumeRunInputs, C3BenchmarkDimensions]:
    shape = (2, 1, 1, 1, 5)
    spatial_shape = shape[1:]
    observed = np.zeros(spatial_shape, dtype=np.bool_)
    observed[..., 0] = True
    target = ~observed
    values = np.zeros(shape, dtype=np.float32)
    values[:, observed] = np.array([[2.0], [-3.0]], dtype=np.float32)
    volume = ObservedC3Volume(
        values=values,
        time_s=np.array([0.0, 0.008]),
        array_rows=np.arange(5, dtype=np.int64).reshape(spatial_shape),
        observed_trace_mask=observed,
        evaluation_target_trace_mask=target,
    )
    case = {
        "case_id": "memory_case",
        "dataset_id": C3_RANDOM80_POC_DATASET_ID,
        "mask": {
            "kind": "random_trace",
            "missing_fraction": 0.8,
            "random_seed": 17,
        },
        "input_files": {
            "mask": {
                "observation_mask.parquet": {"sha256": "a" * 64},
                "interpolation_mask.json": {"sha256": "b" * 64},
            }
        },
    }
    metadata = {
        "volume_id": "memory_volume",
        "shape": list(shape),
        "selection": {
            "time": [0, 2],
            "source_line": [7, 8],
            "shot_in_line": [3, 4],
            "relative_receiver_x": [5, 6],
            "relative_receiver_y": [10, 15],
        },
        "role_counts": {
            OBSERVED_ROLE: 1,
            EVALUATION_TARGET_ROLE: 4,
        },
    }
    inputs = C3VolumeRunInputs(
        observed_volume=volume,
        index_table=pd.DataFrame({"array_row": np.arange(5, dtype=np.int64)}),
        case=case,
        volume_metadata=metadata,
        inputs_lock={
            "benchmark_case": {"case_id": "memory_case"},
            "benchmark_volume": {"volume_id": "memory_volume"},
        },
    )
    dimensions = C3BenchmarkDimensions((0, 2), (7, 7), shape)
    return inputs, dimensions


def _load_memory_inputs(
    monkeypatch: pytest.MonkeyPatch,
    inputs: C3VolumeRunInputs,
    dimensions: C3BenchmarkDimensions,
) -> C3VolumeRunInputs:
    calls: list[dict[str, object]] = []

    def load_once(**arguments: object) -> C3VolumeRunInputs:
        calls.append(arguments)
        return inputs

    monkeypatch.setattr(c3_poc_inputs, "load_c3_volume_run_inputs", load_once)
    config = _poc_config(inputs.volume_metadata)
    result = load_c3_random80_poc_inputs(
        config=config,
        interim_dir=Path("interim"),
        processed_dir=Path("processed"),
        mask_dir=Path("mask"),
        case_dir=Path("case"),
        volume_dir=Path("volume"),
        dimensions=dimensions,
    )
    assert len(calls) == 1
    assert calls[0]["config"] == config
    return result


def test_loads_small_artifacts_as_existing_volume_run_inputs(tmp_path: Path) -> None:
    artifacts = prepare_c3_volume_run_artifacts(
        tmp_path,
        dataset_id=C3_RANDOM80_POC_DATASET_ID,
        missing_fraction=0.8,
    )
    shape = tuple(artifacts.volume_metadata["shape"])

    inputs = load_c3_random80_poc_inputs(
        **_artifact_arguments(artifacts),
        config=_poc_config(artifacts.volume_metadata),
        dimensions=_fixture_dimensions(shape),
    )

    assert isinstance(inputs, C3VolumeRunInputs)
    assert inputs.observed_volume.values.shape == shape
    assert inputs.inputs_lock["benchmark_id"] == C3_RANDOM80_POC_BENCHMARK_ID
    assert inputs.inputs_lock["case_id"] == inputs.case["case_id"]
    assert inputs.inputs_lock["dataset_id"] == C3_RANDOM80_POC_DATASET_ID
    assert inputs.inputs_lock["volume_id"] == inputs.volume_metadata["volume_id"]
    assert inputs.inputs_lock["selection"] == inputs.volume_metadata["selection"]
    assert inputs.inputs_lock["shape"] == list(shape)
    assert (
        inputs.inputs_lock["benchmark_volume"]["files"]
        == load_c3_volume_run_inputs(
            config=_poc_config(artifacts.volume_metadata),
            **_artifact_arguments(artifacts),
        ).inputs_lock["benchmark_volume"]["files"]
    )
    json.dumps(inputs.inputs_lock, allow_nan=False)


@pytest.mark.parametrize(
    "config",
    [{}, {"benchmark_volume": {"selection": {}}}],
)
def test_requires_resolved_benchmark_volume_selection(config: dict[str, object]) -> None:
    with pytest.raises(
        ConfigurationError,
        match="benchmark_volume.selection",
    ):
        load_c3_random80_poc_inputs(
            config=config,
            interim_dir=Path("interim"),
            processed_dir=Path("processed"),
            mask_dir=Path("mask"),
            case_dir=Path("case"),
            volume_dir=Path("volume"),
        )


@pytest.mark.parametrize(
    "axis",
    ["shot_in_line", "relative_receiver_x", "relative_receiver_y"],
)
def test_rejects_same_shape_volume_selection_shift(
    tmp_path: Path,
    axis: str,
) -> None:
    artifacts = prepare_c3_volume_run_artifacts(
        tmp_path,
        dataset_id=C3_RANDOM80_POC_DATASET_ID,
        missing_fraction=0.8,
    )
    config = _poc_config(artifacts.volume_metadata)
    selection = config["benchmark_volume"]["selection"]
    selection[axis] = [bound + 1 for bound in selection[axis]]

    with pytest.raises(
        ConfigurationError,
        match="benchmark_volume.selection does not match the verified volume artifact",
    ):
        load_c3_random80_poc_inputs(
            **_artifact_arguments(artifacts),
            config=config,
            dimensions=_fixture_dimensions(tuple(artifacts.volume_metadata["shape"])),
        )


def test_rejects_non_seg_c3_na_dataset(tmp_path: Path) -> None:
    artifacts = prepare_c3_volume_run_artifacts(tmp_path, missing_fraction=0.8)

    with pytest.raises(ValueError, match="PoC dataset_id must be 'seg_c3_na'"):
        load_c3_random80_poc_inputs(
            **_artifact_arguments(artifacts),
            config=_poc_config(artifacts.volume_metadata),
            dimensions=_fixture_dimensions(tuple(artifacts.volume_metadata["shape"])),
        )


@pytest.mark.parametrize("axis", VOLUME_AXIS_ORDER)
@pytest.mark.parametrize("explicit_dimensions", [False, True])
def test_rejects_config_and_artifact_shift_before_loading(
    monkeypatch: pytest.MonkeyPatch, axis: str, explicit_dimensions: bool
) -> None:
    inputs, _ = _memory_inputs()
    metadata = deepcopy(inputs.volume_metadata)
    metadata["shape"] = list(MAIN_C3_DIMENSIONS.shape)
    metadata["selection"] = deepcopy(C3_RANDOM80_POC_SELECTION)
    metadata["selection"][axis] = [bound + 1 for bound in metadata["selection"][axis]]
    inputs = replace(inputs, volume_metadata=metadata)
    config = _poc_config(metadata)
    assert config["benchmark_volume"]["selection"] == inputs.volume_metadata["selection"]
    assert [stop - start for start, stop in metadata["selection"].values()] == metadata["shape"]
    calls = []

    def load_shifted_artifact(**kwargs):
        calls.append(kwargs)
        return inputs

    monkeypatch.setattr(c3_poc_inputs, "load_c3_volume_run_inputs", load_shifted_artifact)
    dimensions = {"dimensions": replace(MAIN_C3_DIMENSIONS)} if explicit_dimensions else {}
    with pytest.raises(ConfigurationError, match="must match the fixed PoC selection"):
        load_c3_random80_poc_inputs(
            config=config,
            interim_dir=Path("interim"),
            processed_dir=Path("processed"),
            mask_dir=Path("mask"),
            case_dir=Path("case"),
            volume_dir=Path("volume"),
            **dimensions,
        )
    assert calls == []


def test_fixed_benchmark_identity_matches_specification():
    assert C3_RANDOM80_POC_BENCHMARK_ID == "c3_sl25_40_random80_observed_only_v1"
    assert C3_RANDOM80_POC_SELECTION == {
        "time": [0, 384],
        "source_line": [25, 41],
        "shot_in_line": [28, 60],
        "relative_receiver_x": [0, 8],
        "relative_receiver_y": [18, 50],
    }
    assert c3_poc_inputs.C3_RANDOM80_MISSING_FRACTION == 0.8


def test_rejects_overlapping_observed_and_target_masks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, dimensions = _memory_inputs()
    volume = inputs.observed_volume
    overlapping = volume.evaluation_target_trace_mask.copy()
    overlapping[volume.observed_trace_mask] = True
    inputs = replace(
        inputs,
        observed_volume=replace(volume, evaluation_target_trace_mask=overlapping),
    )

    with pytest.raises(ValueError, match="must not overlap"):
        _load_memory_inputs(monkeypatch, inputs, dimensions)


def test_rejects_unclassified_qc_domain_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs, dimensions = _memory_inputs()
    volume = inputs.observed_volume
    target = volume.evaluation_target_trace_mask.copy()
    target[..., -1] = False
    inputs = replace(
        inputs,
        observed_volume=replace(volume, evaluation_target_trace_mask=target),
    )

    with pytest.raises(ValueError, match="exactly cover the QC domain"):
        _load_memory_inputs(monkeypatch, inputs, dimensions)


@pytest.mark.parametrize(
    ("corruption", "match"),
    [
        ("shape", "volume shape"),
        ("mask_dtype", "boolean trace mask"),
        ("mask_kind", "mask kind"),
        ("missing_fraction", "missing_fraction"),
        ("time_range", "time range"),
        ("sail_lines", "source-line range"),
    ],
)
def test_rejects_wrong_shape_mask_contract_or_domain(
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
    match: str,
) -> None:
    inputs, dimensions = _memory_inputs()
    if corruption == "shape":
        metadata = deepcopy(inputs.volume_metadata)
        metadata["shape"] = [2, 1, 1, 1, 4]
        inputs = replace(inputs, volume_metadata=metadata)
    elif corruption == "mask_dtype":
        volume = inputs.observed_volume
        inputs = replace(
            inputs,
            observed_volume=replace(
                volume,
                observed_trace_mask=volume.observed_trace_mask.astype(np.uint8),
            ),
        )
    elif corruption in ("mask_kind", "missing_fraction"):
        case = deepcopy(inputs.case)
        case["mask"]["kind" if corruption == "mask_kind" else "missing_fraction"] = (
            RANDOM_WHOLE_FFID_MASK_KIND if corruption == "mask_kind" else 0.5
        )
        inputs = replace(inputs, case=case)
    else:
        metadata = deepcopy(inputs.volume_metadata)
        key = "time" if corruption == "time_range" else "source_line"
        metadata["selection"][key] = [1, 3] if key == "time" else [6, 7]
        inputs = replace(inputs, volume_metadata=metadata)

    with pytest.raises(ValueError, match=match):
        _load_memory_inputs(monkeypatch, inputs, dimensions)


def test_target_truth_sentinel_is_not_materialized(tmp_path: Path) -> None:
    sentinel = 9.876543e19
    artifacts = prepare_c3_volume_run_artifacts(
        tmp_path,
        dataset_id=C3_RANDOM80_POC_DATASET_ID,
        missing_fraction=0.8,
        target_offset=sentinel,
    )
    shape = tuple(artifacts.volume_metadata["shape"])
    mask, _ = load_interpolation_mask(artifacts.mask)
    target_rows = mask.loc[
        mask[OBSERVATION_ROLE_COLUMN].eq(EVALUATION_TARGET_ROLE),
        "array_row",
    ].to_numpy(dtype=np.int64)
    source = np.load(artifacts.interim / "amplitudes.npy", allow_pickle=False)
    assert np.all(source[target_rows] > sentinel / 2.0)

    inputs = load_c3_random80_poc_inputs(
        **_artifact_arguments(artifacts),
        config=_poc_config(artifacts.volume_metadata),
        dimensions=_fixture_dimensions(shape),
    )

    volume = inputs.observed_volume
    assert np.all(volume.values[:, volume.evaluation_target_trace_mask] == 0)
    assert not np.any(volume.values == np.float32(sentinel))


def test_lock_counts_match_materialized_volume_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, dimensions = _memory_inputs()

    loaded = _load_memory_inputs(monkeypatch, inputs, dimensions)

    role_counts = inputs.volume_metadata["role_counts"]
    assert loaded.inputs_lock["observed_trace_count"] == role_counts[OBSERVED_ROLE]
    assert loaded.inputs_lock["target_trace_count"] == role_counts[EVALUATION_TARGET_ROLE]
    assert loaded.inputs_lock["mask"] == {
        "kind": "random_trace",
        "missing_fraction": 0.8,
        "random_seed": 17,
        "unit": "complete_trace",
        "files": inputs.case["input_files"]["mask"],
    }


def test_existing_canonical_loader_remains_usable(tmp_path: Path) -> None:
    artifacts = prepare_c3_volume_run_artifacts(tmp_path)

    inputs = load_c3_volume_run_inputs(config={}, **_artifact_arguments(artifacts))

    assert isinstance(inputs, C3VolumeRunInputs)
    assert set(inputs.inputs_lock) == {"benchmark_case", "benchmark_volume"}


def test_poc_check_reads_only_observations_and_leaves_inputs_unchanged(tmp_path, monkeypatch):
    from seis_interp.evaluation import c3_volume_metrics
    from seis_interp.pipelines.check_c3_poc import check_c3_poc_inputs
    from seis_interp.training.amplitude_scaling import compute_observed_global_rms

    artifacts = prepare_c3_volume_run_artifacts(
        tmp_path,
        dataset_id=C3_RANDOM80_POC_DATASET_ID,
        missing_fraction=0.8,
        target_offset=1.0e20,
    )

    def unexpected_truth(*args, **kwargs):
        raise AssertionError("poc check must not read target truth")

    monkeypatch.setattr(c3_volume_metrics, "evaluate_c3_volume_prediction", unexpected_truth)
    mask, _ = load_interpolation_mask(artifacts.mask)
    observed_rows = mask.loc[mask[OBSERVATION_ROLE_COLUMN].eq(OBSERVED_ROLE), "array_row"]
    original_load = np.load
    accessed_rows = []

    class ObservedOnlyArray(np.ndarray):
        def __getitem__(self, index):
            rows, time_slice = index
            assert np.isin(rows, observed_rows).all()
            accessed_rows.extend(rows.tolist())
            return self.view(np.ndarray)[rows, time_slice]

    def guarded_load(path, *args, **kwargs):
        values = original_load(path, *args, **kwargs)
        if Path(path) == artifacts.interim / "amplitudes.npy":
            return values.view(ObservedOnlyArray)
        return values

    monkeypatch.setattr(np, "load", guarded_load)
    originals = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    shape = tuple(artifacts.volume_metadata["shape"])
    result = check_c3_poc_inputs(
        **_artifact_arguments(artifacts),
        selection=artifacts.volume_metadata["selection"],
        dimensions=_fixture_dimensions(shape),
    )
    assert len(accessed_rows) == result["observed_trace_count"]
    inputs = load_c3_random80_poc_inputs(
        **_artifact_arguments(artifacts),
        config=_poc_config(artifacts.volume_metadata),
        dimensions=_fixture_dimensions(shape),
    )
    observed = inputs.observed_volume
    assert result["shape"] == list(shape)
    assert result["observed_trace_count"] == int(observed.observed_trace_mask.sum())
    assert result["target_trace_count"] == int(observed.evaluation_target_trace_mask.sum())
    assert result["observed_global_rms"] == compute_observed_global_rms(
        observed.values,
        observed.observed_trace_mask,
    )
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == originals


def test_poc_check_uses_fixed_full_selection(monkeypatch):
    from seis_interp.pipelines import check_c3_poc

    def check_selection(**kwargs):
        assert kwargs["config"]["benchmark_volume"]["selection"] == {
            "time": [0, 384],
            "source_line": [25, 41],
            "shot_in_line": [28, 60],
            "relative_receiver_x": [0, 8],
            "relative_receiver_y": [18, 50],
        }
        raise ValueError("checked expected selection")

    monkeypatch.setattr(check_c3_poc, "load_c3_random80_poc_inputs", check_selection)
    with pytest.raises(ValueError, match="checked expected selection"):
        check_c3_poc.check_c3_poc_inputs(
            **{
                key: Path(key)
                for key in ("interim_dir", "processed_dir", "mask_dir", "case_dir", "volume_dir")
            }
        )
