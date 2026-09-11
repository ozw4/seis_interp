from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from seis_interp.cli import main
from seis_interp.configuration import load_resolved_config
from seis_interp.data.c3_poc_inputs import (
    C3_RANDOM80_POC_BENCHMARK_ID,
    C3_RANDOM80_POC_DATASET_ID,
    load_c3_random80_poc_inputs,
)
from seis_interp.data.c3_volume_index_store import load_c3_volume_index
from seis_interp.data.file_checksums import file_sha256
from seis_interp.evaluation.c3_volume_metrics import evaluate_c3_volume_prediction
from seis_interp.pipelines import interpolate_nersi as pipeline
from seis_interp.pipelines.interpolate_nersi import (
    CHECKPOINT_RELATIVE_PATH,
    METHOD,
    METHOD_VARIANT,
    PREDICTION_RELATIVE_PATH,
    interpolate_nersi_run,
)
from seis_interp.processing.c3_benchmark_contract import C3BenchmarkDimensions
from seis_interp.training.amplitude_scaling import compute_observed_global_rms
from seis_interp.training.c3_volume_nersi_data import (
    PROFILE_AXIS_ORDER,
    PROFILE_COORDINATE_ORDER,
    build_c3_volume_nersi_data,
)
from seis_interp.training.c3_volume_nersi_prediction import predict_c3_volume_nersi
from seis_interp.training.nersi_checkpoints import (
    load_fixed_step_nersi_checkpoint,
    nersi_checkpoint_input_binding,
    validate_fixed_step_nersi_checkpoint_input_binding,
)
from tests.fixtures.c3_volume_run_artifacts import (
    PreparedC3VolumeRunArtifacts,
    prepare_c3_volume_run_artifacts,
)


@pytest.fixture(scope="module")
def nersi_artifacts(tmp_path_factory: pytest.TempPathFactory) -> PreparedC3VolumeRunArtifacts:
    return _prepare_poc_artifacts(tmp_path_factory.mktemp("nersi-volume"))


def _prepare_poc_artifacts(
    path: Path, *, target_offset: float = 0.0
) -> PreparedC3VolumeRunArtifacts:
    return prepare_c3_volume_run_artifacts(
        path,
        dataset_id=C3_RANDOM80_POC_DATASET_ID,
        missing_fraction=0.8,
        target_offset=target_offset,
        time_sample_count=8,
        receiver_y_count=8,
    )


@pytest.fixture(autouse=True)
def _use_synthetic_poc_dimensions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "load_c3_random80_poc_inputs", _load_synthetic_poc_inputs)


def _load_synthetic_poc_inputs(**kwargs):
    _, metadata = load_c3_volume_index(kwargs["volume_dir"])
    selection = metadata["selection"]
    source_start, source_stop = selection["source_line"]
    dimensions = C3BenchmarkDimensions(
        time_range=tuple(selection["time"]),
        sail_line_numbers=(source_start, source_stop - 1),
        shape=tuple(metadata["shape"]),
    )
    return load_c3_random80_poc_inputs(**kwargs, dimensions=dimensions)


def _config(artifacts: PreparedC3VolumeRunArtifacts, *, training_seed: int = 314) -> dict:
    case = json.loads((artifacts.case / "benchmark_case.json").read_text(encoding="utf-8"))
    return {
        "project": {"random_seed": 42},
        "interpolation_mask": {
            "partition": "test",
            "kind": artifacts.mask_kind,
            "missing_fraction": case["mask"]["missing_fraction"],
        },
        "benchmark_case": {"id": "synthetic_case"},
        "benchmark_volume": {
            "id": "synthetic_volume",
            "selection": artifacts.volume_metadata["selection"],
        },
        "model": {
            "name": "nersi",
            "coordinate_order": list(PROFILE_COORDINATE_ORDER),
            "fourier_components": 2,
            "frequency_schedule": "exponential",
            "frequency_base": 1.25,
            "encoder_width": 8,
            "latent_channels": 2,
            "decoder_channels": [2, 2, 1],
            "upsample_scales": [2, 2, 2],
            "kernel_size": 1,
            "activation": "gelu",
            "output_activation": "linear",
        },
        "training": {
            "model_initialization_seed": training_seed,
            "sampling_seed": 201,
            "optimizer": "adam",
            "loss": "masked_trace_relative_mse",
            "amplitude_scaling": "observed_volume_global_rms",
            "learning_rate": 1.0e-3,
            "profiles_per_step": 1,
            "max_steps": 2,
            "report_interval": 1,
            "device": "cpu",
        },
        "prediction": {"batch_size": 5},
        "evaluation": {
            "primary_metric": "physical_amplitude_global_snr_db",
            "domain": "evaluation_target",
        },
    }


def _write_config(
    path: Path,
    artifacts: PreparedC3VolumeRunArtifacts,
    *,
    training_seed: int = 314,
    contents: dict | None = None,
) -> Path:
    value = _config(artifacts, training_seed=training_seed) if contents is None else contents
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def _run(
    artifacts: PreparedC3VolumeRunArtifacts,
    config: Path,
    output: Path,
    *,
    progress_reporter=None,
) -> dict[str, object]:
    return interpolate_nersi_run(
        config_path=config,
        interim_dir=artifacts.interim,
        processed_dir=artifacts.processed,
        mask_dir=artifacts.mask,
        case_dir=artifacts.case,
        volume_dir=artifacts.volume,
        output_dir=output,
        progress_reporter=progress_reporter,
    )


def _loaded_inputs(artifacts: PreparedC3VolumeRunArtifacts, config: Path):
    return _load_synthetic_poc_inputs(
        config=load_resolved_config(config),
        interim_dir=artifacts.interim,
        processed_dir=artifacts.processed,
        mask_dir=artifacts.mask,
        case_dir=artifacts.case,
        volume_dir=artifacts.volume,
    )


def test_model_initialization_and_profile_sampling_seeds_are_independent(
    tmp_path, nersi_artifacts, monkeypatch
):
    original_train = pipeline.train_nersi_fixed_steps
    captures = []

    def train(model, data, **kwargs):
        initial = deepcopy(model.state_dict())
        batches = []
        hook = model.register_forward_pre_hook(
            lambda _model, args: batches.append(args[0].detach().cpu().clone())
        )
        try:
            result = original_train(model, data, **kwargs)
        finally:
            hook.remove()
        captures.append((initial, batches))
        return result

    monkeypatch.setattr(pipeline, "train_nersi_fixed_steps", train)
    for index, (model_seed, sampling_seed) in enumerate(((101, 201), (102, 201), (101, 202))):
        config = _config(nersi_artifacts)
        config["training"].update(model_initialization_seed=model_seed, sampling_seed=sampling_seed)
        path = tmp_path / f"seed_{index}.yaml"
        path.write_text(yaml.safe_dump(config))
        output = tmp_path / f"seed_{index}"
        _run(nersi_artifacts, path, output)
        loaded = load_fixed_step_nersi_checkpoint(output / "final.pt")
        assert loaded.model_initialization_seed == model_seed
        assert loaded.sampling_seed == sampling_seed
    assert all(torch.equal(value, captures[2][0][key]) for key, value in captures[0][0].items())
    assert any(not torch.equal(value, captures[1][0][key]) for key, value in captures[0][0].items())
    assert all(torch.equal(a, b) for a, b in zip(captures[0][1], captures[1][1], strict=True))
    assert any(not torch.equal(a, b) for a, b in zip(captures[0][1], captures[2][1], strict=True))


def test_tiny_pipeline_artifacts_restore_and_independent_rescore(
    tmp_path: Path,
    nersi_artifacts: PreparedC3VolumeRunArtifacts,
) -> None:
    config = _write_config(tmp_path / "nersi.yaml", nersi_artifacts)
    output = tmp_path / "run"

    metrics = _run(nersi_artifacts, config, output)

    assert sorted(
        path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()
    ) == sorted(
        [
            "final.pt",
            "prediction.npy",
            "config.resolved.yaml",
            "inputs.lock.json",
            "metrics.json",
            "metadata.json",
        ]
    )
    stored_metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    run = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    inputs_lock = json.loads((output / "inputs.lock.json").read_text(encoding="utf-8"))
    prediction = np.load(output / PREDICTION_RELATIVE_PATH, allow_pickle=False)
    inputs = _loaded_inputs(nersi_artifacts, config)
    observed = inputs.observed_volume
    observed_count = int(np.count_nonzero(observed.observed_trace_mask))
    target_count = int(np.count_nonzero(observed.evaluation_target_trace_mask))

    assert stored_metrics == {key: metrics[key] for key in stored_metrics}
    assert set(stored_metrics) == {
        "evaluation_domain",
        "amplitude_domain",
        "evaluation_target",
        "zero_fill",
        "observed_max_abs_error",
    }
    for record in (stored_metrics, run, inputs_lock):
        json.dumps(record, allow_nan=False)
    assert metrics["method"] == METHOD == "nersi"
    assert metrics["method_variant"] == METHOD_VARIANT
    assert metrics["evaluation_domain"] == "evaluation_target"
    assert metrics["amplitude_domain"] == "physical"
    assert metrics["evaluation_target"]["trace_count"] == target_count
    assert metrics["evaluation_target"]["covered_target_trace_count"] == target_count
    assert metrics["evaluation_target"]["sample_count"] == target_count * 8
    assert metrics["observed_max_abs_error"] == 0.0
    assert metrics["uncovered_trace_count"] == metrics["uncovered_sample_count"] == 0
    assert metrics["training"]["steps_completed"] == 2
    assert metrics["training"]["optimizer_updates"] == 2
    assert metrics["optimizer_updates"] == 2
    assert metrics["parameter_count"] > 0
    assert math.isfinite(metrics["training"]["final_batch_loss"])
    assert set(metrics["timing"]) == {
        "fit_seconds",
        "prediction_seconds",
        "end_to_end_seconds",
    }
    assert all(math.isfinite(value) and value >= 0.0 for value in metrics["timing"].values())
    assert {
        "snr_db",
        "rmse",
        "relative_l2",
        "mean_trace_relative_mse",
    }.issubset(metrics["evaluation_target"])

    assert prediction.shape == (8, 2, 3, 2, 8)
    assert prediction.dtype == np.float32
    assert prediction.flags.c_contiguous
    assert np.isfinite(prediction).all()
    np.testing.assert_array_equal(
        prediction[:, observed.observed_trace_mask],
        observed.values[:, observed.observed_trace_mask].astype(np.float32),
    )

    assert run["method_details"]["method_variant"] == METHOD_VARIANT
    assert run["method_details"]["fit_domain"] == "O_only"
    assert run["method_details"]["inner_corruption_mask"] is False
    assert run["normalization"] == {
        "type": "global_rms",
        "source": "O_only",
        "scale": compute_observed_global_rms(observed.values, observed.observed_trace_mask),
    }
    assert run["loss_or_native_objective"] == "masked_trace_relative_mse"
    assert run["method_details"]["checkpoint_role"] == "final"
    assert run["method_details"]["random_seed"] == 42
    assert run["method_details"]["model_initialization_seed"] == 314
    assert run["method_details"]["profiles"]["axis_order"] == list(PROFILE_AXIS_ORDER)
    assert run["method_details"]["profiles"]["coordinate_order"] == list(PROFILE_COORDINATE_ORDER)
    assert run["method_details"]["profiles"]["coordinate_normalization"] == (
        "fixed_analysis_domain_index_bounds_to_unit_interval"
    )
    assert (
        run["method_details"]["profiles"]["coordinate_normalization_source"]
        == "fixed_analysis_domain"
    )
    assert run["method_details"]["profiles"]["coordinate_bounds"] == [[0, 1], [0, 2], [0, 1]]
    assert run["method_details"]["profiles"]["count"] == 12
    assert run["method_details"]["profiles"]["shape"] == [8, 8]
    assert run["method_details"]["profiles"]["stable_order"] == (
        "C_order_source_line_shot_in_line_relative_receiver_x"
    )
    assert 0 < run["method_details"]["profiles"]["training_profile_count"] <= 12
    assert run["method_details"]["parameter_count"] > 0
    assert (
        run["method_details"]["paper_alignment"]["paper_specified"][
            "fourier_components_per_coordinate"
        ]
        == 40
    )
    assert (
        run["method_details"]["paper_alignment"]["repository_reimplementation_choices"][
            "configured_fourier_components_per_coordinate"
        ]
        == 2
    )
    assert run["training_or_reconstruction"]["observed_trace_count"] == observed_count
    assert run["training_or_reconstruction"]["observed_sample_count"] == observed_count * 8
    assert run["method_details"]["prediction"]["shape"] == [8, 2, 3, 2, 8]
    assert run["method_details"]["prediction"]["dtype"] == "float32"
    assert run["method_details"]["prediction"]["sha256"] == file_sha256(
        output / PREDICTION_RELATIVE_PATH
    )
    assert run["coverage"] == {
        "complete": True,
        "boundary_targets_included": True,
        "target_trace_count": target_count,
        "covered_target_trace_count": target_count,
        "target_coverage_fraction": 1.0,
        "uncovered_trace_count": 0,
        "uncovered_sample_count": 0,
    }
    for name in (
        "load_and_verification_seconds",
        "training_data_seconds",
        "fit_seconds",
        "prediction_seconds",
        "end_to_end_seconds",
        "evaluation_seconds",
    ):
        assert math.isfinite(run["timing"][name]) and run["timing"][name] >= 0.0
    assert inputs_lock["benchmark_case"]["sha256"] == file_sha256(
        nersi_artifacts.case / "benchmark_case.json"
    )
    assert inputs_lock["benchmark_volume"]["volume_id"] == "synthetic_volume"
    assert inputs_lock["benchmark_id"] == C3_RANDOM80_POC_BENCHMARK_ID
    assert inputs_lock["dataset_id"] == C3_RANDOM80_POC_DATASET_ID
    assert inputs_lock["mask"]["missing_fraction"] == 0.8
    assert inputs_lock["selection"] == nersi_artifacts.volume_metadata["selection"]
    assert run["method_details"]["checkpoint"]["scope"] == "one_verified_case_volume_only"
    assert run["method_details"]["checkpoint"]["sha256"] == file_sha256(
        output / CHECKPOINT_RELATIVE_PATH
    )
    assert run["method_details"]["checkpoint"]["input_binding"] == nersi_checkpoint_input_binding(
        inputs_lock
    )

    loaded = load_fixed_step_nersi_checkpoint(output / CHECKPOINT_RELATIVE_PATH, device="cpu")
    assert loaded.coordinate_order == PROFILE_COORDINATE_ORDER
    assert loaded.profile_axis_order == PROFILE_AXIS_ORDER
    assert loaded.coordinate_bounds == ((0, 1), (0, 2), (0, 1))
    assert loaded.spatial_shape == (2, 3, 2, 8)
    assert loaded.profile_shape == (8, 8)
    assert loaded.global_step == 2
    assert loaded.model_initialization_seed == 314
    assert loaded.sampling_seed == run["method_details"]["sampling_seed"] == 201
    assert run["compute"]["optimizer_updates"] == 2
    assert run["compute"]["parameter_count"] == metrics["parameter_count"]
    assert run["compute"]["supervised_trace_presentations"] > 0
    for key, filename in (("prediction", "prediction.npy"), ("checkpoint", "final.pt")):
        assert run["artifacts"][key] == {"path": filename, "sha256": file_sha256(output / filename)}
    current_data = build_c3_volume_nersi_data(
        observed,
        amplitude_scale=loaded.amplitude_scale,
    )
    validate_fixed_step_nersi_checkpoint_input_binding(
        loaded,
        inputs.inputs_lock,
        current_data,
    )
    restored = predict_c3_volume_nersi(
        loaded.model,
        current_data,
        observed,
        batch_size=run["method_details"]["prediction"]["batch_size"],
        device="cpu",
    )
    np.testing.assert_allclose(restored.values, prediction, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_array_equal(
        restored.values[:, observed.observed_trace_mask],
        observed.values[:, observed.observed_trace_mask].astype(np.float32),
    )

    rescored = evaluate_c3_volume_prediction(
        prediction,
        observed,
        interim_dir=nersi_artifacts.interim,
        volume_metadata=inputs.volume_metadata,
    )
    assert stored_metrics == rescored
    assert rescored == {
        key: metrics[key]
        for key in (
            "evaluation_domain",
            "amplitude_domain",
            "evaluation_target",
            "zero_fill",
            "observed_max_abs_error",
        )
    }

    wrong_lock = deepcopy(inputs.inputs_lock)
    wrong_lock["benchmark_volume"]["volume_id"] = "another-volume"
    with pytest.raises(ValueError, match="case/volume input binding"):
        validate_fixed_step_nersi_checkpoint_input_binding(loaded, wrong_lock, current_data)


def test_real_cli_emits_only_strict_json_on_stdout(
    tmp_path: Path,
    nersi_artifacts: PreparedC3VolumeRunArtifacts,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _write_config(tmp_path / "nersi.yaml", nersi_artifacts)
    output = tmp_path / "cli-run"
    arguments = ["interpolate", "nersi"]
    for option, path in (
        ("config", config),
        ("interim", nersi_artifacts.interim),
        ("processed", nersi_artifacts.processed),
        ("mask", nersi_artifacts.mask),
        ("case", nersi_artifacts.case),
        ("volume", nersi_artifacts.volume),
        ("output", output),
    ):
        arguments.extend([f"--{option}", str(path)])

    assert main([*arguments, "--device", "cpu", "--json"]) == 0

    captured = capsys.readouterr()
    metrics = json.loads(
        captured.out,
        parse_constant=lambda value: pytest.fail(f"non-finite JSON constant: {value}"),
    )
    assert (
        metrics["evaluation_target"]
        == json.loads((output / "metrics.json").read_text(encoding="utf-8"))["evaluation_target"]
    )
    assert metrics["method"] == "nersi"
    assert metrics["training"]["steps_completed"] == 2
    assert "Loading and verifying C3 inputs." in captured.err
    assert "nersi_volume step 2/2:" in captured.err
    assert "Writing final checkpoint, prediction, and immutable run records." in captured.err


def test_same_seed_cpu_runs_are_deterministic(
    tmp_path: Path,
    nersi_artifacts: PreparedC3VolumeRunArtifacts,
) -> None:
    config = _write_config(tmp_path / "nersi.yaml", nersi_artifacts, training_seed=2718)
    outputs = (tmp_path / "first", tmp_path / "second")

    metrics = [_run(nersi_artifacts, config, output) for output in outputs]
    predictions = [
        np.load(output / PREDICTION_RELATIVE_PATH, allow_pickle=False) for output in outputs
    ]
    checkpoints = [
        load_fixed_step_nersi_checkpoint(output / CHECKPOINT_RELATIVE_PATH) for output in outputs
    ]
    runs = [
        json.loads((output / "metadata.json").read_text(encoding="utf-8")) for output in outputs
    ]

    assert {key: value for key, value in metrics[0].items() if key != "timing"} == {
        key: value for key, value in metrics[1].items() if key != "timing"
    }
    assert metrics[0]["training"] == metrics[1]["training"]
    np.testing.assert_array_equal(predictions[0], predictions[1])
    assert (outputs[0] / "config.resolved.yaml").read_bytes() == (
        outputs[1] / "config.resolved.yaml"
    ).read_bytes()
    assert (outputs[0] / "inputs.lock.json").read_bytes() == (
        outputs[1] / "inputs.lock.json"
    ).read_bytes()
    assert runs[0]["method_details"]["model"] == runs[1]["method_details"]["model"]
    assert checkpoints[0].model.constructor_config() == checkpoints[1].model.constructor_config()
    for name, expected in checkpoints[0].model.state_dict().items():
        assert torch.equal(checkpoints[1].model.state_dict()[name], expected), name


def test_target_truth_changes_only_evaluation_not_training_or_prediction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictions = []
    original_predict = pipeline.predict_c3_volume_nersi

    def capture_prediction(*args, **kwargs):
        result = original_predict(*args, **kwargs)
        predictions.append(result.values.copy())
        return result

    monkeypatch.setattr(pipeline, "predict_c3_volume_nersi", capture_prediction)
    checkpoints = []
    metrics = []
    observed_volumes = []
    for name, offset in (("base", 0.0), ("changed", 10000.0)):
        root = tmp_path / name
        artifacts = _prepare_poc_artifacts(root / "data", target_offset=offset)
        config = _write_config(root / "nersi.yaml", artifacts, training_seed=2718)
        output = root / "run"
        observed_volumes.append(_loaded_inputs(artifacts, config).observed_volume)
        metrics.append(_run(artifacts, config, output))
        checkpoints.append(load_fixed_step_nersi_checkpoint(output / CHECKPOINT_RELATIVE_PATH))

    first, changed = observed_volumes
    np.testing.assert_array_equal(first.observed_trace_mask, changed.observed_trace_mask)
    np.testing.assert_array_equal(
        first.evaluation_target_trace_mask, changed.evaluation_target_trace_mask
    )
    np.testing.assert_array_equal(first.values, changed.values)
    assert checkpoints[0].amplitude_scale == checkpoints[1].amplitude_scale
    first_state = checkpoints[0].model.state_dict()
    changed_state = checkpoints[1].model.state_dict()
    assert first_state.keys() == changed_state.keys()
    for name, expected in first_state.items():
        assert torch.equal(changed_state[name], expected), name
    assert len(predictions) == 2
    np.testing.assert_array_equal(predictions[0], predictions[1])
    assert metrics[0]["training"] == metrics[1]["training"]
    assert metrics[0]["evaluation_target"] != metrics[1]["evaluation_target"]


def test_training_and_prediction_complete_before_target_evaluation(
    tmp_path: Path,
    nersi_artifacts: PreparedC3VolumeRunArtifacts,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _write_config(tmp_path / "nersi.yaml", nersi_artifacts)
    output = tmp_path / "ordered-run"
    events: list[str] = []
    original_train = pipeline.train_nersi_fixed_steps
    original_predict = pipeline.predict_c3_volume_nersi
    original_evaluate = pipeline.evaluate_c3_volume_prediction

    def train(*args, **kwargs):
        assert events == []
        assert not output.exists()
        result = original_train(*args, **kwargs)
        events.append("trained")
        return result

    def predict(*args, **kwargs):
        assert events == ["trained"]
        assert not output.exists()
        result = original_predict(*args, **kwargs)
        events.append("predicted")
        return result

    def evaluate(*args, **kwargs):
        assert events == ["trained", "predicted"]
        assert not output.exists()
        coverage = kwargs["target_coverage_mask"]
        assert np.all(coverage[args[1].evaluation_target_trace_mask])
        result = original_evaluate(*args, **kwargs)
        events.append("evaluated")
        return result

    monkeypatch.setattr(pipeline, "train_nersi_fixed_steps", train)
    monkeypatch.setattr(pipeline, "predict_c3_volume_nersi", predict)
    monkeypatch.setattr(pipeline, "evaluate_c3_volume_prediction", evaluate)

    _run(nersi_artifacts, config, output)

    assert events == ["trained", "predicted", "evaluated"]


@pytest.mark.parametrize(
    ("keys", "value", "match"),
    [
        (("benchmark_case", "id"), "wrong_case", "benchmark_case.id"),
        (("benchmark_volume", "id"), "wrong_volume", "benchmark_volume.id"),
        (("interpolation_mask", "kind"), "random_whole_ffid", "interpolation_mask.kind"),
    ],
)
def test_case_volume_and_mask_mismatches_create_no_output(
    tmp_path: Path,
    nersi_artifacts: PreparedC3VolumeRunArtifacts,
    keys: tuple[str, ...],
    value: object,
    match: str,
) -> None:
    invalid = deepcopy(_config(nersi_artifacts))
    section = invalid
    for key in keys[:-1]:
        section = section[key]
    section[keys[-1]] = value
    config = _write_config(tmp_path / "invalid.yaml", nersi_artifacts, contents=invalid)
    output = tmp_path / "invalid-run"

    with pytest.raises(ValueError, match=match):
        _run(nersi_artifacts, config, output)

    assert not output.exists()


def test_existing_output_directory_is_rejected_without_modification(
    tmp_path: Path,
    nersi_artifacts: PreparedC3VolumeRunArtifacts,
) -> None:
    config = _write_config(tmp_path / "nersi.yaml", nersi_artifacts)
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "marker.txt"
    marker.write_text("unchanged", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        _run(nersi_artifacts, config, output)

    assert marker.read_text(encoding="utf-8") == "unchanged"
    assert list(output.iterdir()) == [marker]
