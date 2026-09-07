from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from seis_interp import run_records
from seis_interp.cli import main
from seis_interp.data.c3_volume_adapter import load_observed_c3_volume
from seis_interp.data.file_checksums import file_sha256
from seis_interp.models.siren import Siren
from seis_interp.pipelines import interpolate_siren as pipeline
from seis_interp.pipelines.interpolate_siren import (
    CHECKPOINT_RELATIVE_PATH,
    METHOD,
    METHOD_VARIANT,
    PREDICTION_RELATIVE_PATH,
    interpolate_siren_run,
)
from seis_interp.processing.interpolation_masks import (
    RANDOM_TRACE_MASK_KIND,
    RANDOM_WHOLE_FFID_MASK_KIND,
)
from seis_interp.training.checkpoints import load_fixed_step_siren_checkpoint
from seis_interp.training.fixed_step_siren import FixedStepSirenResult, Reporter
from seis_interp.training.point_sampler import RandomPointSampler
from tests.fixtures.c3_volume_run_artifacts import (
    PreparedC3VolumeRunArtifacts,
    prepare_c3_volume_run_artifacts,
)


def _write_config(
    path: Path,
    artifacts: PreparedC3VolumeRunArtifacts,
    *,
    device: str = "cpu",
    training_random_seed: int = 42,
) -> Path:
    case = json.loads((artifacts.case / "benchmark_case.json").read_text(encoding="utf-8"))
    config = {
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
            "name": "siren",
            "coordinate_features": "cmp_offset_azimuth",
            "input_features": 6,
            "hidden_width": 8,
            "hidden_layers": 2,
            "output_features": 1,
            "omega_0": 10.0,
            "hidden_omega": 1.0,
            "layer_omega_schedule": None,
            "skip_connections": None,
        },
        "training": {
            "random_seed": training_random_seed,
            "optimizer": "adam",
            "loss": "l2",
            "learning_rate": 1e-3,
            "batch_size": 8,
            "max_steps": 3,
            "report_interval": 2,
            "device": device,
        },
        "prediction": {"batch_size": 17},
        "evaluation": {
            "primary_metric": "physical_amplitude_global_snr_db",
            "domain": "evaluation_target",
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _run(
    artifacts: PreparedC3VolumeRunArtifacts,
    config: Path,
    output: Path,
    *,
    device_override: str | None = None,
    progress_reporter: Reporter | None = None,
) -> dict[str, object]:
    return interpolate_siren_run(
        config_path=config,
        interim_dir=artifacts.interim,
        processed_dir=artifacts.processed,
        mask_dir=artifacts.mask,
        case_dir=artifacts.case,
        volume_dir=artifacts.volume,
        output_dir=output,
        device_override=device_override,
        progress_reporter=progress_reporter,
    )


@pytest.mark.parametrize(
    ("mask_kind", "device", "device_override", "dirty"),
    [
        (RANDOM_TRACE_MASK_KIND, "cpu", None, False),
        (RANDOM_WHOLE_FFID_MASK_KIND, "cuda:0", "cpu", True),
    ],
)
def test_run_writes_only_final_artifacts_and_complete_matching_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mask_kind: str,
    device: str,
    device_override: str | None,
    dirty: bool,
) -> None:
    git_metadata = {"git_commit": "a" * 40, "git_worktree_dirty": dirty}
    monkeypatch.setattr(run_records, "current_git_metadata", lambda: dict(git_metadata))
    artifacts = prepare_c3_volume_run_artifacts(tmp_path, mask_kind=mask_kind)
    config = _write_config(tmp_path / "config.yaml", artifacts, device=device)
    original_config = yaml.safe_load(config.read_text(encoding="utf-8"))
    output = tmp_path / "run"
    timestamps = []
    progress = []

    def recorded_timestamp() -> str:
        if not timestamps:
            assert not output.exists()
            timestamp = "2026-09-07T10:00:00Z"
        else:
            checkpoint = torch.load(
                output / CHECKPOINT_RELATIVE_PATH, map_location="cpu", weights_only=True
            )
            assert checkpoint["training"]["global_step"] == 3
            prediction = np.load(output / PREDICTION_RELATIVE_PATH, allow_pickle=False)
            assert np.all(np.isfinite(prediction))
            assert not (output / "run.json").exists()
            timestamp = "2026-09-07T10:00:10Z"
        timestamps.append(timestamp)
        return timestamp

    monkeypatch.setattr(run_records, "utc_timestamp", recorded_timestamp)

    metrics = _run(
        artifacts,
        config,
        output,
        device_override=device_override,
        progress_reporter=progress.append,
    )

    assert sorted(
        path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()
    ) == [
        "artifacts/final.pt",
        "artifacts/prediction.npy",
        "config.resolved.yaml",
        "inputs.lock.json",
        "metrics.json",
        "run.json",
    ]
    stored_metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    inputs_lock = json.loads((output / "inputs.lock.json").read_text(encoding="utf-8"))
    resolved = yaml.safe_load((output / "config.resolved.yaml").read_text(encoding="utf-8"))
    prediction = np.load(output / PREDICTION_RELATIVE_PATH, allow_pickle=False)
    loaded = load_fixed_step_siren_checkpoint(output / CHECKPOINT_RELATIVE_PATH)
    payload = torch.load(output / CHECKPOINT_RELATIVE_PATH, map_location="cpu", weights_only=True)
    observed = load_observed_c3_volume(
        interim_dir=artifacts.interim,
        processed_dir=artifacts.processed,
        mask_dir=artifacts.mask,
        case_dir=artifacts.case,
        volume_dir=artifacts.volume,
    )
    observed_count = int(np.count_nonzero(observed.observed_trace_mask))
    target_count = int(np.count_nonzero(observed.evaluation_target_trace_mask))
    time_count = observed.values.shape[0]
    expected_rms = float(
        np.sqrt(
            np.mean(np.square(observed.values[:, observed.observed_trace_mask], dtype=np.float64))
        )
    )

    assert metrics == stored_metrics
    for record in (stored_metrics, run, inputs_lock):
        json.dumps(record, allow_nan=False)
    assert metrics["method"] == METHOD == "siren_5d"
    assert metrics["method_variant"] == METHOD_VARIANT == "per_volume_internal_learning_fixed_steps"
    assert metrics["case_id"] == "synthetic_case"
    assert metrics["volume_id"] == "synthetic_volume"
    assert metrics["evaluation_domain"] == "evaluation_target"
    assert metrics["amplitude_domain"] == "physical"
    assert metrics["evaluation_target"]["trace_count"] == target_count
    assert metrics["evaluation_target"]["sample_count"] == target_count * time_count
    assert metrics["observed_max_abs_error"] == 0.0
    assert metrics["uncovered_trace_count"] == 0
    assert metrics["uncovered_sample_count"] == 0
    assert metrics["warnings"] == []
    assert metrics["training"]["steps_completed"] == loaded.global_step == 3
    assert metrics["training"]["final_batch_loss"] == loaded.final_batch_loss
    assert math.isfinite(loaded.final_batch_loss)
    assert [record["step"] for record in metrics["training"]["history"]] == [2, 3]
    assert all(set(record) == {"step", "train_loss"} for record in metrics["training"]["history"])
    assert math.isfinite(metrics["observed_model_rmse_before_reinsertion"])
    assert math.isfinite(metrics["observed_model_max_abs_error_before_reinsertion"])
    assert prediction.shape == observed.values.shape
    assert prediction.dtype == observed.values.dtype
    assert np.all(np.isfinite(prediction))
    assert (
        prediction[:, observed.observed_trace_mask].tobytes()
        == observed.values[:, observed.observed_trace_mask].tobytes()
    )
    target_rows = observed.array_rows[observed.evaluation_target_trace_mask]
    reference = np.load(artifacts.interim / "amplitudes.npy", allow_pickle=False)[
        target_rows, :time_count
    ].astype(np.float64)
    target_prediction = prediction[:, observed.evaluation_target_trace_mask].T.astype(np.float64)
    error_energy = float(np.sum((reference - target_prediction) ** 2))
    reference_energy = float(np.sum(reference**2))
    assert metrics["evaluation_target"]["reference_energy"] == pytest.approx(reference_energy)
    assert metrics["evaluation_target"]["error_energy"] == pytest.approx(error_energy)
    assert metrics["evaluation_target"]["snr_db"] == pytest.approx(
        10.0 * math.log10(reference_energy / error_energy)
    )

    expected_model = {
        key: value
        for key, value in original_config["model"].items()
        if key not in {"name", "coordinate_features"}
    }
    assert payload["model_config"] == expected_model
    assert run["model"] == {**expected_model, "parameter_dtype": "float32"}
    assert {parameter.dtype for parameter in loaded.model.parameters()} == {torch.float32}
    for key, value in expected_model.items():
        assert getattr(loaded.model, key) == value
    assert loaded.normalization.amplitude_rms == pytest.approx(expected_rms)
    assert loaded.model_coordinates.coordinate_features == "cmp_offset_azimuth"
    assert loaded.model_coordinates.time_coordinate_scale == 1.0
    assert payload["checkpoint_role"] == "fixed_step_final"
    assert payload["method_variant"] == METHOD_VARIANT
    assert payload["training_domain"] == "benchmark_observed_samples"
    assert payload["training"] == {"global_step": 3, "final_batch_loss": loaded.final_batch_loss}
    assert run["method"] == METHOD
    assert run["method_variant"] == METHOD_VARIANT
    assert run["case_id"] == "synthetic_case"
    assert run["volume_id"] == "synthetic_volume"
    assert run["git_commit"] == git_metadata["git_commit"]
    assert run["git_worktree_dirty"] is dirty
    assert run["status"] == "success"
    assert timestamps == ["2026-09-07T10:00:00Z", "2026-09-07T10:00:10Z"]
    assert run["started_at_utc"] == timestamps[0]
    assert run["finished_at_utc"] == timestamps[1]
    assert run["device"] == resolved["training"]["device"] == "cpu"
    expected_resolved = deepcopy(original_config)
    expected_resolved["training"]["device"] = "cpu"
    assert resolved == expected_resolved
    assert yaml.safe_load(config.read_text(encoding="utf-8")) == original_config
    assert run["python_version"]
    assert run["numpy_version"] == np.__version__
    assert run["torch_version"] == str(torch.__version__)
    assert run["random_seed"] == 42
    assert run["input"]["mask"] == {
        "kind": mask_kind,
        "random_seed": 42,
        "requested_missing_fraction": original_config["interpolation_mask"]["missing_fraction"],
    }
    assert run["input"]["selected_volume"] == {
        "selection": artifacts.volume_metadata["selection"],
        "shape": list(observed.values.shape),
        "dtype": observed.values.dtype.name,
        "observed_trace_count": observed_count,
        "evaluation_target_trace_count": target_count,
        "actual_missing_trace_count": target_count,
        "actual_missing_fraction": target_count / observed.array_rows.size,
    }
    assert run["coordinates"] == {
        "features": "cmp_offset_azimuth",
        "order": ["time_s", "cmp_x_m", "cmp_y_m", "offset_m", "azimuth_sin", "azimuth_cos"],
        "scale_scope": "selected_volume_geometry",
        "coordinate_min": list(loaded.normalization.coordinate_min),
        "coordinate_max": list(loaded.normalization.coordinate_max),
        "time_coordinate_scale": 1.0,
    }
    assert run["amplitude"] == {
        "scaling": "train_global_rms",
        "training_domain": "benchmark_observed_samples",
        "scale_source": "observed_trace_samples_only",
        "amplitude_rms": pytest.approx(expected_rms),
    }
    assert run["parameter_count"] == 137
    assert run["training"] == {
        "random_seed": 42,
        "optimizer": "adam",
        "loss": "l2",
        "input_dtype": "float32",
        "target_dtype": "float32",
        "learning_rate": 1e-3,
        "batch_size": 8,
        "max_steps": 3,
        "report_interval": 2,
        "steps_completed": 3,
        "sampling": "uniform_observed_points_with_replacement",
        "stopping_rule": "fixed_optimizer_steps",
        "observed_trace_count": observed_count,
        "observed_sample_count": observed_count * time_count,
    }
    assert run["prediction"] == {
        "artifact": "artifacts/prediction.npy",
        "batch_size": 17,
        "point_count": prediction.size,
        "axis_order": list(artifacts.volume_metadata["axis_order"]),
        "shape": list(prediction.shape),
        "dtype": prediction.dtype.name,
        "observed_data_consistency": "hard_reinsertion_after_model_prediction",
    }
    assert run["checkpoint"] == {"artifact": "artifacts/final.pt", "role": "fixed_step_final"}
    for key in (
        "load_and_verification_seconds",
        "training_data_seconds",
        "training_seconds",
        "prediction_seconds",
        "evaluation_seconds",
    ):
        assert math.isfinite(run["resources"][key]) and run["resources"][key] >= 0.0
    assert run["resources"]["process_max_rss_kib"] > 0
    assert run["resources"]["process_max_rss_scope"] == "whole_process"
    assert run["resources"]["torch_num_threads"] == torch.get_num_threads()
    assert "cuda_max_memory_allocated_bytes" not in run["resources"]
    assert "cuda_max_memory_reserved_bytes" not in run["resources"]
    assert run["warnings"] == []
    assert inputs_lock["benchmark_case"]["case_id"] == "synthetic_case"
    assert inputs_lock["benchmark_case"]["sha256"] == file_sha256(
        artifacts.case / "benchmark_case.json"
    )
    assert inputs_lock["benchmark_volume"]["volume_id"] == "synthetic_volume"
    assert [message for message in progress if not message.startswith("siren_volume step")] == [
        "Loading and verifying C3 inputs.",
        "Building volume-local SIREN coordinates and observed training data.",
        "Training SIREN on observed benchmark samples.",
        "Predicting the complete benchmark volume.",
        "Evaluating reconstruction on evaluation-target traces.",
        "Writing final checkpoint, prediction, and immutable run records.",
    ]
    assert len(progress) == 8
    assert progress[3].startswith("siren_volume step 2/3:")
    assert progress[4].startswith("siren_volume step 3/3:")


def test_target_truth_rebinding_does_not_change_training_and_cpu_runs_repeat(
    tmp_path: Path,
) -> None:
    first = prepare_c3_volume_run_artifacts(tmp_path / "first")
    changed = prepare_c3_volume_run_artifacts(tmp_path / "changed", target_offset=5000.0)
    first_config = _write_config(tmp_path / "first.yaml", first)
    changed_config = _write_config(tmp_path / "changed.yaml", changed)
    outputs = [tmp_path / name for name in ("first-run", "repeat-run", "changed-run")]
    metrics = [
        _run(first, first_config, outputs[0]),
        _run(first, first_config, outputs[1]),
        _run(changed, changed_config, outputs[2]),
    ]
    predictions = [
        np.load(output / PREDICTION_RELATIVE_PATH, allow_pickle=False) for output in outputs
    ]
    checkpoints = [
        load_fixed_step_siren_checkpoint(output / CHECKPOINT_RELATIVE_PATH) for output in outputs
    ]
    locks = [
        json.loads((output / "inputs.lock.json").read_text(encoding="utf-8")) for output in outputs
    ]

    np.testing.assert_array_equal(predictions[1], predictions[0])
    for name, expected in checkpoints[0].model.state_dict().items():
        assert torch.equal(checkpoints[1].model.state_dict()[name], expected)
    for prediction, checkpoint, result in zip(
        predictions[1:], checkpoints[1:], metrics[1:], strict=True
    ):
        np.testing.assert_allclose(prediction, predictions[0], rtol=1e-6, atol=1e-6)
        assert checkpoint.normalization == checkpoints[0].normalization
        assert checkpoint.model_coordinates == checkpoints[0].model_coordinates
        for name, expected in checkpoints[0].model.state_dict().items():
            torch.testing.assert_close(
                checkpoint.model.state_dict()[name].cpu(), expected.cpu(), rtol=1e-6, atol=1e-7
            )
        assert result["training"] == metrics[0]["training"]
        assert (
            result["observed_model_rmse_before_reinsertion"]
            == metrics[0]["observed_model_rmse_before_reinsertion"]
        )
        assert (
            result["observed_model_max_abs_error_before_reinsertion"]
            == metrics[0]["observed_model_max_abs_error_before_reinsertion"]
        )
    assert metrics[0] == metrics[1]
    assert metrics[0]["evaluation_target"] != metrics[2]["evaluation_target"]
    assert locks[0] == locks[1]
    assert locks[0]["benchmark_case"]["sha256"] != locks[2]["benchmark_case"]["sha256"]


def test_changing_only_training_seed_changes_weights_not_benchmark_inputs(tmp_path: Path) -> None:
    artifacts = prepare_c3_volume_run_artifacts(tmp_path)
    seeds = (42, 43)
    configs = [
        _write_config(tmp_path / f"seed-{seed}.yaml", artifacts, training_random_seed=seed)
        for seed in seeds
    ]
    conditions = [yaml.safe_load(config.read_text(encoding="utf-8")) for config in configs]
    assert [condition["training"].pop("random_seed") for condition in conditions] == list(seeds)
    assert conditions[0] == conditions[1]
    outputs = [tmp_path / f"seed-{seed}-run" for seed in seeds]
    metrics = [
        _run(artifacts, config, output) for config, output in zip(configs, outputs, strict=True)
    ]
    runs = [json.loads((output / "run.json").read_text(encoding="utf-8")) for output in outputs]
    checkpoints = [
        load_fixed_step_siren_checkpoint(output / CHECKPOINT_RELATIVE_PATH) for output in outputs
    ]
    first_state, changed_state = [checkpoint.model.state_dict() for checkpoint in checkpoints]

    assert first_state.keys() == changed_state.keys()
    assert any(not torch.equal(first_state[name], changed_state[name]) for name in first_state)
    assert (outputs[0] / "inputs.lock.json").read_bytes() == (
        outputs[1] / "inputs.lock.json"
    ).read_bytes()
    assert [run["random_seed"] for run in runs] == [42, 42]
    assert [run["training"]["random_seed"] for run in runs] == list(seeds)
    assert runs[0]["case_id"] == runs[1]["case_id"] == "synthetic_case"
    assert runs[0]["volume_id"] == runs[1]["volume_id"] == "synthetic_volume"
    assert runs[0]["input"] == runs[1]["input"]
    assert runs[0]["input"]["mask"]["random_seed"] == 42
    for key in ("trace_count", "sample_count", "reference_energy"):
        assert metrics[0]["evaluation_target"][key] == metrics[1]["evaluation_target"][key]
    assert checkpoints[0].normalization == checkpoints[1].normalization
    assert checkpoints[0].model_coordinates == checkpoints[1].model_coordinates
    assert runs[0]["amplitude"] == runs[1]["amplitude"]
    assert runs[0]["amplitude"]["amplitude_rms"] == checkpoints[0].normalization.amplitude_rms


def test_training_and_prediction_finish_before_target_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = prepare_c3_volume_run_artifacts(tmp_path)
    config = _write_config(tmp_path / "config.yaml", artifacts, training_random_seed=43)
    output = tmp_path / "run"
    events = []
    seeds = []
    original_train = pipeline.train_siren_fixed_steps
    original_predict = pipeline.predict_c3_volume_siren
    original_evaluate = pipeline.evaluate_c3_volume_prediction
    original_seed_model = pipeline.seed_global_model_initialization
    original_build_sampler = pipeline.build_c3_volume_siren_sampler

    def seed_model(random_seed: int, *, device: torch.device) -> None:
        seeds.append(("model", random_seed))
        original_seed_model(random_seed, device=device)

    def build_sampler(data, *, random_seed: int) -> RandomPointSampler:
        seeds.append(("sampler", random_seed))
        return original_build_sampler(data, random_seed=random_seed)

    def train_without_evaluation_inputs(
        model: Siren,
        sampler: RandomPointSampler,
        *,
        device: torch.device | str,
        learning_rate: float,
        batch_size: int,
        max_steps: int,
        report_interval: int,
        reporter: Reporter | None = None,
    ) -> FixedStepSirenResult:
        assert events == []
        assert seeds == [("model", 43), ("sampler", 43)]
        assert not output.exists()
        result = original_train(
            model,
            sampler,
            device=device,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_steps=max_steps,
            report_interval=report_interval,
            reporter=reporter,
        )
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
        result = original_evaluate(*args, **kwargs)
        events.append("evaluated")
        return result

    monkeypatch.setattr(pipeline, "train_siren_fixed_steps", train_without_evaluation_inputs)
    monkeypatch.setattr(pipeline, "predict_c3_volume_siren", predict)
    monkeypatch.setattr(pipeline, "evaluate_c3_volume_prediction", evaluate)
    monkeypatch.setattr(pipeline, "seed_global_model_initialization", seed_model)
    monkeypatch.setattr(pipeline, "build_c3_volume_siren_sampler", build_sampler)

    _run(artifacts, config, output)

    assert events == ["trained", "predicted", "evaluated"]


def test_invalid_configurations_fail_before_loading_inputs_or_creating_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = prepare_c3_volume_run_artifacts(tmp_path)
    base_path = _write_config(tmp_path / "base.yaml", artifacts)
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    variants = []
    for section, required_key in (
        ("model", "hidden_width"),
        ("training", "max_steps"),
        ("prediction", "batch_size"),
    ):
        extra = deepcopy(base)
        extra[section]["unexpected"] = 1
        variants.append((extra, section))
        missing = deepcopy(base)
        del missing[section][required_key]
        variants.append((missing, section))
        wrong_type = deepcopy(base)
        wrong_type[section] = None
        variants.append((wrong_type, section))
    invalid_values = [
        ("model", "name", "mlp", "model.name"),
        ("model", "coordinate_features", "lattice", "model.coordinate_features"),
        ("model", "input_features", 5, "input_features"),
        ("model", "output_features", 2, "output_features"),
        ("model", "hidden_width", 0, "model.hidden_width"),
        ("model", "hidden_layers", True, "model.hidden_layers"),
        ("model", "omega_0", float("nan"), "model.omega_0"),
        ("model", "hidden_omega", -1.0, "model.hidden_omega"),
        ("model", "layer_omega_schedule", "linear", "model.layer_omega_schedule"),
        ("model", "skip_connections", "residual", "model.skip_connections"),
        ("training", "optimizer", "sgd", "training.optimizer"),
        ("training", "loss", "l1", "training.loss"),
        ("training", "learning_rate", float("inf"), "training.learning_rate"),
        ("training", "learning_rate", True, "training.learning_rate"),
        ("training", "batch_size", 0, "training.batch_size"),
        ("training", "max_steps", True, "training.max_steps"),
        ("training", "report_interval", -1, "training.report_interval"),
        ("training", "device", " ", "training.device"),
        ("prediction", "batch_size", 0, "prediction.batch_size"),
        ("prediction", "batch_size", True, "prediction.batch_size"),
        ("evaluation", "domain", "all_traces", "evaluation.domain"),
        ("evaluation", "primary_metric", "correlation", "evaluation.primary_metric"),
    ]
    for section, key, value, match in invalid_values:
        variant = deepcopy(base)
        variant[section][key] = value
        variants.append((variant, match))
    invalid_schedule = deepcopy(base)
    invalid_schedule["model"].update(hidden_layers=1, layer_omega_schedule="exponential")
    variants.append((invalid_schedule, "model.layer_omega_schedule"))

    def unexpected_input_load(**kwargs):
        pytest.fail("invalid configuration must fail before input loading")

    monkeypatch.setattr(pipeline, "load_c3_volume_run_inputs", unexpected_input_load)
    for index, (variant, match) in enumerate(variants):
        path = tmp_path / f"invalid-{index}.yaml"
        path.write_text(yaml.safe_dump(variant, sort_keys=False), encoding="utf-8")
        output = tmp_path / f"invalid-run-{index}"
        with pytest.raises(ValueError, match=match):
            _run(artifacts, path, output)
        assert not output.exists()


def test_input_contradictions_and_broken_binding_create_no_output(tmp_path: Path) -> None:
    artifacts = prepare_c3_volume_run_artifacts(tmp_path)
    base_path = _write_config(tmp_path / "base.yaml", artifacts)
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    contradictions = [
        (("benchmark_case", "id"), "wrong_case", "benchmark_case.id"),
        (("benchmark_volume", "id"), "wrong_volume", "benchmark_volume.id"),
        (("benchmark_volume", "selection", "time"), [1, 4], "benchmark_volume.selection"),
        (("interpolation_mask", "partition"), "validation", "interpolation_mask.partition"),
        (("interpolation_mask", "kind"), RANDOM_WHOLE_FFID_MASK_KIND, "interpolation_mask.kind"),
        (("interpolation_mask", "missing_fraction"), 0.25, "missing_fraction"),
        (("project", "random_seed"), 7, "project.random_seed"),
    ]
    for index, (keys, value, match) in enumerate(contradictions):
        variant = deepcopy(base)
        parent = variant
        for key in keys[:-1]:
            parent = parent[key]
        parent[keys[-1]] = value
        path = tmp_path / f"contradiction-{index}.yaml"
        path.write_text(yaml.safe_dump(variant, sort_keys=False), encoding="utf-8")
        output = tmp_path / f"contradiction-run-{index}"
        with pytest.raises(ValueError, match=match):
            _run(artifacts, path, output)
        assert not output.exists()

    amplitudes = np.load(artifacts.interim / "amplitudes.npy", allow_pickle=False)
    amplitudes[0, 0] += 1.0
    np.save(artifacts.interim / "amplitudes.npy", amplitudes, allow_pickle=False)
    output = tmp_path / "broken-binding-run"
    with pytest.raises(ValueError, match="input_files"):
        _run(artifacts, base_path, output)
    assert not output.exists()


def test_existing_output_directory_is_not_modified(tmp_path: Path) -> None:
    artifacts = prepare_c3_volume_run_artifacts(tmp_path)
    config = _write_config(tmp_path / "config.yaml", artifacts)
    output = tmp_path / "run"
    output.mkdir()
    marker = output / "marker.txt"
    marker.write_text("unchanged", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        _run(artifacts, config, output)

    assert marker.read_text(encoding="utf-8") == "unchanged"
    assert list(output.iterdir()) == [marker]


@pytest.mark.parametrize(("device", "device_override"), [("cuda:0", None), ("cpu", "cuda:0")])
def test_unavailable_cuda_fails_before_input_loading_or_output_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    device_override: str | None,
) -> None:
    artifacts = prepare_c3_volume_run_artifacts(tmp_path)
    config = _write_config(tmp_path / "config.yaml", artifacts, device=device)
    output = tmp_path / "run"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def unexpected_input_load(**kwargs):
        pytest.fail("CUDA preflight must fail before input loading")

    monkeypatch.setattr(pipeline, "load_c3_volume_run_inputs", unexpected_input_load)
    with pytest.raises(RuntimeError, match="CUDA.*unavailable"):
        _run(artifacts, config, output, device_override=device_override)
    assert not output.exists()


def test_real_siren_cli_runs_pipeline_with_strict_json_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    artifacts = prepare_c3_volume_run_artifacts(tmp_path)
    config = _write_config(tmp_path / "config.yaml", artifacts, device="cuda:0")
    output = tmp_path / "cli-run"
    arguments = ["interpolate", "siren"]
    for option, path in (
        ("config", config),
        ("interim", artifacts.interim),
        ("processed", artifacts.processed),
        ("mask", artifacts.mask),
        ("case", artifacts.case),
        ("volume", artifacts.volume),
        ("output", output),
    ):
        arguments.extend([f"--{option}", str(path)])

    assert main([*arguments, "--device", "cpu", "--json"]) == 0

    captured = capsys.readouterr()
    metrics = json.loads(
        captured.out,
        parse_constant=lambda value: pytest.fail(f"non-finite JSON constant: {value}"),
    )
    assert metrics == json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["method"] == "siren_5d"
    assert metrics["training"]["steps_completed"] == 3
    assert metrics["observed_max_abs_error"] == 0.0
    assert "Loading and verifying C3 inputs." in captured.err
    assert "siren_volume step 3/3:" in captured.err
    assert "Writing final checkpoint, prediction, and immutable run records." in captured.err
    assert load_fixed_step_siren_checkpoint(output / CHECKPOINT_RELATIVE_PATH).global_step == 3
    assert np.isfinite(np.load(output / PREDICTION_RELATIVE_PATH, allow_pickle=False)).all()
    assert json.loads((output / "run.json").read_text(encoding="utf-8"))["device"] == "cpu"
