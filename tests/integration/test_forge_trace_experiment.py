import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from seis_interp.data import forge_trace_experiment as data
from seis_interp.pipelines import run_forge_trace_experiment as pipeline
from seis_interp.processing.forge_mvp_contract import make_outer_mask, make_validation_holdout
from seis_interp.processing.forge_time_sampling import resample_forge_traces
from seis_interp.training import ccnet5d_query_training, forge_mvp_methods
from seis_interp.training.forge_c3_methods import pad_ccnet_source_lines, run_forge_c3_method
from tests.fixtures.forge_mvp import tiny_forge_study


@pytest.mark.parametrize(
    "method", ["regsi_real", "ccnet5d_grid", "nersi_profile", "pocs_grid", "drr_grid"]
)
@pytest.mark.parametrize(
    "study_name", ["study_065_forge_expanded_grid", "study_066_forge_c3_reference"]
)
def test_forge_studies_preserve_the_selected_c3_settings(method, study_name):
    repo = Path(__file__).resolve().parents[2]
    study = repo / "studies" / study_name
    contract = yaml.safe_load((study / "config.yaml").read_text())
    reference = contract["c3_reference_settings"][method]
    selection = json.loads((repo / reference["selection"]).read_text())
    original = deepcopy(selection["winner"]["config"])
    actual = yaml.safe_load((study / f"{method}.yaml").read_text())
    original.pop("device")
    actual.pop("device")
    if method in ("regsi_real", "ccnet5d_grid", "nersi_profile"):
        actual.pop("execution")
        assert actual["training"]["max_steps"] == 50000
    if method == "ccnet5d_grid":
        original["patches"]["shape"][0] = 1001
        if study_name == "study_066_forge_c3_reference":
            assert actual["training"]["mixed_precision"] == "bf16"
            original["training"]["mixed_precision"] = "bf16"
            assert actual["patches"]["shape"] == [1001, 4, 8, 4, 8]
            original["patches"]["shape"] = [1001, 4, 8, 4, 8]
            assert actual["prediction"]["core_shape"] == [1001, 4, 8, 4, 8]
            assert actual["prediction"]["mixed_precision"] == "bf16"
            original["prediction"] = {"core_shape": [1001, 4, 8, 4, 8], "mixed_precision": "bf16"}
    if method == "nersi_profile":
        original["model"]["profile_shape"] = [1008, contract["spatial_shape"][-1]]
    assert actual == original
    assert contract["workflow"]["total_runs"] == 5
    assert contract["time_sampling"]["decimation_factor"] == 4


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    study, contract, _, configs, *_, native_values = tiny_forge_study(tmp_path)
    methods = ["regsi_real", "ccnet5d_grid", "nersi_profile", "pocs_grid", "drr_grid"]
    contract.update(
        methods=methods,
        mask_study_id="historical_observation_study",
        validation_study_id="historical_validation_study",
        natural_missing_cell_count=1,
        time_sample_count=3,
        sample_interval_us=4000,
        missing_fraction=0.5,
        observed_trace_count=17,
        test_trace_count=18,
        validation_holdout={"fraction": 0.3, "mask_id": "tiny_validation", "mask_seed": 19},
        matched_training={"loss_traces_per_update": 2},
        time_sampling={
            "native_sample_count": 9,
            "native_sample_interval_us": 1000,
            "decimation_factor": 4,
        },
    )
    contract["preprocessing"].update(
        amplitude_scaling="trace_oracle", alignment={"velocity_m_per_s": 10000.0}
    )
    configs["regsi_real"]["training"]["inner_mask_fraction"] = 0.5
    configs["ccnet5d_grid"]["patches"]["shape"] = [3, 3, 3, 2, 3]
    configs["ccnet5d_grid"]["prediction"]["core_shape"] = [3, 3, 3, 2, 3]
    configs["drr_grid"]["execution"] = {"backend": "batched", "workers": 1}
    profile = deepcopy(configs["nersi_real"])
    profile.update(method_id="nersi_profile", checkpoint_rule="fixed_final_ema")
    profile["model"] = {
        "input_features": 3,
        "fourier_components": 2,
        "frequency_base": 1.15,
        "encoder_width": 4,
        "latent_channels": 2,
        "decoder_channels": [2, 2, 2],
        "profile_shape": [8, 8],
        "upsample_scales": [2, 2, 2],
        "kernel_size": 1,
        "activation": "gelu",
        "output_activation": "linear",
    }
    profile["training"].update(supervised_traces_per_update=2, nuclear_norm_weight=0.001)
    profile["prediction"] = {"profile_batch_size": 2}
    configs["nersi_profile"] = profile
    for method in methods:
        config = configs[method]
        config["checkpoint_rule"] = "none" if "training" not in config else "fixed_final_ema"
        if "training" in config:
            config["training"].update(optimizer="adamw", ema_decay=0.999)
        (study / f"{method}.yaml").write_text(yaml.safe_dump(config))
    (study / "config.yaml").write_text(yaml.safe_dump(contract))
    monkeypatch.setattr(data, "execution_identity", lambda _: {"git_commit": "testcommit"})
    experiment = data.prepare_forge_trace_experiment(tmp_path, study)
    return tmp_path, experiment, contract, configs, native_values


def test_four_ms_inputs_share_trace_roles_and_exclude_natural_missing_and_target_amplitudes(
    prepared,
):
    _, experiment, contract, configs, native_values = prepared
    seal = json.loads((experiment / "experiment.json").read_text())
    assert seal["total_runs"] == 5
    assert seal["counts"] == {"train": 12, "validation": 5, "test": 18}
    assert seal["oracle_target_waveforms_read"] == 23
    expected = resample_forge_traces(native_values[:35], contract)
    np.testing.assert_array_equal(np.load(experiment / "input/waveforms.npy"), expected)
    bindings = []
    for method in configs.keys() & set(seal["methods"]):
        inputs, _, partition = data.load_forge_trace_inputs(experiment, method)
        assert 35 not in inputs.geometry.cell_id.to_numpy()
        assert len(inputs.geometry) == 35
        assert set(inputs.observed_cells).isdisjoint(inputs.test_cells)
        assert set(inputs.observed_cells) == set(partition.cell_id[partition.role.eq("train")])
        np.testing.assert_array_equal(inputs.observed_values, expected[inputs.observed_cells])
        with pytest.raises(ValueError, match="outside outer observed"):
            inputs.read_observed(inputs.test_cells[:1])
        normalized = inputs.model_space()
        assert normalized.test_rms is None and normalized.observed_rms is None
        assert normalized.global_rms == 1.0
        np.testing.assert_allclose(
            np.mean(normalized.observed_values.astype(float) ** 2, axis=1), 1
        )
        bindings.append(inputs.binding)
    assert all(binding == bindings[0] for binding in bindings)


def test_probe_input_loading_does_not_even_hash_heldout_waveforms(prepared, monkeypatch):
    _, experiment, _, _, _ = prepared
    calls = []
    verify = data.verify_hashes
    load = np.load

    def recorded(root, hashes):
        calls.append(set(hashes))
        assert "input/waveforms.npy" not in hashes
        return verify(root, hashes)

    def observed_only(path, **kwargs):
        assert Path(path).name != "waveforms.npy"
        return load(path, **kwargs)

    monkeypatch.setattr(data, "verify_hashes", recorded)
    monkeypatch.setattr(np, "load", observed_only)
    inputs, _, _ = data.load_forge_trace_inputs(
        experiment, "ccnet5d_grid", verify_target_waveforms=False
    )
    assert calls and "input/observed_waveforms.npy" in calls[0]
    assert len(inputs.observed_values) > 0


def test_historical_mask_identities_preserve_roles_in_a_new_study(prepared):
    _, experiment, contract, _, _ = prepared
    manifest = pd.read_parquet(experiment / "input/manifest.parquet")
    actual = pd.read_parquet(experiment / "input/partition.parquet")
    outer = make_outer_mask(manifest, {**contract, "study_id": contract["mask_study_id"]})
    expected = make_validation_holdout(
        outer, {**contract, "study_id": contract["validation_study_id"]}
    )
    pd.testing.assert_frame_equal(actual.drop(columns="role"), expected)
    wrong_outer = make_outer_mask(manifest, contract)
    assert not wrong_outer.split.equals(outer.split)


def test_ccnet_padding_preserves_ids_and_adds_only_unknown_cells(prepared):
    _, experiment, _, _, _ = prepared
    inputs, config, _ = data.load_forge_trace_inputs(experiment, "ccnet5d_grid")
    padded = pad_ccnet_source_lines(inputs, config)
    assert padded.spatial_shape == (3, 3, 2, 3)
    np.testing.assert_array_equal(padded.geometry.cell_id, inputs.geometry.cell_id)
    np.testing.assert_array_equal(padded.observed_cells, inputs.observed_cells)
    volume = padded.volume()
    assert not volume.observed_trace_mask[2].any()
    assert not volume.evaluation_target_trace_mask[2].any()
    assert not volume.values[:, 2].any()


def test_ccnet_reuse_and_spatial_query_execution_keep_full_time_in_training_and_prediction(
    prepared, monkeypatch
):
    _, experiment, _, _, _ = prepared
    inputs, config, _ = data.load_forge_trace_inputs(experiment, "ccnet5d_grid")
    config["training"]["max_steps"] = 3
    config["training"]["supervised_traces_per_update"] = 2
    config["execution"] = {
        "reuse_patch": True,
        "spatial_query_chunk_size": 3,
        "full_trace_prediction": True,
    }
    config["prediction"]["core_shape"][0] = 1
    backward = ccnet5d_query_training.backward_ccnet5d_trace_queries
    predict = forge_mvp_methods.predict_ccnet5d_volume
    times, predictions = [], []

    def record_backward(model, x, y, mask, **kwargs):
        times.append(x.shape[1])
        assert mask.sum() == 2
        assert np.all(np.moveaxis(x, 1, -1)[mask] == 0)
        return backward(model, x, y, mask, **kwargs)

    def record_prediction(model, volume, **kwargs):
        assert kwargs["core_shape"][0] == volume.values.shape[0] > 1
        predictions.append(volume.values.shape[0])
        return predict(model, volume, **kwargs)

    monkeypatch.setattr(forge_mvp_methods, "predict_ccnet5d_volume", record_prediction)
    monkeypatch.setattr(
        "seis_interp.training.ccnet5d_observed_training.backward_ccnet5d_trace_queries",
        record_backward,
    )
    result = run_forge_c3_method(inputs, config, reporter=lambda _: None)
    assert len(times) == 3 and len(set(times)) == 1
    assert predictions == [times[0]]
    assert result.checkpoint["steps_completed"] == 3


def test_ccnet_cached_gradients_use_one_dense_forward_per_patch_and_full_time(
    prepared, monkeypatch
):
    _, experiment, _, _, _ = prepared
    inputs, config, _ = data.load_forge_trace_inputs(experiment, "ccnet5d_grid")
    config["training"].update(max_steps=3, supervised_traces_per_update=2)
    config["execution"] = {
        "reuse_patch": True,
        "cache_patch_gradients": True,
        "full_trace_prediction": True,
    }
    config["prediction"]["core_shape"][0] = 1
    initialize = forge_mvp_methods.initialize_model
    training_shapes, prediction_shapes = [], []

    def record_initialize(cfg):
        model = initialize(cfg)

        def record(module, args, output):
            (training_shapes if module.training else prediction_shapes).append(args[0].shape)

        model.register_forward_hook(record)
        return model

    monkeypatch.setattr(forge_mvp_methods, "initialize_model", record_initialize)
    result = run_forge_c3_method(inputs, config, reporter=lambda _: None)
    assert len(training_shapes) == 1
    assert prediction_shapes
    assert {shape[2] for shape in prediction_shapes} == {training_shapes[0][2]}
    assert training_shapes[0][2] > 1
    assert result.checkpoint["steps_completed"] == 3
    assert result.history[-1]["patch_forwards_completed"] == 1


def test_ccnet_forwards_prediction_precision_and_smaller_spatial_core(prepared, monkeypatch):
    _, experiment, _, _, _ = prepared
    inputs, config, _ = data.load_forge_trace_inputs(experiment, "ccnet5d_grid")
    config["prediction"].update(core_shape=[1, 1, 2, 1, 2], mixed_precision="bf16")
    config["execution"] = {"full_trace_prediction": True}
    original = forge_mvp_methods.predict_ccnet5d_volume
    calls = []

    def predict(model, volume, **kwargs):
        assert kwargs["core_shape"] == (volume.values.shape[0], 1, 2, 1, 2)
        assert kwargs["mixed_precision"] == "bf16"
        calls.append(kwargs["core_shape"])
        # CPU fitting tests routing; the CUDA unit test exercises real BF16 math.
        return original(model, volume, **(kwargs | {"mixed_precision": "off"}))

    monkeypatch.setattr(forge_mvp_methods, "predict_ccnet5d_volume", predict)
    result = run_forge_c3_method(inputs, config, reporter=lambda _: None)
    assert len(calls) == 1 and result.prediction.shape[1] == 3


def test_ccnet_training_checkpoint_survives_prediction_failure(prepared, monkeypatch):
    repo, experiment, _, _, _ = prepared
    output = experiment / "method_runs/ccnet5d_grid"

    def fail_prediction(model, volume, **kwargs):
        checkpoint = torch.load(output / "final.pt", map_location="cpu", weights_only=True)
        assert checkpoint["steps_completed"] == 1
        assert checkpoint["selection_rule"] == "fixed_final_ema"
        for name, value in model.state_dict().items():
            torch.testing.assert_close(
                checkpoint["model_state_dict"][name], value.cpu(), rtol=0, atol=0
            )
        raise RuntimeError("forced prediction failure")

    def unexpected_scoring(*args, **kwargs):
        raise AssertionError("prediction failure must not read scoring targets")

    monkeypatch.setattr(forge_mvp_methods, "predict_ccnet5d_volume", fail_prediction)
    monkeypatch.setattr(pipeline, "read_forge_trace_targets", unexpected_scoring)
    with pytest.raises(RuntimeError, match="forced prediction failure"):
        pipeline.execute_forge_trace_method(repo, experiment, "ccnet5d_grid")
    record = json.loads((output / "manifest.json").read_text())
    assert record["status"] == "failed" and record["metrics"] == {}
    assert (output / "final.pt").is_file() and not (output / "prediction.npy").exists()
    assert record["test_scoring_calls"] == 0


@pytest.mark.parametrize(
    "method", ["regsi_real", "ccnet5d_grid", "nersi_profile", "pocs_grid", "drr_grid"]
)
def test_each_method_finalizes_before_scoring_and_runs_only_once(prepared, monkeypatch, method):
    repo, experiment, _, _, _ = prepared
    original = pipeline.read_forge_trace_targets
    reads = []

    def read_finalized(path, domain):
        output = experiment / "method_runs" / method
        record = json.loads((output / "manifest.json").read_text())
        assert "prediction_finalized_at" in record
        assert (output / "prediction.npy").is_file()
        if method in ("regsi_real", "ccnet5d_grid", "nersi_profile"):
            assert (output / "final.pt").is_file()
        reads.append(domain)
        return original(path, domain)

    monkeypatch.setattr(pipeline, "read_forge_trace_targets", read_finalized)
    result = pipeline.execute_forge_trace_method(repo, experiment, method)
    assert result["status"] == "completed"
    assert reads == ["validation", "test"]
    assert result["test_scoring_calls"] == 1
    assert result["metrics"]["validation"]["all_eligible"]["trace_count"] == 5
    assert result["metrics"]["test"]["all_eligible"]["trace_count"] == 18
    assert np.isfinite(result["validation_score"])
    if method in ("regsi_real", "ccnet5d_grid", "nersi_profile"):
        checkpoint = torch.load(experiment / "method_runs" / method / "final.pt", weights_only=True)
        assert checkpoint["steps_completed"] == 1
        assert checkpoint["selection_rule"] == "fixed_final_ema"
    repeated = pipeline.execute_forge_trace_method(repo, experiment, method)
    assert repeated == result and reads == ["validation", "test"]
    pipeline.summarize_forge_trace_experiment(experiment)
    table = pd.read_csv(experiment / "summary.csv")
    assert len(table) == 4
    assert set(table.method_id) == {method}
