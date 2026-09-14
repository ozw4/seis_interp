from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from seis_interp.configuration import load_resolved_config
from seis_interp.data.c3_volume_adapter import ObservedC3Volume
from seis_interp.evaluation.normalized_trace_reference import evaluate_normalized_trace_reference
from seis_interp.nersi_config import validate_nersi_poc_config
from seis_interp.training.local_nersi import (
    fit_local_nersi_block,
    local_nersi_inputs,
    restore_local_nersi_prediction,
    source_line_ranges,
)


def inputs():
    shape = (3, 2, 1, 8)
    values = np.random.default_rng(7).normal(size=(8, *shape)).astype(np.float32)
    mask = np.zeros(shape, dtype=bool)
    mask[..., ::2] = True
    return ObservedC3Volume(values, np.arange(8), np.arange(48).reshape(shape), mask, ~mask)


def settings():
    config = load_resolved_config("studies/study_040_c3_nersi_target_tuning/local_lines8.yaml")
    config["model"].update(
        fourier_components=2,
        encoder_width=8,
        latent_channels=2,
        decoder_channels=[3, 2, 2],
        kernel_size=1,
    )
    config["training"].update(max_steps=2, report_interval=1, profiles_per_step=2, device="cpu")
    return validate_nersi_poc_config(config)


def binding():
    return {
        "benchmark_case": {"case_id": "case", "sha256": "a" * 64},
        "benchmark_volume": {"volume_id": "volume", "files": {"volume.json": {"sha256": "b" * 64}}},
    }


@pytest.mark.parametrize("width", [1, 4, 8])
def test_local_configs_only_change_partition_width(width):
    root = "studies/study_040_c3_nersi_target_tuning"
    expected = load_resolved_config(f"{root}/local_lines8.yaml")
    expected["local_models"]["source_line_width"] = width
    actual = load_resolved_config(f"{root}/local_lines{width}.yaml")
    assert actual == expected
    validated = validate_nersi_poc_config(actual)
    assert validated.training.loss == "masked_trace_mse"
    assert validated.training.max_steps == 5000


def test_large_local_config_preserves_single_model_architecture_and_preprocessing():
    root = "studies/study_040_c3_nersi_target_tuning"
    expected = load_resolved_config(f"{root}/normalized_ema.yaml")
    expected["local_models"] = {"source_line_width": 8}
    expected["training"]["max_steps"] = 20000
    actual = load_resolved_config(f"{root}/local_lines8_large_steps20000.yaml")
    assert actual == expected
    validate_nersi_poc_config(actual)


def test_accumulation_candidate_declares_budget_and_effective_batch():
    root = "studies/study_040_c3_nersi_target_tuning"
    expected = load_resolved_config(f"{root}/local_lines8_large_steps20000.yaml")
    expected["training"].update(gradient_accumulation_steps=8, max_steps=5000)
    actual = load_resolved_config(f"{root}/local_lines8_large_accumulate8.yaml")
    assert actual == expected
    settings = validate_nersi_poc_config(actual)
    assert (
        settings.training.gradient_accumulation_steps * settings.training.profiles_per_step == 128
    )


def test_fitted_shear_candidate_changes_only_local_alignment_search():
    root = "studies/study_040_c3_nersi_target_tuning"
    baseline = load_resolved_config(f"{root}/local_lines8_large_steps20000.yaml")
    fitted = load_resolved_config(f"{root}/local_lines8_large_fitted_shear.yaml")
    search = fitted["local_models"].pop("alignment_search")
    assert fitted == baseline
    assert 3.0625 in search["candidates"]
    assert search["distances"] == list(range(8, 17))


def test_single_model_pipeline_rejects_local_config_before_loading_inputs(tmp_path):
    from seis_interp.pipelines.interpolate_nersi import interpolate_nersi_run

    with pytest.raises(ValueError, match="independent local NeRSI runner"):
        interpolate_nersi_run(
            config_path="studies/study_040_c3_nersi_target_tuning/local_lines8.yaml",
            output_dir=tmp_path / "new",
            interim_dir=tmp_path,
            processed_dir=tmp_path,
            mask_dir=tmp_path,
            case_dir=tmp_path,
            volume_dir=tmp_path,
        )


@pytest.mark.parametrize(
    "width, expected", [(1, [(0, 1), (1, 2), (2, 3)]), (2, [(0, 2), (2, 3)]), (3, [(0, 3)])]
)
def test_disjoint_full_partition(width, expected):
    assert source_line_ranges(3, width) == expected


@pytest.mark.parametrize("width", [0, -1, True, 1.5, 4])
def test_invalid_width(width):
    with pytest.raises(ValueError):
        source_line_ranges(3, width)


def test_local_data_discards_target_storage_and_preserves_row_mapping():
    observed = inputs()
    altered = replace(observed, values=observed.values.copy())
    altered.values[:, ~observed.observed_trace_mask] = np.nan
    options = dict(
        amplitude_scale=1.0,
        trace_scale=np.ones(observed.values.shape[1:]),
        time_alignment=settings().time_alignment,
    )
    local, data = local_nersi_inputs(observed, 1, 3, **options)
    other, other_data = local_nersi_inputs(altered, 1, 3, **options)
    np.testing.assert_array_equal(local.values, other.values)
    np.testing.assert_array_equal(local.array_rows, observed.array_rows[1:3])
    np.testing.assert_array_equal(data.normalized_profiles, other_data.normalized_profiles)
    assert data.normalized_coordinates[:, 0].min() == 0
    assert data.normalized_coordinates[:, 0].max() == 1


def test_independent_training_restoration_and_target_only_evaluation(tmp_path):
    observed = inputs()
    opts = settings()
    scale = np.ones(observed.values.shape[1:], dtype=np.float64)
    records = []
    predicted = np.empty_like(observed.values)
    for start, stop in source_line_ranges(3, 2):
        record, prediction = fit_local_nersi_block(
            observed,
            opts,
            start=start,
            stop=stop,
            amplitude_scale=1.0,
            trace_scale=scale,
            inputs_lock=binding(),
            checkpoint_path=tmp_path / f"{start}.pt",
            device="cpu",
        )
        records.append(record)
        predicted[:, start:stop] = prediction
        assert record["optimizer_updates"] == 2
    restored = restore_local_nersi_prediction(
        observed,
        opts,
        records=records,
        width=2,
        amplitude_scale=1.0,
        trace_scale=scale,
        inputs_lock=binding(),
        checkpoint_directory=tmp_path,
        device="cpu",
    )
    np.testing.assert_array_equal(predicted, restored)
    np.testing.assert_array_equal(
        predicted[:, observed.observed_trace_mask], observed.values[:, observed.observed_trace_mask]
    )
    altered = replace(observed, values=observed.values.copy())
    altered.values[:, ~observed.observed_trace_mask] *= 1000
    _, repeated = fit_local_nersi_block(
        altered,
        opts,
        start=0,
        stop=2,
        amplitude_scale=1.0,
        trace_scale=scale,
        inputs_lock=binding(),
        checkpoint_path=tmp_path / "altered.pt",
        device="cpu",
    )
    np.testing.assert_array_equal(repeated, predicted[:, :2])
    a = torch.load(tmp_path / "0.pt", weights_only=True)["model_state_dict"]
    b = torch.load(tmp_path / "altered.pt", weights_only=True)["model_state_dict"]
    for name in a:
        torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
    truth = observed.values.reshape(8, -1).T
    np.save(tmp_path / "amplitudes.npy", truth)
    before = evaluate_normalized_trace_reference(
        predicted,
        observed,
        scale,
        interim_dir=tmp_path,
        volume_metadata={"selection": {"time": [0, 8]}},
    )
    truth[~observed.observed_trace_mask.ravel()] *= -1
    np.save(tmp_path / "amplitudes.npy", truth)
    after = evaluate_normalized_trace_reference(
        predicted,
        observed,
        scale,
        interim_dir=tmp_path,
        volume_metadata={"selection": {"time": [0, 8]}},
    )
    assert before["evaluation_target"]["snr_db"] != after["evaluation_target"]["snr_db"]
    for mutation in ("range", "hash", "binding"):
        bad = deepcopy(records)
        lock = binding()
        if mutation == "range":
            bad[1]["source_line_range"] = [1, 3]
        elif mutation == "hash":
            bad[0]["checkpoint"]["sha256"] = "0" * 64
        else:
            lock["benchmark_case"]["sha256"] = "c" * 64
        with pytest.raises(ValueError):
            restore_local_nersi_prediction(
                observed,
                opts,
                records=bad,
                width=2,
                amplitude_scale=1.0,
                trace_scale=scale,
                inputs_lock=lock,
                checkpoint_directory=tmp_path,
                device="cpu",
            )
