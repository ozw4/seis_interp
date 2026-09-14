from copy import deepcopy

import pytest

from seis_interp.configuration import REPOSITORY_ROOT, load_resolved_config
from seis_interp.nersi_config import validate_nersi_poc_config


def test_study_primary_metric_is_physical_mean_trace_snr():
    study = REPOSITORY_ROOT / "studies/study_040_c3_nersi_target_tuning"
    config = load_resolved_config(study / "config.yaml")
    assert config["primary_metric"] == "mean_trace_snr_db"
    assert config["primary_metric_amplitude_domain"] == "physical"
    assert config["selection_domain"] == "evaluation_target"
    assert config["target_snr_db"] == 15.0


@pytest.mark.parametrize(
    "candidate,parent",
    [
        ("normalized_ema_accumulate8", "normalized_ema"),
        ("aligned_ema_accumulate8", "aligned_ema0999"),
    ],
)
def test_single_model_accumulation_preserves_other_settings(candidate, parent):
    study = REPOSITORY_ROOT / "studies/study_040_c3_nersi_target_tuning"
    expected = load_resolved_config(study / f"{parent}.yaml")
    expected["training"].update(gradient_accumulation_steps=8, max_steps=10000)
    actual = load_resolved_config(study / f"{candidate}.yaml")
    assert actual == expected
    assert validate_nersi_poc_config(actual).training.gradient_accumulation_steps == 8


def test_effective_256_changes_only_accumulation_factor():
    study = REPOSITORY_ROOT / "studies/study_040_c3_nersi_target_tuning"
    expected = load_resolved_config(study / "aligned_ema_accumulate8.yaml")
    expected["training"]["gradient_accumulation_steps"] = 16
    actual = load_resolved_config(study / "aligned_ema_accumulate16.yaml")
    assert actual == expected
    assert validate_nersi_poc_config(actual).training.gradient_accumulation_steps == 16


def test_normalized_ema_changes_only_amplitude_scaling():
    study = REPOSITORY_ROOT / "studies/study_040_c3_nersi_target_tuning"
    expected = load_resolved_config(study / "aligned_ema0999.yaml")
    expected["training"]["amplitude_scaling"] = "observed_trace_rms_idw"
    expected["trace_rms_idw"] = {"radius": 2, "power": 2, "axis_scales": [1, 1, 1, 1]}
    actual = load_resolved_config(study / "normalized_ema.yaml")
    assert actual == expected
    validate_nersi_poc_config(actual)


@pytest.mark.parametrize(
    "candidate,change",
    [
        ("normalized_ema_steps200000", "budget"),
        ("normalized_ema_mixup", "mixup"),
    ],
)
def test_normalized_reference_followups_fix_other_settings(candidate, change):
    study = REPOSITORY_ROOT / "studies/study_040_c3_nersi_target_tuning"
    expected = load_resolved_config(study / "normalized_ema.yaml")
    if change == "budget":
        expected["training"]["max_steps"] = 200000
    else:
        expected["augmentation"] = {"profile_mixup_max_fraction": 0.5, "random_seed": 501}
    actual = load_resolved_config(study / f"{candidate}.yaml")
    assert actual == expected
    validate_nersi_poc_config(actual)


@pytest.mark.parametrize(
    "candidate,parent,section,key,value",
    [
        ("aligned_ema09999", "aligned_ema0999", "optimization", "ema_decay", 0.9999),
        ("aligned_nyquist_dense", "aligned_nyquist", "model", "frequency_base", 1.02),
        (
            "aligned_cartesian_nyquist_dense_cosine_ema",
            "aligned_cartesian_nyquist_cosine_ema",
            "model",
            "frequency_base",
            1.02,
        ),
    ],
)
def test_dense_band_and_longer_ema_change_only_one_field(candidate, parent, section, key, value):
    study = REPOSITORY_ROOT / "studies/study_040_c3_nersi_target_tuning"
    expected = load_resolved_config(study / f"{parent}.yaml")
    expected[section][key] = value
    actual = load_resolved_config(study / f"{candidate}.yaml")
    assert actual == expected
    validate_nersi_poc_config(actual)


@pytest.mark.parametrize(
    "candidate,changes",
    [
        (
            "aligned_cosine",
            {
                "optimization": {
                    "learning_rate_schedule": "cosine",
                    "minimum_learning_rate": 0.00001,
                    "ema_decay": None,
                }
            },
        ),
        (
            "aligned_ema0999",
            {
                "optimization": {
                    "learning_rate_schedule": "constant",
                    "minimum_learning_rate": 0.001,
                    "ema_decay": 0.999,
                }
            },
        ),
        ("aligned_nyquist", {"fourier_bandlimit": {"nyquist_fraction": [1.0, 1.0, 1.0]}}),
        ("aligned_cartesian", {"profile_coordinates": "cartesian_cmp_half_offset"}),
        (
            "aligned_cartesian_nyquist_cosine_ema",
            {
                "profile_coordinates": "cartesian_cmp_half_offset",
                "fourier_bandlimit": {"nyquist_fraction": [1.0, 1.0, 1.0]},
                "optimization": {
                    "learning_rate_schedule": "cosine",
                    "minimum_learning_rate": 0.00001,
                    "ema_decay": 0.999,
                },
            },
        ),
    ],
)
def test_encoding_optimization_candidates_fix_every_other_field(candidate, changes):
    study = REPOSITORY_ROOT / "studies/study_040_c3_nersi_target_tuning"
    expected = load_resolved_config(study / "aligned_fractional30625.yaml")
    expected.update(changes)
    actual = load_resolved_config(study / f"{candidate}.yaml")
    assert actual == expected
    validate_nersi_poc_config(actual)


@pytest.mark.parametrize(
    "candidate,section,key,value",
    [
        ("frequency_105", "model", "frequency_base", 1.05),
        ("frequency_102", "model", "frequency_base", 1.02),
        ("frequency_110", "model", "frequency_base", 1.10),
        ("frequency_115", "model", "frequency_base", 1.15),
        ("learning_rate_003", "training", "learning_rate", 0.003),
    ],
)
def test_only_declared_parameter_changes(candidate, section, key, value):
    studies = REPOSITORY_ROOT / "studies"
    baseline = load_resolved_config(
        studies / "study_037_c3_neural_mse_loss_ablation/formal/nersi.yaml"
    )
    actual = load_resolved_config(
        studies / "study_040_c3_nersi_target_tuning" / f"{candidate}.yaml"
    )
    expected = deepcopy(baseline)
    expected[section][key] = value
    assert actual == expected
    assert validate_nersi_poc_config(actual).training.max_steps == 5000


@pytest.mark.parametrize("kernel", [5, 7])
def test_kernel_changes_only_existing_constructor_parameter(kernel):
    study = REPOSITORY_ROOT / "studies/study_040_c3_nersi_target_tuning"
    expected = load_resolved_config(study / "frequency_110.yaml")
    expected["model"]["kernel_size"] = kernel
    actual = load_resolved_config(study / f"kernel_{kernel}.yaml")
    assert actual == expected
    validate_nersi_poc_config(actual)


@pytest.mark.parametrize("steps", [20000, 50000])
def test_extended_budget_changes_only_update_count(steps):
    study = REPOSITORY_ROOT / "studies/study_040_c3_nersi_target_tuning"
    expected = load_resolved_config(study / "kernel_5.yaml")
    expected["training"]["max_steps"] = steps
    actual = load_resolved_config(study / f"kernel_5_steps_{steps}.yaml")
    assert actual == expected
    assert validate_nersi_poc_config(actual).training.max_steps == steps


@pytest.mark.parametrize(
    "candidate,section,changes",
    [
        ("long_lr_0003", "training", {"learning_rate": 0.0003}),
        ("long_frequency_105", "model", {"frequency_base": 1.05}),
        (
            "long_compact",
            "model",
            {"encoder_width": 128, "latent_channels": 16, "decoder_channels": [32, 16, 8]},
        ),
        (
            "long_wide",
            "model",
            {"encoder_width": 768, "latent_channels": 128, "decoder_channels": [128, 64, 32]},
        ),
    ],
)
def test_long_candidates_change_only_declared_parameters(candidate, section, changes):
    study = REPOSITORY_ROOT / "studies/study_040_c3_nersi_target_tuning"
    expected = load_resolved_config(study / "kernel_5_steps_50000.yaml")
    expected[section].update(changes)
    actual = load_resolved_config(study / f"{candidate}.yaml")
    assert actual == expected
    validate_nersi_poc_config(actual)


@pytest.mark.parametrize("candidate,radius", [("jitter_002", 0.02), ("jitter_010", 0.1)])
def test_augmentation_candidates_preserve_all_other_parameters(candidate, radius):
    study = REPOSITORY_ROOT / "studies/study_040_c3_nersi_target_tuning"
    expected = load_resolved_config(study / "long_wide.yaml")
    expected["augmentation"] = {"coordinate_jitter_cells": radius, "random_seed": 501}
    actual = load_resolved_config(study / f"{candidate}.yaml")
    assert actual == expected
    assert validate_nersi_poc_config(actual).augmentation == expected["augmentation"]


def test_original_frequency_and_large_capacity_candidates_are_explicit():
    study = REPOSITORY_ROOT / "studies/study_040_c3_nersi_target_tuning"
    expected = load_resolved_config(study / "long_wide.yaml")
    expected["model"]["frequency_base"] = 1.25
    actual = load_resolved_config(study / "long_wide_frequency125.yaml")
    assert actual == expected
    validate_nersi_poc_config(actual)
    expected["model"].update(
        encoder_width=1536, latent_channels=192, decoder_channels=[192, 96, 48]
    )
    expected["training"]["max_steps"] = 5000
    actual = load_resolved_config(study / "large_frequency125_steps5000.yaml")
    assert actual == expected
    validate_nersi_poc_config(actual)
    expected["training"]["max_steps"] = 50000
    actual = load_resolved_config(study / "large_frequency125_steps50000.yaml")
    assert actual == expected
    validate_nersi_poc_config(actual)


@pytest.mark.parametrize(
    "candidate,field,value",
    [
        ("batch_64", "profiles_per_step", 64),
        ("batch_128", "profiles_per_step", 128),
        ("frequency_110_lr_0003", "learning_rate", 0.0003),
    ],
)
def test_followup_changes_only_declared_training_parameter(candidate, field, value):
    study = REPOSITORY_ROOT / "studies/study_040_c3_nersi_target_tuning"
    expected = load_resolved_config(study / "frequency_110.yaml")
    expected["training"][field] = value
    actual = load_resolved_config(study / f"{candidate}.yaml")
    assert actual == expected
    assert validate_nersi_poc_config(actual).training.max_steps == 5000


@pytest.mark.parametrize(
    "candidate,parent",
    [
        ("idw_base", "../study_037_c3_neural_mse_loss_ablation/formal/nersi"),
        ("idw_large", "large_frequency125_steps50000"),
    ],
)
def test_idw_candidates_change_only_amplitude_preprocessing(candidate, parent):
    study = REPOSITORY_ROOT / "studies/study_040_c3_nersi_target_tuning"
    expected = load_resolved_config(study / f"{parent}.yaml")
    expected["training"]["amplitude_scaling"] = "observed_trace_rms_idw"
    expected["trace_rms_idw"] = {"radius": 2, "power": 2, "axis_scales": [1] * 4}
    actual = load_resolved_config(study / f"{candidate}.yaml")
    assert actual == expected
    assert validate_nersi_poc_config(actual).trace_rms_idw == expected["trace_rms_idw"]


@pytest.mark.parametrize("suffix,rate", [("0001", 0.0001), ("001", 0.001)])
def test_siren_inspired_batch_candidates(suffix, rate):
    study = REPOSITORY_ROOT / "studies/study_040_c3_nersi_target_tuning"
    expected = load_resolved_config(
        study / "../study_037_c3_neural_mse_loss_ablation/formal/nersi.yaml"
    )
    expected["training"].update(profiles_per_step=256, learning_rate=rate, max_steps=10000)
    actual = load_resolved_config(study / f"siren_batch256_lr{suffix}.yaml")
    assert actual == expected
    validate_nersi_poc_config(actual)


@pytest.mark.parametrize(
    "candidate,parent",
    [
        ("aligned_base", "../study_037_c3_neural_mse_loss_ablation/formal/nersi"),
        ("aligned_idw", "idw_base"),
        ("aligned_large", "large_frequency125_steps50000"),
        ("aligned_idw_large", "idw_large"),
        ("aligned_low_frequency", "kernel_5_steps_50000"),
    ],
)
def test_alignment_changes_only_explicit_preprocessing(candidate, parent):
    study = REPOSITORY_ROOT / "studies/study_040_c3_nersi_target_tuning"
    expected = load_resolved_config(study / f"{parent}.yaml")
    expected["time_alignment"] = {"receiver_y_shift_samples_per_cell": 3, "boundary": "circular"}
    actual = load_resolved_config(study / f"{candidate}.yaml")
    assert actual == expected
    assert validate_nersi_poc_config(actual).time_alignment == expected["time_alignment"]


@pytest.mark.parametrize("components", [8, 16, 80, 160])
def test_aligned_fourier_components_change_only_coordinate_mapping(components):
    study = REPOSITORY_ROOT / "studies/study_040_c3_nersi_target_tuning"
    expected = load_resolved_config(study / "aligned_large.yaml")
    expected["model"]["fourier_components"] = components
    actual = load_resolved_config(study / f"aligned_fourier{components}.yaml")
    assert actual == expected
    validate_nersi_poc_config(actual)


def test_decoder_bottleneck_preserves_encoder_capacity_and_training():
    study = REPOSITORY_ROOT / "studies/study_040_c3_nersi_target_tuning"
    expected = load_resolved_config(study / "aligned_large.yaml")
    expected["model"]["decoder_channels"] = [48, 24, 12]
    actual = load_resolved_config(study / "aligned_decoder_bottleneck.yaml")
    assert actual == expected
    validate_nersi_poc_config(actual)


@pytest.mark.parametrize(
    "candidate,fraction", [("aligned_mixup020", 0.2), ("aligned_mixup050", 0.5)]
)
def test_mixup_candidates_only_add_observed_neighbor_augmentation(candidate, fraction):
    study = REPOSITORY_ROOT / "studies/study_040_c3_nersi_target_tuning"
    expected = load_resolved_config(study / "aligned_large.yaml")
    expected["augmentation"] = {"profile_mixup_max_fraction": fraction, "random_seed": 501}
    actual = load_resolved_config(study / f"{candidate}.yaml")
    assert actual == expected
    assert validate_nersi_poc_config(actual).augmentation == expected["augmentation"]


def test_wider_encoder_keeps_decoder_and_training_unchanged():
    study = REPOSITORY_ROOT / "studies/study_040_c3_nersi_target_tuning"
    expected = load_resolved_config(study / "aligned_fourier80.yaml")
    expected["model"]["encoder_width"] = 4096
    actual = load_resolved_config(study / "aligned_encoder4096.yaml")
    assert actual == expected
    validate_nersi_poc_config(actual)


@pytest.mark.parametrize(
    "candidate,field,value",
    [
        ("aligned_latent384", "latent_channels", 384),
        ("aligned_kernel9", "kernel_size", 9),
        ("aligned_kernel1", "kernel_size", 1),
        ("aligned_kernel3", "kernel_size", 3),
    ],
)
def test_aligned_capacity_candidates_change_one_constructor_parameter(candidate, field, value):
    study = REPOSITORY_ROOT / "studies/study_040_c3_nersi_target_tuning"
    expected = load_resolved_config(study / "aligned_fourier160.yaml")
    expected["model"][field] = value
    actual = load_resolved_config(study / f"{candidate}.yaml")
    assert actual == expected
    validate_nersi_poc_config(actual)


@pytest.mark.parametrize(
    "candidate,field,value",
    [
        ("aligned_lr0003", "learning_rate", 0.0003),
        ("aligned_steps200000", "max_steps", 200000),
    ],
)
def test_aligned_optimization_candidates_change_one_training_parameter(candidate, field, value):
    study = REPOSITORY_ROOT / "studies/study_040_c3_nersi_target_tuning"
    expected = load_resolved_config(study / "aligned_fourier160.yaml")
    expected["training"][field] = value
    actual = load_resolved_config(study / f"{candidate}.yaml")
    assert actual == expected
    validate_nersi_poc_config(actual)


def test_zero_padding_changes_only_time_alignment_boundary():
    study = REPOSITORY_ROOT / "studies/study_040_c3_nersi_target_tuning"
    expected = load_resolved_config(study / "aligned_fourier160.yaml")
    expected["time_alignment"]["boundary"] = "zero_pad"
    actual = load_resolved_config(study / "aligned_zero_pad.yaml")
    assert actual == expected
    assert validate_nersi_poc_config(actual).time_alignment == expected["time_alignment"]


def test_large_kernel_lower_learning_rate_changes_only_optimizer_rate():
    study = REPOSITORY_ROOT / "studies/study_040_c3_nersi_target_tuning"
    expected = load_resolved_config(study / "aligned_kernel9.yaml")
    expected["training"]["learning_rate"] = 0.0001
    actual = load_resolved_config(study / "aligned_kernel9_lr0001.yaml")
    assert actual == expected
    validate_nersi_poc_config(actual)


@pytest.mark.parametrize("suffix,radius", [("005", 0.05), ("020", 0.2)])
def test_aligned_jitter_keeps_low_frequency_model_and_sampling(suffix, radius):
    study = REPOSITORY_ROOT / "studies/study_040_c3_nersi_target_tuning"
    expected = load_resolved_config(study / "aligned_fourier16.yaml")
    expected["augmentation"] = {"coordinate_jitter_cells": radius, "random_seed": 501}
    actual = load_resolved_config(study / f"aligned_fourier16_jitter{suffix}.yaml")
    assert actual == expected
    assert validate_nersi_poc_config(actual).augmentation == expected["augmentation"]


@pytest.mark.parametrize("suffix,shift", [("30625", 3.0625), ("29375", 2.9375)])
def test_fractional_alignment_changes_only_explicit_time_preprocessing(suffix, shift):
    study = REPOSITORY_ROOT / "studies/study_040_c3_nersi_target_tuning"
    expected = load_resolved_config(study / "aligned_fourier160.yaml")
    expected["time_alignment"] = {
        "receiver_y_shift_samples_per_cell": shift,
        "boundary": "fourier_periodic",
    }
    actual = load_resolved_config(study / f"aligned_fractional{suffix}.yaml")
    assert actual == expected
    assert validate_nersi_poc_config(actual).time_alignment == expected["time_alignment"]
