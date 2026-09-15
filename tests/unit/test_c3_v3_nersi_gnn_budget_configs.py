import pytest

from seis_interp.configuration import load_resolved_config
from seis_interp.models.nersi import Nersi
from seis_interp.nersi_config import validate_nersi_poc_config


@pytest.mark.parametrize("candidate", ["fourier160", "fourier40", "fourier16"])
def test_fixed_training_contract_and_exact_parameter_budget(candidate):
    baseline = load_resolved_config(
        "studies/study_042_c3_v3_five_method_comparison/nersi_no_time_shear.yaml"
    )
    actual = load_resolved_config(
        f"studies/study_046_c3_v3_nersi_gnn_parameter_budget/{candidate}.yaml"
    )
    settings = validate_nersi_poc_config(actual)
    model = Nersi(**settings.model_constructor_config((384, 32)))
    assert sum(p.numel() for p in model.parameters()) == 363269
    for field in ("fourier_components", "encoder_width", "latent_channels", "decoder_channels"):
        baseline["model"][field] = actual["model"][field]
    assert actual == baseline
    assert settings.training.max_steps == 50000
    assert settings.time_alignment is None
    assert settings.optimization["learning_rate_schedule"] == "constant"


@pytest.mark.parametrize("candidate", ["encoder_heavy", "decoder_heavy", "kernel3"])
def test_allocation_candidates_preserve_learning_rate_and_nonarchitecture_fields(candidate):
    study = "studies/study_046_c3_v3_nersi_gnn_parameter_budget"
    baseline = load_resolved_config(f"{study}/fourier40.yaml")
    actual = load_resolved_config(f"{study}/{candidate}.yaml")
    for field in ("encoder_width", "latent_channels", "decoder_channels"):
        baseline["model"][field] = actual["model"][field]
    if candidate == "kernel3":
        baseline["model"]["kernel_size"] = 3
    assert actual == baseline
    settings = validate_nersi_poc_config(actual)
    assert settings.training.learning_rate == 0.001
    assert settings.training.max_steps == 50000
    model = Nersi(**settings.model_constructor_config((384, 32)))
    assert sum(p.numel() for p in model.parameters()) == 363269


@pytest.mark.parametrize(
    "candidate,base", [("base105", 1.05), ("base110", 1.10), ("base115", 1.15)]
)
def test_frequency_base_is_the_only_change(candidate, base):
    study = "studies/study_046_c3_v3_nersi_gnn_parameter_budget"
    expected = load_resolved_config(f"{study}/fourier40.yaml")
    expected["model"]["frequency_base"] = base
    actual = load_resolved_config(f"{study}/{candidate}.yaml")
    assert actual == expected
    settings = validate_nersi_poc_config(actual)
    model = Nersi(**settings.model_constructor_config((384, 32)))
    assert sum(p.numel() for p in model.parameters()) == 363269
    assert model.fourier_mapping.frequency_base == base


def test_ten_profiles_changes_only_batch_size():
    study = "studies/study_046_c3_v3_nersi_gnn_parameter_budget"
    expected = load_resolved_config(f"{study}/base115.yaml")
    expected["training"]["profiles_per_step"] = 10
    actual = load_resolved_config(f"{study}/base115_profiles10.yaml")
    assert actual == expected
    settings = validate_nersi_poc_config(actual)
    assert settings.training.profiles_per_step == 10
    assert settings.training.gradient_accumulation_steps == 1


def test_adamw_changes_only_optimizer_from_adopted_ten_profiles():
    study = "studies/study_046_c3_v3_nersi_gnn_parameter_budget"
    expected = load_resolved_config(f"{study}/base115_profiles10.yaml")
    expected["training"]["optimizer"] = "adamw"
    actual = load_resolved_config(f"{study}/base115_profiles10_adamw.yaml")
    assert actual == expected
    assert validate_nersi_poc_config(actual).training.optimizer == "adamw"


def test_shot_profile_changes_only_generated_and_coordinate_axes():
    study = "studies/study_046_c3_v3_nersi_gnn_parameter_budget"
    expected = load_resolved_config(f"{study}/base115_profiles10_adamw.yaml")
    expected["profile_axis"] = "shot_in_line"
    expected["model"]["coordinate_order"] = [
        "source_line",
        "relative_receiver_x",
        "relative_receiver_y",
    ]
    actual = load_resolved_config(f"{study}/base115_profiles10_adamw_shot_profile.yaml")

    assert actual == expected
    settings = validate_nersi_poc_config(actual)
    model = Nersi(**settings.model_constructor_config((384, 32)))
    assert settings.profile_axis == "shot_in_line"
    assert sum(parameter.numel() for parameter in model.parameters()) == 363269
    assert settings.training.profiles_per_step == 10
    assert settings.training.max_steps == 50000
    assert settings.training.learning_rate == 0.001
    assert settings.time_alignment is None


@pytest.mark.parametrize(
    "candidate,weight_decay",
    [("base115_profiles10_adamw_wd0001", 0.0001), ("base115_profiles10_adamw_wd001", 0.001)],
)
def test_regularization_candidates_change_only_weight_decay(candidate, weight_decay):
    study = "studies/study_046_c3_v3_nersi_gnn_parameter_budget"
    expected = load_resolved_config(f"{study}/base115_profiles10_adamw.yaml")
    expected["training"]["weight_decay"] = weight_decay
    actual = load_resolved_config(f"{study}/{candidate}.yaml")

    assert actual == expected
    settings = validate_nersi_poc_config(actual)
    assert settings.training.weight_decay == weight_decay
    model = Nersi(**settings.model_constructor_config((384, 32)))
    assert sum(parameter.numel() for parameter in model.parameters()) == 363269


def test_depthwise_candidate_preserves_fixed_training_contract_and_parameter_budget():
    study = "studies/study_046_c3_v3_nersi_gnn_parameter_budget"
    actual = load_resolved_config(f"{study}/base115_profiles10_adamw_depthwise.yaml")
    settings = validate_nersi_poc_config(actual)
    model = Nersi(**settings.model_constructor_config((384, 32)))

    assert settings.model["decoder_convolution"] == "depthwise_separable"
    assert sum(parameter.numel() for parameter in model.parameters()) == 363269
    assert settings.training.optimizer == "adamw"
    assert settings.training.weight_decay == 0.0
    assert settings.training.profiles_per_step == 10
    assert settings.training.max_steps == 50000
    assert settings.training.learning_rate == 0.001
    assert settings.optimization == {
        "learning_rate_schedule": "constant",
        "minimum_learning_rate": 0.001,
        "ema_decay": 0.999,
    }
    assert settings.time_alignment is None


@pytest.mark.parametrize(
    "candidate,bases",
    [
        ("base115_profiles10_adamw_axis_rx105", (1.15, 1.15, 1.05)),
        ("base115_profiles10_adamw_axis_nyquist", (1.07, 1.09, 1.05)),
        ("base115_profiles10_adamw_axis_source107_shot109", (1.07, 1.09, 1.15)),
        ("base115_profiles10_adamw_axis_source107", (1.07, 1.15, 1.15)),
        ("base115_profiles10_adamw_axis_shot109", (1.15, 1.09, 1.15)),
    ],
)
def test_axis_frequency_candidates_change_only_frequency_bases(candidate, bases):
    study = "studies/study_046_c3_v3_nersi_gnn_parameter_budget"
    expected = load_resolved_config(f"{study}/base115_profiles10_adamw.yaml")
    expected["model"]["frequency_base"] = list(bases)
    actual = load_resolved_config(f"{study}/{candidate}.yaml")

    assert actual == expected
    settings = validate_nersi_poc_config(actual)
    assert settings.model["frequency_base"] == bases
    model = Nersi(**settings.model_constructor_config((384, 32)))
    assert sum(parameter.numel() for parameter in model.parameters()) == 363269


def test_cartesian_candidate_changes_only_profile_coordinates():
    study = "studies/study_046_c3_v3_nersi_gnn_parameter_budget"
    expected = load_resolved_config(f"{study}/base115_profiles10_adamw_axis_nyquist.yaml")
    expected["profile_coordinates"] = "cartesian_cmp_half_offset"
    actual = load_resolved_config(f"{study}/base115_profiles10_adamw_axis_nyquist_cartesian.yaml")

    assert actual == expected
    settings = validate_nersi_poc_config(actual)
    model = Nersi(**settings.model_constructor_config((384, 32)))
    assert settings.cartesian_profile_coordinates is True
    assert sum(parameter.numel() for parameter in model.parameters()) == 363269
    assert settings.training.profiles_per_step == 10
    assert settings.training.max_steps == 50000
    assert settings.training.learning_rate == 0.001
    assert settings.training.optimizer == "adamw"
    assert settings.time_alignment is None


@pytest.mark.parametrize(
    "candidate,model_seed,sampling_seed",
    [
        ("base115_profiles10_adamw_axis_nyquist_seed102", 102, 202),
        ("base115_profiles10_adamw_axis_nyquist_seed103", 103, 203),
    ],
)
def test_axis_nyquist_independent_members_change_only_random_seeds(
    candidate, model_seed, sampling_seed
):
    study = "studies/study_046_c3_v3_nersi_gnn_parameter_budget"
    expected = load_resolved_config(f"{study}/base115_profiles10_adamw_axis_nyquist.yaml")
    expected["training"]["model_initialization_seed"] = model_seed
    expected["training"]["sampling_seed"] = sampling_seed
    actual = load_resolved_config(f"{study}/{candidate}.yaml")

    assert actual == expected
    settings = validate_nersi_poc_config(actual)
    model = Nersi(**settings.model_constructor_config((384, 32)))
    assert sum(parameter.numel() for parameter in model.parameters()) == 363269
    assert settings.training.profiles_per_step == 10
    assert settings.training.max_steps == 50000
    assert settings.training.learning_rate == 0.001


def test_nyquist_bandlimit_changes_only_fixed_coordinate_frequency_limits():
    study = "studies/study_046_c3_v3_nersi_gnn_parameter_budget"
    expected = load_resolved_config(f"{study}/base115_profiles10_adamw.yaml")
    expected["fourier_bandlimit"] = {"nyquist_fraction": [1.0, 1.0, 1.0]}
    actual = load_resolved_config(f"{study}/base115_profiles10_adamw_nyquist_bandlimit.yaml")

    assert actual == expected
    settings = validate_nersi_poc_config(actual)
    model = Nersi(**settings.model_constructor_config((384, 32)))
    assert settings.nyquist_fractions == (1.0, 1.0, 1.0)
    assert sum(parameter.numel() for parameter in model.parameters()) == 363269
    assert settings.training.profiles_per_step == 10
    assert settings.training.max_steps == 50000
    assert settings.training.learning_rate == 0.001


def test_profile_embedding_candidate_has_one_fair_single_model_parameter_budget():
    study = "studies/study_046_c3_v3_nersi_gnn_parameter_budget"
    actual = load_resolved_config(
        f"{study}/base115_profiles10_adamw_axis_nyquist_profile_embedding.yaml"
    )
    settings = validate_nersi_poc_config(actual)
    model = Nersi(**settings.model_constructor_config((384, 32)))

    assert settings.model["profile_embedding_channels"] == 32
    assert settings.model["profile_grid_shape"] == (16, 32, 8)
    assert settings.model["frequency_base"] == (1.07, 1.09, 1.05)
    assert sum(parameter.numel() for parameter in model.parameters()) == 363269
    assert settings.training.profiles_per_step == 10
    assert settings.training.max_steps == 50000
    assert settings.training.learning_rate == 0.001
    assert settings.training.optimizer == "adamw"
    assert settings.time_alignment is None


def test_rank_two_latent_candidate_has_one_fair_single_model_parameter_budget():
    study = "studies/study_046_c3_v3_nersi_gnn_parameter_budget"
    actual = load_resolved_config(
        f"{study}/base115_profiles10_adamw_axis_nyquist_rank2_latent.yaml"
    )
    settings = validate_nersi_poc_config(actual)
    model = Nersi(**settings.model_constructor_config((384, 32)))

    assert settings.model["latent_spatial_rank"] == 2
    assert settings.model["frequency_base"] == (1.07, 1.09, 1.05)
    assert sum(parameter.numel() for parameter in model.parameters()) == 363269
    assert settings.training.profiles_per_step == 10
    assert settings.training.max_steps == 50000
    assert settings.training.learning_rate == 0.001
    assert settings.training.optimizer == "adamw"
    assert settings.time_alignment is None


def test_temporal_basis_candidate_has_one_fair_single_model_parameter_budget():
    study = "studies/study_046_c3_v3_nersi_gnn_parameter_budget"
    actual = load_resolved_config(
        f"{study}/base115_profiles10_adamw_axis_nyquist_temporal_basis128.yaml"
    )
    settings = validate_nersi_poc_config(actual)
    model = Nersi(**settings.model_constructor_config((384, 32)))

    assert settings.model["temporal_basis_components"] == 128
    assert settings.model["frequency_base"] == (1.07, 1.09, 1.05)
    assert sum(parameter.numel() for parameter in model.parameters()) == 363269
    assert settings.training.profiles_per_step == 10
    assert settings.training.max_steps == 50000
    assert settings.training.learning_rate == 0.001
    assert settings.training.optimizer == "adamw"
    assert settings.time_alignment is None


def test_fourier48_axis_nyquist_candidate_has_one_fair_single_model_parameter_budget():
    study = "studies/study_046_c3_v3_nersi_gnn_parameter_budget"
    actual = load_resolved_config(f"{study}/fourier48_profiles10_adamw_axis_nyquist.yaml")
    settings = validate_nersi_poc_config(actual)
    model = Nersi(**settings.model_constructor_config((384, 32)))

    assert settings.model["fourier_components"] == 48
    assert settings.model["frequency_base"] == pytest.approx(
        (1.0580395478130187, 1.0741626202136336, 1.0413727500690981)
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == 363269
    assert settings.training.profiles_per_step == 10
    assert settings.training.max_steps == 50000
    assert settings.training.learning_rate == 0.001
    assert settings.training.optimizer == "adamw"
    assert settings.time_alignment is None
