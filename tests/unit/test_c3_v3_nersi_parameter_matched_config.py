from seis_interp.configuration import load_resolved_config
from seis_interp.models.nersi import Nersi
from seis_interp.nersi_config import validate_nersi_poc_config


def test_parameter_matched_nersi_preserves_non_width_conditions():
    expected = load_resolved_config(
        "studies/study_042_c3_v3_five_method_comparison/nersi_no_time_shear.yaml"
    )
    expected["model"].update(encoder_width=40, latent_channels=8, decoder_channels=[8, 4, 2])
    actual = load_resolved_config("studies/study_045_c3_v3_nersi_parameter_matched/config.yaml")
    assert actual == expected
    settings = validate_nersi_poc_config(actual)
    model = Nersi(**settings.model_constructor_config((384, 32)))
    count = sum(p.numel() for p in model.parameters())
    assert count == 111875
    assert abs(count - 111969) / 111969 < 0.001
