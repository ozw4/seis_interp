from seis_interp.data.trace_schema import (
    MODEL_COORDINATE_ORDER,
    MODEL_COORDINATE_UNITS,
    MODEL_SPATIAL_FEATURE_ORDER,
    PHYSICAL_COORDINATE_ORDER,
    PHYSICAL_COORDINATE_UNITS,
)


def test_physical_coordinate_schema_preserves_auditable_azimuth_degrees() -> None:
    assert PHYSICAL_COORDINATE_ORDER == (
        "time_s",
        "cmp_x_m",
        "cmp_y_m",
        "offset_m",
        "azimuth_deg",
    )
    assert PHYSICAL_COORDINATE_UNITS == {
        "time_s": "s",
        "cmp_x_m": "m",
        "cmp_y_m": "m",
        "offset_m": "m",
        "azimuth_deg": "deg",
    }
    assert len(PHYSICAL_COORDINATE_ORDER) == 5


def test_model_coordinate_schema_uses_unit_circle_azimuth_features() -> None:
    assert MODEL_COORDINATE_ORDER == (
        "time_s",
        "cmp_x_m",
        "cmp_y_m",
        "offset_m",
        "azimuth_sin",
        "azimuth_cos",
    )
    assert MODEL_COORDINATE_ORDER[1:] == MODEL_SPATIAL_FEATURE_ORDER
    assert MODEL_COORDINATE_UNITS == {
        "time_s": "s",
        "cmp_x_m": "m",
        "cmp_y_m": "m",
        "offset_m": "m",
        "azimuth_sin": "1",
        "azimuth_cos": "1",
    }
    assert len(MODEL_COORDINATE_ORDER) == 6
    assert len(MODEL_SPATIAL_FEATURE_ORDER) == 5
