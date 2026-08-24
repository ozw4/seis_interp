from seis_interp.data.trace_schema import (
    MODEL_COORDINATE_ORDER,
    MODEL_COORDINATE_UNITS,
    SPATIAL_COORDINATE_ORDER,
)


def test_model_coordinate_schema_is_fixed() -> None:
    assert MODEL_COORDINATE_ORDER == (
        "time_s",
        "cmp_x_m",
        "cmp_y_m",
        "offset_m",
        "azimuth_deg",
    )
    assert MODEL_COORDINATE_ORDER[1:] == SPATIAL_COORDINATE_ORDER
    assert MODEL_COORDINATE_UNITS == {
        "time_s": "s",
        "cmp_x_m": "m",
        "cmp_y_m": "m",
        "offset_m": "m",
        "azimuth_deg": "deg",
    }
