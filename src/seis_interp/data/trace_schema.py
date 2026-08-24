"""Fixed schemas for stored physical coordinates and model input features."""

PHYSICAL_COORDINATE_ORDER = (
    "time_s",
    "cmp_x_m",
    "cmp_y_m",
    "offset_m",
    "azimuth_deg",
)

PHYSICAL_COORDINATE_UNITS = {
    "time_s": "s",
    "cmp_x_m": "m",
    "cmp_y_m": "m",
    "offset_m": "m",
    "azimuth_deg": "deg",
}

MODEL_COORDINATE_ORDER = (
    "time_s",
    "cmp_x_m",
    "cmp_y_m",
    "offset_m",
    "azimuth_sin",
    "azimuth_cos",
)

MODEL_SPATIAL_FEATURE_ORDER = MODEL_COORDINATE_ORDER[1:]

MODEL_COORDINATE_UNITS = {
    "time_s": "s",
    "cmp_x_m": "m",
    "cmp_y_m": "m",
    "offset_m": "m",
    "azimuth_sin": "1",
    "azimuth_cos": "1",
}
