"""Coordinate schema shared by trace storage and model input processing."""

MODEL_COORDINATE_ORDER = (
    "time_s",
    "cmp_x_m",
    "cmp_y_m",
    "offset_m",
    "azimuth_deg",
)

SPATIAL_COORDINATE_ORDER = MODEL_COORDINATE_ORDER[1:]

MODEL_COORDINATE_UNITS = {
    "time_s": "s",
    "cmp_x_m": "m",
    "cmp_y_m": "m",
    "offset_m": "m",
    "azimuth_deg": "deg",
}
