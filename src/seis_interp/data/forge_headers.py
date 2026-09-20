"""Read the fixed-length IEEE SEG-Y headers supplied in FORGE 1-300c.

This intentionally does not change the C3 reader or interpret Rev.1 fields in
the vendor-defined bytes 181-240. No waveform samples are decoded.
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import numpy as np
import pandas as pd

# One-based SEG-Y byte offsets; the full 240 bytes are retained as well.
TRACE_FIELDS = {
    "trace_sequence_line": (1, "i4"),
    "trace_sequence_file": (5, "i4"),
    "ffid": (9, "i4"),
    "trace_number": (13, "i4"),
    "energy_source_point": (17, "i4"),
    "cdp": (21, "i4"),
    "cdp_trace": (25, "i4"),
    "trace_identification_code": (29, "i2"),
    "vertically_summed_traces": (31, "i2"),
    "horizontally_stacked_traces": (33, "i2"),
    "data_use": (35, "i2"),
    "offset_raw": (37, "i4"),
    "receiver_elevation_raw": (41, "i4"),
    "source_elevation_raw": (45, "i4"),
    "source_depth_raw": (49, "i4"),
    "receiver_datum_raw": (53, "i4"),
    "source_datum_raw": (57, "i4"),
    "source_water_depth_raw": (61, "i4"),
    "receiver_water_depth_raw": (65, "i4"),
    "elevation_scalar": (69, "i2"),
    "coordinate_scalar": (71, "i2"),
    "source_x_raw": (73, "i4"),
    "source_y_raw": (77, "i4"),
    "receiver_x_raw": (81, "i4"),
    "receiver_y_raw": (85, "i4"),
    "coordinate_units": (89, "i2"),
    "weathering_velocity_raw": (91, "i2"),
    "subweathering_velocity_raw": (93, "i2"),
    "source_uphole_time_raw": (95, "i2"),
    "receiver_uphole_time_raw": (97, "i2"),
    "source_static_raw": (99, "i2"),
    "receiver_static_raw": (101, "i2"),
    "total_static_raw": (103, "i2"),
    "lag_time_a_ms": (105, "i2"),
    "lag_time_b_ms": (107, "i2"),
    "delay_recording_time_ms": (109, "i2"),
    "mute_start_ms": (111, "i2"),
    "mute_end_ms": (113, "i2"),
    "sample_count": (115, "u2"),
    "sample_interval_us": (117, "u2"),
    "gain_type": (119, "i2"),
    "instrument_gain_constant": (121, "i2"),
    "instrument_initial_gain": (123, "i2"),
    "correlated_raw": (125, "i2"),
    "sweep_start_raw": (127, "i2"),
    "sweep_end_raw": (129, "i2"),
    "sweep_length_ms": (131, "i2"),
    "sweep_type": (133, "i2"),
    "sweep_taper_start_ms": (135, "i2"),
    "sweep_taper_end_ms": (137, "i2"),
    "taper_type": (139, "i2"),
    "alias_filter_frequency_raw": (141, "i2"),
    "alias_filter_slope_raw": (143, "i2"),
    "notch_filter_frequency_raw": (145, "i2"),
    "notch_filter_slope_raw": (147, "i2"),
    "low_cut_frequency_raw": (149, "i2"),
    "high_cut_frequency_raw": (151, "i2"),
    "low_cut_slope_raw": (153, "i2"),
    "high_cut_slope_raw": (155, "i2"),
    "year": (157, "i2"),
    "day_of_year": (159, "i2"),
    "hour": (161, "i2"),
    "minute": (163, "i2"),
    "second": (165, "i2"),
    "time_basis_code": (167, "i2"),
    "trace_weighting_factor_raw": (169, "i2"),
    "geophone_roll_position_raw": (171, "i2"),
    "geophone_first_trace_raw": (173, "i2"),
    "geophone_last_trace_raw": (175, "i2"),
    "gap_size_raw": (177, "i2"),
    "overtravel_raw": (179, "i2"),
    "vendor_181_raw": (181, "i4"),
    "vendor_185_raw": (185, "i4"),
    "vendor_189_raw": (189, "i4"),
    "vendor_193_raw": (193, "i4"),
}


def sha256_file(path: Path) -> str:
    """Hash the entire input, including samples, without decoding amplitudes."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_forge_headers(path: Path) -> tuple[dict, pd.DataFrame]:
    """Read every trace header; reject unsupported/ambiguous physical framing."""
    size = path.stat().st_size
    with path.open("rb") as stream:
        reel = stream.read(3600)
    if len(reel) != 3600:
        raise ValueError("truncated 3600-byte file header")
    layouts = []
    for endian, prefix in (("little", "<"), ("big", ">")):
        interval, samples, fmt = (
            struct.unpack_from(prefix + "H", reel, offset)[0] for offset in (3216, 3220, 3224)
        )
        stride = 240 + 4 * samples
        if fmt == 5 and interval > 0 and samples > 0 and size > 3600:
            if (size - 3600) % stride == 0:
                layouts.append((endian, prefix, interval, samples, stride))
    if len(layouts) != 1:
        raise ValueError("not an unambiguous fixed-length IEEE float32 SEG-Y layout")
    endian, prefix, interval, samples, stride = layouts[0]
    revision, fixed, extended = struct.unpack_from(prefix + "HHh", reel, 3500)
    if revision != 0 or extended != 0:
        raise ValueError("FORGE audit supports revision 0 with no extended text headers")
    count = (size - 3600) // stride
    # Copy only headers out of a read-only map, then release the map.
    with path.open("rb") as stream:
        import mmap

        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            headers = np.ndarray(
                (count, 240), dtype="u1", buffer=mapped, offset=3600, strides=(stride, 1)
            ).copy()
    values = {
        name: np.ndarray(
            (count,), dtype=prefix + dtype, buffer=headers, offset=byte - 1, strides=(240,)
        ).astype(np.int64)
        for name, (byte, dtype) in TRACE_FIELDS.items()
    }
    traces = pd.DataFrame(values)
    traces.insert(0, "trace_index", np.arange(count))
    traces["raw_header"] = [row.tobytes() for row in headers]
    text_bytes = reel[:3200]
    encoding = "ascii" if all(b < 128 for b in text_bytes) else "cp500"
    text = text_bytes.decode(encoding, errors="replace")
    metadata = {
        "size_bytes": size,
        "endian": endian,
        "sample_format_code": 5,
        "revision_raw": revision,
        "fixed_length_flag_raw": fixed,
        "extended_text_headers": extended,
        "measurement_system": struct.unpack_from(prefix + "H", reel, 3254)[0],
        "binary_sample_count": samples,
        "binary_sample_interval_us": interval,
        "binary_data_trace_count": struct.unpack_from(prefix + "H", reel, 3212)[0],
        "binary_aux_trace_count": struct.unpack_from(prefix + "H", reel, 3214)[0],
        "trace_count": count,
        "text_encoding": encoding,
        "text_header": "\n".join(text[i : i + 80] for i in range(0, 3200, 80)),
        "raw_file_header": reel,
    }
    return metadata, traces
