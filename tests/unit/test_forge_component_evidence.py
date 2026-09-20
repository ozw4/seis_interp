import pandas as pd
import pytest

from seis_interp.processing.forge_component_evidence import summarize_component_evidence


def test_vendor_bytes_remain_uninterpreted_and_missing_evidence_is_rejected():
    raw = bytearray(240)
    raw[208:212] = b"3\x00\x00\x00"
    headers = pd.DataFrame(
        {
            "source_file": ["a", "a"],
            "trace_index": [0, 1],
            "header_eligible": [True, True],
            "receiver_line": [101, 101],
            "receiver_point": [517, 518],
            "raw_header": [bytes(raw), bytes(raw)],
        }
    )
    text = b"C24 Geophone - Vertical".ljust(80) + b"C27 Geophone - 3C".ljust(80)
    files = pd.DataFrame({"raw_file_header": [text.ljust(3600)]})
    waves = pd.DataFrame(
        {
            "source_file": ["a", "a"],
            "trace_index": [0, 1],
            "all_zero": [False, True],
            "rms": [2.0, 0.0],
            "amplitude_review": [False, False],
            "power_fraction_sweep_4_84": [1.0, float("nan")],
        }
    )
    summary, groups = summarize_component_evidence(headers, files, waves)
    assert summary["orientation"] == "unconfirmed"
    assert summary["candidate_channel_multiplicity_counts"] == {"1": 2}
    assert len(summary["sensor_declarations_file_counts"]) == 2
    assert groups.vendor_bytes_209_212_hex.tolist() == ["33000000"]
    assert groups.all_zero_count.tolist() == [1]
    with pytest.raises(ValueError, match="cover all"):
        summarize_component_evidence(headers, files, waves.iloc[:1])
