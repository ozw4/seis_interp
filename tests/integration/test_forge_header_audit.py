"""Synthetic header audit tests; no external FORGE files are used."""

import json
import struct
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
import segyio
import yaml

from seis_interp.data.forge_headers import read_forge_headers
from seis_interp.data.forge_navigation import read_forge_navigation
from seis_interp.pipelines import audit_forge_headers as pipeline
from seis_interp.processing.forge_header_qc import (
    attach_navigation,
    classify_forge_headers,
    mark_duplicate_headers,
)


def write_segy(path: Path, endian: str = "<", codes=(1, 2, 4, 101)) -> Path:
    reel = bytearray(3600)
    text = "Little Endian; SOURCE AND RECEIVER LINE AND POINT NUMBERS ARE X 100"
    reel[:3200] = text.encode().ljust(3200, b" ")
    for offset, value in [
        (3212, len(codes)),
        (3214, 0),
        (3216, 1000),
        (3220, 4),
        (3224, 5),
        (3254, 1),
    ]:
        struct.pack_into(endian + "H", reel, offset, value)
    with path.open("wb") as stream:
        stream.write(reel)
        for index, code in enumerate(codes):
            header = bytearray(240)
            for byte, fmt, value in [
                (9, "i", 1),
                (13, "i", index + 1),
                (29, "h", code),
                (35, "h", 1),
                (69, "h", -10),
                (71, "h", -10),
                (73, "i", 3328034),
                (77, "i", 42615934),
                (81, "i", 3328268 if code == 1 else 0),
                (85, "i", 42615899 if code == 1 else 0),
                (89, "h", 1),
                (115, "H", 4),
                (117, "H", 1000),
                (181, "i", 10100),
                (185, "i", (517 + index) * 100),
                (189, "i", 51300),
                (193, "i", 10100),
            ]:
                struct.pack_into(endian + fmt, header, byte - 1, value)
            stream.write(header)
            stream.write(struct.pack(endian + "4f", 1, 2, 3, 4))
    return path


def write_navigation(path: Path, role="receiver") -> Path:
    preamble = ["Datum: NAD83", "System: Zone 12N"] + [""] * 13
    line = list(" " * 80)
    fields = (
        [(19, "101517"), (46, "3328268"), (53, "42615899"), (61, "16067")]
        if role == "receiver"
        else [(19, "513101"), (46, "3328034"), (53, "42615934"), (61, "16054")]
    )
    for start, value in fields:
        line[start : start + len(value)] = value
    path.write_text("\n".join(preamble + ["".join(line)]) + "\n")
    return path


@pytest.mark.parametrize("endian,name", [("<", "little"), (">", "big")])
def test_raw_reader_agrees_with_segyio_and_retains_bytes(tmp_path, endian, name):
    path = write_segy(tmp_path / "tiny.sgy", endian)
    metadata, table = read_forge_headers(path)
    assert metadata["endian"] == name
    assert metadata["binary_sample_count"] == 4
    assert len(table) == 4
    raw = path.read_bytes()
    assert metadata["raw_file_header"] == raw[:3600]
    assert table.raw_header.iloc[1] == raw[3856:4096]
    with segyio.open(str(path), strict=False, ignore_geometry=True, endian=name) as handle:
        np.testing.assert_array_equal(table.ffid, handle.attributes(9)[:])
        np.testing.assert_array_equal(table.trace_identification_code, handle.attributes(29)[:])
        np.testing.assert_array_equal(table.vendor_181_raw, handle.attributes(181)[:])


def test_truncated_or_unsupported_layout_is_rejected(tmp_path):
    path = write_segy(tmp_path / "tiny.sgy")
    content = path.read_bytes()
    path.write_bytes(content[:-1])
    with pytest.raises(ValueError, match="layout"):
        read_forge_headers(path)
    changed = bytearray(content)
    struct.pack_into("<H", changed, 3500, 256)
    path.write_bytes(changed)
    with pytest.raises(ValueError, match="revision 0"):
        read_forge_headers(path)
    path.write_bytes(b"bad")
    with pytest.raises(ValueError, match="truncated"):
        read_forge_headers(path)


def test_classification_is_header_only_and_conservative(tmp_path):
    metadata, raw = read_forge_headers(write_segy(tmp_path / "tiny.sgy"))
    table = classify_forge_headers(raw, metadata)
    assert table.trace_class.tolist() == ["seismic_candidate", "dead", "auxiliary", "unknown_code"]
    assert table.header_eligible.tolist() == [True, False, False, False]
    assert table.waveform_qc_status.eq("not_performed").all()
    assert table.component_status.eq("not_resolved_by_header").all()
    assert table.source_x_m.iloc[0] == pytest.approx(332803.4)
    raw.loc[0, "sample_interval_us"] = 2000
    assert not classify_forge_headers(raw, metadata).header_eligible.any()


@pytest.mark.parametrize(
    "field,value",
    [
        ("coordinate_units", 2),
        ("coordinate_scalar", -3),
        ("sample_count", 3),
        ("data_use", 2),
        ("trace_number", 0),
        ("vendor_181_raw", 10101),
    ],
)
def test_invalid_seismic_headers_are_quarantined(tmp_path, field, value):
    metadata, raw = read_forge_headers(write_segy(tmp_path / "tiny.sgy", codes=(1,)))
    raw.loc[0, field] = value
    table = classify_forge_headers(raw, metadata)
    assert not table.header_eligible.iloc[0]
    assert table.header_issues.iloc[0]


def test_scalar_sign_and_measurement_units(tmp_path):
    metadata, raw = read_forge_headers(write_segy(tmp_path / "tiny.sgy", codes=(1,)))
    raw.loc[0, "coordinate_scalar"] = 10
    metadata["measurement_system"] = 2
    table = classify_forge_headers(raw, metadata)
    assert table.source_x_m.iloc[0] == pytest.approx(3328034 * 10 * 0.3048)
    raw.loc[0, "coordinate_scalar"] = 0
    assert classify_forge_headers(raw, metadata).source_x_m.iloc[0] == pytest.approx(
        3328034 * 0.3048
    )
    metadata["measurement_system"] = 0
    assert not classify_forge_headers(raw, metadata).header_eligible.iloc[0]


def test_navigation_missing_invalid_and_mismatch_are_distinct(tmp_path):
    metadata, raw = read_forge_headers(write_segy(tmp_path / "tiny.sgy", codes=(1, 2, 1, 1)))
    raw.loc[1, "vendor_185_raw"] = 51700
    raw.loc[2, "vendor_185_raw"] = 51700
    raw.loc[2, "receiver_x_raw"] += 100
    table = classify_forge_headers(raw, metadata)
    nav = read_forge_navigation(write_navigation(tmp_path / "nav.sp1"))
    result = attach_navigation(table, nav, "receiver", 0.2)
    assert result.receiver_nav_status.tolist() == [
        "matched",
        "header_coordinates_invalid",
        "coordinate_mismatch",
        "not_listed",
    ]
    assert result.header_eligible.tolist() == [True, False, False, True]
    assert result.receiver_nav_distance_m.iloc[0] == pytest.approx(0)
    assert np.isnan(result.receiver_nav_distance_m.iloc[1])
    assert result.receiver_x_raw.tolist() == raw.receiver_x_raw.tolist()


def test_navigation_rejects_bad_width_and_duplicate_keys(tmp_path):
    path = write_navigation(tmp_path / "nav.sp1")
    content = path.read_text()
    path.write_text(content + content.splitlines()[-1] + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        read_forge_navigation(path)
    path.write_text(content + "short\n")
    with pytest.raises(ValueError, match="width"):
        read_forge_navigation(path)


@pytest.mark.parametrize("other_ffid", [1, 2])
def test_duplicate_observations_across_files_are_retained_and_quarantined(tmp_path, other_ffid):
    metadata, raw = read_forge_headers(write_segy(tmp_path / "tiny.sgy", codes=(1,)))
    table = classify_forge_headers(raw, metadata)
    other = table.copy()
    other["ffid"] = other_ffid
    result = mark_duplicate_headers(pd.concat([table, other], ignore_index=True))
    assert len(result) == 2
    assert result.duplicate_seismic_station_pair.all()
    assert result.duplicate_ffid_trace.all() == (other_ffid == 1)
    assert not result.header_eligible.any()


def setup_audit(tmp_path, monkeypatch):
    root = tmp_path / "input"
    root.mkdir()
    write_segy(root / "1511652857_1_51300_10100_20171202_221233_871.sgy")
    write_navigation(root / "receiver.sp1")
    write_navigation(root / "source.sp1", "source")
    study = tmp_path / "studies/test_audit"
    study.mkdir(parents=True)
    (study / "config.yaml").write_text("navigation_tolerance_m: 0.2\n")
    (study / "inputs.yaml").write_text(
        yaml.safe_dump(
            {
                "dataset_root": "input",
                "expected_ffid_range": [1, 2],
                "navigation": {"source": "source.sp1", "receiver": "receiver.sp1"},
            }
        )
    )
    monkeypatch.setattr(pipeline, "_git", lambda *args: "0" * 40)
    return study


def test_pipeline_artifacts_cover_all_traces_and_inputs(tmp_path, monkeypatch):
    study = setup_audit(tmp_path, monkeypatch)
    run = pipeline.run_forge_header_audit(tmp_path, study)
    summary = json.loads((run / "summary.json").read_text())
    assert summary["complete"]
    assert summary["trace_count"] == 4
    assert summary["header_eligible_count"] == 1
    assert summary["missing_ffids_in_expected_range"] == [2]
    assert summary["station_counts"]["receiver_station_count"] == 4
    assert summary["station_counts"]["seismic_receiver_station_count"] == 1
    assert summary["station_counts"]["seismic_receivers_in_every_ffid"] == 1
    lock = json.loads((run / "input.lock.json").read_text())
    assert len(lock) == 3
    assert all(len(entry["sha256"]) == 64 for entry in lock)
    metadata = json.loads((run / "metadata.json").read_text())
    interim = tmp_path / metadata["interim_directory"]
    table = pd.read_parquet(interim / "trace_headers.parquet")
    assert len(table) == 4
    assert len(table.raw_header.iloc[0]) == 240


def test_pipeline_reports_bad_file_without_losing_good_file(tmp_path, monkeypatch):
    study = setup_audit(tmp_path, monkeypatch)
    (tmp_path / "input/broken.sgy").write_bytes(b"broken")
    with pytest.raises(RuntimeError, match="1 files failed"):
        pipeline.run_forge_header_audit(tmp_path, study)
    summary_path = next((tmp_path / "runs").rglob("summary.json"))
    summary = json.loads(summary_path.read_text())
    assert not summary["complete"]
    assert summary["trace_count"] == 4
    assert summary["failed_files"][0]["source_file"] == "broken.sgy"


def test_pipeline_verifies_configured_download_archives(tmp_path, monkeypatch):
    study = setup_audit(tmp_path, monkeypatch)
    root = tmp_path / "input"
    trace = next(root.glob("*.sgy"))
    with ZipFile(root / "shots.zip", "w") as archive:
        archive.write(trace, arcname=trace.name)
    inputs = yaml.safe_load((study / "inputs.yaml").read_text())
    inputs["download_archives"] = [{"path": "shots.zip", "destination": "."}]
    (study / "inputs.yaml").write_text(yaml.safe_dump(inputs))
    run = pipeline.run_forge_header_audit(tmp_path, study)
    checks = json.loads((run / "archive_checks.json").read_text())
    assert checks[0]["member_count"] == 1
    assert checks[0]["crc_and_extracted_sha256_match"]
    assert json.loads((run / "summary.json").read_text())["verified_archive_count"] == 1


def test_text_header_disagreement_quarantines_file(tmp_path, monkeypatch):
    study = setup_audit(tmp_path, monkeypatch)
    path = next((tmp_path / "input").glob("*.sgy"))
    raw = bytearray(path.read_bytes())
    text = b"SAMPLES/TRACE: 9999"
    raw[160 : 160 + len(text)] = text
    path.write_bytes(raw)
    run = pipeline.run_forge_header_audit(tmp_path, study)
    summary = json.loads((run / "summary.json").read_text())
    assert summary["file_issue_counts"] == {"text_binary_samples_mismatch": 1}
    assert summary["header_eligible_count"] == 0
    assert summary["trace_count"] == 4
