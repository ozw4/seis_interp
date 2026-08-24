from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from seis_interp.data import seg_c3_na_inspection as inspection
from seis_interp.processing.geometry import compute_trace_geometry


class FakeSegyFile:
    def __init__(self) -> None:
        self.tracecount = 4
        self.samples = np.arange(3)
        self.bin = {
            "samples": 3,
            "interval": 2000,
            "format": 5,
        }
        self._attributes = {
            "ffid": np.array([10, 10, 11, 11]),
            "scalar": np.array([1, 1, 1, 1]),
            "source_x": np.array([0, 0, 100, 100]),
            "source_y": np.array([0, 0, 0, 0]),
            "group_x": np.array([10, 20, 110, 120]),
            "group_y": np.array([0, 0, 0, 0]),
            "units": np.array([1, 1, 1, 1]),
            "delay": np.array([0, 0, 0, 0]),
        }
        self.trace = [
            np.array([0.0, 1.0, -1.0], dtype=np.float32),
            np.array([0.5, 1.5, -0.5], dtype=np.float32),
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([-1.0, -2.0, -3.0], dtype=np.float32),
        ]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def mmap(self) -> None:
        return None

    def attributes(self, field):
        return self._attributes[field]


class FakeSegyio:
    BinField = SimpleNamespace(Samples="samples", Interval="interval", Format="format")
    TraceField = SimpleNamespace(
        FieldRecord="ffid",
        SourceGroupScalar="scalar",
        SourceX="source_x",
        SourceY="source_y",
        GroupX="group_x",
        GroupY="group_y",
        CoordinateUnits="units",
        DelayRecordingTime="delay",
    )

    @staticmethod
    def open(path, *, mode, strict, ignore_geometry):
        assert Path(path).name == "part.sgy"
        assert mode == "r"
        assert not strict
        assert ignore_geometry
        return FakeSegyFile()


def _write_manifest(tmp_path: Path) -> Path:
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "dataset_id": "seg_c3_na",
                "files": [
                    {
                        "name": "part.sgy",
                        "url": "https://example.test/part.sgy",
                        "ffid_min": 10,
                        "ffid_max": 11,
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return manifest_path


def _inspect_fake_file(tmp_path: Path, monkeypatch) -> inspection.SegyFileInspection:
    manifest_path = _write_manifest(tmp_path)
    data_root = tmp_path / "data"
    dataset_dir = data_root / "external" / "seg_c3_na"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "part.sgy").write_bytes(b"fake")
    monkeypatch.setattr(inspection, "_import_segyio", lambda: FakeSegyio)

    return inspection.inspect_seg_c3_na(manifest_path, data_root, sample_trace_count=3).files[0]


def test_inspection_azimuth_follows_the_source_minus_receiver_convention(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # Receivers sit east of the sources, so source - receiver points west: 270
    # degrees. The opposite vector would report 90 degrees for the same traces.
    file_report = _inspect_fake_file(tmp_path, monkeypatch)

    assert file_report.azimuth_deg == inspection.ValueRange(270.0, 270.0)


def test_inspection_geometry_matches_the_shared_implementation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    file_report = _inspect_fake_file(tmp_path, monkeypatch)
    attributes = FakeSegyFile()._attributes
    midpoint_x, midpoint_y, offset, azimuth = compute_trace_geometry(
        attributes["source_x"].astype(np.float64),
        attributes["source_y"].astype(np.float64),
        attributes["group_x"].astype(np.float64),
        attributes["group_y"].astype(np.float64),
    )

    assert file_report.midpoint_x == inspection.ValueRange(midpoint_x.min(), midpoint_x.max())
    assert file_report.midpoint_y == inspection.ValueRange(midpoint_y.min(), midpoint_y.max())
    assert file_report.offset == inspection.ValueRange(offset.min(), offset.max())
    assert file_report.azimuth_deg == inspection.ValueRange(azimuth.min(), azimuth.max())


def test_inspect_seg_c3_na_reports_structure_and_geometry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    data_root = tmp_path / "data"
    dataset_dir = data_root / "external" / "seg_c3_na"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "part.sgy").write_bytes(b"fake")
    monkeypatch.setattr(inspection, "_import_segyio", lambda: FakeSegyio)

    report = inspection.inspect_seg_c3_na(
        manifest_path,
        data_root,
        sample_trace_count=3,
    )

    assert inspection.seg_c3_na_inspection_ok(report)
    assert report.dataset_directory == str(dataset_dir)
    assert len(report.files) == 1

    file_report = report.files[0]
    assert file_report.trace_count == 4
    assert file_report.samples_per_trace == 3
    assert (file_report.ffid_min, file_report.ffid_max) == (10, 11)
    assert file_report.unique_ffid_count == 2
    assert file_report.source_x == inspection.ValueRange(0.0, 100.0)
    assert file_report.receiver_x == inspection.ValueRange(10.0, 120.0)
    assert file_report.offset == inspection.ValueRange(10.0, 20.0)
    assert file_report.amplitudes.sampled_trace_count == 3
    assert file_report.amplitudes.finite_ratio == 1.0
    assert file_report.issues == ()


def test_inspection_flags_manifest_ffid_mismatch(tmp_path: Path, monkeypatch) -> None:
    manifest_path = _write_manifest(tmp_path)
    raw_manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    raw_manifest["files"][0]["ffid_max"] = 12
    manifest_path.write_text(yaml.safe_dump(raw_manifest), encoding="utf-8")

    data_root = tmp_path / "data"
    dataset_dir = data_root / "external" / "seg_c3_na"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "part.sgy").write_bytes(b"fake")
    monkeypatch.setattr(inspection, "_import_segyio", lambda: FakeSegyio)

    report = inspection.inspect_seg_c3_na(manifest_path, data_root)

    assert not inspection.seg_c3_na_inspection_ok(report)
    assert report.files[0].issues == ("FFID maximum differs from manifest (11 != 12)",)
