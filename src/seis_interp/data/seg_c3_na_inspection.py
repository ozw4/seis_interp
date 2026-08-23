"""Inspect SEG C3 Narrow-Azimuth SEG-Y structure, geometry, and amplitudes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from seis_interp.data.data_root import external_dataset_dir
from seis_interp.data.seg_c3_na import DATASET_ID, default_manifest_path, load_manifest
from seis_interp.processing.geometry import apply_coordinate_scalar, compute_trace_geometry

DEFAULT_SAMPLE_TRACE_COUNT = 32
EXPECTED_COMPLETE_SHOT_TRACE_COUNT = 544


class SegyInspectionError(RuntimeError):
    """Raised when SEG-Y content cannot be inspected."""


@dataclass(frozen=True)
class ValueRange:
    """Minimum and maximum values for one header-derived quantity."""

    minimum: float
    maximum: float


@dataclass(frozen=True)
class AmplitudeInspection:
    """Statistics calculated from evenly sampled traces."""

    sampled_trace_count: int
    sampled_value_count: int
    finite_ratio: float
    zero_ratio: float
    minimum: float | None
    maximum: float | None
    mean: float | None
    standard_deviation: float | None
    rms: float | None


@dataclass(frozen=True)
class SegyFileInspection:
    """QC summary for one SEG-Y file."""

    name: str
    path: str
    size_bytes: int
    trace_count: int
    samples_per_trace: int
    binary_header_samples: int
    sample_interval_us: int
    record_length_s: float
    sample_format_code: int
    ffid_min: int
    ffid_max: int
    unique_ffid_count: int
    traces_per_ffid_min: int
    traces_per_ffid_median: float
    traces_per_ffid_max: int
    complete_ffid_count: int
    coordinate_units: tuple[int, ...]
    coordinate_scalars: tuple[int, ...]
    source_x: ValueRange
    source_y: ValueRange
    receiver_x: ValueRange
    receiver_y: ValueRange
    midpoint_x: ValueRange
    midpoint_y: ValueRange
    offset: ValueRange
    azimuth_deg: ValueRange
    delay_recording_time_ms: ValueRange
    amplitudes: AmplitudeInspection
    issues: tuple[str, ...]


@dataclass(frozen=True)
class DatasetInspection:
    """QC summary for all SEG C3 Narrow-Azimuth files."""

    dataset_id: str
    dataset_directory: str
    sample_trace_count: int
    files: tuple[SegyFileInspection, ...]
    issues: tuple[str, ...]


def _import_segyio() -> Any:
    try:
        import segyio
    except ImportError as exc:
        raise SegyInspectionError(
            "segyio is required for SEG-Y inspection. Install the project with the "
            "'segy' extra or run this script in the Dev Container."
        ) from exc
    return segyio


def _expected_ffid_ranges(manifest_path: Path) -> dict[str, tuple[int | None, int | None]]:
    try:
        raw_manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SegyInspectionError(f"Cannot read FFID metadata from {manifest_path}: {exc}") from exc

    ranges: dict[str, tuple[int | None, int | None]] = {}
    for raw_file in raw_manifest.get("files", []):
        if not isinstance(raw_file, dict) or not isinstance(raw_file.get("name"), str):
            continue
        ranges[raw_file["name"]] = (raw_file.get("ffid_min"), raw_file.get("ffid_max"))
    return ranges


def _read_attribute(segy_file: Any, field: int) -> np.ndarray:
    return np.asarray(segy_file.attributes(field)[:], dtype=np.int64)


def _scaled_coordinate(segy_file: Any, field: int, scalars: np.ndarray) -> np.ndarray:
    """Read one coordinate header field and apply the SEG-Y coordinate scalar."""
    return apply_coordinate_scalar(_read_attribute(segy_file, field), scalars)


def _value_range(values: np.ndarray) -> ValueRange:
    if values.size == 0:
        raise SegyInspectionError("Cannot calculate a range from an empty array")
    return ValueRange(float(np.min(values)), float(np.max(values)))


def _sample_trace_indices(trace_count: int, sample_trace_count: int) -> np.ndarray:
    if sample_trace_count <= 0:
        raise ValueError("sample_trace_count must be positive")
    if trace_count <= 0:
        return np.empty(0, dtype=np.int64)

    count = min(trace_count, sample_trace_count)
    return np.unique(np.linspace(0, trace_count - 1, num=count, dtype=np.int64))


def _inspect_amplitudes(
    segy_file: Any,
    trace_count: int,
    sample_trace_count: int,
) -> AmplitudeInspection:
    indices = _sample_trace_indices(trace_count, sample_trace_count)
    if indices.size == 0:
        return AmplitudeInspection(0, 0, 0.0, 0.0, None, None, None, None, None)

    sampled = np.concatenate(
        [np.asarray(segy_file.trace[int(index)], dtype=np.float32) for index in indices]
    )
    finite_mask = np.isfinite(sampled)
    finite_values = sampled[finite_mask].astype(np.float64)

    minimum = maximum = mean = standard_deviation = rms = None
    if finite_values.size:
        minimum = float(np.min(finite_values))
        maximum = float(np.max(finite_values))
        mean = float(np.mean(finite_values))
        standard_deviation = float(np.std(finite_values))
        rms = float(np.sqrt(np.mean(finite_values**2)))

    return AmplitudeInspection(
        sampled_trace_count=int(indices.size),
        sampled_value_count=int(sampled.size),
        finite_ratio=float(np.mean(finite_mask)),
        zero_ratio=float(np.mean(sampled == 0)),
        minimum=minimum,
        maximum=maximum,
        mean=mean,
        standard_deviation=standard_deviation,
        rms=rms,
    )


def _collect_issues(
    *,
    trace_count: int,
    samples_per_trace: int,
    binary_header_samples: int,
    sample_interval_us: int,
    ffid_min: int,
    ffid_max: int,
    expected_ffid_range: tuple[int | None, int | None],
    source_x: ValueRange,
    source_y: ValueRange,
    receiver_x: ValueRange,
    receiver_y: ValueRange,
    offset: ValueRange,
    amplitudes: AmplitudeInspection,
) -> tuple[str, ...]:
    issues: list[str] = []
    if trace_count <= 0:
        issues.append("No traces were found")
    if samples_per_trace <= 0:
        issues.append("No samples were found in each trace")
    if samples_per_trace != binary_header_samples:
        issues.append(
            "Trace sample count does not match the binary-header sample count "
            f"({samples_per_trace} != {binary_header_samples})"
        )
    if sample_interval_us <= 0:
        issues.append(f"Invalid sample interval: {sample_interval_us} us")

    expected_min, expected_max = expected_ffid_range
    if expected_min is not None and ffid_min != expected_min:
        issues.append(f"FFID minimum differs from manifest ({ffid_min} != {expected_min})")
    if expected_max is not None and ffid_max != expected_max:
        issues.append(f"FFID maximum differs from manifest ({ffid_max} != {expected_max})")

    if source_x == ValueRange(0.0, 0.0) and source_y == ValueRange(0.0, 0.0):
        issues.append("All source coordinates are zero")
    if receiver_x == ValueRange(0.0, 0.0) and receiver_y == ValueRange(0.0, 0.0):
        issues.append("All receiver coordinates are zero")
    if offset.maximum <= 0.0:
        issues.append("All calculated offsets are zero")
    if amplitudes.finite_ratio < 1.0:
        issues.append("Sampled amplitudes contain NaN or infinite values")
    if amplitudes.rms == 0.0:
        issues.append("Sampled amplitudes are all zero")
    return tuple(issues)


def _inspect_file(
    path: Path,
    *,
    expected_ffid_range: tuple[int | None, int | None],
    sample_trace_count: int,
    segyio_module: Any,
) -> SegyFileInspection:
    with segyio_module.open(
        str(path),
        mode="r",
        strict=False,
        ignore_geometry=True,
    ) as segy_file:
        segy_file.mmap()
        trace_count = int(segy_file.tracecount)
        samples_per_trace = len(segy_file.samples)
        binary_header_samples = int(segy_file.bin[segyio_module.BinField.Samples])
        sample_interval_us = int(segy_file.bin[segyio_module.BinField.Interval])
        sample_format_code = int(segy_file.bin[segyio_module.BinField.Format])

        ffids = _read_attribute(segy_file, segyio_module.TraceField.FieldRecord)
        unique_ffids, traces_per_ffid = np.unique(ffids, return_counts=True)
        if unique_ffids.size == 0:
            raise SegyInspectionError(f"No FFID values found in {path}")

        scalars = _read_attribute(segy_file, segyio_module.TraceField.SourceGroupScalar)
        source_x_values = _scaled_coordinate(segy_file, segyio_module.TraceField.SourceX, scalars)
        source_y_values = _scaled_coordinate(segy_file, segyio_module.TraceField.SourceY, scalars)
        receiver_x_values = _scaled_coordinate(segy_file, segyio_module.TraceField.GroupX, scalars)
        receiver_y_values = _scaled_coordinate(segy_file, segyio_module.TraceField.GroupY, scalars)

        # Geometry comes from processing.geometry so that the values reported
        # here are identical to the ones written into the interim trace table.
        (
            midpoint_x_values,
            midpoint_y_values,
            offset_values,
            azimuth_values,
        ) = compute_trace_geometry(
            source_x_values,
            source_y_values,
            receiver_x_values,
            receiver_y_values,
        )

        source_x = _value_range(source_x_values)
        source_y = _value_range(source_y_values)
        receiver_x = _value_range(receiver_x_values)
        receiver_y = _value_range(receiver_y_values)
        offset = _value_range(offset_values)
        amplitudes = _inspect_amplitudes(segy_file, trace_count, sample_trace_count)

        ffid_min = int(unique_ffids.min())
        ffid_max = int(unique_ffids.max())
        issues = _collect_issues(
            trace_count=trace_count,
            samples_per_trace=samples_per_trace,
            binary_header_samples=binary_header_samples,
            sample_interval_us=sample_interval_us,
            ffid_min=ffid_min,
            ffid_max=ffid_max,
            expected_ffid_range=expected_ffid_range,
            source_x=source_x,
            source_y=source_y,
            receiver_x=receiver_x,
            receiver_y=receiver_y,
            offset=offset,
            amplitudes=amplitudes,
        )

        return SegyFileInspection(
            name=path.name,
            path=str(path),
            size_bytes=path.stat().st_size,
            trace_count=trace_count,
            samples_per_trace=samples_per_trace,
            binary_header_samples=binary_header_samples,
            sample_interval_us=sample_interval_us,
            record_length_s=max(samples_per_trace - 1, 0) * sample_interval_us / 1_000_000,
            sample_format_code=sample_format_code,
            ffid_min=ffid_min,
            ffid_max=ffid_max,
            unique_ffid_count=int(unique_ffids.size),
            traces_per_ffid_min=int(traces_per_ffid.min()),
            traces_per_ffid_median=float(np.median(traces_per_ffid)),
            traces_per_ffid_max=int(traces_per_ffid.max()),
            complete_ffid_count=int(
                np.count_nonzero(traces_per_ffid == EXPECTED_COMPLETE_SHOT_TRACE_COUNT)
            ),
            coordinate_units=tuple(
                int(value)
                for value in np.unique(
                    _read_attribute(segy_file, segyio_module.TraceField.CoordinateUnits)
                )
            ),
            coordinate_scalars=tuple(int(value) for value in np.unique(scalars)),
            source_x=source_x,
            source_y=source_y,
            receiver_x=receiver_x,
            receiver_y=receiver_y,
            midpoint_x=_value_range(midpoint_x_values),
            midpoint_y=_value_range(midpoint_y_values),
            offset=offset,
            azimuth_deg=_value_range(azimuth_values),
            delay_recording_time_ms=_value_range(
                _read_attribute(segy_file, segyio_module.TraceField.DelayRecordingTime)
            ),
            amplitudes=amplitudes,
            issues=issues,
        )


def inspect_seg_c3_na(
    manifest_path: str | Path | None = None,
    data_root: str | Path | None = None,
    *,
    sample_trace_count: int = DEFAULT_SAMPLE_TRACE_COUNT,
) -> DatasetInspection:
    """Inspect all SEG-Y files declared by the SEG C3 NA manifest."""
    if sample_trace_count <= 0:
        raise ValueError("sample_trace_count must be positive")

    manifest = load_manifest(manifest_path or default_manifest_path())
    dataset_directory = external_dataset_dir(data_root, DATASET_ID)
    expected_ffid_ranges = _expected_ffid_ranges(manifest.path)

    missing_files = [
        file_spec.name
        for file_spec in manifest.files
        if not (dataset_directory / file_spec.name).is_file()
    ]
    if missing_files:
        names = ", ".join(missing_files)
        raise SegyInspectionError(f"Missing SEG-Y files under {dataset_directory}: {names}")

    declared_names = {file_spec.name for file_spec in manifest.files}
    unexpected_names = sorted(
        path.name for path in dataset_directory.glob("*.sgy") if path.name not in declared_names
    )
    dataset_issues = (
        (f"Unexpected SEG-Y files: {', '.join(unexpected_names)}",) if unexpected_names else ()
    )

    segyio_module = _import_segyio()
    inspections = tuple(
        _inspect_file(
            dataset_directory / file_spec.name,
            expected_ffid_range=expected_ffid_ranges.get(file_spec.name, (None, None)),
            sample_trace_count=sample_trace_count,
            segyio_module=segyio_module,
        )
        for file_spec in manifest.files
    )
    return DatasetInspection(
        dataset_id=DATASET_ID,
        dataset_directory=str(dataset_directory),
        sample_trace_count=sample_trace_count,
        files=inspections,
        issues=dataset_issues,
    )


def seg_c3_na_inspection_ok(inspection: DatasetInspection) -> bool:
    """Return whether dataset- and file-level inspections found no issues."""
    return not inspection.issues and all(
        not file_inspection.issues for file_inspection in inspection.files
    )


def seg_c3_na_inspection_to_dict(inspection: DatasetInspection) -> dict[str, Any]:
    """Convert an inspection report to a JSON-serializable dictionary."""
    payload = asdict(inspection)
    payload["ok"] = seg_c3_na_inspection_ok(inspection)
    for file_payload, file_inspection in zip(payload["files"], inspection.files, strict=True):
        file_payload["ok"] = not file_inspection.issues
    return payload


def _format_range(value_range: ValueRange) -> str:
    return f"{value_range.minimum:,.3f} to {value_range.maximum:,.3f}"


def _format_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.7g}"


def format_seg_c3_na_inspection(inspection: DatasetInspection) -> str:
    """Format a human-readable SEG C3 NA inspection report."""
    lines = [
        f"Dataset: {inspection.dataset_id}",
        f"Directory: {inspection.dataset_directory}",
        f"Sampled traces per file: {inspection.sample_trace_count}",
    ]
    if inspection.issues:
        lines.extend(f"Dataset issue: {issue}" for issue in inspection.issues)

    for file_inspection in inspection.files:
        status = "OK" if not file_inspection.issues else "ISSUE"
        amplitudes = file_inspection.amplitudes
        lines.extend(
            [
                "",
                f"[{status}] {file_inspection.name}",
                f"  size              : {file_inspection.size_bytes:,} bytes "
                f"({file_inspection.size_bytes / 1024**3:.3f} GiB)",
                f"  traces            : {file_inspection.trace_count:,}",
                f"  samples / trace   : {file_inspection.samples_per_trace:,} "
                f"(binary header: {file_inspection.binary_header_samples:,})",
                f"  sample interval   : {file_inspection.sample_interval_us:,} us",
                f"  record length     : {file_inspection.record_length_s:.3f} s",
                f"  sample format code: {file_inspection.sample_format_code}",
                f"  FFID              : {file_inspection.ffid_min} to "
                f"{file_inspection.ffid_max} "
                f"({file_inspection.unique_ffid_count:,} unique)",
                "  traces / FFID     : "
                f"min={file_inspection.traces_per_ffid_min}, "
                f"median={file_inspection.traces_per_ffid_median:.1f}, "
                f"max={file_inspection.traces_per_ffid_max}",
                f"  complete FFIDs    : {file_inspection.complete_ffid_count:,} "
                f"({EXPECTED_COMPLETE_SHOT_TRACE_COUNT} traces)",
                f"  coordinate units  : {list(file_inspection.coordinate_units)}",
                f"  coordinate scalars: {list(file_inspection.coordinate_scalars)}",
                f"  source X          : {_format_range(file_inspection.source_x)}",
                f"  source Y          : {_format_range(file_inspection.source_y)}",
                f"  receiver X        : {_format_range(file_inspection.receiver_x)}",
                f"  receiver Y        : {_format_range(file_inspection.receiver_y)}",
                f"  midpoint X        : {_format_range(file_inspection.midpoint_x)}",
                f"  midpoint Y        : {_format_range(file_inspection.midpoint_y)}",
                f"  offset            : {_format_range(file_inspection.offset)}",
                f"  azimuth           : {_format_range(file_inspection.azimuth_deg)} deg",
                "  delay time        : "
                f"{_format_range(file_inspection.delay_recording_time_ms)} ms",
                f"  sampled amplitudes: {amplitudes.sampled_trace_count} traces, "
                f"{amplitudes.sampled_value_count:,} values",
                f"    finite ratio    : {amplitudes.finite_ratio * 100:.6f}%",
                f"    zero ratio      : {amplitudes.zero_ratio * 100:.6f}%",
                f"    min / max       : {_format_optional(amplitudes.minimum)} / "
                f"{_format_optional(amplitudes.maximum)}",
                f"    mean / std      : {_format_optional(amplitudes.mean)} / "
                f"{_format_optional(amplitudes.standard_deviation)}",
                f"    RMS             : {_format_optional(amplitudes.rms)}",
            ]
        )
        lines.extend(f"  issue             : {issue}" for issue in file_inspection.issues)

    overall = "OK" if seg_c3_na_inspection_ok(inspection) else "ISSUES FOUND"
    lines.extend(["", f"Overall: {overall}"])
    return "\n".join(lines)
