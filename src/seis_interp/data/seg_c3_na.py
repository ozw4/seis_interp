"""Download and verify the SEG C3 Narrow-Azimuth dataset."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import yaml

from seis_interp.data.data_root import external_dataset_dir, resolve_data_root

DATASET_ID = "seg_c3_na"
LOCK_FILENAME = "download.lock.yaml"
DEFAULT_TIMEOUT_S = 60.0
CHUNK_SIZE_BYTES = 8 * 1024 * 1024
PROGRESS_INTERVAL_BYTES = 128 * 1024 * 1024
Reporter = Callable[[str], None]


class ManifestError(ValueError):
    """Raised when a dataset manifest is missing required information."""


class DataIntegrityError(RuntimeError):
    """Raised when a local file does not match its recorded metadata."""


@dataclass(frozen=True)
class FileSpec:
    """One downloadable file declared by the source manifest."""

    name: str
    url: str
    expected_size_bytes: int | None
    expected_sha256: str | None


@dataclass(frozen=True)
class DatasetManifest:
    """Validated download information for SEG C3 NA."""

    path: Path
    files: tuple[FileSpec, ...]


@dataclass(frozen=True)
class FileRecord:
    """Observed metadata for one local file."""

    name: str
    url: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class VerificationResult:
    """Integrity status for one declared file."""

    name: str
    ok: bool
    status: str
    detail: str


def default_manifest_path() -> Path:
    """Locate the tracked manifest when running from the repository checkout."""
    relative_path = Path("data/external/seg_c3_na/manifest.yaml")
    candidates = (
        Path.cwd() / relative_path,
        Path(__file__).resolve().parents[3] / relative_path,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _required_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestError(f"{field_name} must be a mapping")
    return value


def _required_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{field_name} must be a non-empty string")
    return value.strip()


def _optional_positive_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ManifestError(f"{field_name} must be a positive integer or null")
    return value


def _required_positive_int(value: Any, field_name: str) -> int:
    parsed = _optional_positive_int(value, field_name)
    if parsed is None:
        raise ManifestError(f"{field_name} must be a positive integer")
    return parsed


def _optional_sha256(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    digest = _required_string(value, field_name).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ManifestError(f"{field_name} must be a 64-character hexadecimal SHA-256")
    return digest


def _required_sha256(value: Any, field_name: str) -> str:
    parsed = _optional_sha256(value, field_name)
    if parsed is None:
        raise ManifestError(f"{field_name} must be a SHA-256 value")
    return parsed


def _parse_file_spec(raw_file: Any, index: int) -> FileSpec:
    field_prefix = f"files[{index}]"
    file_mapping = _required_mapping(raw_file, field_prefix)
    name = _required_string(file_mapping.get("name"), f"{field_prefix}.name")
    if Path(name).name != name:
        raise ManifestError(f"{field_prefix}.name must not contain path components")

    url = _required_string(file_mapping.get("url"), f"{field_prefix}.url")
    if urlparse(url).scheme != "https":
        raise ManifestError(f"{field_prefix}.url must use HTTPS")

    return FileSpec(
        name=name,
        url=url,
        expected_size_bytes=_optional_positive_int(
            file_mapping.get("size_bytes"), f"{field_prefix}.size_bytes"
        ),
        expected_sha256=_optional_sha256(
            file_mapping.get("sha256"), f"{field_prefix}.sha256"
        ),
    )


def load_manifest(path: str | Path) -> DatasetManifest:
    """Read and validate the tracked SEG C3 NA source manifest."""
    manifest_path = Path(path).expanduser().resolve()
    try:
        raw_manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestError(f"Manifest not found: {manifest_path}") from exc
    except yaml.YAMLError as exc:
        raise ManifestError(f"Invalid YAML in manifest: {manifest_path}") from exc

    manifest_mapping = _required_mapping(raw_manifest, "manifest")
    dataset_id = _required_string(manifest_mapping.get("dataset_id"), "dataset_id")
    if dataset_id != DATASET_ID:
        raise ManifestError(f"dataset_id must be {DATASET_ID!r}, got {dataset_id!r}")

    raw_files = manifest_mapping.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise ManifestError("files must be a non-empty list")

    files = tuple(_parse_file_spec(raw_file, index) for index, raw_file in enumerate(raw_files))
    names = [file_spec.name for file_spec in files]
    if len(names) != len(set(names)):
        raise ManifestError("files must not contain duplicate names")

    return DatasetManifest(path=manifest_path, files=files)


def sha256_file(path: Path) -> str:
    """Calculate a SHA-256 digest without loading the full file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_file(path: Path, file_spec: FileSpec) -> FileRecord:
    return FileRecord(
        name=file_spec.name,
        url=file_spec.url,
        size_bytes=path.stat().st_size,
        sha256=sha256_file(path),
    )


def _validate_record(
    record: FileRecord,
    *,
    expected_size_bytes: int | None,
    expected_sha256: str | None,
) -> None:
    if expected_size_bytes is not None and record.size_bytes != expected_size_bytes:
        raise DataIntegrityError(
            f"{record.name}: expected {expected_size_bytes} bytes, got {record.size_bytes}"
        )
    if expected_sha256 is not None and record.sha256 != expected_sha256:
        raise DataIntegrityError(
            f"{record.name}: SHA-256 mismatch; rerun the download with --force"
        )


def _manifest_sha256(manifest: DatasetManifest) -> str:
    return sha256_file(manifest.path)


def _read_lock(lock_path: Path, manifest_sha256: str) -> dict[str, FileRecord] | None:
    if not lock_path.is_file():
        return None

    try:
        raw_lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise DataIntegrityError(f"Invalid YAML in download lock: {lock_path}") from exc

    lock_mapping = _required_mapping(raw_lock, "download lock")
    if lock_mapping.get("dataset_id") != DATASET_ID:
        raise DataIntegrityError(f"Unexpected dataset_id in download lock: {lock_path}")

    source_manifest = _required_mapping(lock_mapping.get("source_manifest"), "source_manifest")
    if source_manifest.get("sha256") != manifest_sha256:
        raise DataIntegrityError(
            "The local download lock was generated from a different source manifest. "
            "Remove the lock or rerun the download with --force."
        )

    raw_files = lock_mapping.get("files")
    if not isinstance(raw_files, list):
        raise DataIntegrityError(f"files must be a list in download lock: {lock_path}")

    records: dict[str, FileRecord] = {}
    for index, raw_file in enumerate(raw_files):
        file_mapping = _required_mapping(raw_file, f"download lock files[{index}]")
        name = _required_string(file_mapping.get("name"), f"download lock files[{index}].name")
        records[name] = FileRecord(
            name=name,
            url=_required_string(
                file_mapping.get("url"), f"download lock files[{index}].url"
            ),
            size_bytes=_required_positive_int(
                file_mapping.get("size_bytes"),
                f"download lock files[{index}].size_bytes",
            ),
            sha256=_required_sha256(
                file_mapping.get("sha256"), f"download lock files[{index}].sha256"
            ),
        )
    return records


def _write_lock(
    lock_path: Path,
    manifest: DatasetManifest,
    manifest_sha256: str,
    records: list[FileRecord],
) -> None:
    payload = {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_manifest": {
            "name": manifest.path.name,
            "sha256": manifest_sha256,
        },
        "files": [
            {
                "name": record.name,
                "url": record.url,
                "size_bytes": record.size_bytes,
                "sha256": record.sha256,
            }
            for record in records
        ],
    }
    temporary_path = lock_path.with_suffix(f"{lock_path.suffix}.tmp")
    temporary_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    temporary_path.replace(lock_path)


def _response_status(response: BinaryIO) -> int | None:
    status = getattr(response, "status", None)
    if status is not None:
        return int(status)
    getcode = getattr(response, "getcode", None)
    return int(getcode()) if getcode is not None else None


def _response_content_length(response: BinaryIO) -> int | None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw_length = headers.get("Content-Length")
    if raw_length is None:
        return None
    try:
        return int(raw_length)
    except (TypeError, ValueError) as exc:
        raise OSError(f"Invalid Content-Length response header: {raw_length!r}") from exc


def _response_content_range(response: BinaryIO) -> tuple[int, int] | None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw_range = headers.get("Content-Range")
    if raw_range is None:
        return None

    try:
        unit, value = raw_range.split(" ", maxsplit=1)
        byte_range, total = value.split("/", maxsplit=1)
        start, _end = byte_range.split("-", maxsplit=1)
        if unit != "bytes" or total == "*":
            raise ValueError
        return int(start), int(total)
    except (TypeError, ValueError) as exc:
        raise OSError(f"Invalid Content-Range response header: {raw_range!r}") from exc


def _open_download(file_spec: FileSpec, start_byte: int, timeout_s: float) -> BinaryIO:
    headers = {"User-Agent": "seis-interp/0.1 SEG-C3-NA-downloader"}
    if start_byte:
        headers["Range"] = f"bytes={start_byte}-"
    request = Request(file_spec.url, headers=headers)
    return urlopen(request, timeout=timeout_s)


def _copy_response(
    response: BinaryIO,
    destination: Path,
    *,
    append: bool,
    initial_size_bytes: int,
    file_name: str,
    reporter: Reporter,
) -> int:
    mode = "ab" if append else "wb"
    downloaded_bytes = initial_size_bytes if append else 0
    copied_bytes = 0
    next_report_bytes = downloaded_bytes + PROGRESS_INTERVAL_BYTES

    with destination.open(mode) as output:
        while chunk := response.read(CHUNK_SIZE_BYTES):
            output.write(chunk)
            copied_bytes += len(chunk)
            downloaded_bytes += len(chunk)
            if downloaded_bytes >= next_report_bytes:
                reporter(f"{file_name}: downloaded {downloaded_bytes / 1024**2:.1f} MiB")
                next_report_bytes += PROGRESS_INTERVAL_BYTES
        output.flush()
    return copied_bytes


def _download_file(
    file_spec: FileSpec,
    destination: Path,
    *,
    resume: bool,
    timeout_s: float,
    reporter: Reporter,
) -> None:
    part_path = destination.with_name(f"{destination.name}.part")
    if not resume:
        part_path.unlink(missing_ok=True)

    start_byte = part_path.stat().st_size if part_path.is_file() else 0
    reporter(
        f"{file_spec.name}: {'resuming' if start_byte else 'starting'} download"
        + (f" at {start_byte} bytes" if start_byte else "")
    )

    try:
        response = _open_download(file_spec, start_byte, timeout_s)
    except HTTPError as exc:
        if exc.code != 416 or not start_byte:
            raise
        part_path.unlink(missing_ok=True)
        start_byte = 0
        response = _open_download(file_spec, start_byte, timeout_s)

    with response:
        status = _response_status(response)
        append = bool(start_byte and status == 206)
        content_range = _response_content_range(response)
        if append and (content_range is None or content_range[0] != start_byte):
            raise OSError(
                f"{file_spec.name}: server returned an unexpected Content-Range for resume"
            )
        if start_byte and not append:
            reporter(f"{file_spec.name}: server ignored Range; restarting from byte 0")
        expected_response_bytes = _response_content_length(response)
        copied_bytes = _copy_response(
            response,
            part_path,
            append=append,
            initial_size_bytes=start_byte,
            file_name=file_spec.name,
            reporter=reporter,
        )
        if expected_response_bytes is not None and copied_bytes != expected_response_bytes:
            raise OSError(
                f"{file_spec.name}: incomplete response; expected "
                f"{expected_response_bytes} bytes, received {copied_bytes}"
            )
        if content_range is not None and part_path.stat().st_size != content_range[1]:
            raise OSError(
                f"{file_spec.name}: resumed file size does not match Content-Range total"
            )

    part_path.replace(destination)


def _expected_metadata(
    file_spec: FileSpec,
    lock_records: Mapping[str, FileRecord] | None,
) -> tuple[int | None, str | None]:
    lock_record = lock_records.get(file_spec.name) if lock_records is not None else None
    expected_size = file_spec.expected_size_bytes
    expected_sha256 = file_spec.expected_sha256
    if lock_record is not None:
        expected_size = expected_size or lock_record.size_bytes
        expected_sha256 = expected_sha256 or lock_record.sha256
    return expected_size, expected_sha256


def download_seg_c3_na(
    manifest_path: str | Path,
    data_root: str | Path | None = None,
    *,
    force: bool = False,
    resume: bool = True,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    reporter: Reporter = print,
) -> Path:
    """Download all declared SEG C3 NA files and write a local integrity lock."""
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")

    manifest = load_manifest(manifest_path)
    root = resolve_data_root(data_root, create=True)
    dataset_directory = root / "external" / DATASET_ID
    dataset_directory.mkdir(parents=True, exist_ok=True)

    manifest_digest = _manifest_sha256(manifest)
    lock_path = dataset_directory / LOCK_FILENAME
    lock_records = None if force else _read_lock(lock_path, manifest_digest)
    records: list[FileRecord] = []

    for file_spec in manifest.files:
        destination = dataset_directory / file_spec.name
        if force:
            destination.unlink(missing_ok=True)
            destination.with_name(f"{destination.name}.part").unlink(missing_ok=True)

        if destination.is_file():
            reporter(f"{file_spec.name}: using existing file")
        else:
            _download_file(
                file_spec,
                destination,
                resume=resume,
                timeout_s=timeout_s,
                reporter=reporter,
            )

        record = _record_file(destination, file_spec)
        expected_size, expected_sha256 = _expected_metadata(file_spec, lock_records)
        _validate_record(
            record,
            expected_size_bytes=expected_size,
            expected_sha256=expected_sha256,
        )
        reporter(f"{file_spec.name}: verified SHA-256 {record.sha256}")
        records.append(record)

    _write_lock(lock_path, manifest, manifest_digest, records)
    return lock_path


def _verification_result(
    file_spec: FileSpec,
    destination: Path,
    lock_records: Mapping[str, FileRecord] | None,
) -> VerificationResult:
    if not destination.is_file():
        return VerificationResult(file_spec.name, False, "missing", str(destination))

    expected_size, expected_sha256 = _expected_metadata(file_spec, lock_records)
    if expected_sha256 is None:
        return VerificationResult(
            file_spec.name,
            False,
            "unlocked",
            "No expected SHA-256 is available; run the download command first.",
        )

    actual_size = destination.stat().st_size
    if expected_size is not None and actual_size != expected_size:
        return VerificationResult(
            file_spec.name,
            False,
            "size_mismatch",
            f"expected {expected_size} bytes, got {actual_size}",
        )

    actual_sha256 = sha256_file(destination)
    if actual_sha256 != expected_sha256:
        return VerificationResult(
            file_spec.name,
            False,
            "checksum_mismatch",
            f"expected {expected_sha256}, got {actual_sha256}",
        )

    return VerificationResult(
        file_spec.name,
        True,
        "ok",
        f"{actual_size} bytes, SHA-256 {actual_sha256}",
    )


def verify_seg_c3_na(
    manifest_path: str | Path,
    data_root: str | Path | None = None,
) -> tuple[VerificationResult, ...]:
    """Verify local files against the tracked manifest and generated download lock."""
    manifest = load_manifest(manifest_path)
    dataset_directory = external_dataset_dir(data_root, DATASET_ID)
    lock_path = dataset_directory / LOCK_FILENAME
    lock_records = _read_lock(lock_path, _manifest_sha256(manifest))

    return tuple(
        _verification_result(
            file_spec,
            dataset_directory / file_spec.name,
            lock_records,
        )
        for file_spec in manifest.files
    )
