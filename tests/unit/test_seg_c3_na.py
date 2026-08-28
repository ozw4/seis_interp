from __future__ import annotations

import hashlib
import io
from pathlib import Path
from urllib.request import Request

import pytest
import yaml

from seis_interp.data import seg_c3_na
from seis_interp.data.seg_c3_na import (
    DataIntegrityError,
    DatasetManifest,
    FileSpec,
    ManifestError,
    VerificationResult,
    download_seg_c3_na,
    load_manifest,
    verified_source_sha256,
    verify_seg_c3_na,
)


class FakeResponse(io.BytesIO):
    def __init__(
        self,
        content: bytes,
        status: int,
        *,
        content_length: int | None = None,
        content_range: str | None = None,
    ) -> None:
        super().__init__(content)
        self.status = status
        length = len(content) if content_length is None else content_length
        self.headers = {"Content-Length": str(length)}
        if content_range is not None:
            self.headers["Content-Range"] = content_range

    def getcode(self) -> int:
        return self.status


class FakeOpener:
    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.ranges: list[str | None] = []

    def __call__(self, request: Request, timeout: float) -> FakeResponse:
        assert timeout > 0
        content = self.files[request.full_url]
        range_header = request.get_header("Range")
        self.ranges.append(range_header)
        if range_header is None:
            return FakeResponse(content, status=200)

        start_byte = int(range_header.removeprefix("bytes=").removesuffix("-"))
        return FakeResponse(
            content[start_byte:],
            status=206,
            content_range=f"bytes {start_byte}-{len(content) - 1}/{len(content)}",
        )


def _write_manifest(tmp_path: Path, files: dict[str, bytes]) -> Path:
    manifest_path = tmp_path / "manifest.yaml"
    manifest = {
        "dataset_id": "seg_c3_na",
        "files": [
            {
                "name": name,
                "url": f"https://example.test/{name}",
                "size_bytes": None,
                "sha256": None,
            }
            for name in files
        ],
    }
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return manifest_path


def test_load_manifest_rejects_non_https_url(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "dataset_id": "seg_c3_na",
                "files": [{"name": "part.sgy", "url": "http://example.test/part.sgy"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ManifestError, match="HTTPS"):
        load_manifest(manifest_path)


def test_load_manifest_preserves_optional_ffid_ranges_in_file_order(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "dataset_id": "seg_c3_na",
                "files": [
                    {
                        "name": "second.sgy",
                        "url": "https://example.test/second.sgy",
                        "ffid_min": 20,
                        "ffid_max": 29,
                    },
                    {
                        "name": "first.sgy",
                        "url": "https://example.test/first.sgy",
                        "ffid_min": 10,
                        "ffid_max": 19,
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    manifest = load_manifest(manifest_path)

    assert [(file.name, file.ffid_min, file.ffid_max) for file in manifest.files] == [
        ("second.sgy", 20, 29),
        ("first.sgy", 10, 19),
    ]


def test_load_manifest_allows_an_omitted_ffid_range(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path, {"part.sgy": b"content"})

    file_spec = load_manifest(manifest_path).files[0]

    assert file_spec.ffid_min is None
    assert file_spec.ffid_max is None


@pytest.mark.parametrize(
    "range_fields",
    [
        {"ffid_min": 10},
        {"ffid_max": 20},
        {"ffid_min": 20, "ffid_max": 10},
    ],
)
def test_load_manifest_rejects_an_invalid_ffid_range(
    tmp_path: Path, range_fields: dict[str, int]
) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "dataset_id": "seg_c3_na",
                "files": [
                    {
                        "name": "part.sgy",
                        "url": "https://example.test/part.sgy",
                        **range_fields,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ManifestError, match="ffid"):
        load_manifest(manifest_path)


def test_download_resumes_and_writes_verifiable_lock(tmp_path: Path, monkeypatch) -> None:
    files = {
        "part_1.sgy": b"first-file-content" * 100,
        "part_2.sgy": b"second-file-content" * 120,
    }
    manifest_path = _write_manifest(tmp_path, files)
    data_root = tmp_path / "data-root"
    dataset_dir = data_root / "external" / "seg_c3_na"
    dataset_dir.mkdir(parents=True)
    resume_size = 73
    (dataset_dir / "part_1.sgy.part").write_bytes(files["part_1.sgy"][:resume_size])

    opener = FakeOpener(
        {f"https://example.test/{name}": content for name, content in files.items()}
    )
    monkeypatch.setattr(seg_c3_na, "urlopen", opener)

    lock_path = download_seg_c3_na(
        manifest_path,
        data_root,
        reporter=lambda _message: None,
    )

    assert lock_path == dataset_dir / "download.lock.yaml"
    assert opener.ranges == [f"bytes={resume_size}-", None]
    for name, content in files.items():
        assert (dataset_dir / name).read_bytes() == content

    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    assert lock["dataset_id"] == "seg_c3_na"
    assert {record["name"]: record["sha256"] for record in lock["files"]} == {
        name: hashlib.sha256(content).hexdigest() for name, content in files.items()
    }

    results = verify_seg_c3_na(manifest_path, data_root)
    assert all(result.ok for result in results)
    assert {result.status for result in results} == {"ok"}
    assert [result.path for result in results] == [dataset_dir / name for name in files]
    assert [result.sha256 for result in results] == [
        hashlib.sha256(content).hexdigest() for content in files.values()
    ]


def test_existing_file_must_match_previous_lock(tmp_path: Path, monkeypatch) -> None:
    files = {"part.sgy": b"trusted-content" * 100}
    manifest_path = _write_manifest(tmp_path, files)
    data_root = tmp_path / "data-root"
    opener = FakeOpener({"https://example.test/part.sgy": files["part.sgy"]})
    monkeypatch.setattr(seg_c3_na, "urlopen", opener)

    download_seg_c3_na(manifest_path, data_root, reporter=lambda _message: None)
    data_file = data_root / "external" / "seg_c3_na" / "part.sgy"
    data_file.write_bytes(b"corrupted")

    with pytest.raises(DataIntegrityError, match="expected"):
        download_seg_c3_na(manifest_path, data_root, reporter=lambda _message: None)

    download_seg_c3_na(
        manifest_path,
        data_root,
        force=True,
        reporter=lambda _message: None,
    )
    assert data_file.read_bytes() == files["part.sgy"]


def test_verify_requires_recorded_checksums(tmp_path: Path) -> None:
    files = {"part.sgy": b"content"}
    manifest_path = _write_manifest(tmp_path, files)
    data_root = tmp_path / "data-root"
    dataset_dir = data_root / "external" / "seg_c3_na"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "part.sgy").write_bytes(files["part.sgy"])

    results = verify_seg_c3_na(manifest_path, data_root)

    assert len(results) == 1
    assert not results[0].ok
    assert results[0].status == "unlocked"


def test_verified_source_sha256_returns_manifest_order_after_structural_matching(
    tmp_path: Path,
) -> None:
    specs = (
        FileSpec("a.sgy", "https://example.test/a.sgy", None, None),
        FileSpec("b.sgy", "https://example.test/b.sgy", None, None),
    )
    manifest = DatasetManifest(tmp_path / "manifest.yaml", specs)
    source_paths = tuple(tmp_path / spec.name for spec in specs)
    expected = ("a" * 64, "b" * 64)
    results = tuple(
        VerificationResult(
            spec.name,
            True,
            "ok",
            "verified",
            path=path,
            sha256=digest,
        )
        for spec, path, digest in zip(specs, source_paths, expected, strict=True)
    )

    assert verified_source_sha256(manifest, source_paths, results) == expected


@pytest.mark.parametrize(
    ("path", "sha256", "message"),
    [
        (Path("wrong/a.sgy"), "a" * 64, "refers to"),
        (Path("a.sgy"), "not-a-digest", "valid SHA-256"),
    ],
)
def test_verified_source_sha256_rejects_untrusted_result_metadata(
    tmp_path: Path,
    path: Path,
    sha256: str,
    message: str,
) -> None:
    spec = FileSpec("a.sgy", "https://example.test/a.sgy", None, None)
    manifest = DatasetManifest(tmp_path / "manifest.yaml", (spec,))
    source_path = tmp_path / "a.sgy"
    result_path = source_path if path == Path("a.sgy") else tmp_path / path
    result = VerificationResult(
        spec.name,
        True,
        "ok",
        "verified",
        path=result_path,
        sha256=sha256,
    )

    with pytest.raises(DataIntegrityError, match=message):
        verified_source_sha256(manifest, (source_path,), (result,))


def test_incomplete_http_response_keeps_partial_file(tmp_path: Path, monkeypatch) -> None:
    content = b"incomplete-content"
    manifest_path = _write_manifest(tmp_path, {"part.sgy": content})
    data_root = tmp_path / "data-root"

    def truncated_response(request: Request, timeout: float) -> FakeResponse:
        assert request.full_url == "https://example.test/part.sgy"
        assert timeout > 0
        return FakeResponse(content, status=200, content_length=len(content) + 10)

    monkeypatch.setattr(seg_c3_na, "urlopen", truncated_response)

    with pytest.raises(OSError, match="incomplete response"):
        download_seg_c3_na(manifest_path, data_root, reporter=lambda _message: None)

    dataset_dir = data_root / "external" / "seg_c3_na"
    assert not (dataset_dir / "part.sgy").exists()
    assert (dataset_dir / "part.sgy.part").read_bytes() == content
