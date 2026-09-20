from zipfile import ZIP_STORED, BadZipFile, ZipFile

import pytest

from seis_interp.data.forge_archives import verify_forge_archive


def test_archive_crc_and_extracted_bytes_are_verified(tmp_path):
    archive = tmp_path / "data.zip"
    destination = tmp_path / "extracted"
    destination.mkdir()
    with ZipFile(archive, "w", compression=ZIP_STORED) as handle:
        handle.writestr("trace.sgy", b"test seismic bytes")
    (destination / "trace.sgy").write_bytes(b"test seismic bytes")
    summary = verify_forge_archive(archive, destination)
    assert summary["member_count"] == 1
    assert summary["crc_and_extracted_sha256_match"]
    assert len(summary["members"][0]["sha256"]) == 64
    (destination / "trace.sgy").write_bytes(b"different contents")
    with pytest.raises(ValueError, match="differs from ZIP"):
        verify_forge_archive(archive, destination)
    (destination / "trace.sgy").write_bytes(b"test seismic bytes")
    archive.write_bytes(archive.read_bytes().replace(b"test seismic bytes", b"test seismic byte!"))
    with pytest.raises(BadZipFile, match="CRC"):
        verify_forge_archive(archive, destination)


def test_missing_extraction_and_unsafe_member_are_rejected(tmp_path):
    archive = tmp_path / "data.zip"
    with ZipFile(archive, "w") as handle:
        handle.writestr("trace.sgy", b"abc")
    with pytest.raises(ValueError, match="missing or incomplete"):
        verify_forge_archive(archive, tmp_path / "missing")
    with ZipFile(archive, "w") as handle:
        handle.writestr("../escape.sgy", b"abc")
    with pytest.raises(ValueError, match="outside extraction"):
        verify_forge_archive(archive, tmp_path)


def test_deflate64_uses_installed_unzip_and_checks_extracted_bytes(tmp_path):
    import struct
    from zipfile import ZIP_DEFLATED

    # A short literal-only deflate stream also has valid Deflate64 coding.
    archive = tmp_path / "deflate64.zip"
    payload = b"abcdefghijklmnop"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as handle:
        handle.writestr("trace.sgy", payload)
    raw = bytearray(archive.read_bytes())
    struct.pack_into("<H", raw, 8, 9)
    struct.pack_into("<H", raw, raw.index(b"PK\x01\x02") + 10, 9)
    archive.write_bytes(raw)
    (tmp_path / "trace.sgy").write_bytes(payload)
    assert verify_forge_archive(archive, tmp_path)["crc_and_extracted_sha256_match"]
    (tmp_path / "trace.sgy").write_bytes(payload[::-1])
    with pytest.raises(ValueError, match="differs from ZIP"):
        verify_forge_archive(archive, tmp_path)
