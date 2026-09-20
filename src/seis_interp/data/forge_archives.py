"""Validate downloaded FORGE ZIP members against their extracted local files."""

import hashlib
import subprocess
from pathlib import Path
from zipfile import ZipFile

from seis_interp.data.forge_headers import sha256_file


def _deflate64_sha256(archive: Path, member: str) -> str:
    """Read the provider's Deflate64 members with the installed Info-ZIP tool."""
    checksum = hashlib.sha256()
    with subprocess.Popen(
        ["unzip", "-p", str(archive), member],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as process:
        for chunk in iter(lambda: process.stdout.read(1024 * 1024), b""):
            checksum.update(chunk)
        error = process.stderr.read().decode(errors="replace")
        if process.wait() != 0:
            raise ValueError(f"Deflate64 ZIP validation failed for {member}: {error}")
    return checksum.hexdigest()


def verify_forge_archive(archive: Path, destination: Path) -> dict:
    """Read every member to verify ZIP CRC and compare SHA-256 after extraction.

    This is local integrity evidence, not a publisher-provided checksum. Nothing
    is extracted or overwritten; missing or differing members stop the audit.
    """
    members = []
    destination = destination.resolve()
    with ZipFile(archive) as handle:
        names = set()
        for entry in handle.infolist():
            if entry.is_dir():
                continue
            target = (destination / entry.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"ZIP member outside extraction directory: {entry.filename}")
            if entry.filename in names:
                raise ValueError(f"duplicate ZIP member: {entry.filename}")
            names.add(entry.filename)
            if not target.is_file() or target.stat().st_size != entry.file_size:
                raise ValueError(f"missing or incomplete extracted file: {entry.filename}")
            if entry.compress_type == 9:
                digest = _deflate64_sha256(archive, entry.filename)
            else:
                checksum = hashlib.sha256()
                with handle.open(entry) as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        checksum.update(chunk)
                digest = checksum.hexdigest()
            if sha256_file(target) != digest:
                raise ValueError(f"extracted file differs from ZIP: {entry.filename}")
            members.append(
                {
                    "member": entry.filename,
                    "size_bytes": entry.file_size,
                    "sha256": digest,
                }
            )
    return {
        "member_count": len(members),
        "uncompressed_bytes": sum(m["size_bytes"] for m in members),
        "crc_and_extracted_sha256_match": True,
        "members": members,
    }
