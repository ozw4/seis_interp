"""Calculate checksums for data files."""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK_SIZE_BYTES = 8 * 1024 * 1024


def file_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(_CHUNK_SIZE_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()
