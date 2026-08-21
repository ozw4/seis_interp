"""Resolve the external data root without embedding machine-specific paths."""

from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT_ENV = "SEIS_INTERP_DATA_ROOT"


class DataRootError(RuntimeError):
    """Raised when the external data root is not configured."""


def resolve_data_root(value: str | Path | None = None, *, create: bool = False) -> Path:
    """Return an absolute data-root path from an override or environment variable."""
    raw_value = value if value is not None else os.environ.get(DATA_ROOT_ENV)
    if raw_value is None or not str(raw_value).strip():
        raise DataRootError(
            f"{DATA_ROOT_ENV} is not set. Export it or pass --data-root explicitly."
        )

    path = Path(raw_value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()

    if create:
        path.mkdir(parents=True, exist_ok=True)

    return path


def external_dataset_dir(data_root: str | Path | None, dataset_id: str) -> Path:
    """Return the directory for an external dataset beneath the configured root."""
    return resolve_data_root(data_root) / "external" / dataset_id
