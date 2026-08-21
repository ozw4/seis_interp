"""Resolve the external data root without embedding machine-specific paths."""

from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT_ENV = "SEIS_INTERP_DATA_ROOT"
EXAMPLE_PATH_PREFIXES = (
    "/absolute/path/to/",
    "/absolute/host/path/to/",
    "/replace/with/",
)


class DataRootError(RuntimeError):
    """Raised when the external data root is not configured."""


def _configured_value(value: str | Path | None) -> str:
    raw_value = value if value is not None else os.environ.get(DATA_ROOT_ENV)
    if raw_value is None or not str(raw_value).strip():
        raise DataRootError(
            f"{DATA_ROOT_ENV} is not set. Export it or pass --data-root explicitly."
        )

    configured_value = str(raw_value).strip()
    if configured_value.startswith(EXAMPLE_PATH_PREFIXES):
        raise DataRootError(
            f"{DATA_ROOT_ENV} still contains an example path: {configured_value}. "
            "Replace it with a writable data root outside the repository."
        )
    return configured_value


def resolve_data_root(value: str | Path | None = None, *, create: bool = False) -> Path:
    """Return an absolute data-root path from an override or environment variable."""
    path = Path(_configured_value(value)).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()

    if create:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise DataRootError(
                f"Cannot create data root {path}: {exc}. "
                f"Set {DATA_ROOT_ENV} to a writable directory outside the repository."
            ) from exc

    return path


def external_dataset_dir(data_root: str | Path | None, dataset_id: str) -> Path:
    """Return the directory for an external dataset beneath the configured root."""
    return resolve_data_root(data_root) / "external" / dataset_id
