"""Load repository configuration files with explicit inheritance."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path, PureWindowsPath

import yaml

_REPOSITORY_MARKER = "pyproject.toml"


class ConfigurationError(ValueError):
    """Raised when a configuration file violates the repository contract."""


def _discover_repository_root(start: Path) -> Path | None:
    """Return the nearest marked repository containing ``start``."""
    current = start if start.is_dir() else start.parent
    for candidate in (current, *current.parents):
        if (candidate / _REPOSITORY_MARKER).is_file():
            return candidate
    return None


_PACKAGE_PATH = Path(__file__).resolve()
_PACKAGE_REPOSITORY_ROOT = _discover_repository_root(_PACKAGE_PATH)
REPOSITORY_ROOT = _PACKAGE_REPOSITORY_ROOT or _PACKAGE_PATH.parents[2]
DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "default.yaml"


def load_resolved_config(
    path: Path,
    *,
    repository_root: Path | None = None,
) -> dict[str, object]:
    """Load a YAML mapping and recursively merge its relative ``extends`` chain."""
    canonical_root = _canonical_repository_root(repository_root)
    return _load_resolved_config(
        Path(path),
        active_paths=(),
        repository_root=canonical_root,
    )


def get_required_config_value(config: Mapping[str, object], dotted_path: str) -> object:
    """Return one required value addressed by a dotted mapping path."""
    if not isinstance(config, Mapping):
        raise ConfigurationError("configuration must be a mapping")
    if (
        not isinstance(dotted_path, str)
        or not dotted_path
        or any(not part for part in dotted_path.split("."))
    ):
        raise ConfigurationError("configuration path must be a non-empty dotted path")

    parts = dotted_path.split(".")
    current: object = config
    traversed: list[str] = []
    for part in parts:
        if not isinstance(current, Mapping):
            parent_path = ".".join(traversed)
            raise ConfigurationError(
                f"cannot resolve required configuration value {dotted_path!r}: "
                f"{parent_path!r} is not a mapping"
            )
        if part not in current:
            raise ConfigurationError(f"missing required configuration value: {dotted_path}")
        current = current[part]
        traversed.append(part)
    return current


def repository_relative_config_source(
    path: Path,
    *,
    repository_root: Path | None = None,
) -> str:
    """Return a portable repository-relative source for an existing config file."""
    config_path = Path(path).expanduser().resolve(strict=True)
    if not config_path.is_file():
        raise ConfigurationError("configuration source must be a file")

    canonical_root = _canonical_repository_root(repository_root)
    if canonical_root is None:
        canonical_root = _discover_repository_root(config_path)
    if canonical_root is None:
        raise ConfigurationError(
            f"configuration source is outside a repository marked by {_REPOSITORY_MARKER}"
        )
    try:
        relative_path = config_path.relative_to(canonical_root)
    except ValueError as error:
        raise ConfigurationError("configuration source is outside its repository root") from error
    if relative_path.is_absolute():
        raise ConfigurationError("configuration source must not be an absolute path")
    return relative_path.as_posix()


def _load_resolved_config(
    path: Path,
    *,
    active_paths: tuple[Path, ...],
    repository_root: Path | None,
) -> dict[str, object]:
    """Load one level of a config inheritance chain using canonical paths."""
    canonical_path = path.expanduser().resolve(strict=True)
    _require_path_in_repository(canonical_path, repository_root)
    if canonical_path in active_paths:
        cycle_start = active_paths.index(canonical_path)
        cycle = (*active_paths[cycle_start:], canonical_path)
        cycle_text = " -> ".join(item.name for item in cycle)
        raise ConfigurationError(f"configuration extends cycle detected: {cycle_text}")

    try:
        payload = yaml.safe_load(canonical_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ConfigurationError(
            f"invalid YAML in configuration {canonical_path.name}: {error}"
        ) from error
    if not isinstance(payload, Mapping):
        raise ConfigurationError(
            f"configuration {canonical_path.name} must contain a mapping at its root"
        )
    _validate_mapping_keys(payload, source_name=canonical_path.name)

    child = dict(payload)
    if "extends" not in child:
        return child

    extends = child.pop("extends")
    if not isinstance(extends, str) or not extends.strip():
        raise ConfigurationError(
            f"configuration {canonical_path.name} extends must be a non-empty string"
        )
    extends_value = extends.strip()
    extends_path = Path(extends_value)
    windows_extends_path = PureWindowsPath(extends_value)
    if (
        extends_path.is_absolute()
        or windows_extends_path.is_absolute()
        or windows_extends_path.drive
        or windows_extends_path.root
        or "\\" in extends_value
    ):
        raise ConfigurationError(
            f"configuration {canonical_path.name} extends must be a POSIX relative path"
        )

    base = _load_resolved_config(
        canonical_path.parent / extends_path,
        active_paths=(*active_paths, canonical_path),
        repository_root=repository_root,
    )
    return _merge_mappings(base, child)


def _canonical_repository_root(path: Path | None) -> Path | None:
    if path is None:
        return None
    canonical_root = Path(path).expanduser().resolve(strict=True)
    if not canonical_root.is_dir():
        raise ConfigurationError("configuration repository root must be a directory")
    return canonical_root


def _require_path_in_repository(path: Path, repository_root: Path | None) -> None:
    if repository_root is None:
        return
    try:
        path.relative_to(repository_root)
    except ValueError as error:
        raise ConfigurationError(
            "configuration and its extends chain must stay within the repository root"
        ) from error


def _validate_mapping_keys(value: object, *, source_name: str) -> None:
    """Require string keys in every mapping nested within a config value."""
    if isinstance(value, Mapping):
        for key, nested_value in value.items():
            if not isinstance(key, str):
                raise ConfigurationError(
                    f"configuration {source_name} contains a non-string mapping key: {key!r}"
                )
            _validate_mapping_keys(nested_value, source_name=source_name)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for nested_value in value:
            _validate_mapping_keys(nested_value, source_name=source_name)


def _merge_mappings(
    base: Mapping[str, object],
    override: Mapping[str, object],
) -> dict[str, object]:
    """Recursively merge mappings while replacing every other value type."""
    merged = dict(base)
    for key, override_value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, Mapping) and isinstance(override_value, Mapping):
            merged[key] = _merge_mappings(base_value, override_value)
        else:
            merged[key] = override_value
    return merged
