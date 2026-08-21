from __future__ import annotations

from pathlib import Path

import pytest

from seis_interp.data.data_root import DataRootError, external_dataset_dir, resolve_data_root


def test_resolve_data_root_requires_configuration(monkeypatch) -> None:
    monkeypatch.delenv("SEIS_INTERP_DATA_ROOT", raising=False)

    with pytest.raises(DataRootError, match="SEIS_INTERP_DATA_ROOT"):
        resolve_data_root()


@pytest.mark.parametrize(
    "configured_value",
    [
        "/absolute/path/to/seis_interp_data",
        "/absolute/host/path/to/seis_interp_data",
        "/replace/with/host/path/to/seis_interp_data",
    ],
)
def test_resolve_data_root_rejects_example_paths(monkeypatch, configured_value: str) -> None:
    monkeypatch.setenv("SEIS_INTERP_DATA_ROOT", configured_value)

    with pytest.raises(DataRootError, match="example path"):
        resolve_data_root(create=True)


def test_resolve_data_root_uses_explicit_override(tmp_path: Path) -> None:
    data_root = resolve_data_root(tmp_path / "data", create=True)

    assert data_root == (tmp_path / "data").resolve()
    assert data_root.is_dir()
    assert external_dataset_dir(data_root, "seg_c3_na") == data_root / "external" / "seg_c3_na"
