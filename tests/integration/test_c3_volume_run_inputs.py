from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seis_interp.data import c3_volume_run_inputs
from tests.fixtures.c3_volume_run_artifacts import prepare_c3_volume_run_artifacts


@pytest.mark.parametrize("reordered", [False, True])
def test_verified_index_preserves_volume_flat_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reordered: bool
) -> None:
    artifacts = prepare_c3_volume_run_artifacts(tmp_path)
    index, metadata = c3_volume_run_inputs.load_c3_volume_index(artifacts.volume)
    if reordered:
        index = index.iloc[::-1].reset_index(drop=True)
    calls = []

    def loaded_index(directory: Path) -> tuple[pd.DataFrame, dict[str, object]]:
        calls.append(directory)
        return index, metadata

    monkeypatch.setattr(c3_volume_run_inputs, "load_c3_volume_index", loaded_index)
    arguments = {
        "config": {},
        "interim_dir": artifacts.interim,
        "processed_dir": artifacts.processed,
        "mask_dir": artifacts.mask,
        "case_dir": artifacts.case,
        "volume_dir": artifacts.volume,
    }
    if reordered:
        with pytest.raises(ValueError, match="array_row order.*flat order"):
            c3_volume_run_inputs.load_c3_volume_run_inputs(**arguments)
    else:
        inputs = c3_volume_run_inputs.load_c3_volume_run_inputs(**arguments)
        assert inputs.index_table is index
        assert isinstance(inputs.index_table, pd.DataFrame)
        assert len(inputs.index_table) == inputs.observed_volume.array_rows.size
        np.testing.assert_array_equal(
            inputs.index_table["array_row"].to_numpy(dtype=np.int64),
            inputs.observed_volume.array_rows.reshape(-1),
        )
    assert calls == [artifacts.volume]
