from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from seis_interp.visualization.forge_mvp import plot_preparation


@pytest.mark.parametrize("test_count", [7, 8])
def test_preparation_plot_labels_the_actual_missing_fraction(tmp_path, monkeypatch, test_count):
    manifest = pd.DataFrame({"source_cell": [0] * 5 + [1] * 5, "receiver_cell": list(range(5)) * 2})
    mask = manifest.copy()
    mask["split"] = ["test"] * test_count + ["observed"] * (10 - test_count)
    geometry = pd.DataFrame(
        {
            "source_x_m": manifest.source_cell,
            "source_y_m": np.zeros(10),
            "receiver_x_m": manifest.receiver_cell,
            "receiver_y_m": np.ones(10),
        }
    )
    titles = {}

    def capture(figure, path, **kwargs):
        titles[Path(path).name] = figure.axes[0].get_title()

    monkeypatch.setattr(Figure, "savefig", capture)
    plot_preparation(manifest, mask, {"real": geometry, "grid": geometry}, tmp_path)

    assert titles["outer_mask.png"] == f"Fixed random {test_count * 10}% outer mask"
