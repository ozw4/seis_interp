import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from seis_interp.visualization.forge_geometry_map import (
    create_source_mask_map,
    prepare_source_mask_map,
)


@pytest.fixture
def geometry():
    manifest = pd.DataFrame(
        {
            "cell_id": [0, 1, 2, 3, 4],
            "source_cell": [0, 0, 0, 1, 1],
            "receiver_cell": [0, 1, 2, 0, 1],
            "source_x_m": [100, 100, 100, 200, 200],
            "source_y_m": [300, 300, 300, 400, 400],
            "receiver_x_m": [110, 120, 130, 110, 120],
            "receiver_y_m": [310, 320, 330, 310, 320],
            "source_grid_x_m": [999] * 5,
            "receiver_grid_x_m": [999] * 5,
        }
    )
    mask = pd.DataFrame(
        {"cell_id": [4, 2, 0, 3, 1], "split": ["observed", "test", "observed", "test", "test"]}
    )
    return manifest, mask


def test_actual_map_joins_reordered_mask_and_keeps_natural_missing(geometry):
    manifest, mask = geometry
    before = manifest.copy(deep=True)
    sources, receivers = prepare_source_mask_map(manifest, mask, source_cell=1)
    assert sources.cell_id.tolist() == [0, 1]
    assert sources.x_m.tolist() == [100, 200]
    assert receivers.x_m.tolist() == [110, 120, 130]
    assert receivers.split.tolist() == ["test", "observed", "natural_missing"]
    _, other_receivers = prepare_source_mask_map(manifest, mask, source_cell=0)
    assert other_receivers.split.tolist() == ["observed", "test", "test"]
    pd.testing.assert_frame_equal(manifest, before)


def test_actual_map_rejects_unknown_source_or_inconsistent_coordinates(geometry):
    manifest, mask = geometry
    with pytest.raises(ValueError, match="selected source cell"):
        prepare_source_mask_map(manifest, mask, source_cell=9)
    manifest.loc[4, "receiver_x_m"] = 999
    with pytest.raises(ValueError, match="inconsistent actual receiver coordinates"):
        prepare_source_mask_map(manifest, mask, source_cell=1)


def test_actual_map_rejects_mismatched_or_excluded_mask(geometry):
    manifest, mask = geometry
    with pytest.raises(ValueError, match="trace cells must match"):
        prepare_source_mask_map(manifest, mask.iloc[:-1], source_cell=1)
    mask.loc[0, "split"] = "excluded"
    with pytest.raises(ValueError, match="observed/test"):
        prepare_source_mask_map(manifest, mask, source_cell=1)


def test_map_places_sources_and_mask_states_on_one_equal_scale_axis(geometry):
    sources, receivers = prepare_source_mask_map(*geometry, source_cell=1)
    figure = create_source_mask_map(
        sources, receivers, source_cell=1, missing_fraction=0.7, mask_seed=600701
    )
    assert len(figure.axes) == 1
    ax = figure.axes[0]
    assert ax.get_aspect() == 1
    artists = {artist.get_label(): artist for artist in ax.collections}
    for name, expected in (
        ("Other sources (1)", [[100, 300]]),
        ("Receivers: observed (1)", [[120, 320]]),
        ("Receivers: masked (1)", [[110, 310]]),
        ("Receivers: natural missing (1)", [[130, 330]]),
        ("Selected source (cell 1)", [[200, 400]]),
    ):
        np.testing.assert_array_equal(artists[name].get_offsets(), expected)
    figure.canvas.draw()
    assert "70% missing" in figure.texts[1].get_text()
    renderer = figure.canvas.get_renderer()
    title, subtitle = [item.get_window_extent(renderer) for item in figure.texts]
    assert not title.overlaps(subtitle)
    legend = figure.legends[0].get_window_extent(renderer)
    assert 0 <= legend.x0 < legend.x1 <= figure.bbox.x1
    assert 0 <= legend.y0 < legend.y1 <= figure.bbox.y1
    plt.close(figure)
