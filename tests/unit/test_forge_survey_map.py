import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from seis_interp.visualization.forge_survey_map import create_survey_map, prepare_survey_map


@pytest.fixture
def survey():
    geometry = pd.DataFrame(
        {
            "source_line": [1, 1, 1, 1],
            "source_point": [1, 1, 2, 2],
            "source_x_m": [100, 100, 200, 200],
            "source_y_m": [300, 300, 400, 400],
            "receiver_line": [5, 5, 5, 5],
            "receiver_point": [1, 2, 1, 2],
            "receiver_x_m": [150, 250, 150, 250],
            "receiver_y_m": [200, 200, 200, 200],
        }
    )
    navigation = pd.DataFrame(
        {"line": [1, 1, 1], "point": [3, 2, 1], "x_m": [300, 200, 100], "y_m": [500, 400, 300]}
    )
    return geometry, navigation, geometry.iloc[[2]].copy()


def test_survey_selection_matches_station_keys_and_retains_unrecorded_navigation(survey):
    sources, receivers = prepare_survey_map(*survey)
    assert sources.point.tolist() == [1, 2, 3]
    assert sources.in_analysis.tolist() == [False, True, False]
    assert sources.recorded.tolist() == [True, True, False]
    assert receivers.in_analysis.tolist() == [True, False]
    assert receivers.x_m.tolist() == [150, 250]


def test_survey_selection_rejects_unknown_keys_and_coordinate_mismatch(survey):
    geometry, navigation, analysis = survey
    analysis.loc[:, "source_point"] = 99
    with pytest.raises(ValueError, match="absent from full survey"):
        prepare_survey_map(geometry, navigation, analysis)
    analysis.loc[:, "source_point"] = 2
    analysis.loc[:, "receiver_x_m"] = 999
    with pytest.raises(ValueError, match="coordinates differ"):
        prepare_survey_map(geometry, navigation, analysis)


def test_survey_figure_has_one_actual_coordinate_axis_and_selected_extent(survey):
    sources, receivers = prepare_survey_map(*survey)
    figure = create_survey_map(sources, receivers, unknown_receiver_count=1)
    assert len(figure.axes) == 1
    ax = figure.axes[0]
    assert ax.get_aspect() == 1
    artists = {item.get_label(): item for item in ax.collections}
    np.testing.assert_array_equal(artists["M1 sources (1)"].get_offsets(), [[200, 400]])
    np.testing.assert_array_equal(artists["M1 receivers (1)"].get_offsets(), [[150, 200]])
    extent = ax.patches[0]
    assert extent.get_xy() == (150, 200)
    assert extent.get_width() == 50 and extent.get_height() == 200
    assert not extent.get_fill()
    assert "1 receiver without XY omitted" in figure.texts[-1].get_text()
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    boxes = [item.get_window_extent(renderer) for item in figure.texts]
    for box in boxes:
        assert 0 <= box.x0 < box.x1 <= figure.bbox.x1
        assert 0 <= box.y0 < box.y1 <= figure.bbox.y1
    assert not boxes[0].overlaps(boxes[1])
    plt.close(figure)
