"""Publication layout retains numerical section data and shared scales."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from seis_interp.visualization.publication_sections import create_publication_section


def test_publication_section_preserves_data_and_physical_layout() -> None:
    panels = [np.arange(24).reshape(6, 4) * (index + 1) for index in range(12)]
    original_font = matplotlib.rcParams["font.family"][:]
    titles = [f"Method {index}" for index in range(12)]
    figure = create_publication_section(
        panels,
        time_s=np.arange(6) * 0.008,
        spatial_label="Local source line index",
        clip=12,
        font_family="DejaVu Sans",
        panel_titles=titles,
    )
    try:
        figure.canvas.draw()
        assert figure.get_figwidth() * 25.4 == pytest.approx(178)
        assert figure._suptitle is None
        assert figure._supxlabel is None
        assert figure._supylabel is None
        for index, axis in enumerate(figure.axes[:12]):
            assert axis.get_title() == titles[index]
            letter = axis.texts[0]
            assert letter.get_text() == f"({chr(97 + index)})"
            assert letter.get_fontsize() == 10
            assert letter.get_fontweight() == "bold"
            assert letter.get_position() == (1, 6)
            bounds = letter.get_window_extent(figure.canvas.get_renderer())
            assert bounds.x1 == pytest.approx(axis.bbox.x0 + figure.dpi / 72)
            assert bounds.y0 > axis.bbox.y1
            assert figure.bbox.contains(bounds.x0, bounds.y1)
            assert axis.get_xlabel() == ("Local source line index" if index >= 10 else "")
            assert axis.get_ylabel() == ("Time (s)" if index % 2 == 0 else "")
            artist = axis.images[0]
            np.testing.assert_array_equal(artist.get_array(), panels[index])
            assert artist.get_clim() == (-12, 12)
            assert artist.get_cmap().name == "seismic"
            assert artist.get_interpolation() == "none"
            assert artist.get_extent() == pytest.approx([-0.5, 3.5, 0.044, -0.004])
            assert len(axis.get_xticks()) > 0
            assert len(axis.get_yticks()) > 0
            assert len(axis.get_xticks(minor=True)) == len(axis.get_yticks(minor=True)) == 0
            assert bool(axis.get_xticklabels()) == (index >= 10)
            assert bool(axis.get_yticklabels()) == (index % 2 == 0)
            for tick in [*axis.xaxis.get_major_ticks(), *axis.yaxis.get_major_ticks()]:
                assert tick.tick1line.get_markersize() == tick.tick2line.get_markersize() == 0
        assert matplotlib.rcParams["font.family"] == original_font
    finally:
        plt.close(figure)


@pytest.mark.parametrize("clip", [0, -1, np.nan, np.inf])
def test_invalid_amplitude_scale_is_rejected(clip: float) -> None:
    with pytest.raises(ValueError, match="clip"):
        create_publication_section(
            [np.ones((2, 2))] * 2,
            time_s=np.arange(2),
            spatial_label="Index",
            clip=clip,
        )


def test_wiggle_preserves_trace_positions_and_uses_shared_gain() -> None:
    values = np.array([[-20.0, 0.0, 5.0], [0.0, 0.0, 10.0], [20.0, 0.0, -5.0]])
    original = values.copy()
    time_s = np.array([0.0, 0.008, 0.016])
    figure = create_publication_section(
        [values, values / 2],
        time_s=time_s,
        spatial_label="Index",
        clip=10,
        display="wiggle",
        font_family="DejaVu Sans",
    )
    try:
        assert len(figure.axes) == 2  # No image colorbar for wiggles.
        assert not figure.texts
        for axis, panel in zip(figure.axes, [values, values / 2], strict=True):
            assert not axis.images
            assert len(axis.lines) == len(axis.collections) == 3
            assert axis.get_ylim() == pytest.approx([0.020, -0.004])
            for position, line in enumerate(axis.lines):
                np.testing.assert_allclose(
                    line.get_xdata(),
                    position + 0.45 * np.clip(panel[:, position], -10, 10) / 10,
                )
                np.testing.assert_array_equal(line.get_ydata(), time_s)
            np.testing.assert_array_equal(axis.lines[1].get_xdata(), np.ones(3))
        np.testing.assert_array_equal(values, original)
    finally:
        plt.close(figure)


def test_six_column_wiggle_layout_labels_and_order() -> None:
    panels = [np.full((3, 4), index / 20) for index in range(12)]
    figure = create_publication_section(
        panels,
        time_s=np.arange(3),
        spatial_label="Local\nindex",
        clip=1,
        display="wiggle",
        columns=6,
        font_family="DejaVu Sans",
    )
    try:
        figure.canvas.draw()
        assert figure.get_figwidth() * 25.4 == pytest.approx(178)
        for index, axis in enumerate(figure.axes):
            grid = axis.get_subplotspec().get_gridspec()
            assert (grid.nrows, grid.ncols) == (2, 6)
            assert axis.get_subplotspec().num1 == index
            assert axis.get_ylabel() == ("Time (s)" if index % 6 == 0 else "")
            assert axis.get_xlabel() == ("Local\nindex" if index >= 6 else "")
            np.testing.assert_allclose(axis.lines[0].get_xdata(), 0.45 * index / 20)
    finally:
        plt.close(figure)


def test_six_column_heatmap_colorbar_label_fits_inside_figure() -> None:
    figure = create_publication_section(
        [np.ones((3, 4))] * 12,
        time_s=np.arange(3),
        spatial_label="Receiver y\nindex",
        clip=1,
        columns=6,
        font_family="DejaVu Sans",
    )
    try:
        figure.canvas.draw()
        label = figure.axes[-1].xaxis.label
        assert label.get_text() == "Amplitude"
        bounds = label.get_window_extent(figure.canvas.get_renderer())
        assert figure.bbox.contains(bounds.x0, bounds.y0)
        assert figure.bbox.contains(bounds.x1, bounds.y1)
    finally:
        plt.close(figure)


def test_heatmap_without_colorbar_expands_panels_and_keeps_labels_inside() -> None:
    figure = create_publication_section(
        [np.ones((3, 4))] * 12,
        time_s=np.arange(3),
        spatial_label="Local shot\nin line\nindex",
        clip=1,
        columns=6,
        show_colorbar=False,
        font_family="Arial",
        panel_titles=[
            "Reference",
            "POCS",
            "DRR",
            "SIREN",
            "CCNet-5D",
            "Relational\ntrace graph",
            "Masked\ninput",
            "Residual\n(POCS)",
            "Residual\n(DRR)",
            "Residual\n(SIREN)",
            "Residual\n(CCNet-5D)",
            "Residual\n(trace graph)",
        ],
    )
    try:
        figure.canvas.draw()
        assert len(figure.axes) == 12
        assert figure.subplotpars.left == 0.05
        for axis in figure.axes:
            assert all(spine.get_linewidth() == 0.5 for spine in axis.spines.values())
            assert axis.get_position().width > 0.91 / 8
            assert axis.get_position().height > 0.60 / 2.60
            for text in [axis.title, axis.xaxis.label, axis.yaxis.label, *axis.texts]:
                if text.get_text():
                    bounds = text.get_window_extent(figure.canvas.get_renderer())
                    assert figure.bbox.contains(bounds.x0, bounds.y0)
                    assert figure.bbox.contains(bounds.x1, bounds.y1)
    finally:
        plt.close(figure)


def test_two_five_five_layout_keeps_all_panels_in_order() -> None:
    panels = [np.full((4, 3), index) for index in range(12)]
    figure = create_publication_section(
        panels,
        time_s=np.arange(4),
        spatial_label="Receiver y\nindex",
        clip=12,
        show_colorbar=False,
        layout="2-5-5",
        font_family="DejaVu Sans",
    )
    try:
        figure.canvas.draw()
        assert len(figure.axes) == 12
        assert figure.get_figwidth() * 25.4 == pytest.approx(178)
        for index, axis in enumerate(figure.axes):
            assert axis.get_subplotspec().get_gridspec().ncols == 5
            np.testing.assert_array_equal(axis.images[0].get_array(), panels[index])
            assert bool(axis.get_yticklabels()) == (index in {0, 2, 7})
            assert bool(axis.get_xticklabels()) == (index < 2 or index >= 7)
            assert axis.get_xlabel() == ("Receiver y\nindex" if index >= 7 else "")
        for axis in figure.axes:
            assert axis.get_position().width == pytest.approx(figure.axes[2].get_position().width)
            assert axis.get_position().height == pytest.approx(figure.axes[2].get_position().height)
        assert figure.axes[0].get_position().x0 == pytest.approx(figure.axes[2].get_position().x0)
        assert figure.axes[1].get_position().x0 == pytest.approx(figure.axes[3].get_position().x0)
        assert figure.axes[0].get_position().y0 > figure.axes[2].get_position().y0
        assert figure.axes[2].get_position().y0 > figure.axes[7].get_position().y0
        assert figure.get_figheight() * 25.4 == pytest.approx(142)
        gap = figure.axes[0].get_position().y0 - figure.axes[2].get_position().y1
        assert gap / figure.axes[0].get_position().height == pytest.approx(0.27)
    finally:
        plt.close(figure)
