from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from seis_interp.visualization.forge_mvp import (
    plot_preparation,
    plot_sections,
    plot_training_loss,
)


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


def test_sections_share_full_trace_reference_rms_and_observed_clip(tmp_path, monkeypatch):
    manifest = pd.DataFrame({"source_cell": [1, 1, 2, 2], "receiver_cell": [3, 4, 3, 4]})
    mask = pd.DataFrame({"split": ["observed", "test", "test", "observed"]})
    truth = np.array([[0, 0, 0, 6], [0, 0, 0, 60], [0, 0, 0, 0], [-4, 4, -4, 4]], np.float32)
    low, high = truth.copy(), truth.copy()
    low[1] *= 0.5
    high[1] *= 2
    low[2] = 1  # A prediction on a silent reference must stay finite and visible.
    originals = [values.copy() for values in (truth, low, high)]
    plots = {}

    def capture(figure, path, **kwargs):
        plots[Path(path).name] = {
            ax.get_title(): (ax.images[0].get_array().copy(), ax.images[0].get_clim())
            for ax in figure.axes
            if ax.images
        }
        assert any("reference trace RMS" in ax.get_ylabel() for ax in figure.axes)
        assert all(image.get_cmap().name == "seismic" for ax in figure.axes for image in ax.images)

    monkeypatch.setattr(Figure, "savefig", capture)
    plot_sections(
        manifest,
        mask,
        truth,
        {"low": low, "high": high},
        {
            "time_sample_count": 4,
            "sample_interval_us": 1000,
            "visualization": {
                "time_range_s": [0.002, 0.003],
                "source_cell": 1,
                "receiver_cell": 3,
                "clip_observed_percentile": 100,
            },
        },
        tmp_path,
    )
    # The last two samples retain RMS computed over all four samples.
    expected = np.array([[0, 2], [0, 2], [0, 0], [-1, 1]])
    expected_low = np.array([[0, 2], [0, 1], [1, 1], [-1, 1]])
    expected_high = np.array([[0, 2], [0, 4], [0, 0], [-1, 1]])
    for role, rows in (("source", [0, 1]), ("receiver", [0, 2])):
        panels = plots[f"common_{role}.png"]
        errors = plots[f"common_{role}_errors.png"]
        np.testing.assert_allclose(panels["ground truth"][0], expected[rows].T)
        np.testing.assert_allclose(panels["observed input"][0], [[0, 0], [2, 0]])
        np.testing.assert_allclose(panels["low"][0], expected_low[rows].T)
        np.testing.assert_allclose(panels["high"][0], expected_high[rows].T)
        np.testing.assert_allclose(errors["low error"][0], (expected_low - expected)[rows].T)
        np.testing.assert_allclose(errors["high error"][0], (expected_high - expected)[rows].T)
        assert all(limits == (-2, 2) for _, limits in [*panels.values(), *errors.values()])
    for values, original in zip((truth, low, high), originals, strict=True):
        np.testing.assert_array_equal(values, original)


def test_training_loss_plot_writes_png_and_keeps_history(tmp_path):
    history = [
        {"step": 1, "loss": 1.0, "learning_rate": 0.001},
        {"step": 2, "loss": 0.5, "learning_rate": 0.001},
    ]
    original = [dict(row) for row in history]
    output = tmp_path / "training_loss.png"
    plot_training_loss(history, output, report_interval=1)
    assert output.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert history == original
