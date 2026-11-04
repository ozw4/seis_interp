"""Geometry-only sections, completion and residual rules, and saved-score guards."""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from seis_interp.data.forge_headers import sha256_file
from seis_interp.pipelines.export_common_grid_gathers import export_common_grid_gathers
from seis_interp.visualization.common_grid_gathers import (
    DISPLAY_METHODS,
    PANEL_TITLES,
    central_sections,
    completed_section_panels,
    pooled_reference_clip,
)

SHAPE = (2, 3, 2, 4)
SEED = 5


def test_central_sections_are_geometry_only_ordered_and_complete():
    cells = np.arange(np.prod(SHAPE)).reshape(SHAPE)
    sections = central_sections(cells.ravel()[::-1], SHAPE, (3, 1, 0, 2))
    expected = [cells[1, 1, 1, :], cells[1, :, 1, 2], cells[:, 1, 1, 2], cells[1, 1, :, 2]]
    for section, selected in zip(sections, expected, strict=True):
        assert section["cell_ids"] == selected.tolist()
    assert [s["varying_axis"] for s in central_sections(cells.ravel(), SHAPE, (3, 1, 0))] == [
        "receiver_grid_point",
        "source_grid_point",
        "source_grid_line",
    ]
    with pytest.raises(ValueError, match="missing grid cells"):
        central_sections(np.delete(cells.ravel(), expected[1][0]), SHAPE, (3, 1, 0, 2))


@pytest.mark.parametrize("mode", ["physical", "trace_rms"])
def test_panels_complete_observed_traces_and_subtract_from_reference(mode):
    reference = np.array([[0, 0, 4, 0], [0, 8, 0, 0], [0, 0, 0, 0]], np.float64)
    observed = np.array([True, False, False])
    predictions = {method: reference * (i + 2) for i, method in enumerate(DISPLAY_METHODS)}
    for values in predictions.values():
        values[0] = np.nan  # Observed rows are never read from predictions.
        values[2] = 1  # Silent reference: divisor 1 in the RMS view.
    panels = completed_section_panels(reference, observed, predictions, normalization=mode)
    assert len(panels) == len(PANEL_TITLES) == 12
    scale = np.array([2, 4, 1])[:, None] if mode == "trace_rms" else np.ones((3, 1))
    target = (reference / scale).T
    np.testing.assert_array_equal(panels[0], target)
    np.testing.assert_array_equal(panels[1], np.where(observed[None, :], target, 0))
    for i, method in enumerate(DISPLAY_METHODS):
        completed = np.where(observed[:, None], reference, predictions[method]) / scale
        np.testing.assert_allclose(panels[2 + i], completed.T)
        np.testing.assert_allclose(panels[7 + i], target - completed.T)
        assert np.all(panels[7 + i][:, 0] == 0)
    assert pooled_reference_clip([panels]) == pytest.approx(np.percentile(np.abs(target), 99))


def test_panels_reject_wrong_order_and_missing_hidden_predictions():
    reference = np.ones((2, 3))
    observed = np.array([True, False])
    good = {method: np.ones((2, 3)) for method in DISPLAY_METHODS}
    with pytest.raises(ValueError, match="display order"):
        completed_section_panels(
            reference, observed, dict(reversed(good.items())), normalization="physical"
        )
    bad = {**good, "regsi_real": np.full((2, 3), np.nan)}
    with pytest.raises(ValueError, match="finite predictions"):
        completed_section_panels(reference, observed, bad, normalization="physical")


def write_experiment(root):
    """A FORGE-layout experiment: every cell present, saved scores from the arrays."""
    experiment = root / "runs" / "experiment"
    samples, cells = 6, int(np.prod(SHAPE))
    grid = np.asarray(np.unravel_index(np.arange(cells), SHAPE)).T
    rng = np.random.default_rng(0)
    waveforms = rng.normal(size=(cells, samples)).astype(np.float32)
    roles = np.where(np.arange(cells) % 3 == 0, "train", "test").astype(object)
    roles[1::6] = "validation"
    (experiment / "study").mkdir(parents=True)
    (experiment / "study/config.yaml").write_text(
        yaml.safe_dump(
            {
                "spatial_shape": list(SHAPE),
                "time_sample_count": samples,
                "sample_interval_us": 4000,
                "visualization": {"time_range_s": [0.0, 0.02]},
            }
        )
    )
    seed_dir = experiment / "input" / f"seed_{SEED}"
    seed_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "cell_id": np.arange(cells),
            "source_grid_line": grid[:, 0],
            "source_grid_point": grid[:, 1],
            "receiver_grid_line": grid[:, 2],
            "receiver_grid_point": grid[:, 3],
        }
    ).to_parquet(experiment / "input/manifest.parquet")
    np.save(experiment / "input/waveforms.npy", waveforms)
    partition = pd.DataFrame(
        {
            "cell_id": np.arange(cells),
            "trace_id": [f"t{i}" for i in range(cells)],
            "role": roles,
            "clean_target": np.arange(cells) % 5 != 0,
        }
    )
    partition.to_parquet(seed_dir / "partition.parquet")
    np.save(seed_dir / "observed_waveforms.npy", waveforms[roles == "train"])
    (experiment / "workers").mkdir()
    for i, method in enumerate(DISPLAY_METHODS):
        (experiment / "workers" / f"{method}_selection.json").write_text(
            json.dumps({"winner": {"id": f"a0{i + 1}_b01"}, "test_used_for_selection": False})
        )
        run = experiment / "method_runs" / method / "evaluation" / f"seed_{SEED}"
        run.mkdir(parents=True)
        hidden = np.flatnonzero(roles != "train")
        prediction = (waveforms[hidden] * (0.5 + 0.1 * i)).astype(np.float32)
        np.save(run / "prediction.npy", prediction)
        np.save(run / "cell_ids.npy", hidden)
        for domain in ("validation", "test"):
            rows = roles[hidden] == domain
            y, p = waveforms[hidden][rows].astype(np.float64), prediction[rows]
            pd.DataFrame(
                {
                    "cell_id": hidden[rows],
                    "trace_id": partition.trace_id.to_numpy()[hidden][rows],
                    "target_energy": np.square(y).sum(axis=1),
                    "error_energy": np.square(p - y).sum(axis=1),
                }
            ).to_parquet(run / f"{domain}_per_trace.parquet")
        (run / "manifest.json").write_text(
            json.dumps(
                {
                    "id": f"a0{i + 1}_b01",
                    "trace_seed": SEED,
                    "status": "completed",
                    "artifacts": {"prediction.npy": sha256_file(run / "prediction.npy")},
                }
            )
        )
    return experiment


def export(root, experiment):
    return export_common_grid_gathers(
        root, {"forge": experiment}, root / "results" / "gathers", seed=SEED, dpi=20
    )


def test_export_writes_figures_tables_and_visualization_only_manifest(tmp_path):
    output = export(tmp_path, write_experiment(tmp_path))
    manifest = json.loads((output / "manifest.json").read_text())
    forge = manifest["datasets"]["forge"]
    assert manifest["new_inference"] is False and manifest["new_test_scoring"] is False
    assert forge["acceptance_max_abs_snr_db_difference"] <= 1e-6
    assert len(forge["figures"]) == 4 * 2 * 2
    assert all((output / "forge" / name).is_file() for name in forge["figures"])
    traces = pd.read_csv(output / "forge/section_traces.csv")
    assert set(traces.varying_axis) == {s["varying_axis"] for s in forge["sections"]}
    for section in forge["sections"]:
        assert sum(section["role_counts"].values()) == len(section["cell_ids"])


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("prediction", "hash differs"),
        ("cells", "predicted cells differ"),
        ("score", "disagree with saved scores"),
        ("observed", "observed input waveforms differ"),
    ],
)
def test_export_rejects_mismatched_inputs(tmp_path, damage, message):
    experiment = write_experiment(tmp_path)
    run = experiment / "method_runs/regsi_real/evaluation" / f"seed_{SEED}"
    if damage == "prediction":
        values = np.load(run / "prediction.npy")
        np.save(run / "prediction.npy", values + 1)
    elif damage == "cells":
        cells = np.load(run / "cell_ids.npy")
        np.save(run / "cell_ids.npy", cells + 1000)
    elif damage == "score":
        table = pd.read_parquet(run / "test_per_trace.parquet")
        table["error_energy"] *= 1.01
        table.to_parquet(run / "test_per_trace.parquet")
    else:
        path = experiment / "input" / f"seed_{SEED}" / "observed_waveforms.npy"
        np.save(path, np.load(path) + 1)
    with pytest.raises(ValueError, match=message):
        export(tmp_path, experiment)
