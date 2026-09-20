import numpy as np

from seis_interp.visualization.c3_central_sections import prepare_central_sections


def test_fixed_sections_panel_order_and_reference_only_contrast() -> None:
    rows = np.arange(3 * 4 * 2 * 5).reshape(3, 4, 2, 5)
    source = np.arange(rows.size * 7).reshape(rows.size, 7)
    truth = source[:, 1:5].T.reshape(4, *rows.shape)
    observed = np.zeros_like(truth)
    prediction = truth + 10000
    sections, clip = prepare_central_sections(
        reference_amplitudes=source,
        array_rows=rows,
        observed_values=observed,
        predictions={"first": prediction, "second": truth},
        time_selection=(1, 5),
    )
    assert [s["varying_axis"] for s in sections] == [
        "relative_receiver_y",
        "shot_in_line",
        "source_line",
    ]
    assert sections[0]["array_rows"] == rows[1, 2, 1, :].tolist()
    for section in sections:
        panels = section["panels"]
        np.testing.assert_array_equal(panels[0], source[section["array_rows"], 1:5].T)
        assert np.all(panels[1] == 0)
        np.testing.assert_array_equal(panels[2], panels[0] + 10000)
        np.testing.assert_array_equal(panels[3], panels[0])
        assert np.all(panels[4] == -10000)
        assert np.all(panels[5] == 0)
    assert clip == np.percentile(np.concatenate([s["panels"][0].ravel() for s in sections]), 99)
