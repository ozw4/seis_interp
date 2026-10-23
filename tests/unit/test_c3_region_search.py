import pytest

from seis_interp.processing.c3_region_search import (
    c3_reference_configuration,
    c3_search_candidates,
    resolve_c3_candidate,
    select_c3_candidate,
)


def _candidates():
    return c3_search_candidates(
        {"model": {"dilation": 1}, "graph": {"k": 2}},
        {
            "axis_a": [{"id": str(n), "set": {"model.dilation": n}} for n in range(1, 6)],
            "axis_b": [{"id": str(n), "set": {"graph.k": n}} for n in range(2, 7)],
        },
    )


def test_full_cartesian_search_and_ties_require_all_25_outcomes():
    candidates = _candidates()
    assert len(candidates) == 25
    assert {(c["config"]["model"]["dilation"], c["config"]["graph"]["k"]) for c in candidates} == {
        (a, b) for a in range(1, 6) for b in range(2, 7)
    }
    records = [
        {"id": c["id"], "status": "completed", "validation_score": 0.5}
        for c in reversed(candidates)
    ]
    assert select_c3_candidate(candidates, records)["id"] == "a01_b01"
    records[0]["validation_score"] = 0.4
    assert select_c3_candidate(candidates, records)["id"] == "a05_b05"
    with pytest.raises(ValueError, match="all 25"):
        select_c3_candidate(candidates, records[:-1])


def test_duplicate_axis_candidates_are_rejected():
    with pytest.raises(ValueError, match="distinct"):
        c3_search_candidates({}, {"axis_a": [{"set": {"x": 1}}] * 5, "axis_b": []})


def test_divergence_is_ranked_last_and_an_all_diverged_grid_uses_the_id_tiebreak():
    candidates = _candidates()
    records = [{"id": item["id"], "status": "diverged"} for item in candidates]
    assert select_c3_candidate(candidates, records)["id"] == "a01_b01"
    records[-1].update(status="completed", validation_score=1e30)
    assert select_c3_candidate(candidates, records)["id"] == "a05_b05"
    records[0]["status"] = "running"
    with pytest.raises(ValueError, match="final completed or diverged"):
        select_c3_candidate(candidates, records)


def test_regsi_mask_rounding_has_a_fixed_nominal_rate_and_complete_batches():
    config = {"method_id": "regsi_real", "training": {"inner_mask_fraction": 0.1}}
    for observed in (20800, 20900, 21000):
        resolved = resolve_c3_candidate(config, observed, 64)
        expected = 64 * int(0.1 * observed // 64)
        assert round(resolved["training"]["inner_mask_fraction"] * observed) == expected
    assert config["training"]["inner_mask_fraction"] == 0.1


def test_historical_reference_must_be_inside_the_explicit_grid():
    candidates = _candidates()
    assert c3_reference_configuration(candidates, {"model.dilation": 1, "graph.k": 6}) == "a01_b05"
    with pytest.raises(ValueError, match="historical reference settings are absent"):
        c3_reference_configuration(candidates, {"model.dilation": 6})
