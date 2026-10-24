import hashlib
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from seis_interp.data.experiment_imports import method_config_hash
from seis_interp.forge_mvp_config import canvas_time_stages
from seis_interp.processing.c3_region_search import c3_search_candidates, resolve_c3_candidate
from seis_interp.processing.experiment_commonality import check_fixed_settings
from seis_interp.training.forge_mvp_methods import canvas_method_config, initialize_model
from seis_interp.training.rehearsal import rehearsal_candidates
from seis_interp.training.worker_shards import shard_candidates

STUDY = Path(__file__).resolve().parents[2] / "studies/study_067_forge_c3_common_grid"


def config(dataset, method):
    return yaml.safe_load((STUDY / dataset / f"{method}.yaml").read_text())


@pytest.mark.parametrize(
    "dataset,length,parameters,stages",
    [
        ("c3", 384, 363269, None),
        ("forge", 2172, 341485, [28, 28, 28]),
    ],
)
def test_canvas_rule_keeps_c3_and_normalizes_forge(dataset, length, parameters, stages):
    base = config(dataset, "nersi_profile")
    assert base["canvas_time_upsampling"] is True
    assert "loss_support_mask" not in base["training"]
    assert base["training"]["coordinate_jitter_cells"] == 0.5
    enabled = deepcopy(base)
    base.pop("canvas_time_upsampling")
    assert canvas_time_stages(length) == (0 if dataset == "c3" else 3)
    resolved = canvas_method_config(enabled, length)
    model = initialize_model(resolved)
    assert sum(p.numel() for p in model.parameters()) == parameters
    assert resolved["model"].get("time_upsample_channels") == stages
    if dataset == "c3":
        assert resolved == canvas_method_config(base, length)
        enabled["training"]["loss_support_mask"] = True
        assert method_config_hash(enabled, time_sample_count=length) == method_config_hash(base)
    else:
        assert model.base_profile_shape == (34, 4)
        assert resolved["model"]["time_kernel_size"] == 3
        assert method_config_hash(enabled) != method_config_hash(base)
        enabled["alignment"] = {"velocity_m_per_s": 500.0, "max_delay_samples": 1171}
        before = method_config_hash(enabled)
        enabled["training"]["loss_support_mask"] = True
        assert method_config_hash(enabled) != before


@pytest.mark.parametrize(
    "dataset,observations,expected",
    [
        (
            "c3",
            23593,
            [
                "161a926cce8b5bf7da5cf03f85ca970960af9fb6246cc3872b55b06d281ea04f",
                "74e5ba6b4234617807b49810cc49dc78c8094c9740b1972c34fb2970ecbde3b4",
            ],
        ),
        (
            "forge",
            13271,
            [
                "0c5d6242ee522f622901bf067982331c63dffdeca14cfa253169eca4db450efc",
                "35f8d5ee6be43fb830462f691957b5402302e56a81e62fbd7b79e89ec2c5143a",
            ],
        ),
    ],
)
def test_regsi_p2_and_r1_identity_remains_the_pre_sprint_identity(dataset, observations, expected):
    grid = yaml.safe_load((STUDY / "grid.yaml").read_text())
    candidates = c3_search_candidates(config(dataset, "regsi_real"), grid["methods"]["regsi_real"])
    # Captured before editing Sprint 1; these are P2 V3 and R-1's cells.
    for candidate, digest in zip((candidates[0], candidates[4]), expected, strict=True):
        resolved = resolve_c3_candidate(candidate["config"], observations, 64)
        assert method_config_hash(resolved) == digest
        resolved["training"].pop("ema_eval_decays")
        assert method_config_hash(resolved) == digest


@pytest.mark.parametrize("dataset", ["c3", "forge"])
def test_approved_defaults_seeds_and_k1_option(dataset):
    contract = config(dataset, "config")
    assert contract["trace_split"]["evaluation_seeds"] == [7321, 7322, 7323]
    assert contract["model_seed"] == 7301
    ccnet = config(dataset, "ccnet5d_grid")
    assert ccnet["patches"]["inner_mask_fraction"] == 0.25
    assert not ccnet["execution"].get("reuse_patch", False)
    assert not ccnet["execution"].get("cache_patch_gradients", False)
    # C-0 adopted benchmark on; execution-only identity must still accept off.
    assert ccnet["execution"]["cudnn_benchmark"] is True
    benchmark = deepcopy(ccnet)
    benchmark["execution"]["cudnn_benchmark"] = False
    assert method_config_hash(benchmark) == method_config_hash(ccnet)
    drr = config(dataset, "drr_grid")
    assert drr["algorithm"]["spatial_overlap"] == [0, 3, 0, 3]
    assert drr["algorithm"]["overlap_policy"] == "explicit"
    pocs, k1 = config(dataset, "pocs_grid"), config(dataset, "pocs_k1")
    # P-1 adopted K1 as the frozen POCS base in both datasets.
    assert pocs["algorithm"]["n_iterations"] == 32
    assert pocs["algorithm"]["overlap"][3:] == [3, 6]
    assert k1["algorithm"]["n_iterations"] == 32
    assert k1["algorithm"]["overlap"][3:] == [3, 6]


def test_n1_templates_define_exactly_the_approved_four_arms():
    arms = []
    for path in sorted((STUDY / "forge").glob("nersi_n1_*.yaml")):
        item = yaml.safe_load(path.read_text())
        assert item["canvas_time_upsampling"] is True
        assert item["training"]["ema_decay"] == 0.999
        assert item["training"]["ema_eval_decays"] == [0.999, 0.9995, 0.9998]
        assert item["training"]["max_steps"] == 50000
        arms.append(
            (item["training"].get("coordinate_jitter_cells"), item["training"]["loss_support_mask"])
        )
        if "no_jitter" in path.name:
            assert "coordinate_jitter_cells" not in item["training"]
    assert set(arms) == {(0.5, False), (0.5, True), (None, False), (None, True)}
    base = config("c3", "nersi_profile")
    changed = deepcopy(base)
    changed["training"]["ema_decay"] = 0.9998
    assert method_config_hash(changed) != method_config_hash(base)


def test_no_jitter_c3_identity_matches_study064_and_commonality_passes():
    grid = yaml.safe_load((STUDY / "grid.yaml").read_text())["methods"]["nersi_profile"]
    c3, forge = config("c3", "nersi_profile"), config("forge", "nersi_profile")
    for item in (c3, forge):
        item["training"].pop("coordinate_jitter_cells")
    allowed = yaml.safe_load((STUDY / "dataset_differences.yaml").read_text())
    check_fixed_settings(c3, forge, allowed["methods"]["nersi_profile"], "nersi_profile")
    candidate = next(item for item in c3_search_candidates(c3, grid) if item["id"] == "a02_b04")
    resolved = resolve_c3_candidate(candidate["config"], 23593, 64)
    # Captured from study_064's saved S1/a02_b04/config.resolved.yaml.
    expected = "b39ddd913d0443c47251515bb6d31ec524945d0bd05c1899be179fec5d0796eb"
    assert method_config_hash(resolved, time_sample_count=384) == expected
    resolved["training"]["coordinate_jitter_cells"] = None
    resolved["canvas_time_upsampling"] = True
    resolved["training"]["loss_support_mask"] = True
    assert method_config_hash(resolved, time_sample_count=384) == expected


@pytest.mark.parametrize("yaml_length", [256, 384, 1008, 2176])
def test_nersi_upsampling_identity_uses_runtime_canvas(yaml_length):
    base = config("forge", "nersi_profile")
    base.pop("canvas_time_upsampling")
    base["model"]["profile_shape"][0] = yaml_length
    base["alignment"] = {"velocity_m_per_s": 500.0, "max_delay_samples": 1171}
    enabled = deepcopy(base)
    enabled["canvas_time_upsampling"] = True
    assert canvas_method_config(enabled, 1001 + 1171)["model"]["time_upsample_channels"] == [
        28,
        28,
        28,
    ]
    assert method_config_hash(enabled, time_sample_count=1001) != method_config_hash(base)
    assert method_config_hash(enabled) != method_config_hash(base)
    # Conversely, a large YAML length cannot make the C3 switch effective.
    enabled.pop("alignment")
    base.pop("alignment")
    assert method_config_hash(enabled, time_sample_count=384) == method_config_hash(base)


def test_upsampling_inertness_requires_a_known_alignment_maximum():
    base = config("c3", "nersi_profile")
    base.pop("canvas_time_upsampling")
    base["alignment"] = {"velocity_m_per_s": 500.0}
    enabled = deepcopy(base)
    enabled["canvas_time_upsampling"] = True
    assert method_config_hash(enabled, time_sample_count=384) != method_config_hash(base)


@pytest.mark.parametrize("support", [False, True])
def test_no_jitter_n1_import_identity_accepts_saved_null(support):
    arm = config("forge", f"nersi_n1_no_jitter_{'support' if support else 'canvas'}")
    saved = deepcopy(arm)
    saved["training"]["coordinate_jitter_cells"] = None
    assert method_config_hash(arm) == method_config_hash(saved)
    saved["training"]["coordinate_jitter_cells"] = 0.5
    assert method_config_hash(arm) != method_config_hash(saved)


def test_adopted_forge_nersi_rejects_p3_identity_despite_derived_profile_time():
    grid = yaml.safe_load((STUDY / "grid.yaml").read_text())["methods"]["nersi_profile"]
    base = config("forge", "nersi_profile")
    candidate = next(item for item in c3_search_candidates(base, grid) if item["id"] == "a02_b04")
    resolved = candidate["config"] | {
        "alignment": {"velocity_m_per_s": 500.0, "max_delay_samples": 1171}
    }
    # Captured from the completed p3_forge_nersi_jitter05 resolved config.
    expected = "a84c85133151b3de987610a6feb90e2c42d729574bec036b90d3aa7c31765dc8"
    legacy = deepcopy(resolved)
    legacy.pop("canvas_time_upsampling")
    assert method_config_hash(legacy) == expected
    adopted = method_config_hash(resolved, time_sample_count=1001)
    assert adopted != expected
    assert canvas_method_config(resolved, 2172)["model"]["profile_shape"] == [2176, 32]
    resolved["model"]["profile_shape"][0] = 1008
    assert method_config_hash(resolved, time_sample_count=1001) == adopted
    assert canvas_method_config(resolved, 2172)["model"]["profile_shape"] == [2176, 32]
    resolved["model"]["profile_shape"][1] = 40
    assert method_config_hash(resolved, time_sample_count=1001) != adopted


@pytest.mark.parametrize("dataset", ["c3", "forge"])
def test_round2_import_sources_include_conditional_pilots(dataset):
    sources = config(dataset, "config")["import_from"]
    prefix = "runs/study_067_forge_c3_common_grid/20261022T162308560491Z_05c92e79ce80_"
    assert prefix + f"p1_{dataset}_pocs_k1_iterations32_rx4x8_ov3x6" in sources
    if dataset == "c3":
        assert prefix + "r1_c3_regsi_v3_a01_b05" in sources
    else:
        assert (
            "runs/study_067_forge_c3_common_grid/20261022T183518698062Z_05c92e79ce80_forge"
            in sources
        )


@pytest.mark.parametrize("dataset,parallel", [("c3", 2), ("forge", 4)])
def test_adopted_drr_overlap_keeps_rank_window_axes_and_schedules_heaviest_first(dataset, parallel):
    grid = yaml.safe_load((STUDY / "grid.yaml").read_text())["methods"]["drr_grid"]
    assert [item["id"] for item in grid["axis_a"]] == ["6", "12", "18", "24", "30"]
    windows = [[4, 6, 4, 6], [4, 8, 4, 8], [6, 8, 6, 8], [8, 12, 8, 12], [8, 16, 8, 16]]
    assert [item["id"] for item in grid["axis_b"]] == [str(window) for window in windows]
    candidates = c3_search_candidates(config(dataset, "drr_grid"), grid)
    assert [item["id"] for item in candidates] == [
        f"a{a:02d}_b{b:02d}" for a in range(1, 6) for b in range(1, 6)
    ]
    for item in candidates:
        resolved = resolve_c3_candidate(item["config"], 100, 64)
        algorithm = resolved["algorithm"]
        window = algorithm["spatial_window_shape"]
        assert algorithm["spatial_overlap"] == [0, window[1] // 2, 0, window[3] // 2]
        assert algorithm["backend"] == "drr_batched"
        assert algorithm["boundary_rule"] == "clip_and_shift"
        assert resolved["execution"]["workers"] == 12
    schedule = config(dataset, "config")["method_execution"]["drr_grid"]
    assert schedule["parallel_candidates"] == parallel
    assert schedule["candidate_priority"] == [
        f"a{a:02d}_b{b:02d}" for b in range(5, 0, -1) for a in range(5, 0, -1)
    ]


@pytest.mark.parametrize("dataset", ["c3", "forge"])
def test_launch_neural_priorities_preserve_candidates_and_feed_both_shards(dataset):
    contract = config(dataset, "config")
    grid = yaml.safe_load((STUDY / "grid.yaml").read_text())["methods"]
    for method in ("ccnet5d_grid", "regsi_real", "nersi_profile"):
        candidates = c3_search_candidates(config(dataset, method), grid[method])
        ordered = rehearsal_candidates(candidates, contract, method, None)
        assert {item["id"]: item["config"] for item in ordered} == {
            item["id"]: item["config"] for item in candidates
        }
        ids = [item["id"] for item in ordered]
        if method == "nersi_profile" and dataset == "c3":
            assert ordered == candidates
            continue
        assert len(contract["method_execution"][method]["candidate_priority"]) == 25
        if method == "regsi_real":
            assert ids[:5] == [f"a{a:02d}_b05" for a in range(1, 6)]
            assert [item[-3:] for item in ids] == sorted([item[-3:] for item in ids], reverse=True)
        elif method == "ccnet5d_grid":
            heavy = ids[1:] if dataset == "forge" else ids
            assert [item[:3] for item in heavy] == sorted(
                [item[:3] for item in heavy], reverse=True
            )
            if dataset == "forge":
                assert ids[0] == "a01_b02"
                assert contract["method_execution"][method]["worker_shards"] == 2
                shards = [shard_candidates(ordered, f"{i}/2") for i in range(2)]
                assert [item["id"] for item in shards[0]][:3] == ["a01_b02", "a05_b02", "a05_b04"]
                assert [item["id"] for item in shards[1]][:3] == ["a05_b01", "a05_b03", "a05_b05"]
        else:
            assert ids[:10] == [f"a{a:02d}_b{b:02d}" for a in (3, 4) for b in range(1, 6)]


def test_round2_rules_are_registered_verbatim_and_config_hashes_match():
    design = (STUDY.parents[1] / "reports/study_067_design_round2.md").read_text()
    decisions = (STUDY / "decisions.md").read_text()
    for start, end in [("## (2)", "## (3)"), ("## (3)", "## (4)"), ("## (5)", "## (6)")]:
        assert design[design.index(start) : design.index(end)] in decisions
    for line in design.splitlines():
        if line.strip().startswith(("- (i)", "- (ii)", "- (iii)")):
            assert line.strip() in decisions
    for path in STUDY.rglob("*.yaml"):
        relative = path.relative_to(STUDY).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert f"| [{relative}]({relative}) | `{digest}` |" in decisions
