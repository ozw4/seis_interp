from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from seis_interp.configuration import (
    DEFAULT_CONFIG_PATH,
    REPOSITORY_ROOT,
    ConfigurationError,
    get_required_config_value,
    load_resolved_config,
    repository_relative_config_source,
)
from seis_interp.data.trace_schema import MODEL_COORDINATE_ORDER


def write_config(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def make_repository(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "pyproject.toml").write_text("[project]\nname = 'test-repository'\n", encoding="utf-8")
    return path


def test_package_constants_point_to_the_checkout() -> None:
    assert (REPOSITORY_ROOT / "pyproject.toml").is_file()
    assert DEFAULT_CONFIG_PATH == REPOSITORY_ROOT / "configs" / "default.yaml"
    assert DEFAULT_CONFIG_PATH.is_file()


def test_tracked_study_resolves_default_and_study_values() -> None:
    study_config = REPOSITORY_ROOT / "studies" / "study_001_c3_na_baseline" / "config.yaml"

    resolved = load_resolved_config(study_config, repository_root=REPOSITORY_ROOT)

    assert get_required_config_value(resolved, "project.random_seed") == 42
    assert get_required_config_value(resolved, "sampling.random_trace_holdout_fraction") == 0.2
    assert get_required_config_value(resolved, "sampling.validation_fraction_of_holdout") == 0.25
    assert get_required_config_value(resolved, "model.name") == "siren"
    assert get_required_config_value(resolved, "model.input_features") == len(
        MODEL_COORDINATE_ORDER
    )
    assert get_required_config_value(resolved, "model.omega_0") == 10.0
    assert get_required_config_value(resolved, "model.hidden_omega") == 1.0
    assert get_required_config_value(resolved, "training.optimizer") == "adam"
    assert get_required_config_value(resolved, "training.loss") == "l2"
    assert get_required_config_value(resolved, "training.learning_rate") == 1.0e-4
    assert repository_relative_config_source(
        study_config,
        repository_root=REPOSITORY_ROOT,
    ) == ("studies/study_001_c3_na_baseline/config.yaml")


def test_pocs_study_resolves_without_siren_training_settings() -> None:
    study_config = REPOSITORY_ROOT / "studies" / "study_022_c3_na_pocs" / "config.yaml"

    resolved = load_resolved_config(study_config, repository_root=REPOSITORY_ROOT)

    assert "model" not in resolved
    assert "training" not in resolved
    assert "metrics" not in resolved["evaluation"]
    assert get_required_config_value(resolved, "pocs.n_iterations") == 100
    assert resolved["normalization"] == {
        "coordinates": "train_minmax_linear_plus_azimuth_sin_cos",
        "amplitude": "train_global_rms",
    }

    smoke = load_resolved_config(study_config.with_name("config_smoke.yaml"))

    assert "model" not in smoke
    assert "training" not in smoke
    assert smoke["evaluation"] == resolved["evaluation"]
    assert smoke["normalization"] == resolved["normalization"]
    assert get_required_config_value(smoke, "pocs.n_iterations") == 20


def test_drr_study_resolves_only_its_method_and_shared_benchmark_conditions() -> None:
    study_config = REPOSITORY_ROOT / "studies" / "study_023_c3_na_drr" / "config.yaml"
    pocs_config = REPOSITORY_ROOT / "studies" / "study_022_c3_na_pocs" / "config.yaml"
    for filename, iterations, window in (
        ("config.yaml", 10, [5, 8, 4, 8]),
        ("config_smoke.yaml", 3, [4, 4, 4, 8]),
    ):
        resolved = load_resolved_config(study_config.with_name(filename))
        pocs = load_resolved_config(pocs_config.with_name(filename))
        assert not {"model", "training", "pocs"}.intersection(resolved)
        for section in (
            "project",
            "data",
            "normalization",
            "sampling",
            "interpolation_mask",
            "benchmark_case",
            "benchmark_volume",
            "evaluation",
        ):
            assert resolved[section] == pocs[section]
        assert resolved["drr"] == {
            "rank": 4,
            "damping_power": 3,
            "n_iterations": iterations,
            "frequency_min_hz": 5.0,
            "frequency_max_hz": None,
            "spatial_window_shape": window,
            "spatial_overlap": [0, 0, 0, 0],
        }
    drr_inputs = yaml.safe_load(study_config.with_name("inputs.yaml").read_text(encoding="utf-8"))
    pocs_inputs = yaml.safe_load(pocs_config.with_name("inputs.yaml").read_text(encoding="utf-8"))
    drr_references = drr_inputs.pop("references")
    pocs_inputs.pop("references")
    assert drr_inputs == pocs_inputs
    assert drr_references == [
        {
            "id": "chen_et_al_2016_damped_rank_reduction",
            "title": (
                "Simultaneous denoising and reconstruction of 5-D seismic data via "
                "damped rank-reduction method"
            ),
            "year": 2016,
            "doi": "10.1093/gji/ggw230",
        }
    ]


def test_siren_volume_study_uses_shared_benchmark_and_fixed_step_contract() -> None:
    study_directory = REPOSITORY_ROOT / "studies" / "study_024_c3_na_siren_volume"
    config_path = study_directory / "config.yaml"
    smoke_path = study_directory / "config_smoke.yaml"
    pocs_directory = REPOSITORY_ROOT / "studies" / "study_022_c3_na_pocs"
    drr_directory = REPOSITORY_ROOT / "studies" / "study_023_c3_na_drr"

    formal_document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    smoke_document = yaml.safe_load(smoke_path.read_text(encoding="utf-8"))
    formal = load_resolved_config(config_path, repository_root=REPOSITORY_ROOT)
    smoke = load_resolved_config(smoke_path, repository_root=REPOSITORY_ROOT)

    assert "extends" not in formal_document
    assert set(formal) == {
        "study",
        "project",
        "data",
        "normalization",
        "sampling",
        "interpolation_mask",
        "benchmark_case",
        "benchmark_volume",
        "model",
        "training",
        "prediction",
        "evaluation",
    }
    assert not {"pocs", "drr"}.intersection(formal)
    assert formal["model"] == {
        "name": "siren",
        "coordinate_features": "cmp_offset_azimuth",
        "input_features": 6,
        "hidden_width": 256,
        "hidden_layers": 4,
        "output_features": 1,
        "omega_0": 30.0,
        "hidden_omega": 30.0,
        "layer_omega_schedule": None,
        "skip_connections": None,
    }
    assert formal["training"] == {
        "random_seed": 42,
        "optimizer": "adam",
        "loss": "l2",
        "learning_rate": 1.0e-4,
        "batch_size": 65536,
        "max_steps": 20000,
        "report_interval": 500,
        "device": "cuda:0",
    }
    assert formal["prediction"] == {"batch_size": 262144}
    assert formal["evaluation"] == {
        "primary_metric": "physical_amplitude_global_snr_db",
        "domain": "evaluation_target",
    }
    assert set(formal["sampling"]) == {
        "split_scope",
        "source_line_ranges",
        "duplicate_physical_coordinate_policy",
    }

    assert set(smoke_document) == {"extends", "benchmark_volume", "training", "prediction"}
    assert smoke["model"] == formal["model"]
    assert "random_seed" not in smoke_document["training"]
    assert smoke["training"] == {
        "random_seed": 42,
        "optimizer": "adam",
        "loss": "l2",
        "learning_rate": 1.0e-4,
        "batch_size": 4096,
        "max_steps": 100,
        "report_interval": 20,
        "device": "cpu",
    }
    assert smoke["prediction"] == {"batch_size": 32768}

    shared_sections = (
        "project",
        "data",
        "normalization",
        "sampling",
        "interpolation_mask",
        "benchmark_case",
        "benchmark_volume",
        "evaluation",
    )
    for filename, siren in (("config.yaml", formal), ("config_smoke.yaml", smoke)):
        pocs = load_resolved_config(pocs_directory / filename)
        drr = load_resolved_config(drr_directory / filename)
        for section in shared_sections:
            assert siren[section] == pocs[section] == drr[section]

    siren_inputs_path = study_directory / "inputs.yaml"
    siren_inputs = yaml.safe_load(siren_inputs_path.read_text(encoding="utf-8"))
    pocs_inputs = yaml.safe_load((pocs_directory / "inputs.yaml").read_text(encoding="utf-8"))
    drr_inputs = yaml.safe_load((drr_directory / "inputs.yaml").read_text(encoding="utf-8"))
    assert set(siren_inputs) == {"datasets", "references"}
    assert siren_inputs["datasets"] == pocs_inputs["datasets"] == drr_inputs["datasets"]
    assert siren_inputs["references"] == [
        {
            "id": "sitzmann_et_al_2020_siren",
            "title": "Implicit Neural Representations with Periodic Activation Functions",
            "year": 2020,
            "arxiv": "2006.09661",
        },
        {
            "id": "liu_et_al_2024_isr",
            "title": "5-D Seismic Data Interpolation by Continuous Representation",
            "year": 2024,
            "doi": "10.1109/TGRS.2024.3431439",
        },
    ]
    for path in (config_path, smoke_path, siren_inputs_path):
        assert "schema_version" not in path.read_text(encoding="utf-8")


def test_ccnet5d_training_configs_keep_supervision_and_fixed_budgets_separate() -> None:
    directory = REPOSITORY_ROOT / "studies" / "study_025_c3_na_ccnet5d"
    path = directory / "config_train.yaml"
    assert "extends" not in yaml.safe_load(path.read_text(encoding="utf-8"))
    formal = load_resolved_config(path)
    smoke = load_resolved_config(directory / "config_train_smoke.yaml")
    calibration = load_resolved_config(directory / "config_train_calibration.yaml")
    assert set(formal) == {
        "study",
        "project",
        "data",
        "model",
        "supervision",
        "patches",
        "training",
        "selection",
    }
    assert formal["model"] == {
        "name": "ccnet5d",
        "hidden_channels": 64,
        "intermediate_channels": 64,
        "kernel_size": 5,
        "output_activation": "linear",
    }
    assert formal["patches"] == {
        "shape": [16, 16, 16, 8, 16],
        "fit_count": 10000,
        "selection_count": 1000,
        "missing_fraction": 0.8,
        "mask_kind": "random_trace",
        "random_seed": 42,
    }
    assert formal["training"] == {
        "optimizer": "adam",
        "loss": "mse_complete_patch",
        "random_seed": 42,
        "batch_size": 1,
        "max_epochs": 20,
        "learning_rate": 1e-4,
        "decay_after_epochs": 15,
        "decay_factor": 0.1,
        "validate_every_steps": 2000,
        "report_every_steps": 200,
        "device": "cuda:0",
    }
    assert formal["selection"] == {
        "metric": "missing_global_snr_db",
        "domain": "held_out_train_partition_patch_instances",
    }
    assert smoke["model"] == formal["model"] | {"hidden_channels": 4, "intermediate_channels": 4}
    assert smoke["patches"] == formal["patches"] | {
        "shape": [8, 2, 4, 2, 4],
        "fit_count": 4,
        "selection_count": 2,
    }
    assert smoke["training"] == formal["training"] | {
        "max_epochs": 1,
        "validate_every_steps": 2,
        "report_every_steps": 1,
        "device": "cpu",
    }
    assert calibration["model"] == formal["model"]
    assert calibration["patches"] == formal["patches"] | {"fit_count": 1, "selection_count": 1}
    assert calibration["training"] == formal["training"] | {
        "max_epochs": 1,
        "validate_every_steps": 1,
        "report_every_steps": 1,
    }
    for region, formal_shots, smoke_shots in (
        ("fit_region", [27, 59], [27, 35]),
        ("selection_region", [59, 75], [35, 43]),
    ):
        assert formal["supervision"][region] == {
            "time": [0, 384],
            "source_line": [0, 16],
            "shot_in_line": formal_shots,
            "relative_receiver_x": [0, 8],
            "relative_receiver_y": [18, 34],
        }
        assert smoke["supervision"][region] == {
            "time": [128, 144],
            "source_line": [0, 2],
            "shot_in_line": smoke_shots,
            "relative_receiver_x": [0, 2],
            "relative_receiver_y": [18, 22],
        }
        assert calibration["supervision"][region] == formal["supervision"][region] | {
            "time": [128, 192]
        }
    axes = ("time", "source_line", "shot_in_line", "relative_receiver_x", "relative_receiver_y")
    for config in (formal, smoke, calibration):
        assert config["study"]["status"] == "draft"
        assert config["data"] == {"dataset_id": "seg_c3_na"}
        assert config["project"]["random_seed"] == 42
        assert config["training"]["random_seed"] == config["patches"]["random_seed"] == 42
        assert config["selection"] == formal["selection"]
        supervision = config["supervision"]
        assert supervision["partition"] == "train"
        assert supervision["amplitude_normalization"] == "fit_region_global_rms"
        fit, selection = supervision["fit_region"], supervision["selection_region"]
        assert any(fit[axis][1] <= selection[axis][0] for axis in axes[1:])
        for region in (fit, selection):
            assert 0 <= region["source_line"][0] < region["source_line"][1] <= 25
            assert all(
                0 < width <= region[axis][1] - region[axis][0]
                for axis, width in zip(axes, config["patches"]["shape"], strict=True)
            )


def test_ccnet5d_inference_configs_reuse_benchmark_without_training_or_model_settings() -> None:
    directory = REPOSITORY_ROOT / "studies" / "study_025_c3_na_ccnet5d"
    assert "extends" not in yaml.safe_load((directory / "config.yaml").read_text(encoding="utf-8"))
    for filename, device, core_shape in (
        ("config.yaml", "cuda:0", [32, 8, 16, 8, 16]),
        ("config_smoke.yaml", "cpu", [16, 4, 8, 8, 8]),
    ):
        resolved = load_resolved_config(directory / filename)
        assert set(resolved) == {
            "study",
            "project",
            "data",
            "interpolation_mask",
            "benchmark_case",
            "benchmark_volume",
            "prediction",
            "evaluation",
        }
        assert not {"model", "training", "normalization", "pocs", "drr"}.intersection(resolved)
        assert resolved["data"] == {"dataset_id": "seg_c3_na"}
        assert resolved["prediction"] == {"device": device, "core_shape": core_shape}
        for baseline in (
            "study_022_c3_na_pocs",
            "study_023_c3_na_drr",
            "study_024_c3_na_siren_volume",
        ):
            expected = load_resolved_config(REPOSITORY_ROOT / "studies" / baseline / filename)
            for section in (
                "project",
                "interpolation_mask",
                "benchmark_case",
                "benchmark_volume",
                "evaluation",
            ):
                assert resolved[section] == expected[section]
        for training_filename in (
            "config_train.yaml",
            "config_train_smoke.yaml",
            "config_train_calibration.yaml",
        ):
            teacher = load_resolved_config(directory / training_filename)["supervision"]
            for region in ("fit_region", "selection_region"):
                assert (
                    teacher[region]["source_line"][1]
                    <= (resolved["benchmark_volume"]["selection"]["source_line"][0])
                )


def test_ccnet5d_inputs_reuse_all_shared_data_and_record_the_paper_reference() -> None:
    directory = REPOSITORY_ROOT / "studies" / "study_025_c3_na_ccnet5d"
    inputs = yaml.safe_load((directory / "inputs.yaml").read_text(encoding="utf-8"))

    assert set(inputs) == {"datasets", "references"}
    for baseline in ("study_022_c3_na_pocs", "study_023_c3_na_drr", "study_024_c3_na_siren_volume"):
        baseline_path = REPOSITORY_ROOT / "studies" / baseline / "inputs.yaml"
        assert (
            inputs["datasets"]
            == yaml.safe_load(baseline_path.read_text(encoding="utf-8"))["datasets"]
        )
    assert inputs["references"] == [
        {
            "id": "fang_et_al_2023_ccnet5d",
            "title": "CCNet-5D: 5D convolutional neural network for seismic data interpolation",
            "year": 2023,
            "doi": "10.1190/GEO2022-0420.1",
        }
    ]


def test_recursively_merges_mappings_and_replaces_other_values(tmp_path: Path) -> None:
    base = write_config(
        tmp_path / "base.yaml",
        """
project:
  random_seed: 10
model:
  hidden_width: 128
  nested:
    retained: true
    replaced: base
items: [base, values]
nullable: present
""",
    )
    child = write_config(
        tmp_path / "studies" / "child.yaml",
        """
extends: ../base.yaml
project:
  random_seed: 99
model:
  nested:
    replaced: child
items: [child]
nullable: null
""",
    )

    resolved = load_resolved_config(child)

    assert resolved == {
        "project": {"random_seed": 99},
        "model": {
            "hidden_width": 128,
            "nested": {"retained": True, "replaced": "child"},
        },
        "items": ["child"],
        "nullable": None,
    }
    assert "extends" not in resolved
    assert load_resolved_config(base)["project"] == {"random_seed": 10}


def test_relative_extends_resolution_does_not_depend_on_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = write_config(
        tmp_path / "repo" / "configs" / "default.yaml",
        "project:\n  random_seed: 42\n",
    )
    child = write_config(
        tmp_path / "repo" / "studies" / "study" / "config.yaml",
        "extends: ../../configs/default.yaml\nstudy:\n  status: draft\n",
    )
    unrelated_directory = tmp_path / "elsewhere"
    unrelated_directory.mkdir()
    monkeypatch.chdir(unrelated_directory)

    resolved = load_resolved_config(child)

    assert resolved == {
        "project": {"random_seed": 42},
        "study": {"status": "draft"},
    }
    assert base.is_file()


def test_recursive_inheritance_uses_the_nearest_child_override(tmp_path: Path) -> None:
    write_config(tmp_path / "default.yaml", "project:\n  random_seed: 1\n")
    write_config(
        tmp_path / "middle.yaml",
        "extends: default.yaml\nproject:\n  random_seed: 2\n",
    )
    leaf = write_config(
        tmp_path / "leaf.yaml",
        "extends: middle.yaml\nproject:\n  random_seed: 3\n",
    )

    assert get_required_config_value(load_resolved_config(leaf), "project.random_seed") == 3


@pytest.mark.parametrize(
    "contents,match",
    [
        ("- not\n- a mapping\n", "mapping at its root"),
        ("null\n", "mapping at its root"),
        ("1: value\n", "non-string mapping key"),
        ("outer:\n  2: value\n", "non-string mapping key"),
        ("extends: ''\n", "non-empty string"),
        ("extends: 123\n", "non-empty string"),
        ("extends: /absolute/base.yaml\n", "POSIX relative path"),
        ("extends: 'C:\\\\base.yaml'\n", "POSIX relative path"),
        ("extends: 'C:base.yaml'\n", "POSIX relative path"),
        ("extends: '\\base.yaml'\n", "POSIX relative path"),
        ("extends: '..\\outside.yaml'\n", "POSIX relative path"),
        ("broken: [yaml\n", "invalid YAML"),
    ],
)
def test_rejects_invalid_configuration_documents(
    tmp_path: Path,
    contents: str,
    match: str,
) -> None:
    path = write_config(tmp_path / "config.yaml", contents)

    with pytest.raises(ConfigurationError, match=match):
        load_resolved_config(path)


def test_missing_leaf_or_extended_file_remains_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_resolved_config(tmp_path / "missing.yaml")

    child = write_config(tmp_path / "child.yaml", "extends: missing-base.yaml\n")
    with pytest.raises(FileNotFoundError):
        load_resolved_config(child)


def test_detects_extends_cycles_after_canonical_path_resolution(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    first = write_config(tmp_path / "first.yaml", "extends: nested/../second.yaml\n")
    write_config(tmp_path / "second.yaml", "extends: ./first.yaml\n")

    with pytest.raises(ConfigurationError, match="cycle"):
        load_resolved_config(first)


def test_rejects_extends_that_escape_the_selected_repository(tmp_path: Path) -> None:
    repository = make_repository(tmp_path / "repository")
    outside = write_config(tmp_path / "outside.yaml", "project: {}\n")
    child = write_config(
        repository / "studies" / "study" / "config.yaml",
        "extends: ../../../outside.yaml\n",
    )

    with pytest.raises(ConfigurationError, match="extends chain.*repository root"):
        load_resolved_config(child, repository_root=repository)

    linked_base = repository / "configs" / "linked.yaml"
    linked_base.parent.mkdir()
    linked_base.symlink_to(outside)
    child.write_text("extends: ../../configs/linked.yaml\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="extends chain.*repository root"):
        load_resolved_config(child, repository_root=repository)


def test_get_required_config_value_reads_nested_values_and_explicit_null() -> None:
    config = {"project": {"random_seed": 0}, "optional": {"value": None}}

    assert get_required_config_value(config, "project.random_seed") == 0
    assert get_required_config_value(config, "optional.value") is None


def test_get_required_config_value_reports_missing_or_blocked_paths() -> None:
    with pytest.raises(ConfigurationError, match="missing required.*project.random_seed"):
        get_required_config_value({"project": {}}, "project.random_seed")
    with pytest.raises(ConfigurationError, match="not a mapping"):
        get_required_config_value({"project": 42}, "project.random_seed")


@pytest.mark.parametrize("dotted_path", ["", ".project", "project.", "project..seed"])
def test_get_required_config_value_rejects_invalid_dotted_paths(dotted_path: str) -> None:
    with pytest.raises(ConfigurationError, match="dotted path"):
        get_required_config_value({}, dotted_path)


def test_repository_relative_source_uses_nearest_marker_and_posix_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = make_repository(tmp_path / "repository")
    config = write_config(repository / "studies" / "example" / "config.yaml", "study: {}\n")
    unrelated_directory = tmp_path / "cwd"
    unrelated_directory.mkdir()
    monkeypatch.chdir(unrelated_directory)

    source = repository_relative_config_source(config)

    assert source == "studies/example/config.yaml"
    assert not Path(source).is_absolute()


def test_repository_relative_source_rejects_files_outside_a_marked_repository(
    tmp_path: Path,
) -> None:
    make_repository(tmp_path / "repository")
    outside_config = write_config(tmp_path / "outside" / "config.yaml", "study: {}\n")

    with pytest.raises(ConfigurationError, match="outside a repository"):
        repository_relative_config_source(outside_config)


def test_repository_relative_source_is_anchored_to_the_selected_repository(
    tmp_path: Path,
) -> None:
    selected_repository = make_repository(tmp_path / "selected")
    other_repository = make_repository(tmp_path / "other")
    other_config = write_config(other_repository / "studies" / "study" / "config.yaml", "{}\n")

    with pytest.raises(ConfigurationError, match="outside its repository root"):
        repository_relative_config_source(
            other_config,
            repository_root=selected_repository,
        )


def test_repository_relative_source_resolves_symlinks_before_containment_check(
    tmp_path: Path,
) -> None:
    repository = make_repository(tmp_path / "repository")
    outside_config = write_config(tmp_path / "outside" / "config.yaml", "study: {}\n")
    linked_config = repository / "config.yaml"
    linked_config.symlink_to(outside_config)

    with pytest.raises(ConfigurationError, match="outside a repository"):
        repository_relative_config_source(linked_config)


def test_repository_relative_source_requires_an_existing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        repository_relative_config_source(tmp_path / "missing.yaml")

    repository = make_repository(tmp_path / "repository")
    config_directory = repository / "configs"
    config_directory.mkdir()
    with pytest.raises(ConfigurationError, match="must be a file"):
        repository_relative_config_source(config_directory)
