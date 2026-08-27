from __future__ import annotations

from pathlib import Path

import pytest

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
