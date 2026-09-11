from copy import deepcopy

import pytest
import yaml

from seis_interp.ccnet5d_poc_config import validate_ccnet5d_poc_config
from seis_interp.configuration import REPOSITORY_ROOT, ConfigurationError, load_resolved_config
from seis_interp.nersi_config import validate_nersi_poc_config
from seis_interp.relational_trace_graph_poc_config import validate_relational_trace_graph_poc_config

BASELINE = REPOSITORY_ROOT / "studies/study_036_c3_random80_observed_only_poc"
STUDY = REPOSITORY_ROOT / "studies/study_037_c3_neural_mse_loss_ablation"
VALIDATORS = [
    ("nersi", validate_nersi_poc_config),
    ("ccnet5d", validate_ccnet5d_poc_config),
    ("gnn", validate_relational_trace_graph_poc_config),
]


@pytest.mark.parametrize("method,validate", VALIDATORS)
def test_formal_resolved_config_changes_only_loss(method, validate, tmp_path):
    baseline = load_resolved_config(BASELINE / "formal" / f"{method}.yaml")
    source = STUDY / "formal" / f"{method}.yaml"
    formal = load_resolved_config(source)
    expected = deepcopy(baseline)
    assert baseline["training"]["loss"] == "masked_trace_relative_mse"
    expected["training"]["loss"] = "masked_trace_mse"
    assert formal == expected
    assert validate(baseline) is not None
    assert validate(formal) is not None
    assert "extends" not in yaml.safe_load(source.read_text())
    standalone = tmp_path / source.name
    standalone.write_bytes(source.read_bytes())
    assert load_resolved_config(standalone) == formal
    expected["training"].update(max_steps=2, report_interval=1)
    assert load_resolved_config(STUDY / "smoke" / source.name) == expected


@pytest.mark.parametrize("method,validate", VALIDATORS)
@pytest.mark.parametrize(
    "loss_name", ["masked_trace_mse", "masked_trace_relative_mse", "masked_mse", "mse", "unknown"]
)
def test_validators_preserve_only_two_explicit_poc_loss_names(method, validate, loss_name):
    config = load_resolved_config(BASELINE / "formal" / f"{method}.yaml")
    config["training"]["loss"] = loss_name
    if loss_name in ("masked_trace_mse", "masked_trace_relative_mse"):
        training = validate(config).training
        actual = training["loss"] if isinstance(training, dict) else training.loss
        assert actual == loss_name
    else:
        with pytest.raises(ConfigurationError, match="loss"):
            validate(config)


def test_study_contract_and_inputs_are_fixed_without_classical_methods():
    for folder in ("formal", "smoke"):
        assert {p.name for p in (STUDY / folder).iterdir()} == {
            "nersi.yaml",
            "ccnet5d.yaml",
            "gnn.yaml",
        }
    assert (
        "nominal 80% random mask、crop内のrealized missing fractionは79.8874%"
        in (STUDY / "README.md").read_text()
    )
    config = load_resolved_config(STUDY / "config.yaml")
    assert config["study"] == {"id": STUDY.name, "status": "planned"}
    assert config["ablation"] == {
        "methods": ["nersi", "ccnet5d", "relational_trace_graph"],
        "changed_field": "training.loss",
        "new_value": "masked_trace_mse",
        "all_other_fields_fixed": True,
        "automatic_retry": False,
        "hpo": False,
        "validation_selection": False,
        "multiple_seeds": False,
    }
    baseline = load_resolved_config(BASELINE / "inputs.yaml")
    inputs = load_resolved_config(STUDY / "inputs.yaml")
    assert inputs["paths"] == baseline["paths"]
    assert inputs["benchmark_id"] == baseline["benchmark_id"]
    assert inputs["outputs"] == {
        "runs": f"../../runs/{STUDY.name}",
        "existing_directory_policy": "reject",
    }
