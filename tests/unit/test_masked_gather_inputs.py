from __future__ import annotations

from dataclasses import fields, replace

import pytest
import torch

from seis_interp.data.masked_gather_inputs import (
    MaskedGatherInputs,
    validate_masked_gather_inputs,
)

BATCH_SIZE = 2
CONTEXT_COUNT = 2
TIME_COUNT = 3


def _valid_inputs(*, dtype: torch.dtype = torch.float32) -> MaskedGatherInputs:
    target_mask = torch.ones(BATCH_SIZE, 8, 68, dtype=torch.bool)
    target_mask[0, 0, 0] = False
    target_mask[1] = False
    target = target_mask[..., None].expand(-1, -1, -1, TIME_COUNT).to(dtype=dtype)

    context_mask = torch.ones(BATCH_SIZE, CONTEXT_COUNT, 8, 68, dtype=torch.bool)
    context_mask[0, 0, 0, 0] = False
    contexts = context_mask[..., None].expand(-1, -1, -1, -1, TIME_COUNT).to(dtype=dtype)
    deltas = torch.tensor(
        [[[1.0, 0.0], [0.0, 2.0]], [[-1.0, 1.0], [3.0, -2.0]]],
        dtype=dtype,
    )
    coordinates = torch.tensor([[0.0, 1.0], [0.25, 0.75]], dtype=dtype)
    return MaskedGatherInputs(
        target_observed=target,
        target_observation_mask=target_mask,
        context_gathers=contexts,
        context_availability=context_mask,
        source_deltas_m=deltas,
        target_coordinates=coordinates,
    )


def test_accepts_partial_and_whole_shot_missing_targets_and_returns_same_object() -> None:
    inputs = _valid_inputs()

    validated = validate_masked_gather_inputs(inputs)

    assert validated is inputs
    assert inputs.target_observation_mask[0].any()
    assert not inputs.target_observation_mask[1].any()


@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32, torch.float64],
)
def test_accepts_matching_floating_dtypes(dtype: torch.dtype) -> None:
    inputs = _valid_inputs(dtype=dtype)

    assert validate_masked_gather_inputs(inputs) is inputs


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("target_observed", torch.zeros(2, 8, 68), "target_observed must have shape"),
        (
            "target_observed",
            torch.zeros(0, 8, 68, TIME_COUNT),
            "batch dimension must be positive",
        ),
        (
            "target_observed",
            torch.zeros(BATCH_SIZE, 7, 68, TIME_COUNT),
            "fixed 8 x 68",
        ),
        (
            "target_observed",
            torch.zeros(BATCH_SIZE, 8, 68, 1),
            "at least two samples",
        ),
        (
            "target_observation_mask",
            torch.ones(BATCH_SIZE, 8, 67, dtype=torch.bool),
            "target_observation_mask",
        ),
        (
            "context_gathers",
            torch.zeros(BATCH_SIZE, CONTEXT_COUNT, 8, 68),
            "context_gathers must have shape",
        ),
        (
            "context_gathers",
            torch.zeros(BATCH_SIZE + 1, CONTEXT_COUNT, 8, 68, TIME_COUNT),
            "batch dimension",
        ),
        (
            "context_gathers",
            torch.zeros(BATCH_SIZE, 0, 8, 68, TIME_COUNT),
            "context dimension must be positive",
        ),
        (
            "context_gathers",
            torch.zeros(BATCH_SIZE, CONTEXT_COUNT, 7, 68, TIME_COUNT),
            "fixed 8 x 68",
        ),
        (
            "context_gathers",
            torch.zeros(BATCH_SIZE, CONTEXT_COUNT, 8, 68, TIME_COUNT + 1),
            "time dimension",
        ),
        (
            "context_availability",
            torch.ones(BATCH_SIZE, CONTEXT_COUNT, 8, 67, dtype=torch.bool),
            "context_availability",
        ),
        (
            "source_deltas_m",
            torch.ones(BATCH_SIZE, CONTEXT_COUNT, 3),
            "source_deltas_m must have shape",
        ),
        (
            "target_coordinates",
            torch.zeros(BATCH_SIZE, 3),
            "target_coordinates must have shape",
        ),
    ],
)
def test_rejects_shape_contract_violations(
    field: str,
    value: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_masked_gather_inputs(replace(_valid_inputs(), **{field: value}))


@pytest.mark.parametrize(
    "field",
    [
        "target_observed",
        "target_observation_mask",
        "context_gathers",
        "context_availability",
        "source_deltas_m",
        "target_coordinates",
    ],
)
def test_rejects_non_tensor_fields(field: str) -> None:
    with pytest.raises(TypeError, match=rf"{field} must be a torch.Tensor"):
        validate_masked_gather_inputs(replace(_valid_inputs(), **{field: object()}))


@pytest.mark.parametrize(
    "field",
    ["target_observed", "context_gathers", "source_deltas_m", "target_coordinates"],
)
def test_rejects_nonfloating_fields(field: str) -> None:
    inputs = _valid_inputs()
    integer_value = getattr(inputs, field).to(torch.int64)

    with pytest.raises(TypeError, match=rf"{field} must have a floating-point dtype"):
        validate_masked_gather_inputs(replace(inputs, **{field: integer_value}))


@pytest.mark.parametrize("field", ["target_observation_mask", "context_availability"])
def test_rejects_nonboolean_masks(field: str) -> None:
    inputs = _valid_inputs()

    with pytest.raises(TypeError, match="observation masks must have dtype torch.bool"):
        validate_masked_gather_inputs(
            replace(inputs, **{field: getattr(inputs, field).to(torch.float32)})
        )


def test_rejects_mismatched_floating_dtypes() -> None:
    inputs = _valid_inputs()

    with pytest.raises(TypeError, match="floating input fields must have the same dtype"):
        validate_masked_gather_inputs(
            replace(inputs, source_deltas_m=inputs.source_deltas_m.to(torch.float64))
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_rejects_mismatched_devices() -> None:
    inputs = _valid_inputs()

    with pytest.raises(ValueError, match="all input fields must be on the same device"):
        validate_masked_gather_inputs(
            replace(inputs, target_coordinates=inputs.target_coordinates.cuda())
        )


@pytest.mark.parametrize(
    ("field", "nonfinite"),
    [
        ("target_observed", float("nan")),
        ("context_gathers", float("inf")),
        ("source_deltas_m", float("-inf")),
        ("target_coordinates", float("nan")),
    ],
)
def test_rejects_nonfinite_floating_values(field: str, nonfinite: float) -> None:
    inputs = _valid_inputs()
    value = getattr(inputs, field).clone()
    value.reshape(-1)[0] = nonfinite

    with pytest.raises(ValueError, match=rf"{field} must contain only finite values"):
        validate_masked_gather_inputs(replace(inputs, **{field: value}))


@pytest.mark.parametrize("coordinate", [-0.01, 1.01])
def test_rejects_target_coordinates_outside_inclusive_unit_range(coordinate: float) -> None:
    inputs = _valid_inputs()
    coordinates = inputs.target_coordinates.clone()
    coordinates[0, 0] = coordinate

    with pytest.raises(ValueError, match=r"inclusive range \[0, 1\]"):
        validate_masked_gather_inputs(replace(inputs, target_coordinates=coordinates))


def test_rejects_zero_source_delta() -> None:
    inputs = _valid_inputs()
    deltas = inputs.source_deltas_m.clone()
    deltas[0, 1] = 0.0

    with pytest.raises(ValueError, match="positive Euclidean norm"):
        validate_masked_gather_inputs(replace(inputs, source_deltas_m=deltas))


def test_rejects_nonzero_target_value_at_masked_receiver() -> None:
    inputs = _valid_inputs()
    target = inputs.target_observed.clone()
    target[0, 0, 0, 1] = 1.0

    with pytest.raises(ValueError, match="target_observed must be exactly zero"):
        validate_masked_gather_inputs(replace(inputs, target_observed=target))


def test_rejects_nonzero_context_value_at_unavailable_receiver() -> None:
    inputs = _valid_inputs()
    contexts = inputs.context_gathers.clone()
    contexts[0, 0, 0, 0, 1] = 1.0

    with pytest.raises(ValueError, match="context_gathers must be exactly zero"):
        validate_masked_gather_inputs(replace(inputs, context_gathers=contexts))


def test_rejects_fully_observed_target() -> None:
    inputs = _valid_inputs()
    target_mask = inputs.target_observation_mask.clone()
    target_mask[0] = True

    with pytest.raises(ValueError, match="at least one unobserved receiver"):
        validate_masked_gather_inputs(
            replace(
                inputs,
                target_observation_mask=target_mask,
                target_observed=target_mask[..., None]
                .expand(-1, -1, -1, TIME_COUNT)
                .to(torch.float32),
            )
        )


def test_rejects_fully_unavailable_context_shot() -> None:
    inputs = _valid_inputs()
    context_mask = inputs.context_availability.clone()
    context_mask[1, 0] = False
    contexts = inputs.context_gathers.clone()
    contexts[1, 0] = 0.0

    with pytest.raises(ValueError, match="at least one available receiver"):
        validate_masked_gather_inputs(
            replace(
                inputs,
                context_availability=context_mask,
                context_gathers=contexts,
            )
        )


def test_validation_does_not_modify_caller_tensors() -> None:
    inputs = _valid_inputs(dtype=torch.float64)
    originals = {
        field.name: getattr(inputs, field.name).clone() for field in fields(MaskedGatherInputs)
    }

    validate_masked_gather_inputs(inputs)

    for name, original in originals.items():
        assert torch.equal(getattr(inputs, name), original)
