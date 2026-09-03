from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from seis_interp.cli import build_parser
from seis_interp.data.whole_shot import nearest_train_source_indices
from seis_interp.models.shot_gather_inpainter import (
    DYNAMIC_ATTENTION_INPUT_FEATURE_NAMES,
    DYNAMIC_ATTENTION_SOURCE_WEIGHTING,
    INVERSE_DISTANCE_SOURCE_WEIGHTING,
    LEARNED_FILM_RECEIVER_POSITION_CONDITIONING,
    MOMENTS_SOURCE_FEATURE_MODE,
    NO_RECEIVER_POSITION_CONDITIONING,
    ORDERED_RAW_SOURCE_FEATURE_MODE,
    ShotGatherInpainter,
)
from seis_interp.processing.c3_receiver_grid import RECEIVER_X_COUNT, RECEIVER_Y_COUNT
from seis_interp.training import shot_gather_inpainter_trainer
from seis_interp.training.shot_gather_inpainter_checkpoints import (
    load_shot_gather_inpainter_checkpoint,
    save_shot_gather_inpainter_checkpoint,
)
from seis_interp.training.shot_gather_inpainter_trainer import (
    ShotGatherTrainingResult,
    _masked_mean_square,
    _shot_gather_training_step,
    train_shot_gather_inpainter,
)


def test_checkpoint_round_trip_preserves_constructor_and_selection(tmp_path: Path) -> None:
    model = ShotGatherInpainter(
        width=8,
        temporal_dilations=(1, 2),
        spatial_y_dilations=(1, 3),
        stem_kernel_size=5,
        residual_kernel_size=3,
        distance_epsilon=2.0e-5,
        distance_power=2.0,
    )
    with torch.no_grad():
        model.head[-1].bias.fill_(0.25)
    path = tmp_path / "best.pt"

    save_shot_gather_inpainter_checkpoint(
        path,
        model,
        best_step=0,
        best_validation_global_snr_db=3.5,
    )
    payload = torch.load(path, weights_only=True)
    assert payload["model_config"]["spatial_y_dilations"] == [1, 3]
    loaded = load_shot_gather_inpainter_checkpoint(path)

    assert loaded.best_step == 0
    assert loaded.best_validation_global_snr_db == 3.5
    assert loaded.input_feature_schema_version == 1
    assert loaded.input_feature_names == model.input_feature_names
    assert loaded.source_feature_mode == MOMENTS_SOURCE_FEATURE_MODE
    assert loaded.source_gather_count is None
    assert loaded.source_weighting == INVERSE_DISTANCE_SOURCE_WEIGHTING
    assert loaded.source_weighting_schema_version == 1
    assert loaded.source_weighting_input_feature_names == ()
    assert payload["source_weighting_schema"] == {
        "version": 1,
        "mode": INVERSE_DISTANCE_SOURCE_WEIGHTING,
        "input_feature_names": [],
    }
    assert loaded.receiver_position_conditioning == NO_RECEIVER_POSITION_CONDITIONING
    assert payload["model_config"]["receiver_position_conditioning"] == (
        NO_RECEIVER_POSITION_CONDITIONING
    )
    assert loaded.model.width == 8
    assert loaded.model.temporal_dilations == (1, 2)
    assert loaded.model.spatial_y_dilations == (1, 3)
    assert loaded.model.distance_epsilon == 2.0e-5
    assert loaded.model.distance_power == 2.0
    for expected, actual in zip(
        model.state_dict().values(),
        loaded.model.state_dict().values(),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)


def test_dynamic_attention_checkpoint_round_trip_preserves_schema_and_parameters(
    tmp_path: Path,
) -> None:
    model = ShotGatherInpainter(
        width=8,
        temporal_dilations=(1, 2),
        source_weighting=DYNAMIC_ATTENTION_SOURCE_WEIGHTING,
    )
    with torch.no_grad():
        final_projection = model.dynamic_source_attention.temporal[-1]
        final_projection.weight[0, 2, 0, 0, 0] = 0.25
    path = tmp_path / "dynamic-attention.pt"

    save_shot_gather_inpainter_checkpoint(
        path,
        model,
        best_step=11,
        best_validation_global_snr_db=6.5,
    )
    payload = torch.load(path, weights_only=True)
    assert payload["model_config"]["source_weighting"] == DYNAMIC_ATTENTION_SOURCE_WEIGHTING
    assert payload["model_config"]["dynamic_attention_width"] == 8
    assert payload["model_config"]["dynamic_attention_kernel_size"] == 5
    assert payload["source_weighting_schema"] == {
        "version": 1,
        "mode": DYNAMIC_ATTENTION_SOURCE_WEIGHTING,
        "input_feature_names": list(DYNAMIC_ATTENTION_INPUT_FEATURE_NAMES),
    }

    loaded = load_shot_gather_inpainter_checkpoint(path)

    assert loaded.source_weighting == DYNAMIC_ATTENTION_SOURCE_WEIGHTING
    assert loaded.source_weighting_schema_version == 1
    assert loaded.source_weighting_input_feature_names == DYNAMIC_ATTENTION_INPUT_FEATURE_NAMES
    assert loaded.model.source_weighting == DYNAMIC_ATTENTION_SOURCE_WEIGHTING
    assert loaded.best_step == 11
    for name, expected in model.state_dict().items():
        torch.testing.assert_close(
            loaded.model.state_dict()[name],
            expected,
            rtol=0.0,
            atol=0.0,
        )


def test_ordered_raw_checkpoint_round_trip_preserves_mode_count_and_schema(
    tmp_path: Path,
) -> None:
    model = ShotGatherInpainter(
        width=8,
        temporal_dilations=(1, 2),
        source_feature_mode=ORDERED_RAW_SOURCE_FEATURE_MODE,
        source_gather_count=8,
    )
    with torch.no_grad():
        model.head[-1].bias.fill_(0.125)
    path = tmp_path / "ordered-raw.pt"

    save_shot_gather_inpainter_checkpoint(
        path,
        model,
        best_step=7,
        best_validation_global_snr_db=4.25,
    )
    payload = torch.load(path, weights_only=True)
    assert payload["model_config"]["source_feature_mode"] == ORDERED_RAW_SOURCE_FEATURE_MODE
    assert payload["model_config"]["source_gather_count"] == 8
    assert payload["input_feature_schema"] == {
        "version": 2,
        "source_feature_mode": ORDERED_RAW_SOURCE_FEATURE_MODE,
        "source_gather_count": 8,
        "names": list(model.input_feature_names),
    }

    loaded = load_shot_gather_inpainter_checkpoint(path)

    assert loaded.input_feature_schema_version == 2
    assert loaded.input_feature_names == model.input_feature_names
    assert loaded.source_feature_mode == ORDERED_RAW_SOURCE_FEATURE_MODE
    assert loaded.source_gather_count == 8
    assert loaded.model.source_feature_mode == ORDERED_RAW_SOURCE_FEATURE_MODE
    assert loaded.model.source_gather_count == 8
    assert loaded.best_step == 7
    assert loaded.best_validation_global_snr_db == 4.25
    for name, expected in model.state_dict().items():
        torch.testing.assert_close(loaded.model.state_dict()[name], expected)


def test_learned_receiver_film_checkpoint_round_trip_preserves_mode_and_parameters(
    tmp_path: Path,
) -> None:
    model = ShotGatherInpainter(
        width=8,
        temporal_dilations=(1, 2),
        receiver_position_conditioning=LEARNED_FILM_RECEIVER_POSITION_CONDITIONING,
    )
    with torch.no_grad():
        model.blocks[0].receiver_film.scale[0, 1, 2, 3, 0] = 0.25
        model.blocks[1].receiver_film.shift[0, 4, 5, 6, 0] = -0.5
    path = tmp_path / "learned-film.pt"

    save_shot_gather_inpainter_checkpoint(
        path,
        model,
        best_step=9,
        best_validation_global_snr_db=5.5,
    )
    payload = torch.load(path, weights_only=True)
    assert payload["model_config"]["receiver_position_conditioning"] == (
        LEARNED_FILM_RECEIVER_POSITION_CONDITIONING
    )

    loaded = load_shot_gather_inpainter_checkpoint(path)

    assert loaded.receiver_position_conditioning == (LEARNED_FILM_RECEIVER_POSITION_CONDITIONING)
    assert loaded.model.receiver_position_conditioning == (
        LEARNED_FILM_RECEIVER_POSITION_CONDITIONING
    )
    assert loaded.best_step == 9
    for name, expected in model.state_dict().items():
        torch.testing.assert_close(
            loaded.model.state_dict()[name],
            expected,
            rtol=0.0,
            atol=0.0,
        )


def test_checkpoint_without_source_feature_fields_loads_as_legacy_moments(
    tmp_path: Path,
) -> None:
    torch.manual_seed(17)
    model = ShotGatherInpainter(width=8, temporal_dilations=(1, 2))
    path = tmp_path / "legacy-moments.pt"
    save_shot_gather_inpainter_checkpoint(
        path,
        model,
        best_step=0,
        best_validation_global_snr_db=3.5,
    )
    payload = torch.load(path, weights_only=True)
    del payload["model_config"]["source_feature_mode"]
    del payload["model_config"]["source_gather_count"]
    del payload["model_config"]["receiver_position_conditioning"]
    del payload["model_config"]["distance_power"]
    del payload["model_config"]["source_weighting"]
    del payload["model_config"]["dynamic_attention_width"]
    del payload["model_config"]["dynamic_attention_kernel_size"]
    del payload["input_feature_schema"]["source_feature_mode"]
    del payload["input_feature_schema"]["source_gather_count"]
    del payload["source_weighting_schema"]
    torch.save(payload, path)

    loaded = load_shot_gather_inpainter_checkpoint(path)

    assert loaded.source_feature_mode == MOMENTS_SOURCE_FEATURE_MODE
    assert loaded.source_gather_count is None
    assert loaded.input_feature_schema_version == 1
    assert loaded.receiver_position_conditioning == NO_RECEIVER_POSITION_CONDITIONING
    assert loaded.model.distance_power == 1.0
    assert loaded.source_weighting == INVERSE_DISTANCE_SOURCE_WEIGHTING
    assert loaded.source_weighting_input_feature_names == ()
    assert loaded.model.state_dict().keys() == model.state_dict().keys()
    for name, expected in model.state_dict().items():
        torch.testing.assert_close(
            loaded.model.state_dict()[name],
            expected,
            rtol=0.0,
            atol=0.0,
        )


def test_checkpoint_without_spatial_y_dilations_defaults_to_stage09_behavior(
    tmp_path: Path,
) -> None:
    model = ShotGatherInpainter(width=8, temporal_dilations=(1, 2))
    path = tmp_path / "stage09.pt"
    save_shot_gather_inpainter_checkpoint(
        path,
        model,
        best_step=0,
        best_validation_global_snr_db=3.5,
    )
    payload = torch.load(path, weights_only=True)
    del payload["model_config"]["spatial_y_dilations"]
    torch.save(payload, path)

    loaded = load_shot_gather_inpainter_checkpoint(path)

    assert loaded.model.spatial_y_dilations == (1, 1)
    for block in loaded.model.blocks:
        assert block.spatial.dilation == (1, 1, 1)
        assert block.spatial.padding == (1, 1, 0)


def test_checkpoint_rejects_changed_input_feature_order(tmp_path: Path) -> None:
    model = ShotGatherInpainter(width=8, temporal_dilations=(1,))
    path = tmp_path / "best.pt"
    save_shot_gather_inpainter_checkpoint(
        path,
        model,
        best_step=0,
        best_validation_global_snr_db=3.5,
    )
    payload = torch.load(path, weights_only=True)
    payload["input_feature_schema"]["names"] = list(
        reversed(payload["input_feature_schema"]["names"])
    )
    torch.save(payload, path)

    with pytest.raises(ValueError, match="feature names"):
        load_shot_gather_inpainter_checkpoint(path)


@pytest.mark.parametrize(
    ("field", "replacement", "match"),
    [
        ("version", 1, "schema version must be 2"),
        ("source_feature_mode", MOMENTS_SOURCE_FEATURE_MODE, "source mode"),
        ("source_gather_count", 7, "source count"),
        ("source_gather_count", True, "source count"),
    ],
)
def test_ordered_raw_checkpoint_rejects_changed_feature_schema(
    tmp_path: Path,
    field: str,
    replacement: object,
    match: str,
) -> None:
    model = ShotGatherInpainter(
        width=8,
        temporal_dilations=(1,),
        source_feature_mode=ORDERED_RAW_SOURCE_FEATURE_MODE,
        source_gather_count=8,
    )
    path = tmp_path / "ordered-raw.pt"
    save_shot_gather_inpainter_checkpoint(
        path,
        model,
        best_step=0,
        best_validation_global_snr_db=3.5,
    )
    payload = torch.load(path, weights_only=True)
    payload["input_feature_schema"][field] = replacement
    torch.save(payload, path)

    with pytest.raises(ValueError, match=match):
        load_shot_gather_inpainter_checkpoint(path)


@pytest.mark.parametrize(
    ("field", "replacement", "match"),
    [
        ("version", 2, "schema version must be 1"),
        ("mode", INVERSE_DISTANCE_SOURCE_WEIGHTING, "mode does not match"),
        ("input_feature_names", ["changed"], "feature names"),
    ],
)
def test_dynamic_attention_checkpoint_rejects_changed_weighting_schema(
    tmp_path: Path,
    field: str,
    replacement: object,
    match: str,
) -> None:
    model = ShotGatherInpainter(
        width=8,
        temporal_dilations=(1,),
        source_weighting=DYNAMIC_ATTENTION_SOURCE_WEIGHTING,
    )
    path = tmp_path / "dynamic-attention.pt"
    save_shot_gather_inpainter_checkpoint(
        path,
        model,
        best_step=0,
        best_validation_global_snr_db=3.5,
    )
    payload = torch.load(path, weights_only=True)
    payload["source_weighting_schema"][field] = replacement
    torch.save(payload, path)

    with pytest.raises(ValueError, match=match):
        load_shot_gather_inpainter_checkpoint(path)


def test_masked_mean_square_ignores_unavailable_receiver_cells() -> None:
    values = torch.zeros(1, RECEIVER_X_COUNT, RECEIVER_Y_COUNT, 2)
    values[0, 0, 0] = torch.tensor([2.0, 4.0])
    values[0, 0, 1] = 1000.0
    mask = torch.zeros(1, RECEIVER_X_COUNT, RECEIVER_Y_COUNT, dtype=torch.bool)
    mask[0, 0, 0] = True

    result = _masked_mean_square(values, mask)

    torch.testing.assert_close(result, torch.tensor(10.0))


def test_nearest_train_sources_exclude_target_and_break_ties_by_ffid() -> None:
    train_ffids = np.asarray([30, 10, 20, 40], dtype=np.int64)
    train_sources = np.asarray([[1.0, 0.0], [-1.0, 0.0], [0.0, 0.0], [3.0, 0.0]])

    indices = nearest_train_source_indices(
        train_ffids,
        train_sources,
        np.asarray([20, 99]),
        np.asarray([[0.0, 0.0], [0.0, 0.0]]),
        source_gather_count=2,
    )

    assert train_ffids[indices[0]].tolist() == [10, 30]
    assert train_ffids[indices[1]].tolist() == [10, 30]


def test_trainer_writes_best_checkpoint_for_masked_gather_batch(tmp_path: Path) -> None:
    model = ShotGatherInpainter(width=8, temporal_dilations=(1,))
    neighbors = torch.randn(1, 2, RECEIVER_X_COUNT, RECEIVER_Y_COUNT, 3)
    availability = torch.ones(1, 2, RECEIVER_X_COUNT, RECEIVER_Y_COUNT, dtype=torch.bool)
    source_deltas = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]])
    target_coordinates = torch.zeros(1, 2)
    targets = neighbors.mean(dim=1) + 0.1
    target_mask = torch.ones(1, RECEIVER_X_COUNT, RECEIVER_Y_COUNT, dtype=torch.bool)

    def batch_provider(
        batch_size: int,
        *,
        generator: torch.Generator,
        neighbor_dropout: float,
    ) -> tuple[torch.Tensor, ...]:
        assert batch_size == 1
        assert isinstance(generator, torch.Generator)
        assert neighbor_dropout == 0.0
        return (
            neighbors,
            availability,
            source_deltas,
            target_coordinates,
            targets,
            target_mask,
        )

    validation_values = iter((1.0, 1.25))
    result = train_shot_gather_inpainter(
        model,
        batch_provider,
        lambda _model: next(validation_values),
        device="cpu",
        generator=torch.Generator().manual_seed(4),
        checkpoint_path=tmp_path / "best.pt",
        total_steps=1,
        batch_size=1,
        neighbor_dropout=0.0,
        derivative_weight=0.1,
        learning_rate=1.0e-3,
        weight_decay=0.0,
        validation_interval=1,
        use_bfloat16=False,
        training_ffid_count=3,
        training_trace_count=100,
        reporter=lambda _message: None,
    )

    assert result.best_step == 1
    assert result.best_validation_global_snr_db == 1.25
    assert [row["step"] for row in result.history] == [0, 1]
    assert load_shot_gather_inpainter_checkpoint(tmp_path / "best.pt").best_step == 1


def test_trainer_can_keep_step_zero_and_does_not_force_step_one_validation(
    tmp_path: Path,
) -> None:
    model = ShotGatherInpainter(width=8, temporal_dilations=(1,))
    neighbors = torch.randn(1, 2, RECEIVER_X_COUNT, RECEIVER_Y_COUNT, 3)
    availability = torch.ones(1, 2, RECEIVER_X_COUNT, RECEIVER_Y_COUNT, dtype=torch.bool)
    source_deltas = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]])
    target_coordinates = torch.zeros(1, 2)
    targets = neighbors.mean(dim=1) + 0.1
    target_mask = torch.ones(1, RECEIVER_X_COUNT, RECEIVER_Y_COUNT, dtype=torch.bool)
    validation_values = iter((2.0, 1.0))

    result = train_shot_gather_inpainter(
        model,
        lambda _batch_size, *, generator, neighbor_dropout: (
            neighbors,
            availability,
            source_deltas,
            target_coordinates,
            targets,
            target_mask,
        ),
        lambda _model: next(validation_values),
        device="cpu",
        generator=torch.Generator().manual_seed(4),
        checkpoint_path=tmp_path / "best.pt",
        total_steps=2,
        batch_size=1,
        neighbor_dropout=0.0,
        derivative_weight=0.1,
        learning_rate=1.0e-3,
        weight_decay=0.0,
        validation_interval=2,
        use_bfloat16=False,
        training_ffid_count=3,
        training_trace_count=100,
        reporter=lambda _message: None,
    )

    assert result.best_step == 0
    assert result.best_validation_global_snr_db == 2.0
    assert [row["step"] for row in result.history] == [0, 2]
    assert "loss" not in result.history[0]
    assert load_shot_gather_inpainter_checkpoint(tmp_path / "best.pt").best_step == 0


def _gather_batch_tensors() -> tuple[torch.Tensor, ...]:
    neighbors = torch.randn(1, 2, RECEIVER_X_COUNT, RECEIVER_Y_COUNT, 3)
    availability = torch.ones(1, 2, RECEIVER_X_COUNT, RECEIVER_Y_COUNT, dtype=torch.bool)
    source_deltas = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]])
    target_coordinates = torch.zeros(1, 2)
    targets = neighbors.mean(dim=1) + 0.1
    target_mask = torch.ones(1, RECEIVER_X_COUNT, RECEIVER_Y_COUNT, dtype=torch.bool)
    return (neighbors, availability, source_deltas, target_coordinates, targets, target_mask)


def _wrapper_keywords(tmp_path: Path, **overrides: object) -> dict[str, object]:
    keywords: dict[str, object] = {
        "device": "cpu",
        "generator": torch.Generator().manual_seed(4),
        "checkpoint_path": tmp_path / "best.pt",
        "total_steps": 1,
        "batch_size": 1,
        "neighbor_dropout": 0.0,
        "derivative_weight": 0.1,
        "learning_rate": 1.0e-3,
        "weight_decay": 0.0,
        "validation_interval": 1,
        "use_bfloat16": False,
        "training_ffid_count": 3,
        "training_trace_count": 100,
        "reporter": lambda _message: None,
    }
    keywords.update(overrides)
    return keywords


def test_trainer_keeps_result_type_history_keys_and_progress_format(tmp_path: Path) -> None:
    model = ShotGatherInpainter(width=8, temporal_dilations=(1,))
    batch = _gather_batch_tensors()
    validation_values = iter((1.0, 1.25))
    messages: list[str] = []

    result = train_shot_gather_inpainter(
        model,
        lambda _batch_size, *, generator, neighbor_dropout: batch,
        lambda _model: next(validation_values),
        **_wrapper_keywords(tmp_path, reporter=messages.append),
    )

    assert isinstance(result, ShotGatherTrainingResult)
    row = result.history[1]
    assert list(row) == [
        "step",
        "loss",
        "mse",
        "derivative_mse",
        "learning_rate",
        "validation_global_snr_db",
    ]
    assert messages[0] == ("shot_gather_inpainter 0/1: oracle_per_trace_unit_rms_global_snr_db=1")
    assert messages[1] == (
        f"shot_gather_inpainter 1/1: loss={row['loss']:.8g} mse={row['mse']:.8g} "
        f"derivative_mse={row['derivative_mse']:.8g} "
        f"learning_rate={row['learning_rate']:.8g} "
        "oracle_per_trace_unit_rms_global_snr_db=1.25"
    )


def test_trainer_rejects_negative_and_non_finite_derivative_weight(tmp_path: Path) -> None:
    model = ShotGatherInpainter(width=8, temporal_dilations=(1,))

    def _unused_provider(
        batch_size: int,
        *,
        generator: torch.Generator,
        neighbor_dropout: float,
    ) -> tuple[torch.Tensor, ...]:
        raise AssertionError("batch_provider must not be called")

    with pytest.raises(ValueError, match="derivative_weight must be non-negative"):
        train_shot_gather_inpainter(
            model,
            _unused_provider,
            lambda _model: 0.0,
            **_wrapper_keywords(tmp_path, derivative_weight=-0.1),
        )
    with pytest.raises(ValueError, match="derivative_weight must be a finite number"):
        train_shot_gather_inpainter(
            model,
            _unused_provider,
            lambda _model: 0.0,
            **_wrapper_keywords(tmp_path, derivative_weight=float("nan")),
        )


def test_trainer_rejects_wrong_model_type(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="model must be a ShotGatherInpainter"):
        train_shot_gather_inpainter(
            torch.nn.Linear(2, 2),
            lambda _batch_size, *, generator, neighbor_dropout: _gather_batch_tensors(),
            lambda _model: 0.0,
            **_wrapper_keywords(tmp_path),
        )


def test_shot_gather_step_runs_forward_and_losses_in_one_autocast_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = {"value": False}

    class _FakeAutocast:
        def __init__(self, device_type: str, dtype: torch.dtype) -> None:
            assert device_type == "cuda"
            assert dtype == torch.bfloat16

        def __enter__(self) -> None:
            active["value"] = True

        def __exit__(self, *exc: object) -> bool:
            active["value"] = False
            return False

    monkeypatch.setattr(torch, "autocast", _FakeAutocast)

    real_masked_mean_square = shot_gather_inpainter_trainer._masked_mean_square
    loss_scope_flags: list[bool] = []

    def recording_masked_mean_square(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        loss_scope_flags.append(active["value"])
        return real_masked_mean_square(values, mask)

    monkeypatch.setattr(
        shot_gather_inpainter_trainer,
        "_masked_mean_square",
        recording_masked_mean_square,
    )

    model = ShotGatherInpainter(width=8, temporal_dilations=(1,))
    forward_scope_flags: list[bool] = []
    original_forward = model.forward

    def recording_forward(*args: object, **kwargs: object) -> torch.Tensor:
        forward_scope_flags.append(active["value"])
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(model, "forward", recording_forward)

    result = _shot_gather_training_step(
        model,
        _gather_batch_tensors(),
        derivative_weight=0.5,
        use_cuda_bfloat16=True,
    )

    assert forward_scope_flags == [True]
    assert loss_scope_flags == [True, True]
    assert active["value"] is False
    metrics = dict(result.history_metrics)
    torch.testing.assert_close(result.loss, metrics["mse"] + 0.5 * metrics["derivative_mse"])
    assert [name for name, _ in result.finite_checks] == [
        "training MSE",
        "training derivative MSE",
    ]


def test_shot_gather_step_skips_autocast_without_cuda_bfloat16(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _forbidden_autocast(*args: object, **kwargs: object) -> None:
        raise AssertionError("torch.autocast must not be entered on CPU")

    monkeypatch.setattr(torch, "autocast", _forbidden_autocast)
    model = ShotGatherInpainter(width=8, temporal_dilations=(1,))

    result = _shot_gather_training_step(
        model,
        _gather_batch_tensors(),
        derivative_weight=0.1,
        use_cuda_bfloat16=False,
    )

    assert bool(torch.isfinite(result.loss))


def test_cli_parser_exposes_shot_gather_command_without_overwrite() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "train",
            "shot-gather-inpainter",
            "--config",
            "config.yaml",
            "--interim",
            "interim",
            "--processed",
            "processed",
            "--output",
            "run",
        ]
    )

    assert args.train_command == "shot-gather-inpainter"
    assert not hasattr(args, "overwrite")
