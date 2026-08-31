from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from seis_interp.pipelines.train_neighbor_inpainter import _NeighborTensorSource
from seis_interp.processing.source_bracketing import SameLineReceiverBracketingLookup


def _trace_table() -> pd.DataFrame:
    source_y = np.array((0.0, 10.0, 20.0, 30.0, 20.0, 20.0))
    source_x = np.array((100.0, 100.0, 100.0, 100.0, 200.0, 100.0))
    relative_x = np.array((5.0, 5.0, 5.0, 5.0, 5.0, 15.0))
    relative_y = np.array((-10.0, -10.0, -10.0, -10.0, -10.0, -10.0))
    return pd.DataFrame(
        {
            "array_row": np.arange(len(source_y), dtype=np.int64),
            "source_x_m": source_x,
            "source_y_m": source_y,
            "receiver_x_m": source_x + relative_x,
            "receiver_y_m": source_y + relative_y,
        }
    )


def test_lookup_returns_strict_linear_brackets_and_nearest_one_sided_source() -> None:
    table = _trace_table()
    available = np.array((True, True, False, True, True, True))
    ffids = np.array((10, 11, 12, 13, 14, 15), dtype=np.int64)
    lookup = SameLineReceiverBracketingLookup(
        table,
        available,
        ffids_by_position=ffids,
    )

    batch = lookup.batch(np.array((0, 1, 2, 3, 4, 5), dtype=np.int64))

    np.testing.assert_array_equal(
        batch.positions,
        np.array(
            (
                (-1, 1),
                (0, 3),
                (1, 3),
                (1, -1),
                (-1, -1),
                (-1, -1),
            ),
            dtype=np.int64,
        ),
    )
    np.testing.assert_allclose(
        batch.weights,
        np.array(
            (
                (0.0, 1.0),
                (2.0 / 3.0, 1.0 / 3.0),
                (0.5, 0.5),
                (1.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
            ),
            dtype=np.float32,
        ),
    )
    audit = lookup.audit(np.arange(len(table), dtype=np.int64))
    assert audit == {
        "row_count": 6,
        "bracketed_rows": 2,
        "one_sided_rows": 2,
        "unresolved_rows": 2,
        "source_entry_count": 6,
        "source_split_counts": {"train": 6, "non_train": 0},
        "target_ffid_reference_entries": 0,
        "same_source_y_reference_entries": 0,
    }


def test_lookup_skips_candidate_from_target_ffid_even_at_another_source_y() -> None:
    table = _trace_table().iloc[:4].copy()
    available = np.array((True, True, False, True))
    ffids = np.array((12, 11, 12, 13), dtype=np.int64)
    lookup = SameLineReceiverBracketingLookup(
        table,
        available,
        ffids_by_position=ffids,
    )

    batch = lookup.batch(np.array((2,), dtype=np.int64))

    np.testing.assert_array_equal(batch.positions, np.array(((1, 3),), dtype=np.int64))
    np.testing.assert_allclose(batch.weights, np.array(((0.5, 0.5),), dtype=np.float32))
    assert lookup.audit(np.array((2,), dtype=np.int64))["target_ffid_reference_entries"] == 0


def test_tensor_source_appends_reference_after_neighbor_dropout() -> None:
    table = _trace_table().iloc[:4].copy()
    available = np.array((True, True, False, True))
    ffids = np.array((10, 11, 12, 13), dtype=np.int64)
    bracketing = SameLineReceiverBracketingLookup(
        table,
        available,
        ffids_by_position=ffids,
    )

    class _Geometry:
        row_count = 4

        @staticmethod
        def neighbor_positions(positions: np.ndarray) -> np.ndarray:
            return np.tile(np.array((0, 3), dtype=np.int64), (len(positions), 1))

        @staticmethod
        def target_coordinates(positions: np.ndarray) -> np.ndarray:
            return np.zeros((len(positions), 3), dtype=np.float32)

    source = _NeighborTensorSource(
        _Geometry(),  # type: ignore[arg-type]
        train_positions=np.array((0, 1, 3), dtype=np.int64),
        train_amplitudes=torch.tensor(((0.0,), (10.0,), (30.0,))),
        device=torch.device("cpu"),
        source_bracketing=bracketing,
    )

    neighbors, availability, _ = source.gather(
        np.array((2,), dtype=np.int64),
        generator=torch.Generator().manual_seed(7),
        neighbor_dropout=0.9,
    )

    assert neighbors.shape == (1, 3, 1)
    assert availability.shape == (1, 3)
    assert availability[0, -1]
    torch.testing.assert_close(neighbors[0, -1], torch.tensor((20.0,)))
