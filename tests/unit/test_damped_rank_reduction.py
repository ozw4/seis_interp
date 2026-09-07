from __future__ import annotations

import numpy as np
import pytest

from seis_interp.processing.damped_rank_reduction import damped_rank_reduce


def test_diagonal_values_use_the_first_discarded_singular_value() -> None:
    matrix = np.diag([10.0, 5.0, 2.0, 1.0]).astype(np.complex128)

    result = damped_rank_reduce(matrix, rank=2, damping_power=2)

    np.testing.assert_allclose(result, np.diag([9.6, 4.2, 0.0, 0.0]), rtol=1.0e-14, atol=1.0e-14)


@pytest.mark.parametrize("shape", [(6, 4), (4, 6)])
def test_reduced_matrix_has_at_most_the_requested_rank(shape: tuple[int, int]) -> None:
    rng = np.random.default_rng(9)
    matrix = rng.normal(size=shape) + 1j * rng.normal(size=shape)

    result = damped_rank_reduce(matrix, rank=2, damping_power=3)

    assert result.shape == shape
    assert np.linalg.matrix_rank(result, tol=1.0e-12) <= 2


def test_complex_singular_vectors_reconstruct_with_their_phase_and_adjoint() -> None:
    rng = np.random.default_rng(43)
    left, _ = np.linalg.qr(rng.normal(size=(5, 4)) + 1j * rng.normal(size=(5, 4)))
    right, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
    matrix = (left * [10.0, 5.0, 2.0, 1.0]) @ right.conj().T
    expected = 9.6 * np.outer(left[:, 0], right[:, 0].conj())
    expected += 4.2 * np.outer(left[:, 1], right[:, 1].conj())

    result = damped_rank_reduce(matrix, rank=2, damping_power=2)

    np.testing.assert_allclose(result, expected, rtol=1.0e-13, atol=1.0e-13)
    assert np.max(np.abs(result.imag)) > 1.0


def test_increasing_damping_power_approaches_fixed_rank_tsvd() -> None:
    matrix = np.diag([10.0, 5.0, 2.0, 1.0]).astype(np.complex128)
    tsvd = np.diag([10.0, 5.0, 0.0, 0.0])

    errors = [
        np.linalg.norm(damped_rank_reduce(matrix, rank=2, damping_power=power) - tsvd)
        for power in (1, 2, 8, 32)
    ]

    assert errors[0] > errors[1] > errors[2] > errors[3]
    assert errors[-1] < 1.0e-11


@pytest.mark.parametrize("diagonal", [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
def test_zero_singular_values_remain_finite_without_division_by_zero(
    diagonal: list[float],
) -> None:
    matrix = np.diag(diagonal).astype(np.complex128)

    with np.errstate(divide="raise", invalid="raise"):
        result = damped_rank_reduce(matrix, rank=2, damping_power=2)

    assert np.all(np.isfinite(result))
    np.testing.assert_array_equal(result, matrix)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_result_is_contiguous_complex128_and_input_is_unchanged(
    dtype: type[np.complexfloating],
) -> None:
    matrix = np.diag([10.0, 5.0, 2.0, 1.0]).astype(dtype)[:, ::-1]
    original = matrix.copy()

    result = damped_rank_reduce(matrix, rank=np.int64(2), damping_power=np.int64(2))

    assert result.dtype == np.complex128
    assert result.flags.c_contiguous
    np.testing.assert_array_equal(matrix, original)


def test_svd_uses_exact_numpy_economy_decomposition(monkeypatch: pytest.MonkeyPatch) -> None:
    original_svd = np.linalg.svd
    calls: list[bool] = []

    def recorded_svd(matrix: np.ndarray, *, full_matrices: bool) -> tuple[np.ndarray, ...]:
        calls.append(full_matrices)
        return original_svd(matrix, full_matrices=full_matrices)

    monkeypatch.setattr(np.linalg, "svd", recorded_svd)

    damped_rank_reduce(np.eye(3, dtype=np.complex128), rank=1, damping_power=2)

    assert calls == [False]


@pytest.mark.parametrize("rank", [0, -1, 3, 4, True, np.bool_(False), 1.5, 2.0])
def test_invalid_rank_is_rejected(rank: object) -> None:
    with pytest.raises(ValueError, match="rank"):
        damped_rank_reduce(np.eye(3, dtype=np.complex128), rank=rank, damping_power=2)  # type: ignore[arg-type]


@pytest.mark.parametrize("power", [0, -1, True, np.bool_(False), 1.5, 2.0])
def test_invalid_damping_power_is_rejected(power: object) -> None:
    with pytest.raises(ValueError, match="damping_power"):
        damped_rank_reduce(np.eye(3, dtype=np.complex128), rank=1, damping_power=power)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, complex(1.0, np.inf)])
def test_nonfinite_matrix_values_are_rejected(value: complex) -> None:
    matrix = np.eye(3, dtype=np.complex128)
    matrix[0, 2] = value

    with pytest.raises(ValueError, match="finite"):
        damped_rank_reduce(matrix, rank=1, damping_power=2)


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        (np.ones((2, 2, 2), dtype=np.complex128), "two-dimensional"),
        (np.eye(3, dtype=np.float64), "complex dtype"),
        (np.ones((0, 3), dtype=np.complex128), "rank"),
        ([[1.0j, 2.0j]], "two-dimensional"),
    ],
)
def test_invalid_array_contracts_are_rejected(matrix: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        damped_rank_reduce(matrix, rank=1, damping_power=2)  # type: ignore[arg-type]
