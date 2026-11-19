"""Unit tests for src/bermudan.py: equations (11)-(13), (16)-(18), the
Proposition 3.1 determinant, and the three induction steps of Theorems 3.1-3.2
(Section 3)."""

import numpy as np
import pytest

from src.bermudan import (PricingGrid, bermudan_call_values, bermudan_put_values,
                           build_continuation_operator, continuation_put_value_direct,
                           rr2_determinant, tp2_determinant)
from src.bs_pricing import log_return_params


@pytest.fixture
def grid():
    return PricingGrid.build(n_points=301, log_half_width=4.0)


def test_grid_rejects_too_few_points():
    with pytest.raises(ValueError):
        PricingGrid.build(n_points=2, log_half_width=1.0)


def test_continuation_operator_matches_direct_quadrature(grid):
    """The matrix-based operator (vectorized equation 12) must agree with a
    direct scipy.integrate.quad evaluation of the same continuation value."""
    r, dt, sigma = 0.1, 1.0, 0.3
    mu_z, sigma_z = log_return_params(r, sigma, dt)
    A = build_continuation_operator(grid, r, dt, mu_z, sigma_z)
    p0 = np.maximum(grid.k - 1.0, 0.0)
    h_matrix = A @ p0

    def p0_fn(k):
        return np.maximum(k - 1.0, 0.0)

    test_idx = np.linspace(50, len(grid.k) - 50, 6).astype(int)
    h_direct = continuation_put_value_direct(grid.k[test_idx], p0_fn, r, dt, mu_z, sigma_z)
    np.testing.assert_allclose(h_matrix[test_idx], h_direct, rtol=5e-3, atol=1e-4)


def test_bermudan_put_dominates_intrinsic_and_is_monotone_in_n(grid):
    r, dt, sigma = 0.1, 1.0, 0.3
    mu_z, sigma_z = log_return_params(r, sigma, dt)
    history = bermudan_put_values(grid, r, dt, mu_z, sigma_z, n_steps=3)
    intrinsic = np.maximum(grid.k - 1.0, 0.0)
    for p in history:
        assert np.all(p >= intrinsic - 1e-10)
    # more exercise opportunities cannot decrease the option's value
    for n in range(len(history) - 1):
        assert np.all(history[n + 1] >= history[n] - 1e-9)


def test_bermudan_call_dominates_intrinsic(grid):
    r, dt, sigma = 0.1, 1.0, 0.3
    mu_z, sigma_z = log_return_params(r, sigma, dt)
    history = bermudan_call_values(grid, r, dt, mu_z, sigma_z, n_steps=2)
    intrinsic = np.maximum(1.0 - grid.k, 0.0)
    for c in history:
        assert np.all(c >= intrinsic - 1e-10)


def test_rr2_determinant_nonpositive_for_bermudan_put(grid):
    """Proposition 3.1 / equation (14): the 2x2 determinant of intrinsic vs.
    Bermudan put price must be <= 0 for K1 < K2, every n."""
    r, dt, sigma = 0.1, 1.0, 0.3
    mu_z, sigma_z = log_return_params(r, sigma, dt)
    history = bermudan_put_values(grid, r, dt, mu_z, sigma_z, n_steps=2)
    intrinsic = np.maximum(grid.k - 1.0, 0.0)
    rng = np.random.default_rng(0)
    idx_pairs = rng.choice(len(grid.k), size=(200, 2), replace=True)
    for p in history:
        for i, j in idx_pairs:
            if grid.k[i] >= grid.k[j]:
                i, j = j, i
            if grid.k[i] == grid.k[j]:
                continue
            det = rr2_determinant(p[i], p[j], intrinsic[i], intrinsic[j])
            assert det <= 1e-8, f"RR2 violation at K1={grid.k[i]}, K2={grid.k[j]}"


def test_tp2_determinant_sign_convention():
    assert tp2_determinant(1.0, 2.0, 3.0, 6.0) == pytest.approx(0.0)
    assert tp2_determinant(1.0, 1.0, 1.0, 4.0) > 0
