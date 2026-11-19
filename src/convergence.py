"""Bermudan-to-American convergence (Section 6 of the paper).

Lemma 6.1 states that the Bermudan value on a grid Delta^N = T / 2^N
converges to the American value as N -> infinity for right-continuous paths,
a continuous payoff and an integrable running supremum.  Theorem 6.2 and
Corollary 6.1 then transport the RR2/TP2 properties of Bermudan puts/calls
(Sections 3, 5) to their American counterparts in strike and
time-to-expiration, for any r >= 0 in the Black-Scholes model (because the
Gaussian log-return density is log-concave at every refinement level Delta^N).

This module reuses the exact Bermudan recursion of ``src.bermudan`` at
geometrically refined time grids and reports the resulting price sequence,
which is the paper's own convergence argument (no library implements this
specific limiting procedure).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from src.bermudan import PricingGrid, bermudan_put_values
from src.bs_pricing import log_return_params


def _cfl_scaled_grid_points(N: int, base_grid_points: int, base_N: int) -> int:
    """Number of spatial grid points to use at refinement level ``N``.

    ``build_continuation_operator`` represents the conditional-expectation
    operator of equation (12) by linear interpolation onto a log-moneyness
    grid of fixed half-width. The one-step log-return standard deviation is
    ``sigma_z = sigma * sqrt(dt)`` (Section 3.6), which shrinks by a factor
    ``sqrt(2)`` every time N increases by one (``dt = T / 2**N``). If the
    spatial grid spacing were kept fixed, the interpolation error introduced
    by the kink from the ``max`` in equation (13) would no longer be
    smoothed out by the (now narrower) one-step convolution, and this error
    accumulates over the ``2**N`` steps of the recursion, so the Bermudan
    sequence would stop converging monotonically for large N. Scaling the
    number of grid points by ``2**((N - base_N) / 2)`` keeps the grid spacing
    proportional to ``sigma_z`` (a CFL-type resolution condition for this
    fixed-grid random-walk recursion), which restores the monotone
    convergence Lemma 6.1 predicts (verified against this module's tests).
    """
    scaled = base_grid_points * 2.0 ** ((N - base_N) / 2.0)
    n_points = int(round(scaled))
    return n_points if n_points % 2 == 1 else n_points + 1


def bermudan_price_at_refinement(K: float, S0: float, r: float, sigma: float, T: float, N: int,
                                  grid_points: int, grid_log_half_width: float,
                                  base_N: int = 1) -> float:
    """Bermudan put price P_{2^N}(K, S0) on the grid Delta^N = T / 2^N with
    2**N equally spaced exercise opportunities up to maturity T.

    ``grid_points`` is the spatial resolution at ``base_N``; the resolution
    actually used at level ``N`` is scaled by ``_cfl_scaled_grid_points`` so
    that refining the time grid also refines the spatial grid consistently.
    """
    n_steps = 2 ** N
    dt = T / n_steps
    mu_z, sigma_z = log_return_params(r, sigma, dt)
    resolved_grid_points = _cfl_scaled_grid_points(N, grid_points, base_N)
    grid = PricingGrid.build(n_points=resolved_grid_points, log_half_width=grid_log_half_width, center=np.log(K / S0))
    history = bermudan_put_values(grid, r, dt, mu_z, sigma_z, n_steps=n_steps)
    p_n = history[-1]
    # Lemma 3.4: P_n(K, S0) = S0 * p_n(K/S0)
    k_over_S0 = K / S0
    return float(S0 * np.interp(np.log(k_over_S0), grid.x, p_n))


def convergence_sequence(K: float, S0: float, r: float, sigma: float, T: float,
                          n_values: List[int], grid_points: int, grid_log_half_width: float,
                          base_N: int = 1) -> Dict[int, float]:
    """Return {N: P_{2^N}(K, S0)} for N in ``n_values``, an increasing
    sequence of exercise-opportunity refinements converging to the American
    put value (Lemma 6.1)."""
    return {
        N: bermudan_price_at_refinement(K, S0, r, sigma, T, N, grid_points, grid_log_half_width, base_N=min(n_values))
        for N in n_values
    }
