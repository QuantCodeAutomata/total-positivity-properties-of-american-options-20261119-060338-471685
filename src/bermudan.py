"""Bermudan put/call pricing by backward induction and the RR2/TP2 checks of
Sections 3 and 6 of the paper.

The dynamic-programming recursion follows equations (11)-(13):

    P_0(K, S) = (K - S)^+                                              (put)
    H_{n+1}(K, S) = exp(-r*dt) * E[P_n(K, S_1) | S_0 = S]              (12)
    P_{n+1}(K, S) = max((K - S)^+, H_{n+1}(K, S))                      (13)

and the analogous equations (16)-(18) for calls.  By Lemma 3.4,
``P_n(K, S) = S * P_n(K/S, 1)``, so it suffices to track the single-variable
function ``p_n(k) = P_n(k, 1)`` on a grid of ``k`` values; this is the
"key identity" referenced in the paper's numerical-illustration notes.  The
continuation step becomes

    h_{n+1}(k) = exp(-r*dt) * E[exp(Z) * p_n(k * exp(-Z))]

which we evaluate by trapezoidal quadrature over a Gaussian Z-grid, encoded
once as a linear operator (a matrix) so that repeated backward-induction
steps reduce to a single matrix-vector product.  No option-pricing library
implements this induction (it is the paper's own construction), so it is a
custom implementation of equations (11)-(13)/(16)-(18) and Lemma 3.4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
from scipy.stats import norm


@dataclass
class PricingGrid:
    """A log-spaced grid of strikes/moneyness levels used to represent
    ``p_n(k) = P_n(k, 1)`` as a vector of values."""

    k: np.ndarray       # grid points in level space (k = exp(x))
    x: np.ndarray       # grid points in log space (x = log(k))

    @classmethod
    def build(cls, n_points: int, log_half_width: float, center: float = 0.0) -> "PricingGrid":
        if n_points < 3:
            raise ValueError("n_points must be >= 3")
        x = np.linspace(center - log_half_width, center + log_half_width, n_points)
        return cls(k=np.exp(x), x=x)


def _quadrature_z_grid(mu_z: float, sigma_z: float, n_points: int = 241, n_std: float = 8.0) -> Tuple[np.ndarray, np.ndarray]:
    """Gaussian density g(z) sampled on an evenly spaced grid, with trapezoidal weights."""
    z = np.linspace(mu_z - n_std * sigma_z, mu_z + n_std * sigma_z, n_points)
    g = norm.pdf(z, loc=mu_z, scale=sigma_z)
    weights = np.gradient(z)  # trapezoidal spacing weights
    return z, g * weights


def build_continuation_operator(grid: PricingGrid, r: float, dt: float, mu_z: float, sigma_z: float,
                                 z_points: int = 241) -> np.ndarray:
    """Matrix A such that ``h_{n+1} = A @ p_n`` approximates equation (12)
    via Lemma 3.4, for puts and calls alike (the operator only depends on
    the underlying's log-return distribution, not on the payoff)."""
    z, w = _quadrature_z_grid(mu_z, sigma_z, z_points)
    n = grid.x.size
    A = np.zeros((n, n))
    disc = np.exp(-r * dt)
    for m, (zm, wm) in enumerate(zip(z, w)):
        # target log-moneyness k_i * exp(-z_m) -> x_i - z_m
        target_x = grid.x - zm
        # linear interpolation weights of target_x onto grid.x, vectorized
        idx = np.searchsorted(grid.x, target_x) - 1
        idx = np.clip(idx, 0, n - 2)
        x0 = grid.x[idx]
        x1 = grid.x[idx + 1]
        frac = np.clip((target_x - x0) / (x1 - x0), 0.0, 1.0)
        coeff = disc * np.exp(zm) * wm
        np.add.at(A, (np.arange(n), idx), coeff * (1 - frac))
        np.add.at(A, (np.arange(n), idx + 1), coeff * frac)
    return A


def bermudan_put_values(grid: PricingGrid, r: float, dt: float, mu_z: float, sigma_z: float,
                         n_steps: int) -> list[np.ndarray]:
    """Return [p_0, p_1, ..., p_{n_steps}] on ``grid`` for the Bermudan put."""
    A = build_continuation_operator(grid, r, dt, mu_z, sigma_z)
    p0 = np.maximum(grid.k - 1.0, 0.0)  # P_0(K, 1) = (K - 1)^+ ; recall S = 1 here, strikes expressed as k = K/S
    history = [p0]
    p = p0
    for _ in range(n_steps):
        h = A @ p
        p = np.maximum(grid.k - 1.0, 0.0)
        p = np.maximum(p, h)
        history.append(p)
    return history


def bermudan_call_values(grid: PricingGrid, r: float, dt: float, mu_z: float, sigma_z: float,
                          n_steps: int) -> list[np.ndarray]:
    """Return [c_0, c_1, ..., c_{n_steps}] on ``grid`` for the Bermudan call
    using equations (16)-(18); the continuation operator A is identical to
    the put's because it depends only on the law of the log return."""
    A = build_continuation_operator(grid, r, dt, mu_z, sigma_z)
    c0 = np.maximum(1.0 - grid.k, 0.0)  # C_0(K, 1) = (1 - K)^+
    history = [c0]
    c = c0
    for _ in range(n_steps):
        h = A @ c
        c = np.maximum(1.0 - grid.k, 0.0)
        c = np.maximum(c, h)
        history.append(c)
    return history


def continuation_put_value_direct(k: np.ndarray, p_n: Callable[[np.ndarray], np.ndarray],
                                   r: float, dt: float, mu_z: float, sigma_z: float) -> np.ndarray:
    """Reference (slow but transparent) evaluation of H_{n+1}(K, 1) via
    direct Gaussian-weighted quadrature, used only to validate the
    operator-based recursion in tests."""
    from scipy.integrate import quad

    k = np.atleast_1d(np.asarray(k, dtype=float))
    out = np.zeros_like(k)
    disc = np.exp(-r * dt)
    for i, ki in enumerate(k):
        integrand = lambda z: np.exp(z) * p_n(np.array([ki * np.exp(-z)]))[0] * norm.pdf(z, loc=mu_z, scale=sigma_z)
        val, _ = quad(integrand, mu_z - 10 * sigma_z, mu_z + 10 * sigma_z, limit=200)
        out[i] = disc * val
    return out


def rr2_determinant(prices_k1: np.ndarray, prices_k2: np.ndarray, intrinsic_k1: np.ndarray,
                     intrinsic_k2: np.ndarray) -> np.ndarray:
    """Determinant of equation (14)/(20):

        | intrinsic(K1)   intrinsic(K2) |
        | price_n(K1)     price_n(K2)   |

    which Proposition 3.1/Theorem 3.1 require to be <= 0 for all n, K1 < K2.
    """
    return intrinsic_k1 * prices_k2 - intrinsic_k2 * prices_k1


def tp2_determinant(prices_n1_k1: float, prices_n1_k2: float, prices_n2_k1: float, prices_n2_k2: float) -> float:
    """Generic 2x2 determinant used for the RR2/TP2 cross-checks of
    equation (22): price(n1,K1)*price(n2,K2) - price(n1,K2)*price(n2,K1)."""
    return prices_n1_k1 * prices_n2_k2 - prices_n1_k2 * prices_n2_k1
