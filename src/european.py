"""European option RR2/TP2 conditions and violations (Section 4 of the paper).

Proposition 4.1: European puts are RR2 in (n, K) when mu = E[exp(Z)] <= 1;
European calls are TP2 in (n, K) when mu >= 1.  In Black-Scholes, mu <= 1 is
equivalent to r - q <= 0.

Proposition 4.2: if E[S_1] > S_0 (positive drift), the European put has RR2
violations (equation 28) for strikes beyond K_p* = inf{x: E[S_1|S_1<=x] > S_0}.
Symmetrically, negative drift produces TP2 violations (equation 29) for the
European call at strikes below K_c*.

These are custom implementations of the paper's own equations; no
general-purpose option-pricing library reproduces this induction/violation
argument.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import optimize
from scipy.integrate import quad
from scipy.stats import norm

from src.bs_pricing import bs_european_call, bs_european_put, log_return_params


def rr2_violation_determinant(K1: float, K2: float, p1_K1: float, p1_K2: float) -> float:
    """Equation (28): violation when this determinant is > 0.

        | (K1-1)^+   (K2-1)^+ |
        | p1(K1)     p1(K2)   |
    """
    return max(K1 - 1.0, 0.0) * p1_K2 - max(K2 - 1.0, 0.0) * p1_K1


def tp2_violation_determinant(K1: float, K2: float, c1_K1: float, c1_K2: float) -> float:
    """Equation (29): violation when this determinant is < 0.

        | (1-K1)^+   (1-K2)^+ |
        | c1(K1)     c1(K2)   |
    """
    return max(1.0 - K1, 0.0) * c1_K2 - max(1.0 - K2, 0.0) * c1_K1


def truncated_expected_underlying(x: float, mu_z: float, sigma_z: float, S0: float = 1.0) -> float:
    """E[S_1 | S_1 <= x] for S_1 = S0 * exp(Z), Z ~ N(mu_z, sigma_z^2) (Section 4.2)."""
    if x <= 0:
        return 0.0
    log_x = np.log(x / S0)

    def integrand(z):
        return S0 * np.exp(z) * norm.pdf(z, loc=mu_z, scale=sigma_z)

    numerator, _ = quad(integrand, -40, log_x, limit=200)
    denominator = norm.cdf(log_x, loc=mu_z, scale=sigma_z)
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def find_kp_star(mu_z: float, sigma_z: float, S0: float = 1.0, bracket: Tuple[float, float] = (1.0, 10.0)) -> float:
    """K_p* = inf{x : E[S_1 | S_1 <= x] > S0} (Section 4.2, Proposition 4.2)."""

    def objective(x):
        return truncated_expected_underlying(x, mu_z, sigma_z, S0) - S0

    lo, hi = bracket
    f_lo, f_hi = objective(lo), objective(hi)
    if f_lo > 0:
        return lo
    while f_hi < 0 and hi < 1e6:
        hi *= 2
        f_hi = objective(hi)
    return optimize.brentq(objective, lo, hi)


def find_kc_star(mu_z: float, sigma_z: float, S0: float = 1.0, bracket: Tuple[float, float] = (1e-4, 1.0)) -> float:
    """K_c* for the symmetric call-side result under negative drift (Section 4.2)."""

    def truncated_expected_above(x: float) -> float:
        log_x = np.log(x / S0)

        def integrand(z):
            return S0 * np.exp(z) * norm.pdf(z, loc=mu_z, scale=sigma_z)

        numerator, _ = quad(integrand, log_x, 40, limit=200)
        denominator = 1.0 - norm.cdf(log_x, loc=mu_z, scale=sigma_z)
        if denominator <= 0:
            return S0
        return numerator / denominator

    def objective(x):
        return truncated_expected_above(x) - S0

    lo, hi = bracket
    try:
        return optimize.brentq(objective, lo, hi)
    except ValueError:
        return lo


def hazard_rate_ratio(K: np.ndarray, S0: float, r: float, q: float, sigma: float, T: float) -> np.ndarray:
    """R(K) = Phi(-d + sigma*sqrt(T)) / Phi(-d), equation (32) condition check."""
    K = np.asarray(K, dtype=float)
    d = (np.log(S0 / K) + (r - q) * T) / (sigma * np.sqrt(T)) + 0.5 * sigma * np.sqrt(T)
    return norm.cdf(-d + sigma * np.sqrt(T)) / norm.cdf(-d)
