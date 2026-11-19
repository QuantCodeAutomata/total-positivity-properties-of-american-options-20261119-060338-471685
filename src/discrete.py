"""Discrete-distribution extension of the TP2/RR2 results (Section 5).

Theorem 5.1: if the integer-valued log-return steps Z_i have a log-concave
probability mass function p(.), Theorems 3.1-3.2 hold for strikes on the grid
K_0 = {K_0 * e^{bj} : j in Z}.  Equation (34) gives the discrete continuation
value

    H_{n+1}(e^{bj} K_0, S) = e^{-r dt} * sum_m e^{b(j-m)} p(j-m) P_n(e^{bm} K_0, S)

A probability mass function is log-concave when p(j-1) p(j+1) <= p(j)^2 for
all integers j.  The standard binomial model (p(0) = 0) is not log-concave,
and Section 5 exhibits a specific violation using it; this module reproduces
both the positive (trinomial) and negative (binomial) results.  This is a
custom implementation of equation (34); no option-pricing library implements
this discrete TP2/RR2 argument.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def is_log_concave_pmf(p: Dict[int, float]) -> bool:
    """Check p(j-1) p(j+1) <= p(j)^2 for every integer j with mass on both sides."""
    for j in p:
        left = p.get(j - 1, 0.0)
        right = p.get(j + 1, 0.0)
        center = p.get(j, 0.0)
        if left * right > center**2 + 1e-15:
            return False
    return True


def trinomial_pmf(p0: float, c: float) -> Dict[int, float]:
    """Trinomial model p(0) = p0, p(+-1) = (1-p0)/2 +- c (Section 5)."""
    base = (1.0 - p0) / 2.0
    pmf = {-1: base - c, 0: p0, 1: base + c}
    total = sum(pmf.values())
    assert abs(total - 1.0) < 1e-10, "trinomial pmf must sum to 1"
    assert min(pmf.values()) >= 0, "trinomial pmf must be nonnegative"
    return pmf


def binomial_pmf(r: float, sigma: float, dt: float) -> Tuple[Dict[int, float], float]:
    """Risk-neutral binomial up/down probabilities on the grid step b = sigma*sqrt(dt)
    (Section 5 violation example): p(1) = (e^{r dt} - e^{-b}) / (e^{b} - e^{-b})."""
    b = sigma * np.sqrt(dt)
    p_up = (np.exp(r * dt) - np.exp(-b)) / (np.exp(b) - np.exp(-b))
    pmf = {1: p_up, -1: 1.0 - p_up, 0: 0.0}
    return pmf, b


def discrete_bermudan_put_grid(pmf: Dict[int, float], b: float, r: float, dt: float,
                                 j_half_width: int, n_steps: int, K0: float = 1.0) -> Dict[int, np.ndarray]:
    """Backward induction of equation (34) for the put, on the integer grid
    K_0 * e^{bj}; returns {n: array of P_n(K_0*e^{bj}, 1) for j in range}.

    ``K0`` anchors the lattice so that specific strikes of interest (e.g. the
    Section 5 violation example K1 = 0.4, K2 = 0.4*e) land exactly on grid
    points rather than being interpolated.
    """
    j_values = np.arange(-j_half_width, j_half_width + 1)
    k_values = K0 * np.exp(b * j_values)  # strikes K_0 * e^{bj}, S fixed at 1
    p0 = np.maximum(k_values - 1.0, 0.0)
    offsets = sorted(m for m in pmf if pmf[m] > 0)
    history = {0: p0}
    p = p0
    disc = np.exp(-r * dt)
    n = len(j_values)
    for step in range(1, n_steps + 1):
        h = np.zeros(n)
        for m in offsets:
            prob = pmf[m]
            # equation (34): term e^{b*(j-m)} p(j-m) P_n(e^{b*m} K_0, S) with roles
            # matched to the j-index convolution (sum over the step size m the
            # chain can take, shifting the continuation lookup by -m).
            shifted = np.roll(p, m)
            if m > 0:
                shifted[:m] = p[0]  # clamp at the grid boundary (deep ITM/OTM region, flat payoff tail)
            elif m < 0:
                shifted[m:] = p[-1]
            h += disc * np.exp(b * m) * prob * shifted
        p = np.maximum(k_values - 1.0, 0.0)
        p = np.maximum(p, h)
        history[step] = p
    return history


def h_values_binomial(K1: float, K2: float, r: float, sigma: float, dt: float, n_steps: Tuple[int, int]) -> Dict[str, float]:
    """H_n(K, 1) continuation values for the specific binomial violation
    example of Section 5 (S0=1, r=0.1, sigma=1, dt=1, K1=0.4, K2=0.4e).

    With b = sigma*sqrt(dt) = 1, K2 = K1 * e^b, so anchoring the lattice at
    K0 = K1 places K1 exactly at grid index j=0 and K2 exactly at j=1: no
    interpolation is needed and the induction is exact on the binomial tree.
    """
    pmf, b = binomial_pmf(r, sigma, dt)
    assert abs(np.log(K2 / K1) - b) < 1e-9, "K2 must equal K1 * e^b for the exact-lattice shortcut"
    max_n = max(n_steps) + 1
    j_half_width = max_n + 5
    history = discrete_bermudan_put_grid(pmf, b, r, dt, j_half_width=j_half_width, n_steps=max_n, K0=K1)
    j0 = j_half_width  # index of j=0 (K1)
    j1 = j_half_width + 1  # index of j=1 (K2)

    disc = np.exp(-r * dt)
    out = {}
    for n in n_steps:
        p_prev = history[n - 1]
        h_K1 = 0.0
        h_K2 = 0.0
        for m, prob in pmf.items():
            if prob <= 0:
                continue
            # H_n(K0*e^{bj}, 1) = e^{-r dt} sum_m p(m) e^{bm} p_{n-1}(K0*e^{b(j-m)})
            h_K1 += disc * prob * np.exp(b * m) * p_prev[j0 - m]
            h_K2 += disc * prob * np.exp(b * m) * p_prev[j1 - m]
        out[f"H{n}_K1"] = h_K1
        out[f"H{n}_K2"] = h_K2
    return out
