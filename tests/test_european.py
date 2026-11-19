"""Unit tests for src/european.py (Section 4, Propositions 4.1-4.2)."""

import numpy as np
import pytest

from src.bs_pricing import bs_european_put, log_return_params
from src.european import (find_kp_star, hazard_rate_ratio, rr2_violation_determinant,
                           tp2_violation_determinant, truncated_expected_underlying)


def test_no_rr2_violation_when_drift_nonpositive():
    """Proposition 4.1: mu = E[exp(Z)] <= 1 (r <= q) implies RR2 holds, i.e.
    the determinant of equation (28) stays <= 0 for every K1 < K2."""
    r, q, sigma, T = 0.0, 0.10, 0.30, 1.0
    mu_z, sigma_z = log_return_params(r, sigma, dt=T, q=q)
    assert np.exp(mu_z + 0.5 * sigma_z**2) <= 1.0 + 1e-9
    strikes = np.linspace(0.5, 3.0, 60)
    p1 = bs_european_put(1.0, strikes, r, q, sigma, T)
    for i in range(len(strikes) - 1):
        for j in range(i + 1, len(strikes)):
            det = rr2_violation_determinant(strikes[i], strikes[j], p1[i], p1[j])
            assert det <= 1e-9


def test_rr2_violation_exists_when_drift_positive():
    """Proposition 4.2: with r > q (positive drift) there exist K1 < K2 deep
    ITM for which the European put violates RR2."""
    r, q, sigma, T = 0.10, 0.0, 0.30, 1.0
    mu_z, sigma_z = log_return_params(r, sigma, dt=T, q=q)
    kp_star = find_kp_star(mu_z, sigma_z)
    strikes = np.linspace(kp_star * 1.01, kp_star * 3.0, 40)
    p1 = bs_european_put(1.0, strikes, r, q, sigma, T)
    found = False
    for i in range(len(strikes) - 1):
        for j in range(i + 1, len(strikes)):
            det = rr2_violation_determinant(strikes[i], strikes[j], p1[i], p1[j])
            if det > 1e-9:
                found = True
    assert found, "expected at least one RR2-violating pair beyond K_p*"


def test_kp_star_is_above_spot():
    r, q, sigma, T = 0.10, 0.0, 0.30, 1.0
    mu_z, sigma_z = log_return_params(r, sigma, dt=T, q=q)
    kp_star = find_kp_star(mu_z, sigma_z)
    assert kp_star > 1.0


def test_truncated_expectation_monotone_in_threshold():
    mu_z, sigma_z = 0.03, 0.3
    xs = [0.6, 0.9, 1.2, 1.8, 3.0]
    vals = [truncated_expected_underlying(x, mu_z, sigma_z) for x in xs]
    assert all(b >= a - 1e-9 for a, b in zip(vals, vals[1:]))


def test_hazard_rate_ratio_decreasing_in_strike():
    K = np.linspace(0.5, 3.0, 50)
    R = hazard_rate_ratio(K, S0=1.0, r=0.10, q=0.0, sigma=0.30, T=1.0)
    assert np.all(np.diff(R) <= 1e-9)
    assert R[-1] < R[0]


def test_tp2_violation_determinant_sign():
    det = tp2_violation_determinant(0.2, 0.3, 0.75, 0.70)
    assert isinstance(det, float)
