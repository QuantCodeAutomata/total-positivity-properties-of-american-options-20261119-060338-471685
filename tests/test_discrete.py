"""Unit tests for src/discrete.py (Section 5: binomial/trinomial models)."""

import numpy as np
import pytest

from src.discrete import (binomial_pmf, h_values_binomial, is_log_concave_pmf, trinomial_pmf)


def test_trinomial_is_log_concave():
    pmf = trinomial_pmf(p0=2.0 / 3.0, c=0.05)
    assert is_log_concave_pmf(pmf)
    # explicit check from Section 5: p(-1) p(1) <= p(0)^2
    assert pmf[-1] * pmf[1] <= pmf[0] ** 2 + 1e-12


def test_binomial_is_not_log_concave():
    pmf, b = binomial_pmf(r=0.10, sigma=1.0, dt=1.0)
    assert not is_log_concave_pmf(pmf)
    assert pmf[0] == 0.0
    assert pmf[-1] * pmf[1] > pmf[0] ** 2  # 0 == 0 would not violate; check explicitly
    assert pmf[1] > 0 and pmf[-1] > 0  # both nonzero while center mass is zero


def test_binomial_rr2_violation_matches_paper_example():
    """Section 5 violation example: S0=1, r=0.1, sigma=1, dt=1,
    K1=0.4, K2=0.4e; expects H2(K1)H3(K2) - H2(K2)H3(K1) > 0 approx 0.0032."""
    K1, K2 = 0.4, 0.4 * np.e
    out = h_values_binomial(K1, K2, r=0.10, sigma=1.0, dt=1.0, n_steps=(2, 3))
    det = out["H2_K1"] * out["H3_K2"] - out["H2_K2"] * out["H3_K1"]
    assert det > 0, "expected an RR2 violation for the binomial example"
    assert det == pytest.approx(0.0032, abs=5e-3)


def test_trinomial_pmf_sums_to_one_and_rejects_bad_c():
    pmf = trinomial_pmf(p0=0.5, c=0.1)
    assert sum(pmf.values()) == pytest.approx(1.0)
    with pytest.raises(AssertionError):
        trinomial_pmf(p0=0.5, c=0.4)  # negative probability
