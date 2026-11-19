"""Unit tests for src/bs_pricing.py (Sections 2.2, 3, 4)."""

import numpy as np
import pytest

from src.bs_pricing import (bs_european_call, bs_european_put, is_log_concave_gaussian,
                             log_return_params)


def test_log_concave_gaussian_is_true():
    assert is_log_concave_gaussian() is True


def test_log_return_params_matches_bs_drift():
    mean, std = log_return_params(r=0.1, sigma=0.3, dt=1.0, q=0.0)
    assert mean == pytest.approx(0.1 - 0.5 * 0.3**2)
    assert std == pytest.approx(0.3)


def test_log_return_params_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        log_return_params(r=0.1, sigma=-0.1, dt=1.0)
    with pytest.raises(ValueError):
        log_return_params(r=0.1, sigma=0.3, dt=0.0)


def test_put_call_parity():
    S0, r, q, sigma, T = 1.0, 0.1, 0.0, 0.3, 1.0
    K = np.array([0.8, 1.0, 1.2])
    put = bs_european_put(S0, K, r, q, sigma, T)
    call = bs_european_call(S0, K, r, q, sigma, T)
    lhs = call - put
    rhs = S0 * np.exp(-q * T) - K * np.exp(-r * T)
    np.testing.assert_allclose(lhs, rhs, atol=1e-10)


def test_prices_are_nonnegative():
    S0, r, q, sigma, T = 1.0, 0.1, 0.0, 0.3, 1.0
    K = np.linspace(0.2, 5.0, 25)
    assert np.all(bs_european_put(S0, K, r, q, sigma, T) >= -1e-10)
    assert np.all(bs_european_call(S0, K, r, q, sigma, T) >= -1e-10)
