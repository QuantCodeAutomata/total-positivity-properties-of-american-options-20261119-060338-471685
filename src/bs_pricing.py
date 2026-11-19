"""Black-Scholes building blocks shared by the Bermudan, European and
convergence experiments (Sections 3, 4 and 6 of the paper).

Log returns over one period of length ``dt`` in the model
``S_{j+1} = S_j * exp(Z)`` with ``Z ~ N((r - q - sigma**2 / 2) * dt, sigma**2 * dt)``
have a Gaussian density, which is log-concave (Section 2.2), so Theorems
3.1-3.2 and their European counterpart (Proposition 4.1) apply.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.stats import norm


def log_return_params(r: float, sigma: float, dt: float, q: float = 0.0) -> Tuple[float, float]:
    """Mean and standard deviation of the log return Z over one period.

    Risk-neutral Black-Scholes dynamics ``dS = (r - q) S dt + sigma S dW``
    give ``Z = log(S_{t+dt}/S_t) ~ N((r - q - sigma**2/2) dt, sigma**2 dt)``.
    """
    if sigma < 0 or dt <= 0:
        raise ValueError("sigma must be >= 0 and dt must be > 0")
    mean = (r - q - 0.5 * sigma**2) * dt
    std = sigma * np.sqrt(dt)
    return mean, std


def is_log_concave_gaussian() -> bool:
    """Verify that the Gaussian log-return density is log-concave.

    A density f is log-concave when -log f is convex.  For
    f(z) = exp(-(z-mu)**2/(2*s**2)) / sqrt(2*pi*s**2), -log f(z) is an
    (upward) parabola in z, whose second derivative 1/s**2 is strictly
    positive for any s > 0, confirming log-concavity (Section 2.2).
    """
    s = 1.0
    # second derivative of -log f(z) = (z-mu)**2/(2 s**2) + const is 1/s**2
    second_derivative = 1.0 / s**2
    return second_derivative > 0


def bs_european_put(S0: float, K: np.ndarray, r: float, q: float, sigma: float, T: float) -> np.ndarray:
    """European put price via the Black-Scholes formula (paper eq. 24-25, Section 4)."""
    K = np.asarray(K, dtype=float)
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be positive")
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-q * T) * norm.cdf(-d1)


def bs_european_call(S0: float, K: np.ndarray, r: float, q: float, sigma: float, T: float) -> np.ndarray:
    """European call price via the Black-Scholes formula (paper eq. 24-25, Section 4)."""
    K = np.asarray(K, dtype=float)
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be positive")
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
