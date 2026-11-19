"""Shared pytest fixtures and marker registration."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "market_data: test needs a live Dataloader/Massive call (skipped without API keys)"
    )


@pytest.fixture(scope="session")
def has_api_keys() -> bool:
    return bool(os.environ.get("DATALOADER_API_KEY")) and bool(os.environ.get("MASSIVE_API_KEY"))


@pytest.fixture(scope="session")
def reduced_spy_bars(request: pytest.FixtureRequest, has_api_keys: bool) -> pd.DataFrame:
    """A short slice of SPY daily bars loaded through ``data/data_loader.py``,
    cached across test runs with pytest's own cache so it is downloaded once.
    """
    if not has_api_keys:
        pytest.skip("DATALOADER_API_KEY / MASSIVE_API_KEY not set")
    cached = request.config.cache.get("spy_bars_2023", None)
    if cached is not None:
        return pd.DataFrame(cached)
    from data.data_loader import load_ohlcv

    bars, _meta = load_ohlcv(["SPY"], "2023-01-01", "2023-03-31", timeframe="1d", use_cache=True)
    request.config.cache.set("spy_bars_2023", bars.astype(str).to_dict(orient="list"))
    return bars


def _make_option_panel(seed: int, n_dates: int, strikes: np.ndarray, expirations_days: list[int],
                        underlying: str = "SYN", drift: float = 0.0, sigma: float = 0.2,
                        s0: float = 100.0) -> pd.DataFrame:
    """Synthetic end-of-day option panel with the schema of
    ``src/options_violations.py``'s ``REQUIRED_COLUMNS``, built from
    Black-Scholes prices on a geometric-Brownian-motion path so that it has
    the sign and monotonicity structure of real option data without being
    real market data. Used only to test the methodology (src/) on fixtures,
    never to produce an experiment's reported numbers.
    """
    from scipy.stats import norm

    rng = np.random.default_rng(seed)
    rows = []
    dates = pd.bdate_range("2022-01-03", periods=n_dates)
    log_path = np.cumsum(rng.normal(drift, sigma / np.sqrt(252), size=n_dates))
    spot_path = s0 * np.exp(log_path)
    r, q = 0.02, 0.0
    for i, date in enumerate(dates):
        s = spot_path[i]
        for dte in expirations_days:
            expiration = date + pd.Timedelta(days=dte)
            T = dte / 365.0
            if T <= 0:
                continue
            for K in strikes:
                d1 = (np.log(s / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
                d2 = d1 - sigma * np.sqrt(T)
                call = s * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
                put = K * np.exp(-r * T) * norm.cdf(-d2) - s * np.exp(-q * T) * norm.cdf(-d1)
                call_delta = np.exp(-q * T) * norm.cdf(d1)
                put_delta = call_delta - 1.0
                spread = 0.01 * max(call, 0.01)
                rows.append({
                    "date": date, "underlying": underlying, "expiration": expiration,
                    "dte": dte, "strike": K, "option_type": "call",
                    "bid": max(call - spread, 0.0), "ask": call + spread,
                    "volume": 100, "delta": call_delta, "implied_volatility": sigma,
                })
                spread_p = 0.01 * max(put, 0.01)
                rows.append({
                    "date": date, "underlying": underlying, "expiration": expiration,
                    "dte": dte, "strike": K, "option_type": "put",
                    "bid": max(put - spread_p, 0.0), "ask": put + spread_p,
                    "volume": 100, "delta": put_delta, "implied_volatility": sigma,
                })
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_option_panel() -> pd.DataFrame:
    """A small, well-behaved synthetic panel (no TP2/RR2 violations expected
    in distribution) for exercising the cleaning and violation-rate pipeline."""
    strikes = np.arange(80, 121, 5.0)
    return _make_option_panel(seed=0, n_dates=10, strikes=strikes, expirations_days=[10, 20, 40])
