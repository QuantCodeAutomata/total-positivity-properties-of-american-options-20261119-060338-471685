"""Unit tests for src/options_violations.py (Section 7 methodology).

All tests use small synthetic fixtures or hand-built panels with a known
violation structure; no market data is downloaded here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.options_violations import (
    annual_violation_rates,
    clean_option_panel,
    enforce_single_maturity_monotonicity,
    ex_dividend_safe_subset,
    mid_price,
    short_term_subset,
    strategy_profit,
    tp2_rr2_violation_rate,
    trading_strategy_table,
    violation_survival_rates,
)


def _base_row(**overrides) -> dict:
    row = {
        "date": pd.Timestamp("2022-01-03"),
        "underlying": "SYN",
        "expiration": pd.Timestamp("2022-02-03"),
        "dte": 31,
        "strike": 100.0,
        "option_type": "call",
        "bid": 4.9,
        "ask": 5.1,
        "volume": 10,
        "delta": 0.5,
        "implied_volatility": 0.2,
    }
    row.update(overrides)
    return row


def test_clean_option_panel_removes_bad_rows_and_records_counts():
    rows = [
        _base_row(),  # kept
        _base_row(delta=np.nan),  # missing delta -> dropped
        _base_row(volume=0),  # zero volume -> dropped
        _base_row(bid=0.0),  # zero bid -> dropped
        _base_row(delta=0.005),  # |delta| below floor -> dropped
        _base_row(dte=0),  # expiration day -> dropped
    ]
    panel = pd.DataFrame(rows)
    cleaned = clean_option_panel(panel, delta_min=0.01, delta_max=0.99)
    assert cleaned.attrs["n_before_cleaning"] == 6
    assert cleaned.attrs["n_after_cleaning"] == 1
    assert len(cleaned) == 1


def test_clean_option_panel_requires_schema():
    with pytest.raises(ValueError):
        clean_option_panel(pd.DataFrame({"date": [1]}), 0.01, 0.99)


def test_mid_price_is_average_of_bid_ask():
    df = pd.DataFrame({"bid": [1.0, 2.0], "ask": [1.2, 2.4]})
    mid = mid_price(df)
    assert np.allclose(mid.to_numpy(), [1.1, 2.2])


def test_enforce_single_maturity_monotonicity_drops_bad_group_above_tolerance():
    # A single (date, expiration, call) group where mid price increases with
    # strike (violates the Section 7.2 single-maturity monotonicity rule).
    rows = [
        _base_row(strike=90.0, bid=9.9, ask=10.1),
        _base_row(strike=100.0, bid=11.9, ask=12.1),  # higher strike, higher price: violation
        _base_row(strike=110.0, bid=1.9, ask=2.1),
    ]
    panel = pd.DataFrame(rows)
    out = enforce_single_maturity_monotonicity(panel, tolerance=0.0)
    assert out.attrs["monotonicity_violation_fraction"] > 0
    assert len(out) == 0  # the whole offending group is removed


def test_enforce_single_maturity_monotonicity_keeps_group_within_tolerance():
    rows = [
        _base_row(strike=90.0, bid=9.9, ask=10.1),
        _base_row(strike=100.0, bid=11.9, ask=12.1),  # one violating pair
        _base_row(strike=110.0, bid=1.9, ask=2.1),
    ]
    panel = pd.DataFrame(rows)
    out = enforce_single_maturity_monotonicity(panel, tolerance=1.0)
    assert len(out) == 3  # tolerance covers the violation, nothing removed


def test_tp2_violation_detected_for_calls():
    # One date, two expirations, two strikes: construct prices with a known
    # TP2 violation C(K1,T1) C(K2,T2) < C(K1,T2) C(K2,T1).
    date = pd.Timestamp("2022-01-03")
    t1, t2 = pd.Timestamp("2022-01-10"), pd.Timestamp("2022-02-10")
    rows = [
        _base_row(date=date, expiration=t1, strike=90.0, bid=4.95, ask=5.05),   # C(K1,T1)=5
        _base_row(date=date, expiration=t1, strike=100.0, bid=1.95, ask=2.05),  # C(K2,T1)=2
        _base_row(date=date, expiration=t2, strike=90.0, bid=5.95, ask=6.05),   # C(K1,T2)=6
        _base_row(date=date, expiration=t2, strike=100.0, bid=0.95, ask=1.05),  # C(K2,T2)=1
    ]
    panel = pd.DataFrame(rows)
    result = tp2_rr2_violation_rate(panel, "call")
    # 5*1=5 < 6*2=12 -> violation
    assert result["n_tuples"] == 1
    assert result["n_violations"] == 1
    assert result["violation_rate"] == 1.0


def test_tp2_no_violation_for_calls_when_consistent():
    date = pd.Timestamp("2022-01-03")
    t1, t2 = pd.Timestamp("2022-01-10"), pd.Timestamp("2022-02-10")
    rows = [
        _base_row(date=date, expiration=t1, strike=90.0, bid=4.95, ask=5.05),   # C(K1,T1)=5
        _base_row(date=date, expiration=t1, strike=100.0, bid=1.95, ask=2.05),  # C(K2,T1)=2
        _base_row(date=date, expiration=t2, strike=90.0, bid=5.95, ask=6.05),   # C(K1,T2)=6
        _base_row(date=date, expiration=t2, strike=100.0, bid=2.95, ask=3.05),  # C(K2,T2)=3
    ]
    panel = pd.DataFrame(rows)
    result = tp2_rr2_violation_rate(panel, "call")
    # 5*3=15 >= 6*2=12 -> no violation
    assert result["n_violations"] == 0
    assert result["violation_rate"] == 0.0


def test_rr2_violation_detected_for_puts():
    date = pd.Timestamp("2022-01-03")
    t1, t2 = pd.Timestamp("2022-01-10"), pd.Timestamp("2022-02-10")
    rows = [
        _base_row(date=date, expiration=t1, strike=90.0, bid=0.95, ask=1.05, option_type="put"),   # P(K1,T1)=1
        _base_row(date=date, expiration=t1, strike=100.0, bid=1.95, ask=2.05, option_type="put"),   # P(K2,T1)=2
        _base_row(date=date, expiration=t2, strike=90.0, bid=1.95, ask=2.05, option_type="put"),   # P(K1,T2)=2
        _base_row(date=date, expiration=t2, strike=100.0, bid=5.95, ask=6.05, option_type="put"),  # P(K2,T2)=6
    ]
    panel = pd.DataFrame(rows)
    result = tp2_rr2_violation_rate(panel, "put")
    # P11*P22 = 1*6 = 6 > P12*P21 = 2*2 = 4 -> RR2 violation
    assert result["n_violations"] == 1
    assert result["violation_rate"] == 1.0


def test_bid_ask_adjusted_rate_is_never_above_mid_rate_on_synthetic_panel(synthetic_option_panel):
    cleaned = clean_option_panel(synthetic_option_panel, delta_min=0.01, delta_max=0.99)
    cleaned = enforce_single_maturity_monotonicity(cleaned, tolerance=0.005)
    for option_type in ("call", "put"):
        mid_rate = tp2_rr2_violation_rate(cleaned, option_type)["violation_rate"]
        ba_rate = tp2_rr2_violation_rate(cleaned, option_type, use_bid_ask=True)["violation_rate"]
        # the bid-ask test is strictly more stringent (Section 7.3)
        assert ba_rate <= mid_rate + 1e-9


def test_strategy_profit_matches_normalized_formula():
    c11, c12, c21, c22 = 5.0, 6.0, 2.0, 1.0  # the TP2-violating example above
    profit = strategy_profit(c11, c12, c21, c22, "T1", "call")
    expected = (c12 * c21 - c11 * c22) / (c11 * c22 + c12 * c21)
    assert profit == pytest.approx(expected)
    assert profit > 0  # inception cash is positive at a violation by construction


def test_strategy_profit_nan_when_normalizer_nonpositive():
    assert np.isnan(strategy_profit(0.0, 0.0, 0.0, 0.0, "T1", "call"))


def test_trading_strategy_table_has_four_denominations_and_positive_hit_rate():
    date = pd.Timestamp("2022-01-03")
    t1, t2 = pd.Timestamp("2022-01-10"), pd.Timestamp("2022-02-10")
    rows = [
        _base_row(date=date, expiration=t1, strike=90.0, bid=4.95, ask=5.05),
        _base_row(date=date, expiration=t1, strike=100.0, bid=1.95, ask=2.05),
        _base_row(date=date, expiration=t2, strike=90.0, bid=5.95, ask=6.05),
        _base_row(date=date, expiration=t2, strike=100.0, bid=0.95, ask=1.05),
    ]
    panel = pd.DataFrame(rows)
    table = trading_strategy_table(panel, "call")
    assert set(table["denomination"]) == {"T1", "T2", "K1", "K2"}
    assert (table["n"] == 1).all()
    assert (table["hit_rate"] >= 0.0).all() and (table["hit_rate"] <= 1.0).all()


def test_violation_survival_rates_tracks_persistent_and_corrected_violations():
    # Two trading days: a call 4-tuple violates on day 0 and keeps violating
    # on day 1, then corrects (becomes non-violating) on day 2.
    dates = pd.bdate_range("2022-01-03", periods=3)
    t1 = dates[0] + pd.Timedelta(days=3)
    t2 = dates[0] + pd.Timedelta(days=20)
    rows = []
    # day0, day1: violating (5*1 < 6*2); day2: corrected (5*3 >= 6*2)
    prices_by_day = [
        {(90.0, t1): 5.0, (100.0, t1): 2.0, (90.0, t2): 6.0, (100.0, t2): 1.0},
        {(90.0, t1): 5.0, (100.0, t1): 2.0, (90.0, t2): 6.0, (100.0, t2): 1.0},
        {(90.0, t1): 5.0, (100.0, t1): 2.0, (90.0, t2): 6.0, (100.0, t2): 3.0},
    ]
    for date, prices in zip(dates, prices_by_day):
        for (strike, expiration), price in prices.items():
            dte = (expiration - date).days
            rows.append(_base_row(date=date, expiration=expiration, strike=strike,
                                   dte=dte, bid=price - 0.01, ask=price + 0.01))
    panel = pd.DataFrame(rows)
    table = violation_survival_rates(panel, "call", t1_grid=[3], t2_max_calendar_days=30, horizon=2)
    row = table.iloc[0]
    assert row["T1_trading_days"] == 3
    assert row["n_violations_tracked"] == 1
    # still violating one day later (day1), corrected by day2 (two days later)
    assert row["T+1"] == pytest.approx(100.0)
    assert row["T+2"] == pytest.approx(0.0)


def test_ex_dividend_safe_subset_filters_month_and_dte():
    rows = [
        _base_row(date=pd.Timestamp("2022-01-15"), dte=20),   # safe month, short DTE -> kept
        _base_row(date=pd.Timestamp("2022-02-15"), dte=20),   # unsafe month -> dropped
        _base_row(date=pd.Timestamp("2022-01-15"), dte=60),   # safe month, long DTE -> dropped
    ]
    panel = pd.DataFrame(rows)
    out = ex_dividend_safe_subset(panel, safe_months=(1, 4, 7, 10), dte_max=30)
    assert len(out) == 1
    assert out.iloc[0]["date"] == pd.Timestamp("2022-01-15")


def test_short_term_subset_filters_dte_only():
    rows = [_base_row(dte=10), _base_row(dte=40)]
    panel = pd.DataFrame(rows)
    out = short_term_subset(panel, dte_max=30)
    assert len(out) == 1
    assert out.iloc[0]["dte"] == 10


def test_annual_violation_rates_groups_by_year():
    date = pd.Timestamp("2022-01-03")
    t1, t2 = pd.Timestamp("2022-01-10"), pd.Timestamp("2022-02-10")
    rows = [
        _base_row(date=date, expiration=t1, strike=90.0, bid=4.95, ask=5.05),
        _base_row(date=date, expiration=t1, strike=100.0, bid=1.95, ask=2.05),
        _base_row(date=date, expiration=t2, strike=90.0, bid=5.95, ask=6.05),
        _base_row(date=date, expiration=t2, strike=100.0, bid=0.95, ask=1.05),
        _base_row(date=pd.Timestamp("2023-01-03"), expiration=pd.Timestamp("2023-01-10"),
                   strike=90.0, bid=4.95, ask=5.05),
    ]
    panel = pd.DataFrame(rows)
    rates = annual_violation_rates(panel, "call")
    assert list(rates.index) == [2022, 2023]
    assert rates.loc[2022] == pytest.approx(1.0)
    assert np.isnan(rates.loc[2023])  # a single quote has no pair to test
