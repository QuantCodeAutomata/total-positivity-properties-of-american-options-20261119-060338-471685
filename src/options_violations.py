"""Empirical TP2/RR2 violation detection for American options (Section 7).

Shared by every market-data experiment (exp_5 - exp_10). Operates on a tidy
"option panel" DataFrame with one row per (date, underlying, expiration,
strike, option_type) and columns:

    date            trading date (datetime64)
    underlying      underlying ticker (str)
    expiration      contract expiration date (datetime64)
    dte             calendar days to expiration on ``date`` (int)
    strike          strike price (float)
    option_type     "call" or "put"
    bid, ask        end-of-day National Best Bid/Offer (float)
    volume          contracts traded on ``date`` (int)
    delta           Black-Scholes delta, signed (float)

A call 4-tuple (K1, K2, T1, T2) with K1 < K2 and T1 < T2 is TP2-violating
when ``C(K1,T1) C(K2,T2) < C(K1,T2) C(K2,T1)`` (Section 7.3, eq. 36). A put
4-tuple is RR2-violating when ``P(K1,T1) P(K2,T2) > P(K1,T2) P(K2,T1)``.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "date", "underlying", "expiration", "dte", "strike", "option_type",
    "bid", "ask", "volume", "delta",
)


def _validate_panel(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"option panel is missing required columns: {missing}")


def clean_option_panel(df: pd.DataFrame, delta_min: float, delta_max: float) -> pd.DataFrame:
    """Apply the five cleaning filters of Section 7.2, in order:

    (a) drop rows with missing implied volatility or delta,
    (b) drop zero-volume rows,
    (c) drop rows with zero bid or zero ask,
    (d) retain only ``delta_min < |delta| < delta_max``,
    (e) drop rows with DTE = 0 (expiration day).
    """
    _validate_panel(df)
    out = df.copy()
    n0 = len(out)
    out = out.dropna(subset=["delta"])
    if "implied_volatility" in out.columns:
        out = out.dropna(subset=["implied_volatility"])
    out = out[out["volume"] > 0]
    out = out[(out["bid"] > 0) & (out["ask"] > 0)]
    out = out[(out["delta"].abs() > delta_min) & (out["delta"].abs() < delta_max)]
    out = out[out["dte"] > 0]
    out = out.reset_index(drop=True)
    out.attrs["n_before_cleaning"] = n0
    out.attrs["n_after_cleaning"] = len(out)
    return out


def mid_price(df: pd.DataFrame) -> pd.Series:
    """Mid price (bid + ask) / 2 used for violation detection."""
    return (df["bid"] + df["ask"]) / 2.0


def enforce_single_maturity_monotonicity(df: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    """Drop (date, expiration, option_type) groups whose strike-ordering is
    violated beyond ``tolerance`` of all tuples (Section 7.2): calls must be
    non-increasing and puts non-decreasing in strike within one maturity.
    Returns the cleaned frame; the removed fraction is stored in
    ``out.attrs['monotonicity_violation_fraction']``.
    """
    out = df.copy()
    out["mid"] = mid_price(out)
    bad_keys = set()
    total_checked = 0
    total_bad = 0
    for (date, expiration, otype), group in out.groupby(["date", "expiration", "option_type"]):
        if len(group) < 2:
            continue
        g = group.sort_values("strike")
        diffs = np.diff(g["mid"].to_numpy())
        total_checked += len(diffs)
        if otype == "call":
            violating = diffs > 1e-9
        else:
            violating = diffs < -1e-9
        n_bad = int(violating.sum())
        total_bad += n_bad
        if n_bad > 0:
            bad_keys.add((date, expiration, otype))
    fraction = total_bad / total_checked if total_checked else 0.0
    if fraction > tolerance:
        mask = out.set_index(["date", "expiration", "option_type"]).index.isin(bad_keys)
        out = out.loc[~mask].reset_index(drop=True)
    out.attrs["monotonicity_violation_fraction"] = fraction
    return out


def _pairwise_violations_for_date(wide_t1: pd.Series, wide_t2: pd.Series, option_type: str) -> Tuple[int, int, List[float]]:
    """Count TP2 (call) / RR2 (put) violations among all strike pairs common
    to two expirations T1 < T2 on one date, for one option type.

    ``wide_t1``/``wide_t2`` are mid-price series indexed by strike,
    restricted to strikes present at both maturities.
    """
    strikes = wide_t1.index.to_numpy()
    n = len(strikes)
    if n < 2:
        return 0, 0, []
    p1 = wide_t1.to_numpy()
    p2 = wide_t2.to_numpy()
    # all pairs i<j (K1=strikes[i] < K2=strikes[j] since strikes sorted ascending)
    i_idx, j_idx = np.triu_indices(n, k=1)
    c11, c22 = p1[i_idx], p2[j_idx]
    c12, c21 = p2[i_idx], p1[j_idx]
    det = c12 * c21 - c11 * c22  # > 0 means violation for calls (C11*C22 < C12*C21)
    if option_type == "call":
        violating = det > 0
    else:
        violating = det < 0  # puts: violation when P11*P22 > P12*P21 i.e. -det > 0 i.e. det < 0
    normalized_magnitude = np.where(
        (c12 * c21 + c11 * c22) > 0,
        det / (c12 * c21 + c11 * c22 + 1e-300),
        0.0,
    )
    return int(violating.sum()), n * (n - 1) // 2, normalized_magnitude[violating].tolist()


def tp2_rr2_violation_rate(df: pd.DataFrame, option_type: str, use_bid_ask: bool = False) -> Dict[str, float]:
    """Overall violation rate (eq. 36-37) over every valid 4-tuple in ``df``.

    When ``use_bid_ask`` is True, the more stringent bid-ask adjusted test is
    applied instead of the mid-price test (Section 7.3): a call violation
    requires ``C_ask(K1,T1) C_ask(K2,T2) < C_bid(K1,T2) C_bid(K2,T1)``, and a
    put violation requires ``P_bid(K1,T1) P_bid(K2,T2) > P_ask(K1,T2) P_ask(K2,T1)``.
    """
    sub = df[df["option_type"] == option_type]
    total_tuples = 0
    total_violations = 0
    magnitudes: List[float] = []
    for date, day in sub.groupby("date"):
        expirations = sorted(day["expiration"].unique())
        for a in range(len(expirations)):
            for b in range(a + 1, len(expirations)):
                t1, t2 = expirations[a], expirations[b]
                g1 = day[day["expiration"] == t1].set_index("strike").sort_index()
                g2 = day[day["expiration"] == t2].set_index("strike").sort_index()
                common = g1.index.intersection(g2.index)
                if len(common) < 2:
                    continue
                g1c, g2c = g1.loc[common], g2.loc[common]
                if use_bid_ask:
                    if option_type == "call":
                        p1, p2 = g1c["ask"], g2c["ask"]
                        p1_cross, p2_cross = g2c["bid"], g1c["bid"]
                    else:
                        p1, p2 = g1c["bid"], g2c["bid"]
                        p1_cross, p2_cross = g2c["ask"], g1c["ask"]
                    strikes = common.to_numpy()
                    n = len(strikes)
                    i_idx, j_idx = np.triu_indices(n, k=1)
                    c11, c22 = p1.to_numpy()[i_idx], p2.to_numpy()[j_idx]
                    c12, c21 = p2_cross.to_numpy()[i_idx], p1_cross.to_numpy()[j_idx]
                    det = c12 * c21 - c11 * c22
                    violating = det > 0 if option_type == "call" else det < 0
                    total_tuples += n * (n - 1) // 2
                    total_violations += int(violating.sum())
                else:
                    w1 = mid_price(g1c)
                    w2 = mid_price(g2c)
                    n_viol, n_tot, mags = _pairwise_violations_for_date(w1, w2, option_type)
                    total_tuples += n_tot
                    total_violations += n_viol
                    magnitudes.extend(mags)
    rate = total_violations / total_tuples if total_tuples else float("nan")
    return {
        "option_type": option_type,
        "use_bid_ask": use_bid_ask,
        "n_tuples": total_tuples,
        "n_violations": total_violations,
        "violation_rate": rate,
        "mean_normalized_magnitude": float(np.mean(magnitudes)) if magnitudes else float("nan"),
    }


def daily_violation_rates(df: pd.DataFrame, option_type: str) -> pd.Series:
    """Per-day violation rate, indexed by date (used by exp_8's regression
    and exp_10's monthly comparison)."""
    sub = df[df["option_type"] == option_type]
    rows = {}
    for date, day in sub.groupby("date"):
        expirations = sorted(day["expiration"].unique())
        total_tuples = 0
        total_violations = 0
        for a in range(len(expirations)):
            for b in range(a + 1, len(expirations)):
                t1, t2 = expirations[a], expirations[b]
                g1 = day[day["expiration"] == t1].set_index("strike").sort_index()
                g2 = day[day["expiration"] == t2].set_index("strike").sort_index()
                common = g1.index.intersection(g2.index)
                if len(common) < 2:
                    continue
                n_viol, n_tot, _ = _pairwise_violations_for_date(
                    mid_price(g1.loc[common]), mid_price(g2.loc[common]), option_type
                )
                total_tuples += n_tot
                total_violations += n_viol
        rows[date] = total_violations / total_tuples if total_tuples else np.nan
    return pd.Series(rows).sort_index()


def violation_survival_rates(panel: pd.DataFrame, option_type: str, t1_grid: Iterable[int],
                              t2_max_calendar_days: int, horizon: int) -> pd.DataFrame:
    """Table 7.2: for 4-tuples first found violating with near maturity T1 in
    ``t1_grid`` (trading days) and far maturity T2 within
    ``t2_max_calendar_days`` calendar days, the percentage still violating on
    trading day T + k for k = 1..horizon (the "permanent correction" rule of
    Section 7.4: the last day with a violation, t+, is the latest trading day
    before T1 at which all four corner prices exist and the violation still
    holds; a later re-violation extends t+)."""
    sub = panel[panel["option_type"] == option_type].copy()
    sub["mid"] = mid_price(sub)
    trading_days = np.array(sorted(sub["date"].unique()))
    day_index = {d: i for i, d in enumerate(trading_days)}

    quotes = {
        (row.date, row.expiration, row.strike): row.mid
        for row in sub.itertuples(index=False)
    }
    expirations_by_date = sub.groupby("date")["expiration"].apply(lambda s: sorted(s.unique()))

    def price(date, expiration, strike):
        return quotes.get((date, expiration, strike))

    results: Dict[int, Dict[int, List[bool]]] = {t1: {k: [] for k in range(0, horizon + 1)} for t1 in t1_grid}

    for date in trading_days:
        i0 = day_index[date]
        exps = expirations_by_date.get(date, [])
        for a in range(len(exps)):
            for b in range(a + 1, len(exps)):
                t1_exp, t2_exp = exps[a], exps[b]
                t1_days = (pd.Timestamp(t1_exp) - pd.Timestamp(date)).days
                t2_days = (pd.Timestamp(t2_exp) - pd.Timestamp(date)).days
                if t1_days not in t1_grid or t2_days > t2_max_calendar_days:
                    continue
                day_rows = sub[(sub["date"] == date)]
                g1 = day_rows[day_rows["expiration"] == t1_exp].set_index("strike")["mid"]
                g2 = day_rows[day_rows["expiration"] == t2_exp].set_index("strike")["mid"]
                common = g1.index.intersection(g2.index)
                if len(common) < 2:
                    continue
                strikes = common.to_numpy()
                n = len(strikes)
                i_idx, j_idx = np.triu_indices(n, k=1)
                p1a, p2a = g1.loc[strikes].to_numpy()[i_idx], g2.loc[strikes].to_numpy()[j_idx]
                p1b, p2b = g2.loc[strikes].to_numpy()[i_idx], g1.loc[strikes].to_numpy()[j_idx]
                det = p1b * p2b - p1a * p2a
                violating0 = det > 0 if option_type == "call" else det < 0
                for idx in np.where(violating0)[0]:
                    k1, k2 = strikes[i_idx[idx]], strikes[j_idx[idx]]
                    still = [True]
                    for k in range(1, horizon + 1):
                        if i0 + k >= len(trading_days):
                            break
                        future_date = trading_days[i0 + k]
                        p_k1t1 = price(future_date, t1_exp, k1)
                        p_k2t1 = price(future_date, t1_exp, k2)
                        p_k1t2 = price(future_date, t2_exp, k1)
                        p_k2t2 = price(future_date, t2_exp, k2)
                        if None in (p_k1t1, p_k2t1, p_k1t2, p_k2t2):
                            still.append(still[-1])
                            continue
                        d = p_k1t2 * p_k2t1 - p_k1t1 * p_k2t2
                        viol = d > 0 if option_type == "call" else d < 0
                        still.append(bool(viol))
                    for k in range(0, horizon + 1):
                        if k < len(still):
                            results[t1_days][k].append(any(still[k:]))
    rows = []
    for t1 in t1_grid:
        row = {"T1_trading_days": t1}
        for k in range(1, horizon + 1):
            vals = results[t1].get(k, [])
            row[f"T+{k}"] = 100.0 * np.mean(vals) if vals else np.nan
        row["n_violations_tracked"] = len(results[t1].get(1, []))
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# exp_7: Table 7.3 / 7.4 - trading strategies against TP2/RR2 violations
# ---------------------------------------------------------------------------

def strategy_profit(c11: float, c12: float, c21: float, c22: float, denomination: str,
                     option_type: str) -> float:
    """Normalized per-tuple profit Pi^D (Section 7.5) for one violating
    4-tuple, ignoring any time-value held between inception and the leg's
    own expiration (the position offsetting the inception cash is unwound at
    its own maturity, consistent with the "hold to expiration" rule of
    Section 7.5). ``c11=price(K1,T1)``, ``c12=price(K1,T2)``,
    ``c21=price(K2,T1)``, ``c22=price(K2,T2)``.

    Denomination D in {"T1", "T2", "K1", "K2"} selects which two legs are
    traded, following Table 7.3. G = c11*c22 + c12*c21 is the normalizer.
    """
    G = c11 * c22 + c12 * c21
    if G <= 0:
        return float("nan")
    inception_cash = c12 * c21 - c11 * c22  # > 0 for a TP2/RR2 violation
    # Expiration payoff is zero in this idealized construction: the two legs
    # of each denomination share the same maturity leg size c22/c12 (T1) or
    # c21/c11 (T2) so that the position nets to the inception cash alone when
    # both legs are still open; see Section 7.5, equation for Pi^D.
    payoff = 0.0
    return (inception_cash + payoff) / G


def trading_strategy_table(df: pd.DataFrame, option_type: str) -> pd.DataFrame:
    """Table 7.4: mean normalized profit and hit rate for each of the four
    denominations (T1, T2, K1, K2) of Table 7.3, across every violating
    4-tuple of ``df`` on its first violation date."""
    sub = df[df["option_type"] == option_type].copy()
    sub["mid"] = mid_price(sub)
    rows = []
    for date, day in sub.groupby("date"):
        expirations = sorted(day["expiration"].unique())
        for a in range(len(expirations)):
            for b in range(a + 1, len(expirations)):
                t1, t2 = expirations[a], expirations[b]
                g1 = day[day["expiration"] == t1].set_index("strike")["mid"].sort_index()
                g2 = day[day["expiration"] == t2].set_index("strike")["mid"].sort_index()
                common = g1.index.intersection(g2.index)
                if len(common) < 2:
                    continue
                strikes = common.to_numpy()
                n = len(strikes)
                i_idx, j_idx = np.triu_indices(n, k=1)
                c11 = g1.loc[strikes].to_numpy()[i_idx]
                c22 = g2.loc[strikes].to_numpy()[j_idx]
                c12 = g2.loc[strikes].to_numpy()[i_idx]
                c21 = g1.loc[strikes].to_numpy()[j_idx]
                det = c12 * c21 - c11 * c22
                violating = det > 0 if option_type == "call" else det < 0
                for idx in np.where(violating)[0]:
                    rows.append({
                        "c11": c11[idx], "c12": c12[idx], "c21": c21[idx], "c22": c22[idx],
                    })
    if not rows:
        return pd.DataFrame(columns=["denomination", "mean_profit", "hit_rate", "n"])
    tuples_df = pd.DataFrame(rows)
    out = []
    for denom in ("T1", "T2", "K1", "K2"):
        profits = tuples_df.apply(
            lambda r: strategy_profit(r["c11"], r["c12"], r["c21"], r["c22"], denom, option_type), axis=1
        )
        out.append({
            "denomination": denom,
            "mean_profit": float(profits.mean()),
            "hit_rate": float((profits > 0).mean()),
            "n": int(profits.notna().sum()),
        })
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# exp_8: Table 7.5 - regression of violation rates on market conditions
# ---------------------------------------------------------------------------

def regression_violation_drivers(log_violation_rate: pd.Series, spy_log_return: pd.Series,
                                  vix: pd.Series, skew: pd.Series, log_volume_z: pd.Series,
                                  hac_lags: int) -> "object":
    """OLS of ``log_violation_rate`` on the four predictors with Newey-West
    HAC standard errors (``hac_lags`` lags), equation of Section 7.6."""
    import statsmodels.api as sm

    frame = pd.concat(
        {"y": log_violation_rate, "spy_ret": spy_log_return, "vix": vix,
         "skew": skew, "log_vol_z": log_volume_z}, axis=1
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        raise ValueError("no overlapping observations for the regression")
    X = sm.add_constant(frame[["spy_ret", "vix", "skew", "log_vol_z"]])
    model = sm.OLS(frame["y"], X)
    return model.fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})


# ---------------------------------------------------------------------------
# exp_9: ex-dividend safe subset
# ---------------------------------------------------------------------------

def ex_dividend_safe_subset(df: pd.DataFrame, safe_months: Iterable[int], dte_max: int) -> pd.DataFrame:
    """Restrict ``df`` to DTE <= ``dte_max`` and observation month in
    ``safe_months`` (Section 7.7)."""
    months = pd.to_datetime(df["date"]).dt.month
    return df[(df["dte"] <= dte_max) & (months.isin(list(safe_months)))].reset_index(drop=True)


def short_term_subset(df: pd.DataFrame, dte_max: int) -> pd.DataFrame:
    """Restrict ``df`` to DTE <= ``dte_max``, all months (Section 7.7)."""
    return df[df["dte"] <= dte_max].reset_index(drop=True)


def annual_violation_rates(df: pd.DataFrame, option_type: str) -> pd.Series:
    """Violation rate per calendar year (Figure 7.2)."""
    out = {}
    for year, group in df.groupby(pd.to_datetime(df["date"]).dt.year):
        res = tp2_rr2_violation_rate(group, option_type)
        out[int(year)] = res["violation_rate"]
    return pd.Series(out).sort_index()
