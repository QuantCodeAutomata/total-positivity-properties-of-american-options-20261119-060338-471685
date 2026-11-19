"""Experiment 5: empirical TP2/RR2 violation rates for American options on
SPY and four individual stocks (Section 7.1-7.3, Table 7.1).

For each ticker this script attempts to load the OptionMetrics-equivalent
end-of-day option panel (bid, ask, delta, implied volatility, volume, one
row per listed contract per trading day) over the sample period of Section
7.1, applies the five cleaning filters of Section 7.2, enforces the
single-maturity monotonicity check, and computes the overall and bid-ask
adjusted TP2 (calls) / RR2 (puts) violation rates of Section 7.3 (Table
7.1). If the panel cannot be loaded for a ticker, the failure is recorded
instead of a fabricated rate; see ``data/data_loader.py`` for the data
source actually available and the diagnosed cause.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from data.data_loader import OptionsDataUnavailable, load_option_eod_panel
from src import config
from src.options_violations import (
    clean_option_panel,
    enforce_single_maturity_monotonicity,
    tp2_rr2_violation_rate,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Table 7.1 of the paper: reported violation rates, kept here only as the
# comparison target recorded in the results file, never computed from it.
PAPER_TABLE_7_1 = {
    "SPY": {"call_rate_pct": 0.804, "put_rate_pct": 1.965, "ba_call_rate_pct": 0.038, "ba_put_rate_pct": 0.348},
    "AMZN": {"call_rate_pct": 1.855, "put_rate_pct": 2.150},
    "GOOGL": {"call_rate_pct": 1.614, "put_rate_pct": 2.244},
    "META": {"call_rate_pct": 0.748, "put_rate_pct": 1.157},
    "TSLA": {"call_rate_pct": 0.710, "put_rate_pct": 2.289},
}


def run_ticker(ticker: str, start: str, end: str) -> Dict:
    """Load, clean, and compute violation rates for one ticker; a failed
    load is reported with its own error message."""
    try:
        panel = load_option_eod_panel(ticker, start, end)
    except OptionsDataUnavailable as exc:
        return {"ticker": ticker, "status": "data_unavailable", "error": str(exc)}

    cleaned = clean_option_panel(panel, config.EXP5_DELTA_MIN, config.EXP5_DELTA_MAX)
    cleaned = enforce_single_maturity_monotonicity(cleaned, config.EXP5_MONOTONICITY_VIOLATION_TOLERANCE)
    call = tp2_rr2_violation_rate(cleaned, "call")
    put = tp2_rr2_violation_rate(cleaned, "put")
    ba_call = tp2_rr2_violation_rate(cleaned, "call", use_bid_ask=True)
    ba_put = tp2_rr2_violation_rate(cleaned, "put", use_bid_ask=True)
    return {
        "ticker": ticker,
        "status": "ok",
        "n_before_cleaning": cleaned.attrs.get("n_before_cleaning"),
        "n_after_cleaning": cleaned.attrs.get("n_after_cleaning"),
        "monotonicity_violation_fraction": cleaned.attrs.get("monotonicity_violation_fraction"),
        "call": call,
        "put": put,
        "bid_ask_call": ba_call,
        "bid_ask_put": ba_put,
    }


def main() -> Dict:
    results = {}
    for ticker, (start, end) in config.EXP_EMPIRICAL_TICKERS.items():
        print(f"exp_5: attempting end-of-day option panel for {ticker} ({start}..{end})")
        results[ticker] = run_ticker(ticker, start, end)
        print(f"exp_5: {ticker} -> {results[ticker]['status']}")

    any_ok = any(r["status"] == "ok" for r in results.values())
    output = {
        "experiment": "exp_5",
        "description": "TP2 (calls) / RR2 (puts) violation rates, Section 7.1-7.3, Table 7.1",
        "paper_table_7_1": PAPER_TABLE_7_1,
        "results": results,
        "conclusion": "reproduced" if all(r["status"] == "ok" for r in results.values())
        else ("partially_reproduced" if any_ok else "not_reproduced_data_unavailable"),
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "exp5_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"exp_5: wrote {out_path}")
    return output


if __name__ == "__main__":
    main()
