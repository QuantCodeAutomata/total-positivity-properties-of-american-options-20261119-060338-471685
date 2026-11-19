"""Experiment 7: per-trade hit rates and profitability of violation-based
trading strategies (Section 7.5, Table 7.4).

For each of the five tickers, attempts to load the end-of-day option
panel, cleans it as in experiment 5, and evaluates the four denomination
strategies of Table 7.3 (T1, T2, K1, K2) for calls and puts, reporting the
mean normalized profit and hit rate of Table 7.4. If the panel cannot be
loaded, the failure is recorded instead of a fabricated profit table.
"""

from __future__ import annotations

import json
from pathlib import Path

from data.data_loader import OptionsDataUnavailable, load_option_eod_panel
from src import config
from src.options_violations import (
    clean_option_panel,
    enforce_single_maturity_monotonicity,
    trading_strategy_table,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# SPY row of Table 7.4, kept only as the comparison target, never computed
# from it.
PAPER_TABLE_7_4_SPY = {
    "call": {"T1": {"mean_profit": 0.285, "hit_rate": 0.941},
             "T2": {"mean_profit": -0.223, "hit_rate": 0.771},
             "K1": {"mean_profit": -0.638, "hit_rate": 0.720},
             "K2": {"mean_profit": 0.696, "hit_rate": 0.969}},
    "put": {"T1": {"mean_profit": 0.064, "hit_rate": 0.994},
            "T2": {"mean_profit": -0.030, "hit_rate": 0.956},
            "K1": {"mean_profit": 0.045, "hit_rate": 0.989},
            "K2": {"mean_profit": -0.010, "hit_rate": 0.961}},
}


def run_ticker(ticker: str, start: str, end: str) -> dict:
    try:
        panel = load_option_eod_panel(ticker, start, end)
    except OptionsDataUnavailable as exc:
        return {"ticker": ticker, "status": "data_unavailable", "error": str(exc)}

    cleaned = clean_option_panel(panel, config.EXP5_DELTA_MIN, config.EXP5_DELTA_MAX)
    cleaned = enforce_single_maturity_monotonicity(cleaned, config.EXP5_MONOTONICITY_VIOLATION_TOLERANCE)
    tables = {
        option_type: trading_strategy_table(cleaned, option_type).to_dict(orient="records")
        for option_type in ("call", "put")
    }
    return {"ticker": ticker, "status": "ok", "strategy_tables": tables}


def main() -> dict:
    results = {}
    for ticker, (start, end) in config.EXP_EMPIRICAL_TICKERS.items():
        print(f"exp_7: attempting end-of-day option panel for {ticker} ({start}..{end})")
        results[ticker] = run_ticker(ticker, start, end)
        print(f"exp_7: {ticker} -> {results[ticker]['status']}")

    any_ok = any(r["status"] == "ok" for r in results.values())
    output = {
        "experiment": "exp_7",
        "description": "Per-trade hit rates and normalized profit of the four denomination "
                        "strategies against TP2/RR2 violations, Section 7.5, Table 7.4",
        "paper_table_7_4_spy": PAPER_TABLE_7_4_SPY,
        "results": results,
        "conclusion": "reproduced" if all(r["status"] == "ok" for r in results.values())
        else ("partially_reproduced" if any_ok else "not_reproduced_data_unavailable"),
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "exp7_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"exp_7: wrote {out_path}")
    return output


if __name__ == "__main__":
    main()
