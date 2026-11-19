"""Experiment 6: violation correction times for SPY options (Section 7.4,
Table 7.2).

Attempts to load the SPY end-of-day option panel, cleans it as in
experiment 5, and tracks each violating 4-tuple (K1, K2, T1, T2) with
T1 in {3, 5, 10, 15} trading days and T2 within 30 calendar days across
consecutive trading days, reporting the percentage still violating at
T+1 .. T+14 (the "permanent correction" rule of Section 7.4). If the
panel cannot be loaded, the failure is recorded instead of a fabricated
survival table.
"""

from __future__ import annotations

import json
from pathlib import Path

from data.data_loader import OptionsDataUnavailable, load_option_eod_panel
from src import config
from src.options_violations import (
    clean_option_panel,
    enforce_single_maturity_monotonicity,
    violation_survival_rates,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Table 7.2 of the paper (T1 = 15 trading-day row), kept only as the
# comparison target, never computed from it.
PAPER_TABLE_7_2_T1_15 = {
    "call": {"T+1": 58.80, "T+14": 1.35},
    "put": {"T+1": 71.47, "T+14": 1.44},
}


def main() -> dict:
    ticker = "SPY"
    start, end = config.EXP_EMPIRICAL_TICKERS[ticker]
    print(f"exp_6: attempting end-of-day option panel for {ticker} ({start}..{end})")
    try:
        panel = load_option_eod_panel(ticker, start, end)
        status = "ok"
        error = None
    except OptionsDataUnavailable as exc:
        panel = None
        status = "data_unavailable"
        error = str(exc)

    tables = {}
    if status == "ok":
        cleaned = clean_option_panel(panel, config.EXP5_DELTA_MIN, config.EXP5_DELTA_MAX)
        cleaned = enforce_single_maturity_monotonicity(cleaned, config.EXP5_MONOTONICITY_VIOLATION_TOLERANCE)
        for option_type in ("call", "put"):
            table = violation_survival_rates(
                cleaned, option_type,
                t1_grid=config.EXP6_T1_GRID_TRADING_DAYS,
                t2_max_calendar_days=config.EXP6_T2_MAX_CALENDAR_DAYS,
                horizon=config.EXP6_SURVIVAL_HORIZON_DAYS,
            )
            tables[option_type] = table.to_dict(orient="records")

    output = {
        "experiment": "exp_6",
        "description": "Violation survival rates at T+1..T+14, Section 7.4, Table 7.2",
        "paper_table_7_2_t1_15": PAPER_TABLE_7_2_T1_15,
        "status": status,
        "error": error,
        "survival_tables": tables,
        "conclusion": "reproduced" if status == "ok" else "not_reproduced_data_unavailable",
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "exp6_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"exp_6: wrote {out_path}")
    return output


if __name__ == "__main__":
    main()
