"""Experiment 9: effect of dividends on violation rates, ex-dividend safe
subset analysis for SPY options (Section 7.7, Figure 7.2).

Attempts to load the SPY end-of-day option panel and, on the cleaned
panel, builds three subsets - the full dataset, the short-term (DTE <= 30)
subset, and the ex-dividend safe subset (DTE <= 30 and observation month
in Jan/Apr/Jul/Oct, Section 7.7) - and compares their annual violation
rates for calls and puts. If the panel cannot be loaded, the failure is
recorded instead of fabricated annual rates.
"""

from __future__ import annotations

import json
from pathlib import Path

from data.data_loader import OptionsDataUnavailable, load_option_eod_panel
from src import config
from src.options_violations import (
    annual_violation_rates,
    clean_option_panel,
    enforce_single_maturity_monotonicity,
    ex_dividend_safe_subset,
    short_term_subset,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def main() -> dict:
    ticker = "SPY"
    start, end = config.EXP_EMPIRICAL_TICKERS[ticker]
    print(f"exp_9: attempting end-of-day option panel for {ticker} ({start}..{end})")
    try:
        panel = load_option_eod_panel(ticker, start, end)
        status = "ok"
        error = None
    except OptionsDataUnavailable as exc:
        panel = None
        status = "data_unavailable"
        error = str(exc)

    subsets = {}
    if status == "ok":
        cleaned = clean_option_panel(panel, config.EXP5_DELTA_MIN, config.EXP5_DELTA_MAX)
        cleaned = enforce_single_maturity_monotonicity(cleaned, config.EXP5_MONOTONICITY_VIOLATION_TOLERANCE)
        short_term = short_term_subset(cleaned, config.EXP9_DTE_MAX)
        ex_div_safe = ex_dividend_safe_subset(cleaned, config.EXP9_SAFE_MONTHS, config.EXP9_DTE_MAX)
        for name, frame in (("full", cleaned), ("dte_le_30", short_term), ("ex_dividend_safe", ex_div_safe)):
            subsets[name] = {
                "call": annual_violation_rates(frame, "call").to_dict(),
                "put": annual_violation_rates(frame, "put").to_dict(),
            }

    output = {
        "experiment": "exp_9",
        "description": "Annual violation rates for the full, DTE<=30, and ex-dividend safe "
                        "subsets of SPY options, Section 7.7, Figure 7.2",
        "status": status,
        "error": error,
        "subsets": subsets,
        "conclusion": "reproduced" if status == "ok" else "not_reproduced_data_unavailable",
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "exp9_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"exp_9: wrote {out_path}")
    return output


if __name__ == "__main__":
    main()
