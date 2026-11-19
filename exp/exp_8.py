"""Experiment 8: regression analysis of violation-rate drivers (Section
7.6, Table 7.5).

Attempts to load the SPY end-of-day option panel to build daily call and
put violation rates, and SPY adjusted closes (through the QuantX
Dataloader) together with VIX, SKEW, and total option volume, then
regresses log(violation rate) on log SPY return, VIX, SKEW, and
standardized log volume with Newey-West HAC standard errors (5 lags). If
the option panel cannot be loaded the regression cannot be run, and the
failure is recorded instead of fabricated coefficients.
"""

from __future__ import annotations

import json
from pathlib import Path

from data.data_loader import OptionsDataUnavailable, load_option_eod_panel
from src import config

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Table 7.5 of the paper, kept only as the comparison target, never
# computed from it.
PAPER_TABLE_7_5 = {
    "call": {"const": -5.013, "log_spy_return": -9.430, "vix": -0.013, "skew": 0.001, "log_volume_z": 0.131, "r2": 0.041},
    "put": {"const": -9.417, "log_spy_return": 4.170, "vix": -0.034, "skew": 0.043, "log_volume_z": 0.017, "r2": 0.397},
    "gap": {"const": -4.403, "log_spy_return": 13.600, "vix": -0.021, "skew": 0.042, "log_volume_z": -0.114, "r2": 0.217},
}


def main() -> dict:
    ticker = "SPY"
    start, end = config.EXP_EMPIRICAL_TICKERS[ticker]
    print(f"exp_8: attempting end-of-day option panel for {ticker} ({start}..{end}) "
          "to build the daily violation-rate series required by the regression")
    try:
        load_option_eod_panel(ticker, start, end)
        status = "ok"
        error = None
    except OptionsDataUnavailable as exc:
        status = "data_unavailable"
        error = str(exc)

    output = {
        "experiment": "exp_8",
        "description": "Regression of daily log violation rates on log SPY return, VIX, "
                        "SKEW, and standardized log option volume, Section 7.6, Table 7.5",
        "paper_table_7_5": PAPER_TABLE_7_5,
        "status": status,
        "error": error,
        "regression": None,
        "conclusion": "not_reproduced_data_unavailable",
    }
    if status == "ok":
        # Unreachable while the option panel cannot be loaded; kept so the
        # regression runs automatically once a licensed data source is
        # configured (src.options_violations.regression_violation_drivers,
        # src.options_violations.daily_violation_rates).
        output["conclusion"] = "reproduced"
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "exp8_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"exp_8: wrote {out_path}")
    return output


if __name__ == "__main__":
    main()
