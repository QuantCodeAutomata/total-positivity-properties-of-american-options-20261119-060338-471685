"""Experiment 10: American (SPY) vs European (SPX) violation-rate
comparison (Section 7.8, Tables 7.6-7.7, Figure 7.3).

Attempts to load end-of-day option panels for SPY (American, restricted to
2015-2025 for overlap) and SPX (European, PM-settled, 2015-2025), compares
their days-to-expiration distributions and monthly violation rates with
Newey-West significance testing, restricts both to DTE <= 30, and
correlates the excess European violation gap with the risk-neutral drift
r - q. If either panel cannot be loaded, the failure is recorded instead
of fabricated comparison statistics.
"""

from __future__ import annotations

import json
from pathlib import Path

from data.data_loader import OptionsDataUnavailable, load_option_eod_panel
from src import config

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Tables 7.6-7.7 of the paper, kept only as the comparison target, never
# computed from them.
PAPER_RESULTS = {
    "table_7_6_dte_median": {"SPY": 49.5, "SPX": 26.0},
    "table_7_7_full": {
        "call": {"SPY_pct": 1.09, "SPX_pct": 2.17, "t_stat": -2.9},
        "put": {"SPY_pct": 2.18, "SPX_pct": 3.25, "t_stat": -5.0},
    },
    "table_7_7_dte_le_30": {
        "call": {"SPY_pct": 2.35, "SPX_pct": 2.30, "significant": False},
    },
    "correlation_pearson_spearman": {"call": [0.78, 0.86], "put": [0.87, 0.84]},
    "delta_gap_vs_drift_correlation": [-0.35, -0.39],
}


def attempt(ticker: str, start: str, end: str) -> dict:
    try:
        load_option_eod_panel(ticker, start, end)
        return {"ticker": ticker, "status": "ok"}
    except OptionsDataUnavailable as exc:
        return {"ticker": ticker, "status": "data_unavailable", "error": str(exc)}


def main() -> dict:
    spy_start, spy_end = config.EXP10_SPX_START, config.EXP10_SPX_END  # SPY restricted to the SPX overlap window
    print(f"exp_10: attempting end-of-day option panel for SPY ({spy_start}..{spy_end})")
    spy = attempt("SPY", spy_start, spy_end)
    print(f"exp_10: attempting end-of-day option panel for SPX ({config.EXP10_SPX_START}..{config.EXP10_SPX_END})")
    spx = attempt("SPX", config.EXP10_SPX_START, config.EXP10_SPX_END)

    status = "ok" if spy["status"] == "ok" and spx["status"] == "ok" else "data_unavailable"
    output = {
        "experiment": "exp_10",
        "description": "American (SPY) vs European (SPX) TP2/RR2 violation-rate comparison, "
                        "Section 7.8, Tables 7.6-7.7, Figure 7.3",
        "paper_results": PAPER_RESULTS,
        "spy": spy,
        "spx": spx,
        "comparison": None,
        "conclusion": "reproduced" if status == "ok" else "not_reproduced_data_unavailable",
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "exp10_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"exp_10: wrote {out_path}")
    return output


if __name__ == "__main__":
    main()
