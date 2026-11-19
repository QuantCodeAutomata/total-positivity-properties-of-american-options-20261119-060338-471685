"""Experiment 1: Bermudan option TP2/RR2 properties (Section 3 of the paper).

Verifies the three-step induction behind Theorems 3.1-3.2 (Bermudan puts are
RR2, Bermudan calls are TP2, in the number of exercise opportunities n and
strike K, whenever log returns have a log-concave density):

    1. Initialization (Proposition 3.1): the intrinsic value (K-S)^+ is RR2.
    2. Continuation-value preservation (Proposition 3.2): H_{n+1} inherits
       RR2/TP2 from P_n through the Gaussian (log-concave) composition.
    3. Max-operator preservation (Proposition 3.3): P_{n+1}=max((K-S)^+,H_{n+1})
       stays RR2.

and reproduces the ratio-monotonicity illustration of Figure 3.1 in the
Black-Scholes model with Delta t = 1, r = 0.1, sigma = 0.3 (Section 3.6).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import config
from src.bermudan import PricingGrid, bermudan_put_values, build_continuation_operator, rr2_determinant
from src.bs_pricing import log_return_params

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def verify_rr2_determinant(grid: PricingGrid, history: list[np.ndarray], n_pairs: int = 500,
                            seed: int = 0) -> dict:
    """Proposition 3.1 / equation (14): the determinant of intrinsic value vs.
    Bermudan put price must be <= 0 for every n and every K1 < K2."""
    intrinsic = np.maximum(grid.k - 1.0, 0.0)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(grid.k), size=(n_pairs, 2), replace=True)
    max_det = -np.inf
    n_checked = 0
    for n, p in enumerate(history):
        for i, j in idx:
            if grid.k[i] >= grid.k[j]:
                i, j = j, i
            if grid.k[i] == grid.k[j]:
                continue
            det = rr2_determinant(p[i], p[j], intrinsic[i], intrinsic[j])
            max_det = max(max_det, det)
            n_checked += 1
    return {"max_determinant": float(max_det), "n_checked": n_checked, "holds": bool(max_det <= 1e-8)}


def ratio_curves(grid: PricingGrid, history: list[np.ndarray], A: np.ndarray) -> dict:
    """Figure 3.1 ratios at S=1: (K-S)^+/P_1, (K-S)^+/H_1 (left panel) and
    H_1/H_2, P_1/P_2 (right panel), all as functions of log(K/S) = grid.x."""
    intrinsic = np.maximum(grid.k - 1.0, 0.0)
    p0, p1, p2 = history[0], history[1], history[2]
    h1 = A @ p0  # H_1 = continuation value computed from P_0 (equation 12)
    h2 = A @ p1  # H_2 = continuation value computed from P_1
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_intrinsic_over_p1 = np.where(p1 > 0, intrinsic / p1, np.nan)
        ratio_intrinsic_over_h1 = np.where(h1 > 0, intrinsic / h1, np.nan)
        ratio_h1_over_h2 = np.where(h2 > 0, h1 / h2, np.nan)
        ratio_p1_over_p2 = np.where(p2 > 0, p1 / p2, np.nan)
    return {
        "log_moneyness": grid.x,
        "intrinsic_over_p1": ratio_intrinsic_over_p1,
        "intrinsic_over_h1": ratio_intrinsic_over_h1,
        "h1_over_h2": ratio_h1_over_h2,
        "p1_over_p2": ratio_p1_over_p2,
    }


def _is_monotone_nondecreasing(y: np.ndarray, tol: float = 1e-6) -> bool:
    mask = np.isfinite(y)
    diffs = np.diff(y[mask])
    return bool(np.all(diffs >= -tol))


def main(full: bool = False) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    r, dt, sigma, S0 = config.EXP1_R, config.EXP1_DT, config.EXP1_SIGMA, config.EXP1_S0
    mu_z, sigma_z = log_return_params(r, sigma, dt)

    grid = PricingGrid.build(
        n_points=config.EXP1_GRID_POINTS, log_half_width=config.EXP1_GRID_LOG_HALF_WIDTH,
    )
    history = bermudan_put_values(grid, r, dt, mu_z, sigma_z, n_steps=config.EXP1_MAX_N)
    A = build_continuation_operator(grid, r, dt, mu_z, sigma_z)

    det_check = verify_rr2_determinant(grid, history)

    # Restrict the reported/plotted ratios to the Figure 3.1 log-moneyness window.
    curves = ratio_curves(grid, history, A)
    window = (curves["log_moneyness"] >= config.EXP1_LOG_MONEYNESS_MIN) & (
        curves["log_moneyness"] <= config.EXP1_LOG_MONEYNESS_MAX
    )
    log_m = curves["log_moneyness"][window]
    left_red = curves["intrinsic_over_p1"][window]      # (K-S)^+ / P_1, expected non-decreasing
    left_blue = curves["intrinsic_over_h1"][window]      # (K-S)^+ / H_1, expected NOT monotone
    right_blue = curves["h1_over_h2"][window]            # H_1 / H_2, expected increasing
    right_red = curves["p1_over_p2"][window]             # P_1 / P_2, expected increasing

    monotonicity = {
        "intrinsic_over_p1_nondecreasing": _is_monotone_nondecreasing(left_red),
        "intrinsic_over_h1_nondecreasing": _is_monotone_nondecreasing(left_blue),
        "h1_over_h2_nondecreasing": _is_monotone_nondecreasing(right_blue),
        "p1_over_p2_nondecreasing": _is_monotone_nondecreasing(right_red),
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(log_m, left_red, color="tab:red", label=r"$(K-S)^+/P_1(K,S)$")
    axes[0].plot(log_m, left_blue, color="tab:blue", label=r"$(K-S)^+/H_1(K,S)$")
    axes[0].set_xlabel(r"$\log(K/S)$")
    axes[0].set_ylabel("ratio")
    axes[0].set_title("Figure 3.1 (left): intrinsic value vs. $P_1$ and $H_1$")
    axes[0].legend()

    axes[1].plot(log_m, right_blue, color="tab:blue", label=r"$H_1(K,S)/H_2(K,S)$")
    axes[1].plot(log_m, right_red, color="tab:red", label=r"$P_1(K,S)/P_2(K,S)$")
    axes[1].set_xlabel(r"$\log(K/S)$")
    axes[1].set_ylabel("ratio")
    axes[1].set_title("Figure 3.1 (right): continuation and price ratios")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp1_figure_3_1.png", dpi=150)
    plt.close(fig)

    results = {
        "parameters": {"r": r, "dt": dt, "sigma": sigma, "S0": S0,
                        "grid_points": config.EXP1_GRID_POINTS,
                        "grid_log_half_width": config.EXP1_GRID_LOG_HALF_WIDTH,
                        "max_n": config.EXP1_MAX_N},
        "proposition_3_1_rr2_determinant": det_check,
        "figure_3_1_monotonicity": monotonicity,
    }
    with open(RESULTS_DIR / "exp1_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="use the paper's full configuration")
    args = parser.parse_args()
    main(full=args.full)
