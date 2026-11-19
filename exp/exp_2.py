"""Experiment 2: European option TP2/RR2 conditions and violations (Section 4).

Verifies Proposition 4.1 (European puts are RR2 when mu = E[exp(Z)] <= 1,
i.e. r <= q; calls are TP2 when mu >= 1) and Proposition 4.2 (positive drift
produces RR2 violations for European puts beyond a boundary K_p*), and
reproduces the Figure 4.1 illustration of log p_1(K) vs log(K-1) in the
Black-Scholes model.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import config
from src.bs_pricing import bs_european_call, bs_european_put, log_return_params
from src.european import (find_kc_star, find_kp_star, hazard_rate_ratio,
                           rr2_violation_determinant, tp2_violation_determinant)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def check_proposition_4_1(r: float, q: float, sigma: float, T: float, S0: float,
                           strikes: np.ndarray) -> dict:
    """Proposition 4.1: mu<=1 (r<=q) => RR2 holds for puts over every K1<K2."""
    mu_z, sigma_z = log_return_params(r, sigma, T, q=q)
    mu = np.exp(mu_z + 0.5 * sigma_z**2)
    p1 = bs_european_put(S0, strikes, r, q, sigma, T)
    max_det = -np.inf
    for i in range(len(strikes) - 1):
        for j in range(i + 1, len(strikes)):
            det = rr2_violation_determinant(strikes[i], strikes[j], p1[i], p1[j])
            max_det = max(max_det, det)
    return {"mu": float(mu), "max_rr2_determinant": float(max_det), "rr2_holds": bool(max_det <= 1e-8)}


def check_proposition_4_2(r: float, q: float, sigma: float, T: float, S0: float) -> dict:
    """Proposition 4.2: positive drift (r>q) produces RR2 violations for the
    European put beyond K_p*; verify condition (31): log p_1(K)-log(K-1) is
    increasing for K > K_p*."""
    mu_z, sigma_z = log_return_params(r, sigma, T, q=q)
    kp_star = find_kp_star(mu_z, sigma_z, S0=S0)
    strikes = np.linspace(kp_star * 1.001, kp_star * 4.0, 200)
    p1 = bs_european_put(S0, strikes, r, q, sigma, T)
    log_ratio = np.log(p1) - np.log(strikes - S0)
    is_increasing = bool(np.all(np.diff(log_ratio) >= -1e-6))
    violation_found = False
    for i in range(len(strikes) - 1):
        for j in range(i + 1, min(i + 10, len(strikes))):
            det = rr2_violation_determinant(strikes[i], strikes[j], p1[i], p1[j])
            if det > 1e-9:
                violation_found = True
    return {
        "kp_star": float(kp_star),
        "kp_star_paper": config.EXP2_KP_STAR_PAPER,
        "log_ratio_increasing_beyond_kp_star": is_increasing,
        "violation_found_beyond_kp_star": violation_found,
    }


def check_black_scholes_drift_condition(r: float, q: float, sigma: float, T: float, S0: float) -> dict:
    """Equation (32): verify E[S_T]/S0 = e^{(r-q)T} vs. the hazard-rate ratio
    R(K) = Phi(-d+sigma*sqrt(T))/Phi(-d), which is decreasing in K and -> 1."""
    strikes = np.geomspace(0.3 * S0, 5.0 * S0, 200)
    R = hazard_rate_ratio(strikes, S0, r, q, sigma, T)
    drift_ratio = np.exp((r - q) * T)
    is_decreasing = bool(np.all(np.diff(R) <= 1e-6))
    return {
        "drift_ratio_E_ST_over_S0": float(drift_ratio),
        "hazard_rate_ratio_decreasing_in_K": is_decreasing,
        "hazard_rate_ratio_min": float(R.min()),
        "hazard_rate_ratio_max": float(R.max()),
    }


def main(full: bool = False) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    S0, sigma, T = config.EXP2_S0, config.EXP2_SIGMA, config.EXP2_T
    r_pos, q_pos = config.EXP2_R_POSITIVE_DRIFT, config.EXP2_Q_POSITIVE_DRIFT
    r_non, q_non = config.EXP2_R_NONPOSITIVE_DRIFT, config.EXP2_Q_NONPOSITIVE_DRIFT

    strikes = S0 * np.exp(np.linspace(config.EXP2_LOG_MONEYNESS_MIN, config.EXP2_LOG_MONEYNESS_MAX,
                                       config.EXP2_N_STRIKES))

    prop_4_1_nonpositive = check_proposition_4_1(r_non, q_non, sigma, T, S0, strikes)
    prop_4_1_positive = check_proposition_4_1(r_pos, q_pos, sigma, T, S0, strikes)
    prop_4_2 = check_proposition_4_2(r_pos, q_pos, sigma, T, S0)
    bs_drift = check_black_scholes_drift_condition(r_pos, q_pos, sigma, T, S0)

    # Call-side TP2 check under negative drift (mu <= 1 for calls requires mu>=1;
    # here we check the symmetric call violation under r < q, equation 29).
    mu_z_neg, sigma_z_neg = log_return_params(r_non, sigma, T, q=q_non)
    kc_star = find_kc_star(mu_z_neg, sigma_z_neg, S0=S0)
    call_strikes = np.linspace(0.05 * S0, kc_star * 0.999, 150)
    c1 = bs_european_call(S0, call_strikes, r_non, q_non, sigma, T)
    tp2_violation_found = False
    for i in range(len(call_strikes) - 1):
        for j in range(i + 1, min(i + 10, len(call_strikes))):
            det = tp2_violation_determinant(call_strikes[i], call_strikes[j], c1[i], c1[j])
            if det < -1e-9:
                tp2_violation_found = True

    # Figure 4.1: log p_1(K) and log(K-1) vs K, positive-drift case.
    fig_strikes = np.linspace(1.01, 4.0, 300)
    p1_fig = bs_european_put(S0, fig_strikes, r_pos, q_pos, sigma, T)
    kp_star = prop_4_2["kp_star"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fig_strikes, np.log(p1_fig), label=r"$\log p_1(K)$")
    ax.plot(fig_strikes, np.log(np.maximum(fig_strikes - S0, 1e-12)), label=r"$\log(K-1)$", linestyle="--")
    ax.axvline(kp_star, color="grey", linestyle=":", label=rf"$K_p^*\approx{kp_star:.3f}$")
    ax.set_xlabel("K")
    ax.set_ylabel("log price")
    ax.set_title("Figure 4.1: European put log-price vs. intrinsic value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp2_figure_4_1.png", dpi=150)
    plt.close(fig)

    results = {
        "parameters": {"S0": S0, "sigma": sigma, "T": T,
                        "r_positive_drift": r_pos, "q_positive_drift": q_pos,
                        "r_nonpositive_drift": r_non, "q_nonpositive_drift": q_non},
        "proposition_4_1_nonpositive_drift_puts": prop_4_1_nonpositive,
        "proposition_4_1_positive_drift_puts_control": prop_4_1_positive,
        "proposition_4_2_positive_drift_violation": prop_4_2,
        "black_scholes_drift_condition_eq_32": bs_drift,
        "call_side_tp2_violation_negative_drift": {
            "kc_star": float(kc_star), "violation_found": tp2_violation_found,
        },
    }
    with open(RESULTS_DIR / "exp2_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="use the paper's full configuration")
    args = parser.parse_args()
    main(full=args.full)
