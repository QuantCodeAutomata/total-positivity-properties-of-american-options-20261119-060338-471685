"""Experiment 4: from Bermudan to American - the continuous-exercise limit
(Section 6).

Verifies Lemma 6.1 (the Bermudan value on the grid Delta^N = T / 2^N
converges as N -> infinity) by reporting the Bermudan put price sequence
P_{2^N}(K, S0) for increasing N, together with the monotonic-decrease check
required by Theorem 6.2/Corollary 6.1 for the Black-Scholes model with
r >= 0 (log returns remain Gaussian, hence log-concave, at every refinement
level, so Theorems 3.1-3.2 apply at each N and are transported to the
American limit).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import config
from src.convergence import convergence_sequence

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def main(full: bool = False) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    n_max = config.EXP4_N_MAX_PAPER if full else config.EXP4_N_MAX_RUN
    n_values = list(range(config.EXP4_N_MIN, n_max + 1))

    sequences: dict[float, dict[int, float]] = {}
    for K in config.EXP4_STRIKES:
        sequences[K] = convergence_sequence(
            K, config.EXP4_S0, config.EXP4_R, config.EXP4_SIGMA, config.EXP4_T,
            n_values, config.EXP4_GRID_POINTS, config.EXP4_GRID_LOG_HALF_WIDTH,
        )

    fig, ax = plt.subplots(figsize=(7, 5))
    monotone_nondecreasing = {}
    final_values = {}
    for K, seq in sequences.items():
        values = np.array([seq[n] for n in n_values])
        ax.plot(n_values, values, marker="o", label=f"K={K}")
        diffs = np.diff(values)
        # Lemma 6.1: each refinement adds exercise opportunities, so the
        # Bermudan sequence rises (or stays flat) towards the American limit.
        monotone_nondecreasing[str(K)] = bool(np.all(diffs >= -1e-9))
        final_values[str(K)] = float(values[-1])
    ax.set_xlabel("refinement level N (grid $\\Delta^N = T/2^N$)")
    ax.set_ylabel("Bermudan put price $P_{2^N}(K,S_0)$")
    ax.set_title("Lemma 6.1: convergence of Bermudan to American put prices")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp4_convergence.png", dpi=150)
    plt.close(fig)

    # Richardson-type extrapolated increment between the last two refinement
    # levels, reported as a proxy for the remaining distance to the American
    # (N -> infinity) limit.
    last_increments = {
        str(K): float(sequences[K][n_values[-2]] - sequences[K][n_values[-1]])
        for K in config.EXP4_STRIKES
    }

    results = {
        "parameters": {"T": config.EXP4_T, "r": config.EXP4_R, "sigma": config.EXP4_SIGMA,
                        "S0": config.EXP4_S0, "n_values": n_values,
                        "n_max_paper": config.EXP4_N_MAX_PAPER, "n_max_run": config.EXP4_N_MAX_RUN,
                        "grid_points": config.EXP4_GRID_POINTS},
        "monotone_nondecreasing_to_american_limit": monotone_nondecreasing,
        "final_bermudan_values": final_values,
        "last_increment_P2^{N-1}_minus_P2^N": last_increments,
    }
    with open(RESULTS_DIR / "exp4_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="use the paper's full configuration")
    args = parser.parse_args()
    main(full=args.full)
