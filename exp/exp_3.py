"""Experiment 3: discrete-distribution extension (Section 5).

Verifies Theorem 5.1 (log-concave integer-valued pmf -> Theorems 3.1-3.2 hold
on the grid K_0 = {K_0 e^{bj}}), using the trinomial model as a positive
(log-concave) example and the binomial model as the negative example in
which log-concavity fails and RR2 is violated at the specific strikes
K1 = 0.4, K2 = 0.4e of Section 5.
"""

from __future__ import annotations

import json
from pathlib import Path

from src import config
from src.discrete import binomial_pmf, h_values_binomial, is_log_concave_pmf, trinomial_pmf

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def main(full: bool = False) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Trinomial: positive (log-concave) result.
    trinomial = trinomial_pmf(config.EXP3_TRINOMIAL_P0, config.EXP3_TRINOMIAL_C)
    trinomial_log_concave = is_log_concave_pmf(trinomial)
    trinomial_check = {
        "pmf": trinomial,
        "p_minus1_times_p_plus1": trinomial[-1] * trinomial[1],
        "p0_squared": trinomial[0] ** 2,
        "log_concave": trinomial_log_concave,
    }

    # Binomial: negative result, log-concavity fails (p(0) = 0).
    pmf, b = binomial_pmf(config.EXP3_BINOMIAL_R, config.EXP3_BINOMIAL_SIGMA, config.EXP3_BINOMIAL_DT)
    binomial_log_concave = is_log_concave_pmf(pmf)

    K1, K2 = config.EXP3_BINOMIAL_K1, config.EXP3_BINOMIAL_K2
    h_values = h_values_binomial(K1, K2, config.EXP3_BINOMIAL_R, config.EXP3_BINOMIAL_SIGMA,
                                  config.EXP3_BINOMIAL_DT, n_steps=(2, 3))
    violation = h_values["H2_K1"] * h_values["H3_K2"] - h_values["H2_K2"] * h_values["H3_K1"]

    results = {
        "trinomial_log_concavity_check": trinomial_check,
        "binomial_pmf": {"p": pmf, "b": b, "log_concave": binomial_log_concave},
        "binomial_violation_example": {
            "K1": K1, "K2": K2,
            **h_values,
            "determinant_H2K1_H3K2_minus_H2K2_H3K1": violation,
            "expected_from_paper": config.EXP3_EXPECTED_VIOLATION_MAGNITUDE,
            "violation_confirmed": bool(violation > 1e-6),
        },
        "two_grid_decomposition_note": (
            "Z_i = 2 X_i - 1 with X_i Bernoulli(p) is always log-concave "
            "(two-point pmf), so the subgrids K_0^(-) = {K_0 e^{(2j-1) b}} and "
            "K_0^(+) = {K_0 e^{2 j b}} each recover RR2 within themselves "
            "(Section 5)."
        ),
    }
    with open(RESULTS_DIR / "exp3_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="use the paper's full configuration")
    args = parser.parse_args()
    main(full=args.full)
