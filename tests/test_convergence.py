"""Unit tests for src/convergence.py (Section 6: Bermudan-to-American limit)."""

import numpy as np

from src.convergence import convergence_sequence


def test_bermudan_prices_converge_and_are_monotone_nondecreasing_in_n():
    """Lemma 6.1: the Bermudan sequence stabilizes as N grows. Each
    refinement adds exercise opportunities, so the price sequence cannot
    decrease (more exercise opportunities raise or preserve value)."""
    seq = convergence_sequence(
        K=1.0, S0=1.0, r=0.10, sigma=0.30, T=1.0,
        n_values=[1, 2, 3, 4, 5], grid_points=161, grid_log_half_width=3.0,
    )
    values = [seq[n] for n in sorted(seq)]
    for a, b in zip(values, values[1:]):
        assert b >= a - 1e-6
    # the last refinement must be closer to the first than the first gap was,
    # confirming the sequence settles rather than drifting (Lemma 6.1)
    diffs = [abs(b - a) for a, b in zip(values, values[1:])]
    assert diffs[-1] <= diffs[0]


def test_convergence_prices_are_nonnegative():
    seq = convergence_sequence(
        K=1.2, S0=1.0, r=0.10, sigma=0.30, T=1.0,
        n_values=[1, 3], grid_points=161, grid_log_half_width=3.0,
    )
    for v in seq.values():
        assert v >= -1e-9
