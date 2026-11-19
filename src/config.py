"""Paper-derived parameters for every experiment.

Each variable is a module-level constant with a comment naming its meaning,
unit, and source (the paper section/equation, or "not specified in the
paper" when no value is given there).  No classes or dictionaries are used
here so every parameter can be grepped by name.
"""

# ---------------------------------------------------------------------------
# exp_1: Bermudan option TP2/RR2 properties (Section 3, Figure 3.1)
# ---------------------------------------------------------------------------
EXP1_DT = 1.0          # time step Delta t between exercise opportunities, Section 3.6
EXP1_R = 0.10           # annualized risk-free rate, Section 3.6 numerical example
EXP1_SIGMA = 0.30       # Black-Scholes volatility, Section 3.6 numerical example
EXP1_S0 = 1.0           # normalized spot price used for all ratio plots, Section 3.6
EXP1_LOG_MONEYNESS_MIN = -2.0  # left end of log(K/S) range for Figure 3.1, Section 3.6
EXP1_LOG_MONEYNESS_MAX = 2.0   # right end of log(K/S) range for Figure 3.1, Section 3.6
EXP1_N_STRIKES = 161    # number of strikes spanning the log-moneyness range (not specified; chosen for a smooth curve)
EXP1_GRID_POINTS = 801  # number of points in the internal pricing grid (not specified in the paper)
EXP1_GRID_LOG_HALF_WIDTH = 6.0  # half-width in log-moneyness of the internal pricing grid (not specified in the paper)
EXP1_MAX_N = 2          # highest exercise-date index used in the illustration (P_0, P_1, P_2), Section 3.6

# ---------------------------------------------------------------------------
# exp_2: European option TP2/RR2 conditions and violations (Section 4, Figure 4.1)
# ---------------------------------------------------------------------------
EXP2_S0 = 1.0           # normalized spot price, Section 4
EXP2_SIGMA = 0.30       # Black-Scholes volatility for Figure 4.1 (not specified exactly; set to the Section 3.6 value so (r - q) alone drives the sign of the drift)
EXP2_T = 1.0            # time to expiration, Section 4 (matches Delta t = 1 used throughout Section 3.6)
EXP2_R_POSITIVE_DRIFT = 0.10  # risk-free rate for the positive-drift (r > q) illustration, Section 3.6 value reused for consistency
EXP2_Q_POSITIVE_DRIFT = 0.0   # dividend yield for the positive-drift illustration (not specified in the paper)
EXP2_R_NONPOSITIVE_DRIFT = 0.0   # risk-free rate for the r - q <= 0 control case, Proposition 4.1
EXP2_Q_NONPOSITIVE_DRIFT = 0.10  # dividend yield for the r - q <= 0 control case, Proposition 4.1
EXP2_KP_STAR_PAPER = 1.24  # reported value of the positive-drift exercise boundary K_p*, Section 4.2 numerical illustration
EXP2_LOG_MONEYNESS_MIN = -1.0  # left end of the strike range for Figure 4.1 (not specified; chosen to show both sides of K_p*)
EXP2_LOG_MONEYNESS_MAX = 1.0   # right end of the strike range for Figure 4.1 (not specified in the paper)
EXP2_N_STRIKES = 161    # number of strikes used for Figure 4.1 (not specified in the paper)

# ---------------------------------------------------------------------------
# exp_3: Discrete distributions - binomial and trinomial models (Section 5)
# ---------------------------------------------------------------------------
EXP3_TRINOMIAL_P0 = 2.0 / 3.0  # trinomial center probability p(0), Section 5 example
EXP3_TRINOMIAL_C = 0.05        # perturbation c in p(+-1) = 1/6 +- c (value of c not specified in the paper; chosen small and nonzero to keep p(1) != p(-1))
EXP3_BINOMIAL_S0 = 1.0   # initial underlying price, Section 5 violation example
EXP3_BINOMIAL_R = 0.10   # interest rate, Section 5 violation example
EXP3_BINOMIAL_SIGMA = 1.0  # up/down scale sigma, Section 5 violation example
EXP3_BINOMIAL_DT = 1.0    # time step, Section 5 violation example
EXP3_BINOMIAL_K1 = 0.4           # lower strike K_1, Section 5 violation example
# upper strike K_2 = 0.4 * e, Section 5 violation example (kept as an expression below)
import math as _math
EXP3_BINOMIAL_K2 = 0.4 * _math.e
EXP3_EXPECTED_VIOLATION_MAGNITUDE = 0.0032  # reported value of H2(K1)H3(K2) - H2(K2)H3(K1), Section 5

# ---------------------------------------------------------------------------
# exp_4: From Bermudan to American - continuous exercise limit (Section 6)
# ---------------------------------------------------------------------------
EXP4_T = 1.0            # option maturity, consistent with Delta t = 1 at N = 0 in Section 3.6
EXP4_R = 0.10            # risk-free rate, reused from Section 3.6 so r >= 0 as required by Corollary 6.1
EXP4_SIGMA = 0.30        # Black-Scholes volatility, reused from Section 3.6
EXP4_S0 = 1.0            # normalized spot price
EXP4_N_MIN = 1           # smallest refinement level N (grid Delta^N = T / 2^N), Section 6
EXP4_N_MAX_PAPER = 15    # largest refinement level N quoted as an illustration range, Section 6
EXP4_N_MAX_RUN = 9       # largest refinement level actually executed (2^9 = 512 exercise steps); the CFL-consistent spatial grid grows with N (see src/convergence.py), so the dense continuation-operator cost grows faster than N; this level keeps the run within the time available while still covering a two-order-of-magnitude refinement range. Use --full to request EXP4_N_MAX_PAPER.
EXP4_GRID_POINTS = 241   # number of points in the internal pricing grid used for the convergence study (not specified in the paper)
EXP4_GRID_LOG_HALF_WIDTH = 3.0  # half-width in log-moneyness of the internal pricing grid (not specified in the paper)
EXP4_STRIKES = (0.8, 1.0, 1.2)  # representative strikes used to report convergence of Bermudan to American values (not specified in the paper)

# ---------------------------------------------------------------------------
# exp_5-exp_10: Empirical American option violation studies (Section 7)
# ---------------------------------------------------------------------------
EXP_EMPIRICAL_TICKERS = {
    # ticker: (sample start, sample end) as stated in Section 7.1 / Table 7.1
    "SPY": ("2005-01-01", "2025-08-29"),
    "AMZN": ("2000-01-01", "2025-08-29"),
    "GOOGL": ("2004-01-01", "2025-08-29"),
    "META": ("2012-01-01", "2025-08-29"),
    "TSLA": ("2010-01-01", "2025-08-29"),
}
EXP5_DELTA_MIN = 0.01    # minimum admissible |delta| in the cleaning filter, Section 7.2
EXP5_DELTA_MAX = 0.99    # maximum admissible |delta| in the cleaning filter, Section 7.2
EXP5_MONOTONICITY_VIOLATION_TOLERANCE = 0.005  # maximum fraction of single-maturity monotonicity violations tolerated before removal, Section 7.2 ("<0.5% of tuples")

EXP6_T1_GRID_TRADING_DAYS = (3, 5, 10, 15)  # near-term maturities (trading days) tracked for violation survival, Section 7.4 / Table 7.2
EXP6_T2_MAX_CALENDAR_DAYS = 30               # upper bound on the far maturity T2, Section 7.4
EXP6_SURVIVAL_HORIZON_DAYS = 14              # number of T+k survival percentages reported, Table 7.2

EXP9_SAFE_MONTHS = (1, 4, 7, 10)  # ex-dividend safe months (Jan/Apr/Jul/Oct), Section 7.7
EXP9_DTE_MAX = 30                 # days-to-expiration cutoff for the short-term and ex-dividend safe subsets, Section 7.7

EXP10_SPX_START = "2015-01-02"   # SPX (PM-settled) sample start, Section 7.8 / Table 7.6
EXP10_SPX_END = "2025-08-29"     # SPX sample end, Section 7.8
EXP10_NEWEY_WEST_LAGS = 3        # Newey-West lag order used for the SPY-vs-SPX t-statistics, Table 7.7
EXP8_NEWEY_WEST_LAGS = 5         # Newey-West lag order used for the regression of violation rates, Section 7.6 / Table 7.5

# Data-availability note: a full download of EOD options analytics (bid,
# ask, implied volatility, delta, volume) for five tickers over 5-25 years
# each would be the OptionMetrics-equivalent dataset the paper uses. This
# repository attempts that load through the Massive options analytics
# endpoint; see results/RESULTS.md for the outcome of that attempt.
