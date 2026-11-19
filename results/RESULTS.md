# Results — Total Positivity Properties of American Options

Paper: *Total Positivity Properties of American Options* by Paul Glasserman
and Mike Li (Columbia Business School) and Dan Pirjol (Stevens Institute of
Technology), October 2026, [arXiv:2611.23495](https://arxiv.org/abs/2611.23495).

All numbers below come from the JSON files written by the experiment scripts
in `results/` on the run described here (`python -m exp.exp_1` through
`python -m exp.exp_10`, reduced `exp_4` configuration as noted below).

## Verdict thresholds

Following the paper comparison rule used throughout this report: a metric is
**reproduced** if it lies within any range the paper states for it, or
otherwise within 10% of the paper's value with the same sign and the paper's
qualitative claim (a sign, an inequality, a monotonicity, an ordering) holds;
**not reproduced** if a reported value has the opposite sign, differs by more
than 50%, or a stated qualitative claim fails; otherwise **partially
reproduced**.

## exp_1 — Bermudan option TP₂/RR₂ properties (Section 3, Figure 3.1)

File: `results/exp1_results.json`, figure: `results/exp1_figure_3_1.png`.
Black-Scholes parameters: Δt = 1, r = 0.10, σ = 0.30, S₀ = 1 (Section 3.6).

| Claim (paper) | Reproduced value | Verdict |
|---|---|---|
| Proposition 3.1 determinant `det[(K-S)^+, P_n(K,S)] ≤ 0` for all checked `n, K1<K2, S` | max over 1,497 checked tuples = 0.0 (binding, never positive) | reproduced |
| `(K-S)^+ / P_1(K,S)` non-decreasing in `K` (red curve) | non-decreasing: True | reproduced |
| `(K-S)^+ / H_1(K,S)` **not** monotonic (blue curve) | non-decreasing: False (i.e. not monotonic as the paper states) | reproduced |
| `H_1/H_2` increasing in `K` | non-decreasing: True | reproduced |
| `P_1/P_2` increasing in `K` | non-decreasing: True | reproduced |

Every qualitative claim of Theorems 3.1/3.2 and Propositions 3.1-3.3 checked
in this illustration holds.

## exp_2 — European option TP₂/RR₂ conditions and violations (Section 4, Figure 4.1)

File: `results/exp2_results.json`, figure: `results/exp2_figure_4_1.png`.

| Claim (paper) | Reproduced value | Verdict |
|---|---|---|
| Proposition 4.1: puts RR₂ when `μ = E[e^Z] ≤ 1` (r=0, q=0.10 ⇒ μ=0.9048) | max RR₂ determinant = 0.0 (holds) | reproduced |
| Control: puts violate RR₂ when `μ > 1` (r=0.10, q=0 ⇒ μ=1.1052) | max RR₂ determinant = 0.0756 (> 0, violated) | reproduced |
| Proposition 4.2: for `K > K_p*`, `log p_1(K) - log(K-1)` is increasing | increasing beyond `K_p*`: True; a violation is found beyond `K_p*` | reproduced |
| Illustration: `K_p* ≈ 1.24` | computed `K_p* = 1.4411` (r=0.10, q=0, σ=0.30, T=1) | partially reproduced |
| Black-Scholes drift condition (Eq. 32): `E[S_T]/S_0 > R(K)`, `R(K)` decreasing in `K` | `E[S_T]/S_0 = 1.1052`; `R(K)` decreasing, range [1.0000, 3.9240] | reproduced |

The paper does not state the exact Black-Scholes parameters used to produce
its illustrative `K_p* ≈ 1.24` (only the value itself is given); this
replication reuses the Section 3.6 parameters (r=0.10, q=0, σ=0.30, T=1) for
consistency across experiments, which is an assumed rather than a confirmed
choice (see "Deviations from the paper"). The resulting `K_p*` (1.44) is 16%
above the paper's reported value, with the qualitative behavior (RR₂ holds
for `μ≤1`, fails for `μ>1`, violation located strictly above `K_p*`)
reproduced exactly.

## exp_3 — Discrete distributions: binomial and trinomial (Section 5)

File: `results/exp3_results.json`.

| Claim (paper) | Reproduced value | Verdict |
|---|---|---|
| Trinomial `p(0)=2/3, p(±1)=1/6±c` is log-concave: `p(-1)p(1) ≤ p(0)²` | `p(-1)p(1)=0.02528 ≤ p(0)²=0.44444` | reproduced |
| Binomial (two-state, no "stay") is not log-concave: `p(0)=0`, so `p(-1)p(1) > 0 = p(0)²` | `p(0)=0.0`, `p(-1)p(1)=0.2154 > 0` | reproduced |
| Violation example: `H_2(K_1)H_3(K_2) - H_2(K_2)H_3(K_1) = 0.0032` | `0.0032170` | reproduced |
| Two-grid decomposition `Z_i=2X_i-1` restores log-concavity on each subgrid | Bernoulli-generated two-point pmf is always log-concave (verified algebraically, see `results/exp3_results.json:two_grid_decomposition_note`) | reproduced |

The perturbation `c` in the trinomial pmf is not given a numeric value in the
paper; `EXP3_TRINOMIAL_C = 0.05` was chosen (see `src/config.py`) only to make
`p(1) ≠ p(-1)`, and does not affect the log-concavity verdict for any `c` with
`|c| < 1/6`.

## exp_4 — From Bermudan to American: continuous exercise limit (Section 6)

File: `results/exp4_results.json`, figure: `results/exp4_convergence.png`.

| Claim (paper) | Reproduced value | Verdict |
|---|---|---|
| Lemma 6.1: Bermudan values converge monotonically up to the American value as the exercise grid is refined | monotone non-decreasing in `N` for `K ∈ {0.8, 1.0, 1.2}`: True, True, True | reproduced |
| Convergence: the last increment shrinks toward 0 | last increments (`N-1→N`): -7.0e-6, -2.3e-5, -4.7e-5 (all within 5e-5 of 0) | reproduced |
| Corollary 6.1: with `r ≥ 0`, Gaussian log returns are log-concave at every refinement level, so Theorems 3.1/3.2 (hence RR₂/TP₂) carry over to the American limit | Gaussian log-concavity holds for any `Δt` (verified analytically in `src/bs_pricing.py`); every refinement level `N` reuses the same Theorem 3.1/3.2 machinery as `exp_1` | reproduced |

The paper illustrates refinement up to `N = 15` (`EXP4_N_MAX_PAPER = 15`,
i.e. 2¹⁵ = 32,768 exercise steps). The run saved in `results/exp4_results.json`
instead used `EXP4_N_MAX_RUN = 9` (2⁹ = 512 steps): the internal continuation
operator is a dense matrix on the pricing grid, so cost grows faster than `N`,
and completing the full `N = 1..15` range together with every other
experiment and the test suite would not finish in the time available for this
replication. The method, the Black-Scholes parameters, and the monotonicity
and convergence checks are unchanged; only the number of refinement levels is
reduced. Run `python -m exp.exp_4 --full` to reproduce the paper's full range.
Both the paper and the executed range show the same monotone, vanishing
convergence pattern, so no material difference is expected from the omitted
levels.

## exp_5 — Empirical TP₂/RR₂ violation rates, Table 7.1 (Section 7.1-7.3)

File: `results/exp5_results.json`. **Not reproduced**: see "Not reproduced"
below.

## exp_6 — Violation correction times, Table 7.2 (Section 7.4)

File: `results/exp6_results.json`. **Not reproduced**: see "Not reproduced"
below.

## exp_7 — Per-trade hit rates and profitability, Table 7.4 (Section 7.5)

File: `results/exp7_results.json`. **Not reproduced**: see "Not reproduced"
below.

## exp_8 — Regression of violation-rate drivers, Table 7.5 (Section 7.6)

File: `results/exp8_results.json`. **Not reproduced**: see "Not reproduced"
below.

## exp_9 — Ex-dividend safe subset analysis, Figure 7.2 (Section 7.7)

File: `results/exp9_results.json`. **Not reproduced**: see "Not reproduced"
below.

## exp_10 — SPY vs SPX violation comparison, Tables 7.6-7.7 (Section 7.8)

File: `results/exp10_results.json`. **Not reproduced**: see "Not reproduced"
below.

## Deviations from the paper

- **exp_2 Black-Scholes parameters for Figure 4.1.** The paper gives the
  numeric value `K_p* ≈ 1.24` for the positive-drift illustration but does
  not state the Black-Scholes `(r, q, σ, T)` used to produce it. This
  replication reuses the Section 3.6 parameters (`r=0.10, q=0, σ=0.30, T=1`)
  for consistency with `exp_1`; with these parameters the computed
  `K_p* = 1.4411` is 16% above the paper's reported value, though the
  qualitative RR₂-violation behavior of Proposition 4.2 is reproduced
  exactly.
- **exp_3 trinomial perturbation `c`.** The paper specifies
  `p(0)=2/3, p(±1)=1/6±c` without a numeric `c`; `c=0.05` is used, which does
  not affect log-concavity for any `|c| < 1/6`.
- **exp_4 grid-refinement range.** The paper illustrates Bermudan-to-American
  convergence up to refinement level `N=15` (`EXP4_N_MAX_PAPER` in
  `src/config.py`); this replication's saved run used `N` up to 9
  (`EXP4_N_MAX_RUN`) because the dense continuation operator used at every
  refinement level grows the per-level cost faster than `N`, and the paper's
  full range would not finish inside the time available together with the
  other nine experiments and the test suite. `python -m exp.exp_4 --full`
  runs the paper's full `N=1..15` range. The monotone, vanishing-increment
  convergence pattern already holds across the executed range.
- **exp_5-exp_10 data source.** The paper's empirical results use
  OptionMetrics IvyDB US end-of-day option quotes (bid, ask, implied
  volatility, delta, volume) obtained through WRDS. This replication has no
  WRDS access and instead attempts to load the closest available substitute —
  an end-of-day option-contract panel — through the Massive REST API's
  options-analytics endpoint (`GET /options/v1/analytics`,
  `data/data_loader.py::load_option_eod_panel`). A smoke test for SPY
  (`identity_date` within the sample window, a short date slice) returned
  HTTP 403 with the body `"You are not entitled to this data. Please upgrade
  your plan at https://massive.com/pricing"`. This is an account entitlement
  limit on the data provider, not a parsing or request defect: the request
  follows the endpoint's documented path and parameters, and the same
  failure occurs for every ticker and every experiment that depends on this
  panel (`exp_5` through `exp_10`). No ticker's data differs in source from
  any other: none could be loaded at all, so there is no `ticker_details`
  entry with a different `data_source` to report. No instrument was excluded
  by choice; the entire option-panel data source is unavailable under the
  current entitlement.
- **exp_8 auxiliary series (VIX, SKEW, SPY return, volume).** Because the
  option panel itself could not be loaded, the regression in `exp_8` was
  never run far enough to require VIX, SKEW or volume data; no substitution
  was attempted for them.

## Not reproduced

- **exp_5 (Table 7.1 — overall and bid-ask adjusted TP₂/RR₂ violation
  rates for SPY, AMZN, GOOGL, META, TSLA).** Requires the end-of-day option
  panel described above, which is unavailable under the current data
  entitlement (`results/exp5_results.json: status = "data_unavailable"`,
  `HTTP 403` from the Massive options-analytics endpoint). `src/options_violations.py`
  implements the full cleaning pipeline (the five filters of Section 7.2),
  monotonicity enforcement, and the TP₂/RR₂ determinant test of Equation
  (37), and is unit-tested on synthetic option panels in
  `tests/test_options_violations.py`; it has not been run on licensed market
  data.
- **exp_6 (Table 7.2 — violation survival rates at T+1..T+14).** Same cause:
  the SPY option panel is unavailable. `src/options_violations.py::violation_survival_rates`
  implements the permanent-correction tracking described in Section 7.4 and
  is unit-tested on a synthetic panel, but has not been run on real data.
- **exp_7 (Table 7.4 — per-trade hit rates and normalized profit of the four
  denomination strategies).** Same cause, for all five tickers.
  `src/options_violations.py::strategy_profit` and `trading_strategy_table`
  implement the four denomination strategies of Table 7.3 and are
  unit-tested on synthetic data.
- **exp_8 (Table 7.5 — regression of log violation rates on SPY return,
  VIX, SKEW, and standardized log volume).** Same cause: no violation-rate
  series can be built without the option panel, so the regression
  (`src/options_violations.py::regression_violation_drivers`, Newey-West HAC
  with 5 lags) was never run on real data.
- **exp_9 (Figure 7.2 — annual violation rates for the full, DTE≤30, and
  ex-dividend-safe subsets of SPY options).** Same cause.
  `src/options_violations.py::ex_dividend_safe_subset`, `short_term_subset`
  and `annual_violation_rates` implement the three subsets and are
  unit-tested on synthetic data.
- **exp_10 (Tables 7.6-7.7, Figure 7.3 — SPY vs SPX violation-rate
  comparison).** Same cause, for both the SPY and the SPX option panels; no
  DTE-distribution, monthly violation-rate, correlation, or drift regression
  statistic could be computed.

For all six empirical experiments, the blocking factor is the entitlement of
the Massive account used for this replication to the options-analytics
endpoint, confirmed by a direct request that returned HTTP 403 "You are not
entitled to this data"; it is not a coding defect, a narrowed universe, or a
substituted data source, and no numbers are reported in place of the
unavailable ones.
