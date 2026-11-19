# Total Positivity Properties of American Options

Replication of *Total Positivity Properties of American Options* by Paul
Glasserman and Mike Li (Columbia Business School) and Dan Pirjol (Stevens
Institute of Technology), October 2026, [arXiv:2611.23495](https://arxiv.org/abs/2611.23495).
The paper proves that Bermudan/American put prices are
reverse-rule-of-order-2 (RR₂) and call prices are totally positive of order 2
(TP₂) in the number of exercise opportunities and the strike, under a
log-concave density assumption on log returns, and documents how often this
property is empirically violated by listed American options.

This repository reproduces the ten experiments selected from the paper:

| Experiment | Title | Data |
|---|---|---|
| exp_1 | Bermudan option TP₂/RR₂ properties (Section 3, Figure 3.1) | None (Black-Scholes, closed form) |
| exp_2 | European option TP₂/RR₂ conditions and violations (Section 4, Figure 4.1) | None (Black-Scholes, closed form) |
| exp_3 | Discrete distributions: binomial and trinomial (Section 5) | None (lattice computation) |
| exp_4 | Bermudan-to-American continuous exercise limit (Section 6) | None (Black-Scholes, closed form) |
| exp_5 | Empirical TP₂/RR₂ violation rates, SPY/AMZN/GOOGL/META/TSLA (Section 7.1-7.3, Table 7.1) | OptionMetrics-equivalent end-of-day option panel |
| exp_6 | Violation correction times for SPY (Section 7.4, Table 7.2) | same |
| exp_7 | Per-trade hit rates of violation-based trading strategies (Section 7.5, Table 7.4) | same |
| exp_8 | Regression of violation-rate drivers (Section 7.6, Table 7.5) | same, plus SPY price, VIX, SKEW |
| exp_9 | Ex-dividend safe subset analysis (Section 7.7, Figure 7.2) | same |
| exp_10 | SPY (American) vs SPX (European) violation comparison (Section 7.8, Tables 7.6-7.7) | same, plus SPX option panel |

Experiments 1-4 are purely theoretical/numerical: they implement the
Bermudan/European pricing recursions of Sections 3-6 and verify the
determinant conditions, monotonicity and convergence claims in the
Black-Scholes model with the paper's parameters (Δt = 1, r = 0.10, σ = 0.30).
Experiments 5-10 require an end-of-day panel of listed American (SPY and
single-stock) and European (SPX) option quotes with bid, ask, implied
volatility and delta for every listed contract and trading day over up to 25
years — the data OptionMetrics IvyDB US supplies on WRDS. See
`results/RESULTS.md` for the outcome of attempting to load the closest
available substitute through the Massive options-analytics endpoint.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in DATALOADER_API_KEY and MASSIVE_API_KEY
```

No market data is required for exp_1-exp_4. exp_5-exp_10 read
`DATALOADER_API_KEY` and `MASSIVE_API_KEY` from `.env` and attempt to load the
end-of-day option panel through Massive; see `data/data_loader.py`.

## Running the experiments

From the repository root:

```bash
python -m exp.exp_1
python -m exp.exp_2
python -m exp.exp_3
python -m exp.exp_4
python -m exp.exp_5
python -m exp.exp_6
python -m exp.exp_7
python -m exp.exp_8
python -m exp.exp_9
python -m exp.exp_10
```

`exp_4` additionally accepts `--full` to run the grid-refinement study at the
paper's illustration range (`EXP4_N_MAX_PAPER = 15`, i.e. up to 2¹⁵ exercise
steps) instead of the reduced range executed by default
(`EXP4_N_MAX_RUN = 9`); see `src/config.py` and "Deviations from the paper" in
`results/RESULTS.md`.

Every script writes its metrics as JSON (and, for exp_1/exp_2/exp_4, a PNG
figure) directly into `results/`, overwriting any earlier output.

## Tests

```bash
python -m pytest -m "not market_data"   # unit tests, no network access needed
python -m pytest                         # also runs the market_data tests
                                          # (skipped automatically without API keys)
```

## Reproduction entry point

```bash
./reproduce.sh
```

runs the full test suite (excluding market-data tests) followed by every
experiment script in order.

## Repository layout

```
src/     Bermudan/European/discrete pricing recursions, Black-Scholes
         formulas, convergence utilities, option-panel cleaning and
         violation-rate statistics, and src/config.py (all parameters)
exp/     one script per experiment, calling only the functions in src/ and data/
data/    data/data_loader.py: OHLCV bars through the QuantX Dataloader REST
         API, and the end-of-day option panel through Massive
tests/   pytest unit tests (synthetic/fixture data) and market_data-marked
         tests (skipped without API keys)
results/ JSON metrics, PNG figures, and RESULTS.md
```

See `AGENTS.md` for conventions and `results/RESULTS.md` for the comparison
against the paper's reported numbers.
