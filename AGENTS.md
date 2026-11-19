# AGENTS.md

Guide for coding agents and contributors working on this repository.

## What this implements

A replication of *Total Positivity Properties of American Options*: Bermudan
and American put (call) prices are shown to be reverse-rule-of-order-2
(RR₂, resp. totally positive of order 2, TP₂) in the number of exercise
opportunities and the strike when log returns have a log-concave density
(Sections 3-6), and the paper measures how often this property is violated by
listed American options (Section 7).

- `exp_1`-`exp_4`: theoretical/numerical verification in the Black-Scholes
  model and on binomial/trinomial lattices. No external data.
- `exp_5`-`exp_10`: empirical violation rates, survival times, trading
  strategies, regression drivers, dividend effects, and an American-vs-European
  comparison, all built on an end-of-day option panel (bid, ask, delta,
  implied volatility per listed contract and trading day).

## Layout

```
src/      flat library modules (no subpackages): bs_pricing.py, bermudan.py,
          european.py, discrete.py, convergence.py, options_violations.py,
          config.py (every experiment parameter, as UPPER_CASE constants
          with a source comment)
exp/      one script per experiment (python -m exp.exp_N), calling only
          the functions in src/ and data/
data/     data/data_loader.py: OHLCV bars through the QuantX Dataloader REST
          API (load_ohlcv) and the end-of-day option panel through the
          Massive options-analytics endpoint (load_option_eod_panel)
tests/    pytest tests; tests/conftest.py registers the market_data marker
          and builds synthetic option panels for the src/ unit tests
results/  JSON metrics, PNG figures, and RESULTS.md (no subdirectories)
```

## Commands

```bash
pip install -r requirements.txt
python -m pytest -m "not market_data"      # unit tests, default
python -m pytest                            # also market_data tests (need API keys)
python -m exp.exp_1   # ... through exp_10
python -m exp.exp_4 --full   # paper's full grid-refinement range (N up to 15)
./reproduce.sh
```

## Conventions

- All experiment parameters live only in `src/config.py` as module-level
  `UPPER_CASE` constants, each with a comment naming its meaning, unit and
  paper section (or "not specified in the paper"). No parameter is set
  inside `exp/` or `src/*.py` function bodies.
- Market data is loaded only through `data/data_loader.py`
  (`load_ohlcv` for OHLCV bars via the QuantX Dataloader REST API,
  `load_option_eod_panel` for the end-of-day option panel via Massive). No
  other module calls an external API, and downloaded data is never committed
  (see `.gitignore`, `data/cache/`).
- API keys (`DATALOADER_API_KEY`, `MASSIVE_API_KEY`) are read from an
  uncommitted `.env` via `python-dotenv`; `.env.example` holds only empty
  placeholders.
- Tests: unit tests of `src/` run on synthetic fixtures built in
  `tests/conftest.py`. Tests that need a live data call are marked
  `@pytest.mark.market_data` and are skipped automatically when the API keys
  are absent; run `python -m pytest -m "not market_data"` for the default,
  offline test run.
- After re-running the experiment scripts, every file in `results/` is
  overwritten by the new run. The narrative, numbers and verdicts in
  `results/RESULTS.md` are not regenerated automatically: check them against
  the freshly written files and update them by hand if a number changed.

## Known data limitation

The empirical experiments (`exp_5`-`exp_10`) need an OptionMetrics-equivalent
end-of-day option panel (bid, ask, delta, implied volatility per listed
contract and trading day) for SPY, AMZN, GOOGL, META, TSLA and SPX over up to
25 years. `data/data_loader.py` attempts this through the Massive
options-analytics endpoint (`GET /options/v1/analytics`); the account used for
this replication returned HTTP 403 "You are not entitled to this data" on a
smoke test, so `exp_5`-`exp_10` run to completion but record a
`data_unavailable` status instead of fabricated numbers. See
`results/RESULTS.md` for details and for what a licensed data source would
need to supply (`src/options_violations.py` implements the full cleaning and
violation-rate pipeline once such a panel is available).
