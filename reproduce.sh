#!/usr/bin/env bash
# Runs the offline test suite and every experiment script in order,
# overwriting the files in results/.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "=== pytest (excluding market_data tests) ==="
python -m pytest -m "not market_data" || exit 1

for n in 1 2 3 4 5 6 7 8 9 10; do
    echo "=== exp_${n} ==="
    python -m "exp.exp_${n}" || echo "exp_${n} exited with a non-zero status (see results/exp${n}_results.json)"
done

echo "=== done: see results/RESULTS.md for the comparison against the paper ==="
