#!/usr/bin/env bash
# dispatch_all.sh — Run all experiments in sequence.
#
# Usage: ./scripts/dispatch_all.sh [binary] [datadir] [resultsdir]
# Example: ./scripts/dispatch_all.sh ./build/run_sssp ./data/cache ./results

set -euo pipefail

BINARY="${1:-./build/run_sssp}"
DATADIR="${2:-./data/cache}"
OUTDIR="${3:-./results}"

# Environment check
bash scripts/env_lock.sh || { echo "Env lock failed — aborting"; exit 1; }

echo "=== Starting full experiment suite ==="
echo "Binary:   $BINARY"
echo "Data dir: $DATADIR"
echo "Out dir:  $OUTDIR"
echo ""

bash scripts/run_e1.sh "$BINARY" "$DATADIR" "$OUTDIR/e1"
bash scripts/run_e3.sh "$BINARY" "$DATADIR" "$OUTDIR/e3"
bash scripts/run_e6.sh "$BINARY" "$DATADIR" "$OUTDIR/e6"
bash scripts/run_e7.sh "$BINARY" "$DATADIR" "$OUTDIR/e7"
bash scripts/run_e4.sh "$BINARY" "$DATADIR" "$OUTDIR/e4"
bash scripts/run_e2.sh "$BINARY" "$BINARY"  "$DATADIR" "$OUTDIR/e2"
bash scripts/run_e5.sh "$BINARY" "$DATADIR" "$OUTDIR/e5"
bash scripts/run_e8.sh "$BINARY" "$DATADIR" "$OUTDIR/e8"
bash scripts/run_e9.sh "$BINARY" "$BINARY"  "$DATADIR" "$OUTDIR/e9"

echo ""
echo "=== All experiments done. Generating figures... ==="

python3 -m analysis.overhead_summary \
    --e6-dir "$OUTDIR/e6" \
    --e7-dir "$OUTDIR/e7" \
    --output "$OUTDIR/overhead.csv" \
    --print-table

python3 -m analysis.coverage_compare \
    --e4-dir "$OUTDIR/e4" \
    --output "$OUTDIR/coverage.csv" \
    --print-table

python3 -m analysis.make_figures \
    --results-dir "$OUTDIR" \
    --output paper/figures/

echo ""
echo "=== Done. Figures in paper/figures/ ==="
