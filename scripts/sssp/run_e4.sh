#!/usr/bin/env bash
# E4: Controlled error injection + detection rate comparison
# Requires: run_sssp (with --emit-cert), inject_errors binary, verify_cert binary
# For now, error injection is done via Python until C++ injector is wired in.

set -euo pipefail
BINARY="${1:-./build/run_sssp}"
DATADIR="${2:-./data/cache}"
OUTDIR="${3:-./results/sssp/e4}"
REPS=1

mkdir -p "$OUTDIR"

DATASETS=(ny_road livejournal)
ERROR_KINDS=(DISTANCE_PERTURB PREDECESSOR_RANDOM INCONSISTENT MISSED_UNREACHABLE)
N_SEEDS=50

for ds in "${DATASETS[@]}"; do
    GR="$DATADIR/${ds}.gr"
    [[ -f "$GR" ]] || { echo "[skip] $ds"; continue; }

    # Step 1: generate clean certificate
    CLEAN_JSONL="$OUTDIR/${ds}_clean.jsonl"
    "$BINARY" --dataset="$GR" --dataset-name="$ds" \
        --algo=dijkstra_cpu --precision=fp32 \
        --emit-cert=1 --verify=1 --reps=1 \
        --output="$CLEAN_JSONL"

    # Step 2: inject errors and verify (Python harness)
    for kind in "${ERROR_KINDS[@]}"; do
        python3 scripts/inject_and_verify.py \
            --clean-jsonl="$CLEAN_JSONL" \
            --graph="$GR" \
            --error-kind="$kind" \
            --n-seeds="$N_SEEDS" \
            --output="$OUTDIR/${ds}_${kind}.jsonl"
    done
done

echo "E4 done. Aggregate with:"
echo "  python3 src/analysis/coverage_compare.py --e4-dir $OUTDIR"
