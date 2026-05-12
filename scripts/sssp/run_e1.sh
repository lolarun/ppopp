#!/usr/bin/env bash
# E1: Cross-platform drift measurement
# Runs run_sssp on all (dataset × precision) combos, outputs JSONL.
# Run separately on NVIDIA and AMD hosts; compare results offline.

set -euo pipefail
BINARY="${1:-./build/run_sssp}"
DATADIR="${2:-./data/cache}"
OUTDIR="${3:-./results/sssp/e1}"
REPS=3

mkdir -p "$OUTDIR"

DATASETS=(ny_road usa_road livejournal web_google)
PRECISIONS=(fp32 fp64)
SOURCE=0

for ds in "${DATASETS[@]}"; do
    GR="$DATADIR/${ds}.gr"
    if [[ ! -f "$GR" ]]; then
        echo "[skip] $ds not found at $GR"
        continue
    fi
    for prec in "${PRECISIONS[@]}"; do
        OUT="$OUTDIR/${ds}_${prec}_$(hostname).jsonl"
        echo "[run] $ds $prec → $OUT"
        "$BINARY" \
            --dataset="$GR" \
            --dataset-name="$ds" \
            --algo=delta_stepping_gpu \
            --precision="$prec" \
            --source="$SOURCE" \
            --emit-cert=1 \
            --verify=1 \
            --reps="$REPS" \
            --output="$OUT"
    done
done

echo ""
echo "E1 done. Results in $OUTDIR"
echo "Next: copy all .jsonl to analysis host, run:"
echo "  python3 src/analysis/drift_compare.py --e1-dir $OUTDIR"
