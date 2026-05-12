#!/usr/bin/env bash
# E6: Certificate emission overhead
# Compares TEPS (traversed edges per second) with and without cert emission.
# Overhead target from paper: < 15%.
#
# Output: results/sssp/e6/*.jsonl  (paired baseline/augmented runs per dataset/GPU)
# Analysis: python3 -m analysis.overhead_summary --e6-dir results/sssp/e6

set -euo pipefail
BINARY="${1:-./build/run_sssp}"
DATADIR="${2:-./data/cache}"
OUTDIR="${3:-./results/sssp/e6}"
REPS=5   # more reps → tighter confidence interval on timing

mkdir -p "$OUTDIR"

DATASETS=(ny_road usa_road livejournal twitter kron_25)
PRECISIONS=(fp32 fp64)
SOURCE=0

# Warmup: 1 throwaway run per binary to warm GPU caches
echo "[E6] Warming up GPU..."
"$BINARY" --dataset="$DATADIR/ny_road.gr" --algo=delta_stepping_gpu \
          --emit-cert=0 --verify=0 --reps=1 --output=/dev/null 2>/dev/null || true

for ds in "${DATASETS[@]}"; do
    GR="$DATADIR/${ds}.gr"
    [[ -f "$GR" ]] || { echo "[skip] $ds"; continue; }

    for prec in "${PRECISIONS[@]}"; do
        TAG="${ds}_${prec}"

        # Baseline: no cert emission
        echo "[E6] baseline $TAG ..."
        "$BINARY" \
            --dataset="$GR" \
            --dataset-name="$ds" \
            --algo=delta_stepping_gpu \
            --precision="$prec" \
            --emit-cert=0 \
            --verify=0 \
            --source="$SOURCE" \
            --reps="$REPS" \
            --output="$OUTDIR/base_${TAG}.jsonl"

        # Augmented: with cert emission
        echo "[E6] augmented $TAG ..."
        "$BINARY" \
            --dataset="$GR" \
            --dataset-name="$ds" \
            --algo=delta_stepping_gpu \
            --precision="$prec" \
            --emit-cert=1 \
            --verify=0 \
            --source="$SOURCE" \
            --reps="$REPS" \
            --output="$OUTDIR/aug_${TAG}.jsonl"

        # Quick overhead report
        python3 - "$OUTDIR/base_${TAG}.jsonl" "$OUTDIR/aug_${TAG}.jsonl" <<'PYEOF'
import json, sys
def teps(path):
    with open(path) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    ms_vals = [r["sssp_ms"] for r in rows if "sssp_ms" in r]
    if not ms_vals: return 0
    med = sorted(ms_vals)[len(ms_vals)//2]
    ne = rows[0].get("dataset", {}).get("n_e", 1)
    return ne / (med / 1000.0)
t_base = teps(sys.argv[1])
t_aug  = teps(sys.argv[2])
oh = (t_base / t_aug - 1) * 100 if t_aug > 0 else 0
print(f"  overhead: {oh:+.1f}%  (base={t_base:.2e} TEPS, aug={t_aug:.2e} TEPS)")
PYEOF
    done
done

# Merge all jsonl into e6 aggregate dir for overhead_summary.py
cat "$OUTDIR"/*.jsonl > "$OUTDIR/all_e6.jsonl" 2>/dev/null || true

echo "E6 done. Analyse with:"
echo "  python3 -m analysis.overhead_summary --e6-dir $OUTDIR --output results/sssp/overhead.csv"
