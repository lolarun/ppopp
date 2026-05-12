#!/usr/bin/env bash
# E7: Verifier cost vs SSSP recompute
# Measures: verify(cert) wall time vs re-running SSSP from scratch.
# Key claim: verifier << recompute (soundness at low marginal cost).
#
# Runs on both CPU-only verifier and (if GPU_VERIFIER=1) GPU verifier.
# Output: results/sssp/e7/*.jsonl
# Analysis: python3 -m analysis.overhead_summary --e7-dir results/sssp/e7

set -euo pipefail
BINARY="${1:-./build/run_sssp}"
DATADIR="${2:-./data/cache}"
OUTDIR="${3:-./results/sssp/e7}"
REPS=3

mkdir -p "$OUTDIR"

DATASETS=(ny_road usa_road livejournal twitter friendster kron_25)
PRECISIONS=(fp32)   # fp64 for supplemental
SOURCE=0

for ds in "${DATASETS[@]}"; do
    GR="$DATADIR/${ds}.gr"
    [[ -f "$GR" ]] || { echo "[skip] $ds"; continue; }

    for prec in "${PRECISIONS[@]}"; do
        TAG="${ds}_${prec}"

        # Step 1: generate cert (verify=0 → pure SSSP time)
        "$BINARY" \
            --dataset="$GR" \
            --dataset-name="$ds" \
            --algo=delta_stepping_gpu \
            --precision="$prec" \
            --emit-cert=1 \
            --verify=0 \
            --source="$SOURCE" \
            --reps="$REPS" \
            --output="$OUTDIR/sssp_${TAG}.jsonl"

        # Step 2: verify only (verify=1, no GPU re-run → only verifier timed)
        # We pass --reps=1 since verification is deterministic
        "$BINARY" \
            --dataset="$GR" \
            --dataset-name="$ds" \
            --algo=delta_stepping_gpu \
            --precision="$prec" \
            --emit-cert=1 \
            --verify=1 \
            --source="$SOURCE" \
            --reps=1 \
            --output="$OUTDIR/verify_${TAG}.jsonl"

        # Report ratio
        python3 - \
            "$OUTDIR/sssp_${TAG}.jsonl" \
            "$OUTDIR/verify_${TAG}.jsonl" <<'PYEOF'
import json, sys
def load(p):
    with open(p) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    return rows
sssp_rows = load(sys.argv[1])
ver_rows  = load(sys.argv[2])
sssp_ms   = sorted(r["sssp_ms"] for r in sssp_rows if "sssp_ms" in r)[len(sssp_rows)//2]
ver_ms    = ver_rows[0].get("verifier_ms", 0) if ver_rows else 0
ratio     = ver_ms / sssp_ms if sssp_ms > 0 else 0
print(f"  SSSP={sssp_ms:.0f}ms  Verif={ver_ms:.0f}ms  ratio={ratio:.3f}x")
PYEOF
        echo "[E7] done: $TAG"
    done
done

echo "E7 done. Analyse with:"
echo "  python3 -m analysis.overhead_summary --e7-dir $OUTDIR --output results/sssp/overhead.csv"
