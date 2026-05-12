#!/usr/bin/env bash
# E9: Stress test — weight distribution × precision × graph structure
# Sweeps:
#   weight_dist: uniform[0.001,1], gaussian(0.5,0.2), powerlaw(alpha=2), adversarial
#   precision:   fp32, fp64
#   datasets:    ny_road, livejournal, kron_25  (one per structure class)
#
# Measures: % vertices with d[] byte-diff cross-platform, verifier verdict.
# Output: results/sssp/e9/stress.csv

set -euo pipefail
BINARY_CUDA="${1:-./build_cuda/run_sssp}"
BINARY_ROCM="${2:-./build_rocm/run_sssp}"
DATADIR="${3:-./data/cache}"
OUTDIR="${4:-./results/sssp/e9}"
REPS=3
SOURCE=0

mkdir -p "$OUTDIR"

DATASETS=(ny_road livejournal kron_25)
PRECISIONS=(fp32 fp64)
WEIGHT_DISTS=(uniform gaussian powerlaw adversarial)

for ds in "${DATASETS[@]}"; do
    GR="$DATADIR/${ds}.gr"
    [[ -f "$GR" ]] || { echo "[skip] $ds"; continue; }

    for prec in "${PRECISIONS[@]}"; do
        for wdist in "${WEIGHT_DISTS[@]}"; do
            TAG="${ds}_${prec}_${wdist}"
            echo "[E9] $TAG ..."

            # NVIDIA run
            if [[ -f "$BINARY_CUDA" ]]; then
                "$BINARY_CUDA" \
                    --dataset="$GR" \
                    --dataset-name="$ds" \
                    --algo=delta_stepping_gpu \
                    --precision="$prec" \
                    --weight-dist="$wdist" \
                    --emit-cert=1 \
                    --verify=1 \
                    --source="$SOURCE" \
                    --reps="$REPS" \
                    --output="$OUTDIR/cuda_${TAG}.jsonl" || \
                    echo "[warn] CUDA run failed for $TAG"
            fi

            # AMD run
            if [[ -f "$BINARY_ROCM" ]]; then
                "$BINARY_ROCM" \
                    --dataset="$GR" \
                    --dataset-name="$ds" \
                    --algo=delta_stepping_gpu \
                    --precision="$prec" \
                    --weight-dist="$wdist" \
                    --emit-cert=1 \
                    --verify=1 \
                    --source="$SOURCE" \
                    --reps="$REPS" \
                    --output="$OUTDIR/rocm_${TAG}.jsonl" || \
                    echo "[warn] AMD run failed for $TAG"
            fi
        done
    done
done

# ── Compare CUDA vs ROCm, aggregate to stress.csv ─────────────────────────────
python3 - "$OUTDIR" <<'PYEOF'
import json, pathlib, csv, sys, math

outdir = pathlib.Path(sys.argv[1])
rows_out = []

for cuda_f in sorted(outdir.glob("cuda_*.jsonl")):
    tag    = cuda_f.stem[5:]        # strip "cuda_"
    rocm_f = outdir / f"rocm_{tag}.jsonl"
    if not rocm_f.exists():
        continue

    def load(p):
        with open(p) as f:
            rows = [json.loads(l) for l in f if l.strip()]
        return rows

    c_rows = load(cuda_f)
    r_rows = load(rocm_f)
    if not c_rows or not r_rows:
        continue

    d_c = c_rows[0].get("cert_d", [])
    d_r = r_rows[0].get("cert_d", [])
    n = min(len(d_c), len(d_r))
    INF = 1e30
    n_diff = sum(1 for i in range(n)
                 if abs((d_c[i] or INF) - (d_r[i] or INF)) > 1e-5)
    pct = 100.0 * n_diff / max(1, n)

    parts = tag.split("_")
    ds, prec = parts[0], parts[1]
    wdist = "_".join(parts[2:])

    rows_out.append({
        "dataset":       ds,
        "precision":     prec,
        "weight_dist":   wdist,
        "n_vertices":    n,
        "n_d_diff":      n_diff,
        "n_d_diff_pct":  round(pct, 3),
        "verdict_cuda":  c_rows[0].get("verifier_verdict", "?"),
        "verdict_rocm":  r_rows[0].get("verifier_verdict", "?"),
    })

out_csv = outdir / "stress.csv"
if rows_out:
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows_out[0].keys())
        w.writeheader()
        w.writerows(rows_out)
    print(f"[E9] stress.csv → {out_csv}  ({len(rows_out)} rows)")
else:
    print("[E9] No cross-platform pairs found; check if both CUDA and AMD binaries ran")
PYEOF

echo "E9 done. Visualise with:"
echo "  python3 -m analysis.make_figures --figs fig8 --results-dir results/sssp/"
