#!/usr/bin/env bash
# E3: Verifier soundness check
# Cross-check GPU SSSP outputs against Boost.Graph CPU reference.
# For each (dataset, GPU, precision), runs:
#   1. Boost Dijkstra reference → d_ref[]
#   2. GPU Δ-stepping           → d_gpu[] + cert
#   3. Verifier on d_gpu[] + cert
#   4. Direct comparison: d_ref vs d_gpu (golden comparison)
#
# Records: #vertices where verifier SAT but d differs > ε (verifier miss)
#          and #vertices where verifier UNSAT (true positive)

set -euo pipefail
BINARY="${1:-./build/run_sssp}"
DATADIR="${2:-./data/cache}"
OUTDIR="${3:-./results/sssp/e3}"
REPS=3

mkdir -p "$OUTDIR"

DATASETS=(ny_road usa_road livejournal web_google)
PRECISIONS=(fp32 fp64)

for ds in "${DATASETS[@]}"; do
    GR="$DATADIR/${ds}.gr"
    [[ -f "$GR" ]] || { echo "[skip] $ds"; continue; }

    for prec in "${PRECISIONS[@]}"; do
        TAG="${ds}_${prec}"

        # CPU Dijkstra reference
        "$BINARY" \
            --dataset="$GR" \
            --dataset-name="$ds" \
            --algo=dijkstra_cpu \
            --precision="$prec" \
            --emit-cert=1 \
            --verify=1 \
            --reps=1 \
            --output="$OUTDIR/ref_${TAG}.jsonl"

        # GPU Δ-stepping
        "$BINARY" \
            --dataset="$GR" \
            --dataset-name="$ds" \
            --algo=delta_stepping_gpu \
            --precision="$prec" \
            --emit-cert=1 \
            --verify=1 \
            --reps="$REPS" \
            --output="$OUTDIR/gpu_${TAG}.jsonl"

        echo "[E3] done: $TAG"
    done
done

# ── Compare ref vs GPU certs ───────────────────────────────────────────────────
python3 - "$OUTDIR" <<'PYEOF'
import pathlib, json, sys, csv
from collections import defaultdict

outdir = pathlib.Path(sys.argv[1])

def load_first(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)

results = []
for ref_f in sorted(outdir.glob("ref_*.jsonl")):
    tag = ref_f.stem[4:]  # strip "ref_"
    gpu_f = outdir / f"gpu_{tag}.jsonl"
    if not gpu_f.exists():
        continue
    ref = load_first(ref_f)
    gpu = load_first(gpu_f)
    if ref is None or gpu is None:
        continue

    d_ref = ref.get("cert_d", [])
    d_gpu = gpu.get("cert_d", [])
    n = min(len(d_ref), len(d_gpu))

    n_diff = sum(1 for i in range(n) if abs((d_ref[i] or 0) - (d_gpu[i] or 0)) > 1e-4)
    n_unsat = int(gpu.get("verifier_verdict", "SAT") != "SAT")

    results.append({
        "tag": tag,
        "n_vertices": n,
        "n_d_diff": n_diff,
        "n_unsat": n_unsat,
        "verdict_ref": ref.get("verifier_verdict", "?"),
        "verdict_gpu": gpu.get("verifier_verdict", "?"),
    })

out_csv = outdir / "verifier_soundness.csv"
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=results[0].keys() if results else [])
    w.writeheader()
    w.writerows(results)

print(f"[E3] verifier_soundness.csv → {out_csv}  ({len(results)} rows)")
for r in results:
    print(f"  {r['tag']:30s}  d_diff={r['n_d_diff']}  unsat={r['n_unsat']}")
PYEOF

echo "E3 done."
