#!/usr/bin/env bash
# E8: Scaling study — RMAT kron-22 through kron-26
# Generates in-memory RMAT graphs at each scale, runs GPU Δ-stepping,
# measures TEPS (with and without cert).
#
# Requires: run_sssp with --algo=delta_stepping_gpu --rmat-scale=N support
#           (or pre-generated .csr files in DATADIR)
# Output: results/sssp/e8/scaling.csv

set -euo pipefail
BINARY="${1:-./build/run_sssp}"
DATADIR="${2:-./data/cache}"
OUTDIR="${3:-./results/sssp/e8}"
REPS=3

mkdir -p "$OUTDIR"

SCALES=(22 23 24 25 26)
EDGEFACTOR=32
PRECISIONS=(fp32)

for scale in "${SCALES[@]}"; do
    CSR="$DATADIR/rmat_${scale}.csr"

    for prec in "${PRECISIONS[@]}"; do
        TAG="rmat_${scale}_${prec}"

        # If CSR not pre-generated, generate via binary
        if [[ ! -f "$CSR" ]]; then
            echo "[E8] generating RMAT scale=$scale ..."
            "$BINARY" \
                --rmat-scale="$scale" \
                --rmat-edgefactor="$EDGEFACTOR" \
                --rmat-seed=42 \
                --save-csr="$CSR"
        fi

        # Run with cert
        "$BINARY" \
            --dataset="$CSR" \
            --dataset-name="rmat_${scale}" \
            --algo=delta_stepping_gpu \
            --precision="$prec" \
            --emit-cert=1 \
            --verify=1 \
            --reps="$REPS" \
            --output="$OUTDIR/${TAG}.jsonl"

        echo "[E8] done scale=$scale  prec=$prec"
    done
done

# ── Aggregate to scaling.csv ───────────────────────────────────────────────────
python3 - "$OUTDIR" <<'PYEOF'
import json, pathlib, csv, sys, statistics

outdir = pathlib.Path(sys.argv[1])
rows_out = []

for jf in sorted(outdir.glob("rmat_*.jsonl")):
    stem  = jf.stem           # "rmat_25_fp32"
    parts = stem.split("_")
    scale = int(parts[1])
    prec  = parts[2]

    with open(jf) as f:
        rows = [json.loads(l) for l in f if l.strip()]

    if not rows:
        continue

    sssp_ms_list = [r["sssp_ms"] for r in rows if "sssp_ms" in r]
    med_ms = statistics.median(sssp_ms_list) if sssp_ms_list else 0
    ne = rows[0].get("dataset", {}).get("n_e", 1)
    teps = ne / (med_ms / 1000.0) if med_ms > 0 else 0
    gpu  = rows[0].get("hardware", {}).get("gpu", "unknown")

    rows_out.append({
        "scale":     scale,
        "precision": prec,
        "gpu":       gpu,
        "n_v":       rows[0].get("dataset", {}).get("n_v", 0),
        "n_e":       ne,
        "sssp_ms":   round(med_ms, 2),
        "teps":      round(teps, 0),
        "verdict":   rows[-1].get("verifier_verdict", "?"),
    })

out_csv = outdir / "scaling.csv"
if rows_out:
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows_out[0].keys())
        w.writeheader()
        w.writerows(rows_out)
    print(f"[E8] scaling.csv → {out_csv}  ({len(rows_out)} rows)")
PYEOF

echo "E8 done. Visualise with:"
echo "  python3 -m analysis.make_figures --figs fig7 --results-dir results/sssp/"
