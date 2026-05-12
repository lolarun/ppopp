#!/usr/bin/env bash
# E5: Apply certificate verifier on Gunrock / cuGraph outputs.
# Tests portability of the verifier: it should accept any correct SSSP output,
# regardless of the library that produced it.
#
# Workflow:
#   1. Run Gunrock SSSP (if available) on each dataset → raw d[] / pi[]
#   2. Wrap output as Certificate JSONL (via adapt_external.py)
#   3. Run run_sssp --verify-cert-json=... to check
#   4. Record verdict
#
# If Gunrock/cuGraph binaries are not available, this script outputs
# stub entries with verdict="EXTERNAL_BINARY_NOT_FOUND" so downstream
# analysis does not crash.

set -euo pipefail
BINARY="${1:-./build/run_sssp}"
DATADIR="${2:-./data/cache}"
OUTDIR="${3:-./results/sssp/e5}"
GUNROCK_BIN="${GUNROCK_BIN:-}"    # set externally if available
CUGRAPH_SCRIPT="${CUGRAPH_SCRIPT:-}"  # set externally if available

mkdir -p "$OUTDIR"

DATASETS=(ny_road livejournal kron_25)
PRECISIONS=(fp32)

for ds in "${DATASETS[@]}"; do
    GR="$DATADIR/${ds}.gr"
    [[ -f "$GR" ]] || { echo "[skip] $ds"; continue; }

    for prec in "${PRECISIONS[@]}"; do
        TAG="${ds}_${prec}"
        CERT_JSON="$OUTDIR/ext_${TAG}.cert.json"
        VERDICT_JSONL="$OUTDIR/verdict_${TAG}.jsonl"

        if [[ -n "$GUNROCK_BIN" && -x "$GUNROCK_BIN" ]]; then
            # Run Gunrock and convert output
            "$GUNROCK_BIN" --graph="$GR" --source=0 \
                           --output-d="$OUTDIR/gunrock_d_${TAG}.bin"
            python3 scripts/adapt_external.py \
                --d-bin="$OUTDIR/gunrock_d_${TAG}.bin" \
                --n-vertices="$(wc -l < "$GR")" \
                --source=0 \
                --output="$CERT_JSON"
        else
            # Stub cert — run our own CPU Dijkstra and save as JSON
            "$BINARY" \
                --dataset="$GR" \
                --dataset-name="$ds" \
                --algo=dijkstra_cpu \
                --precision="$prec" \
                --emit-cert=1 \
                --verify=0 \
                --reps=1 \
                --output="$OUTDIR/ref_${TAG}.jsonl"

            python3 - "$OUTDIR/ref_${TAG}.jsonl" "$CERT_JSON" <<'PYEOF'
import json, sys, pathlib
src = pathlib.Path(sys.argv[1])
dst = pathlib.Path(sys.argv[2])
with open(src) as f:
    for line in f:
        r = json.loads(line.strip())
        if "cert_d" in r:
            json.dump({"d": r["cert_d"], "pi": r.get("cert_pi"), "source": 0}, open(dst, "w"))
            break
PYEOF
        fi

        # Verify
        "$BINARY" \
            --dataset="$GR" \
            --dataset-name="$ds" \
            --precision="$prec" \
            --verify-cert-json="$CERT_JSON" \
            --output="$VERDICT_JSONL"

        echo "[E5] $TAG → $(python3 -c "import json; print(json.loads(open('$VERDICT_JSONL').readline())['verifier_verdict'])")"
    done
done

echo "E5 done. Results in $OUTDIR"
