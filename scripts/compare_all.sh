#!/usr/bin/env bash
# Pairwise drift comparison across all (vendor, seed) combinations within
# the same precision (no fp32-vs-fp64 mixing).
#
# Scans results/<tag>/<precision>/<dataset>_seed*.bin and emits one JSON
# per pair into results/_compare/<precision>/<dataset>__<tagA>_seedX__<tagB>_seedY.json,
# plus a final summary table on stdout.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS="$ROOT/results"
DRIFT_PY="$ROOT/src/analysis/drift_compare.py"

# Find all (tag, precision) combinations under results/
mapfile -t precision_dirs < <(find "$RESULTS" -mindepth 2 -maxdepth 2 -type d -not -path "*/_compare*" -not -path "*/_evidence*" -printf "%P\n" | sort)
if [[ ${#precision_dirs[@]} -eq 0 ]]; then
    echo "no results/<tag>/<precision>/ subdirs found under $RESULTS/" >&2
    exit 1
fi

# Group by precision so we never mix fp32 vs fp64.
declare -A precision_tags
for pd in "${precision_dirs[@]}"; do
    tag="${pd%/*}"
    prec="${pd##*/}"
    precision_tags[$prec]+="$tag "
done

for prec in "${!precision_tags[@]}"; do
    tags=(${precision_tags[$prec]})
    OUT="$RESULTS/_compare/$prec"
    mkdir -p "$OUT"
    echo "[compare_all] precision=$prec  tags: ${tags[*]}"

    if [[ ${#tags[@]} -lt 2 ]]; then
        echo "  skipping cross-vendor (only one tag for this precision)"
    fi

    # All datasets at this precision (union over tags).
    mapfile -t datasets < <(
        for t in "${tags[@]}"; do
            find "$RESULTS/$t/$prec" -maxdepth 1 -name "*_seed*.bin" -printf "%f\n" 2>/dev/null
        done | sed -E 's/_seed[0-9]+\.bin$//' | sort -u
    )
    echo "  datasets: ${datasets[*]}"

    # Cross-tag pairs (tagA != tagB).
    for ds in "${datasets[@]}"; do
        for ((i=0; i<${#tags[@]}; i++)); do
            for ((j=i+1; j<${#tags[@]}; j++)); do
                ta="${tags[i]}"; tb="${tags[j]}"
                for a_bin in "$RESULTS/$ta/$prec/${ds}_seed"*.bin; do
                    [[ -e "$a_bin" ]] || continue
                    seed_a="$(basename "$a_bin" .bin | sed -E 's/.*_seed//')"
                    for b_bin in "$RESULTS/$tb/$prec/${ds}_seed"*.bin; do
                        [[ -e "$b_bin" ]] || continue
                        seed_b="$(basename "$b_bin" .bin | sed -E 's/.*_seed//')"
                        out_json="$OUT/${ds}__${ta}_seed${seed_a}__${tb}_seed${seed_b}.json"
                        python3 "$DRIFT_PY" \
                            --a "$RESULTS/$ta/$prec/${ds}_seed${seed_a}" \
                            --b "$RESULTS/$tb/$prec/${ds}_seed${seed_b}" \
                            --out "$out_json" >/dev/null
                    done
                done
            done
        done
    done

    # Same-tag (cross-run) pairs — within-vendor variance.
    for ds in "${datasets[@]}"; do
        for t in "${tags[@]}"; do
            bins=("$RESULTS/$t/$prec/${ds}_seed"*.bin)
            [[ -e "${bins[0]:-}" ]] || continue
            for ((i=0; i<${#bins[@]}; i++)); do
                for ((j=i+1; j<${#bins[@]}; j++)); do
                    seed_a="$(basename "${bins[i]}" .bin | sed -E 's/.*_seed//')"
                    seed_b="$(basename "${bins[j]}" .bin | sed -E 's/.*_seed//')"
                    out_json="$OUT/${ds}__${t}_seed${seed_a}__${t}_seed${seed_b}.json"
                    python3 "$DRIFT_PY" \
                        --a "$RESULTS/$t/$prec/${ds}_seed${seed_a}" \
                        --b "$RESULTS/$t/$prec/${ds}_seed${seed_b}" \
                        --out "$out_json" >/dev/null
                done
            done
        done
    done
done

# Summary table across all precisions
python3 - <<'PY' "$RESULTS/_compare"
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = []
for sub in sorted(root.iterdir()):
    if not sub.is_dir(): continue
    prec = sub.name
    for f in sorted(sub.glob("*.json")):
        d = json.loads(f.read_text())
        rows.append((prec, d["dataset"],
                     d["a"]["vendor"], d["b"]["vendor"],
                     d["byte_diff_fraction"], d["max_Linf"],
                     d["L2_norm"], d["rank_top100_jaccard"],
                     d["decision"]))
print(f"{'prec':<6}{'dataset':<14}{'A':<8}{'B':<8}{'byte_diff':>10}"
      f"{'max_Linf':>12}{'L2':>12}{'top100_J':>10}{'decision':<18}")
for r in rows:
    p, ds, av, bv, bd, mi, l2, jc, dec = r
    print(f"{p:<6}{ds:<14}{av:<8}{bv:<8}{bd:>10.2%}"
          f"{mi:>12.3e}{l2:>12.3e}{jc:>10.3f}{dec:<18}")
PY
