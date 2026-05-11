#!/usr/bin/env bash
# Pairwise drift comparison across all (vendor,seed) combinations for each dataset.
# Reads results/<tag-A>/<dataset>_seed*.bin and results/<tag-B>/<dataset>_seed*.bin,
# emits one JSON per pair into results/_compare/<dataset>__<tagA>_seedX__<tagB>_seedY.json,
# plus a final summary table on stdout.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS="$ROOT/results"
OUT="$RESULTS/_compare"
DRIFT_PY="$ROOT/src/analysis/drift_compare.py"
mkdir -p "$OUT"

# Find all device tags (subdirectories of results/, excluding _compare)
mapfile -t tags < <(find "$RESULTS" -maxdepth 1 -mindepth 1 -type d -not -name "_compare" -printf "%f\n" | sort)
if [[ ${#tags[@]} -lt 2 ]]; then
    echo "need >=2 device tags under $RESULTS/, got: ${tags[*]}" >&2
    exit 1
fi
echo "[compare_all] tags: ${tags[*]}"

# All datasets (union over tags)
mapfile -t datasets < <(
    for t in "${tags[@]}"; do
        find "$RESULTS/$t" -maxdepth 1 -name "*_seed*.bin" -printf "%f\n"
    done | sed -E 's/_seed[0-9]+\.bin$//' | sort -u
)
echo "[compare_all] datasets: ${datasets[*]}"

# Cross-vendor comparisons (tagA != tagB, all seed pairs).
for ds in "${datasets[@]}"; do
    for ((i=0; i<${#tags[@]}; i++)); do
        for ((j=i+1; j<${#tags[@]}; j++)); do
            ta="${tags[i]}"; tb="${tags[j]}"
            for a_bin in "$RESULTS/$ta/${ds}_seed"*.bin; do
                [[ -e "$a_bin" ]] || continue
                seed_a="$(basename "$a_bin" .bin | sed -E 's/.*_seed//')"
                for b_bin in "$RESULTS/$tb/${ds}_seed"*.bin; do
                    [[ -e "$b_bin" ]] || continue
                    seed_b="$(basename "$b_bin" .bin | sed -E 's/.*_seed//')"
                    out="$OUT/${ds}__${ta}_seed${seed_a}__${tb}_seed${seed_b}.json"
                    python3 "$DRIFT_PY" \
                        --a "$RESULTS/$ta/${ds}_seed${seed_a}" \
                        --b "$RESULTS/$tb/${ds}_seed${seed_b}" \
                        --out "$out" >/dev/null
                done
            done
        done
    done
done

# Same-tag (cross-run) comparisons too — measure within-vendor variance
for ds in "${datasets[@]}"; do
    for t in "${tags[@]}"; do
        bins=("$RESULTS/$t/${ds}_seed"*.bin)
        [[ -e "${bins[0]:-}" ]] || continue
        for ((i=0; i<${#bins[@]}; i++)); do
            for ((j=i+1; j<${#bins[@]}; j++)); do
                seed_a="$(basename "${bins[i]}" .bin | sed -E 's/.*_seed//')"
                seed_b="$(basename "${bins[j]}" .bin | sed -E 's/.*_seed//')"
                out="$OUT/${ds}__${t}_seed${seed_a}__${t}_seed${seed_b}.json"
                python3 "$DRIFT_PY" \
                    --a "$RESULTS/$t/${ds}_seed${seed_a}" \
                    --b "$RESULTS/$t/${ds}_seed${seed_b}" \
                    --out "$out" >/dev/null
            done
        done
    done
done

# Summary table
python3 - <<'PY' "$OUT"
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
rows = []
for f in sorted(out.glob("*.json")):
    d = json.loads(f.read_text())
    rows.append((d["dataset"],
                 d["a"]["vendor"], d["a"]["device"], d["a"]["crc32"],
                 d["b"]["vendor"], d["b"]["device"], d["b"]["crc32"],
                 d["byte_diff_fraction"], d["max_Linf"],
                 d["L2_norm"], d["rank_top100_jaccard"],
                 d["decision"]))
print(f"{'dataset':<14} {'A vendor':<8} {'B vendor':<8} "
      f"{'byte_diff':>10} {'max_Linf':>10} {'L2':>10} "
      f"{'top100_J':>9} {'decision':<18}")
for r in rows:
    ds, av, _, _, bv, _, _, bd, mi, l2, jc, dec = r
    print(f"{ds:<14} {av:<8} {bv:<8} "
          f"{bd:>10.4%} {mi:>10.3e} {l2:>10.3e} "
          f"{jc:>9.3f} {dec:<18}")
PY
