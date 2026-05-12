#!/usr/bin/env bash
# Fetch and preprocess benchmark datasets into data/cache/*.gr
# Run once after cloning. Requires: wget, curl.

set -euo pipefail
CACHE_DIR="$(dirname "$0")/../data/cache"
mkdir -p "$CACHE_DIR"

# ── DIMACS road networks ──────────────────────────────────────────────────────
# 9th DIMACS Challenge: http://www.diag.uniroma1.it/challenge9/download.shtml

fetch_dimacs() {
    local name="$1" url="$2"
    local dest="$CACHE_DIR/${name}.gr.gz"
    if [[ -f "$CACHE_DIR/${name}.gr" ]]; then
        echo "[skip] $name already cached"
        return
    fi
    echo "[fetch] $name ..."
    wget -q -O "$dest" "$url"
    gunzip "$dest"
    echo "[done]  $CACHE_DIR/${name}.gr"
}

# New York
fetch_dimacs "ny_road" \
    "http://www.diag.uniroma1.it/challenge9/data/USA-road-d/USA-road-d.NY.gr.gz"

# USA full
fetch_dimacs "usa_road" \
    "http://www.diag.uniroma1.it/challenge9/data/USA-road-d/USA-road-d.USA.gr.gz"

# ── SNAP social networks ──────────────────────────────────────────────────────
# These are edge-list format — need conversion to DIMACS .gr
# Conversion done by scripts/snap_to_dimacs.py

fetch_snap() {
    local name="$1" url="$2"
    local raw="$CACHE_DIR/${name}.txt.gz"
    local out="$CACHE_DIR/${name}.gr"
    if [[ -f "$out" ]]; then
        echo "[skip] $name already cached"; return
    fi
    echo "[fetch] $name ..."
    wget -q -O "$raw" "$url"
    gunzip "$raw"
    python3 "$(dirname "$0")/snap_to_dimacs.py" \
        "$CACHE_DIR/${name}.txt" "$out"
    echo "[done]  $out"
}

fetch_snap "livejournal" \
    "https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz"

fetch_snap "web_google" \
    "https://snap.stanford.edu/data/web-Google.txt.gz"

echo ""
echo "Cached datasets in $CACHE_DIR:"
ls -lh "$CACHE_DIR"/*.gr 2>/dev/null || echo "(none yet)"
