#!/usr/bin/env bash
# One-shot extension batch run on A10 — generates 4 more canonical CSRs,
# runs fp32 + fp64 matrices over all 7 datasets (idempotent on existing 3),
# emits sha256 manifest.  Intended to run under nohup so it survives SSH drops.

set -euo pipefail
export PATH=/usr/local/cuda/bin:$PATH
cd /root/ppopp-2027

LOG=/tmp/a10_extend.log
exec > >(tee -a "$LOG") 2>&1

echo "=== START $(date -Is) ==="

mkdir -p data/raw data/cache

declare -A SOURCES=(
    [road-CA]=https://snap.stanford.edu/data/roadNet-CA.txt.gz
    [as-skitter]=https://snap.stanford.edu/data/as-skitter.txt.gz
    [wiki-Talk]=https://snap.stanford.edu/data/wiki-Talk.txt.gz
)

# Originally also included cit-Patents (140 MB compressed) and soc-Pokec (140 MB compressed).
# Both hit SNAP throttling from this region — sustained ~17-65 KB/s, meaning multi-hour download
# times that were not worth the marginal dataset variety. Final 6-dataset list satisfies RE1's
# 6-8 target; livejournal already covers the social high-skew slot soc-Pokec would have provided.

for name in road-CA as-skitter wiki-Talk; do
    url="${SOURCES[$name]}"
    txt="data/raw/${name}.txt"
    gz="data/raw/${name}.txt.gz"
    csr="data/cache/${name}.csr.bin"

    if [[ -f "$csr" ]]; then
        echo "[$name] csr already exists, skipping"
        continue
    fi

    if [[ ! -f "$txt" ]]; then
        echo "[$name] downloading $url"
        curl -fsSL -o "$gz" "$url"
        echo "[$name] gunzipping"
        gunzip -f "$gz"
        # Some SNAP files unzip to a different basename — normalise.
        if [[ ! -f "$txt" ]]; then
            # Try common SNAP names
            for candidate in "data/raw/${name}.txt" \
                             "data/raw/roadNet-CA.txt" \
                             "data/raw/cit-Patents.txt" \
                             "data/raw/wiki-Talk.txt" \
                             "data/raw/soc-pokec-relationships.txt"; do
                if [[ -f "$candidate" && "$candidate" != "$txt" ]]; then
                    mv "$candidate" "$txt"
                    break
                fi
            done
        fi
    fi

    echo "[$name] converting to CSR"
    python3 scripts/snap_to_csr.py --input "$txt" --output "$csr"
done

echo
echo "=== sha256 of all CSRs (canonical anchors) ==="
sha256sum data/cache/*.csr.bin

echo
echo "=== fp32 matrix ==="
GPU_BACKEND=CUDA CUDA_ARCH=86 DEVICE_TAG=a10_sm86 PRECISION=fp32 ./scripts/run_re0.sh

echo
echo "=== fp64 matrix ==="
GPU_BACKEND=CUDA CUDA_ARCH=86 DEVICE_TAG=a10_sm86 PRECISION=fp64 ./scripts/run_re0.sh

echo
echo "=== final tally ==="
echo "fp32 results:"
ls results/a10_sm86/fp32 | wc -l
echo "fp64 results:"
ls results/a10_sm86/fp64 | wc -l

echo "=== END $(date -Is) ==="
