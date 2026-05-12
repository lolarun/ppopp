#!/usr/bin/env bash
# One-shot batch for MI300X — regenerates all 6 canonical CSRs on this host
# (F6 guarantees byte-identical to local data/cache/), builds with HIP/gfx942,
# runs fp32 + fp64 matrices.  Intended for nohup execution.

set -euo pipefail
export PATH=/opt/rocm/bin:$PATH
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm
cd /root/ppopp-2027

LOG=/tmp/mi300x_batch.log
exec > >(tee -a "$LOG") 2>&1

echo "=== START $(date -Is) ==="

mkdir -p data/raw data/cache

declare -A SOURCES=(
    [web-google]=https://snap.stanford.edu/data/web-Google.txt.gz
    [livejournal]=https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz
    [road-CA]=https://snap.stanford.edu/data/roadNet-CA.txt.gz
    [as-skitter]=https://snap.stanford.edu/data/as-skitter.txt.gz
    [wiki-Talk]=https://snap.stanford.edu/data/wiki-Talk.txt.gz
)

# Map SNAP gz name -> our canonical name (for files whose gz unpacks to a different basename)
declare -A SNAP_FILE=(
    [web-google]=web-Google.txt
    [livejournal]=soc-LiveJournal1.txt
    [road-CA]=roadNet-CA.txt
    [as-skitter]=as-skitter.txt
    [wiki-Talk]=wiki-Talk.txt
)

# Download + convert SNAP datasets
for name in web-google livejournal road-CA as-skitter wiki-Talk; do
    csr="data/cache/${name}.csr.bin"
    if [[ -f "$csr" ]]; then
        echo "[$name] csr already exists, skipping"
        continue
    fi
    txt_canonical="data/raw/${name}.txt"
    if [[ ! -f "$txt_canonical" ]]; then
        url="${SOURCES[$name]}"
        snap_basename="${SNAP_FILE[$name]}"
        echo "[$name] downloading $url"
        curl -fsSL -o "data/raw/${name}.txt.gz" "$url"
        gunzip -f "data/raw/${name}.txt.gz"
        if [[ -f "data/raw/${snap_basename}" && "data/raw/${snap_basename}" != "$txt_canonical" ]]; then
            mv "data/raw/${snap_basename}" "$txt_canonical"
        fi
    fi
    echo "[$name] converting to CSR"
    python3 scripts/snap_to_csr.py --input "$txt_canonical" --output "$csr"
done

# Synthetic RMAT
if [[ ! -f data/cache/rmat-22.csr.bin ]]; then
    echo "[rmat-22] generating"
    python3 scripts/gen_rmat.py --scale 22 --seed 42 --output data/cache/rmat-22.csr.bin
fi

echo
echo "=== CSR sha256 (should match local data/csr_anchors.txt) ==="
sha256sum data/cache/*.csr.bin

echo
echo "=== fp32 matrix ==="
GPU_BACKEND=ROCM HIP_ARCH=gfx942 DEVICE_TAG=mi300x_vf PRECISION=fp32 ./scripts/run_re0.sh

echo
echo "=== fp64 matrix ==="
GPU_BACKEND=ROCM HIP_ARCH=gfx942 DEVICE_TAG=mi300x_vf PRECISION=fp64 ./scripts/run_re0.sh

echo
echo "=== final tally ==="
echo "fp32 results: $(ls results/mi300x_vf/fp32 | wc -l)"
echo "fp64 results: $(ls results/mi300x_vf/fp64 | wc -l)"
echo "=== END $(date -Is) ==="
