#!/usr/bin/env bash
# RE2e_v2 — fully deterministic pull (ZERO atomicAdd) byte-identity check.
#
# Runs pagerank_pull_v2 on all CSRs × 5 seeds, then checks sha256 identity.
# If ALL datasets are byte-identical across seeds → atomicAdd is the sole
# drift source (F12). If any still drift → third mechanism exists.
#
# Outputs go to results/pagerank/<DEVICE_TAG>/<PRECISION>_pull_v2/
#
# Usage:
#   GPU_BACKEND=CUDA CUDA_ARCH=75 DEVICE_TAG=t4_sm75 PRECISION=fp32 ./scripts/run_re2e_v2.sh
#   PRECISION=fp64 ./scripts/run_re2e_v2.sh   # after fp32

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GPU_BACKEND="${GPU_BACKEND:-CUDA}"
CUDA_ARCH="${CUDA_ARCH:-86}"
HIP_ARCH="${HIP_ARCH:-gfx942}"
DEVICE_TAG="${DEVICE_TAG:-}"
PRECISION="${PRECISION:-fp32}"
SEEDS="${SEEDS:-0 1 2 3 4}"
DAMPING="${DAMPING:-0.85}"
MAX_ITER="${MAX_ITER:-100}"
TOL="${TOL:-1e-6}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build_gpu}"

if [[ -z "$DEVICE_TAG" ]]; then
    if   [[ "$GPU_BACKEND" == "CUDA" ]]; then DEVICE_TAG="cuda_${CUDA_ARCH}"
    elif [[ "$GPU_BACKEND" == "ROCM" ]]; then DEVICE_TAG="rocm_${HIP_ARCH}"
    fi
fi

OUT_DIR="$ROOT/results/pagerank/${DEVICE_TAG}/${PRECISION}_pull_v2"
mkdir -p "$OUT_DIR"

echo "[re2e_v2] backend=$GPU_BACKEND tag=$DEVICE_TAG precision=$PRECISION"

# --- Build ---
if [[ ! -x "$BUILD_DIR/pagerank_pull_v2" ]]; then
    if [[ "$GPU_BACKEND" == "CUDA" ]]; then
        cmake -S "$ROOT" -B "$BUILD_DIR" \
              -DGPU_BACKEND=CUDA \
              -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
              -DCMAKE_BUILD_TYPE=Release
    else
        cmake -S "$ROOT" -B "$BUILD_DIR" \
              -DGPU_BACKEND=ROCM \
              -DCMAKE_HIP_ARCHITECTURES="$HIP_ARCH" \
              -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/amdclang++ \
              -DCMAKE_PREFIX_PATH=/opt/rocm \
              -DCMAKE_BUILD_TYPE=Release
    fi
fi
cmake --build "$BUILD_DIR" -j

BIN="$BUILD_DIR/pagerank_pull_v2"

# --- Run ---
shopt -s nullglob
csrs=("$ROOT"/data/cache/*.csr.bin)
if [[ ${#csrs[@]} -eq 0 ]]; then
    echo "no CSRs in $ROOT/data/cache/" >&2; exit 1
fi

echo "[re2e_v2] === pull_v2 (zero atomicAdd) ==="
for csr in "${csrs[@]}"; do
    base="$(basename "$csr" .csr.bin)"
    for seed in $SEEDS; do
        out="$OUT_DIR/${base}_seed${seed}"
        if [[ -f "$out.bin" && -f "$out.json" ]]; then
            echo "  [pull_v2] skip $base seed=$seed"
            continue
        fi
        echo "  [pull_v2] $base seed=$seed"
        "$BIN" --graph "$csr" --out "$out" --dataset "$base" \
               --precision "$PRECISION" --damping "$DAMPING" \
               --max-iter "$MAX_ITER" --tol "$TOL"
    done
done

# --- Byte-identity check ---
echo
echo "[re2e_v2] === byte-identity check (sha256) ==="
all_pass=true
for csr in "${csrs[@]}"; do
    base="$(basename "$csr" .csr.bin)"
    bins=("$OUT_DIR/${base}_seed"*.bin)
    if [[ ${#bins[@]} -lt 2 ]]; then continue; fi
    ref_hash=$(sha256sum "${bins[0]}" | awk '{print $1}')
    ds_match=true
    for b in "${bins[@]:1}"; do
        h=$(sha256sum "$b" | awk '{print $1}')
        if [[ "$h" != "$ref_hash" ]]; then
            ds_match=false
            all_pass=false
            echo "  $base: MISMATCH — $(basename "${bins[0]}") vs $(basename "$b")"
        fi
    done
    if $ds_match; then
        echo "  $base: ALL ${#bins[@]} seeds BYTE-IDENTICAL (crc=$(sha256sum "${bins[0]}" | awk '{print substr($1,1,8)}'))"
    fi
done

echo
if $all_pass; then
    echo "[re2e_v2] VERDICT: ALL datasets byte-identical — atomicAdd confirmed as SOLE drift source"
else
    echo "[re2e_v2] VERDICT: some datasets still drift — third mechanism beyond atomicAdd exists"
fi

echo "[re2e_v2] done."
