#!/usr/bin/env bash
# RE0 / RE1 driver — runs PageRank on every CSR in data/cache/ for the current
# host's GPU and selected precision, writes outputs to
# results/<DEVICE_TAG>/<PRECISION>/<dataset>_seed<n>.{bin,json}.
#
# Usage (matches Paper 2.1 sibling repo build conventions):
#   GPU_BACKEND=CUDA CUDA_ARCH=86      DEVICE_TAG=a10        PRECISION=fp32 ./scripts/run_re0.sh
#   GPU_BACKEND=CUDA CUDA_ARCH=86      DEVICE_TAG=a10        PRECISION=fp64 ./scripts/run_re0.sh
#   GPU_BACKEND=ROCM HIP_ARCH=gfx942   DEVICE_TAG=mi300x_vf  PRECISION=fp32 ./scripts/run_re0.sh
#   GPU_BACKEND=ROCM HIP_ARCH=gfx942   DEVICE_TAG=mi300x_vf  PRECISION=fp64 ./scripts/run_re0.sh
#
# After running on BOTH vendors (each on its own host), copy the results/
# subdirectories together and run scripts/compare_all.sh.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GPU_BACKEND="${GPU_BACKEND:-ROCM}"          # CUDA | ROCM
CUDA_ARCH="${CUDA_ARCH:-86}"                # used only when GPU_BACKEND=CUDA
HIP_ARCH="${HIP_ARCH:-gfx942}"              # used only when GPU_BACKEND=ROCM
DEVICE_TAG="${DEVICE_TAG:-}"                # output subdir name (e.g. a10, mi300x_vf)
PRECISION="${PRECISION:-fp32}"              # fp32 | fp64
SEEDS="${SEEDS:-0 1 2 3 4}"
DAMPING="${DAMPING:-0.85}"
MAX_ITER="${MAX_ITER:-100}"
TOL="${TOL:-1e-6}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build_gpu}"

if [[ -z "$DEVICE_TAG" ]]; then
    if   [[ "$GPU_BACKEND" == "CUDA" ]]; then DEVICE_TAG="cuda_${CUDA_ARCH}"
    elif [[ "$GPU_BACKEND" == "ROCM" ]]; then DEVICE_TAG="rocm_${HIP_ARCH}"
    else echo "unknown GPU_BACKEND=$GPU_BACKEND" >&2; exit 2
    fi
fi

if [[ "$PRECISION" != "fp32" && "$PRECISION" != "fp64" ]]; then
    echo "PRECISION must be fp32 or fp64, got $PRECISION" >&2; exit 2
fi

OUT_DIR="$ROOT/results/${DEVICE_TAG}/${PRECISION}"
mkdir -p "$OUT_DIR"

echo "[run_re0] backend=$GPU_BACKEND tag=$DEVICE_TAG precision=$PRECISION build=$BUILD_DIR out=$OUT_DIR"

# --- 1. Build ---
if [[ ! -x "$BUILD_DIR/pagerank" ]]; then
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

PR_BIN="$BUILD_DIR/pagerank"

# --- 2. Run on every CSR in data/cache/ ---
shopt -s nullglob
csrs=("$ROOT"/data/cache/*.csr.bin)
if [[ ${#csrs[@]} -eq 0 ]]; then
    echo "no CSRs in $ROOT/data/cache/ — generate them first:" >&2
    echo "  python3 scripts/snap_to_csr.py --input ... --output data/cache/web-google.csr.bin" >&2
    echo "  python3 scripts/gen_rmat.py --scale 22 --output data/cache/rmat-22.csr.bin" >&2
    exit 1
fi

for csr in "${csrs[@]}"; do
    base="$(basename "$csr" .csr.bin)"
    for seed in $SEEDS; do
        out="$OUT_DIR/${base}_seed${seed}"
        if [[ -f "$out.bin" && -f "$out.json" ]]; then
            echo "[run_re0] skip (exists) $out"
            continue
        fi
        echo "[run_re0] $base seed=$seed precision=$PRECISION -> $out.bin"
        # Note: --seed wires no kernel state today; multiple runs still reveal
        # cross-run variance because atomicAdd<T> ordering depends on warp
        # scheduling, not on RNG.
        "$PR_BIN" \
            --graph "$csr" \
            --out "$out" \
            --dataset "$base" \
            --precision "$PRECISION" \
            --damping "$DAMPING" \
            --max-iter "$MAX_ITER" \
            --tol "$TOL"
    done
done

echo "[run_re0] done. outputs in $OUT_DIR"
echo "[run_re0] next: scp this dir to a host with the OTHER vendor's results/"
echo "[run_re0]       then run scripts/compare_all.sh"
