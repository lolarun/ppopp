#!/usr/bin/env bash
# RE2e + RE3 driver — runs BOTH push (atomic) and pull (deterministic) PageRank
# on every CSR, then compares:
#   RE2e: are pull-variant outputs byte-identical across seeds? (expected: yes)
#   RE3:  push wall_ms vs pull wall_ms (cost of determinism)
#
# Outputs go to results/pagerank/<DEVICE_TAG>/<PRECISION>_pull/ for the pull variant.
# Push results reuse existing results/pagerank/<DEVICE_TAG>/<PRECISION>/ from RE0/RE1.
#
# Usage:
#   GPU_BACKEND=CUDA CUDA_ARCH=75 DEVICE_TAG=t4_sm75 PRECISION=fp32 ./scripts/run_re2e_re3.sh

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

PUSH_DIR="$ROOT/results/pagerank/${DEVICE_TAG}/${PRECISION}"
PULL_DIR="$ROOT/results/pagerank/${DEVICE_TAG}/${PRECISION}_pull"
mkdir -p "$PUSH_DIR" "$PULL_DIR"

echo "[re2e_re3] backend=$GPU_BACKEND tag=$DEVICE_TAG precision=$PRECISION"

# --- Build (both targets) ---
if [[ ! -x "$BUILD_DIR/pagerank" || ! -x "$BUILD_DIR/pagerank_pull" ]]; then
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

PUSH_BIN="$BUILD_DIR/pagerank"
PULL_BIN="$BUILD_DIR/pagerank_pull"

# --- Run push (skip if already have results from RE0/RE1) ---
shopt -s nullglob
csrs=("$ROOT"/data/cache/*.csr.bin)
if [[ ${#csrs[@]} -eq 0 ]]; then
    echo "no CSRs in $ROOT/data/cache/" >&2; exit 1
fi

echo "[re2e_re3] === push (atomic) ==="
for csr in "${csrs[@]}"; do
    base="$(basename "$csr" .csr.bin)"
    for seed in $SEEDS; do
        out="$PUSH_DIR/${base}_seed${seed}"
        if [[ -f "$out.bin" && -f "$out.json" ]]; then
            echo "  [push] skip $base seed=$seed"
            continue
        fi
        echo "  [push] $base seed=$seed"
        "$PUSH_BIN" --graph "$csr" --out "$out" --dataset "$base" \
                     --precision "$PRECISION" --damping "$DAMPING" \
                     --max-iter "$MAX_ITER" --tol "$TOL"
    done
done

echo "[re2e_re3] === pull (deterministic) ==="
for csr in "${csrs[@]}"; do
    base="$(basename "$csr" .csr.bin)"
    for seed in $SEEDS; do
        out="$PULL_DIR/${base}_seed${seed}"
        if [[ -f "$out.bin" && -f "$out.json" ]]; then
            echo "  [pull] skip $base seed=$seed"
            continue
        fi
        echo "  [pull] $base seed=$seed"
        "$PULL_BIN" --graph "$csr" --out "$out" --dataset "$base" \
                     --precision "$PRECISION" --damping "$DAMPING" \
                     --max-iter "$MAX_ITER" --tol "$TOL"
    done
done

# --- RE2e quick check: are all pull outputs byte-identical per dataset? ---
echo
echo "[re2e_re3] === RE2e: pull byte-identity check ==="
for csr in "${csrs[@]}"; do
    base="$(basename "$csr" .csr.bin)"
    bins=("$PULL_DIR/${base}_seed"*.bin)
    if [[ ${#bins[@]} -lt 2 ]]; then continue; fi
    ref_hash=$(sha256sum "${bins[0]}" | awk '{print $1}')
    all_match=true
    for b in "${bins[@]:1}"; do
        h=$(sha256sum "$b" | awk '{print $1}')
        if [[ "$h" != "$ref_hash" ]]; then
            all_match=false
            echo "  $base: MISMATCH — $(basename "${bins[0]}") vs $(basename "$b")"
        fi
    done
    if $all_match; then
        echo "  $base: ALL ${#bins[@]} seeds BYTE-IDENTICAL ✓"
    fi
done

# --- RE3 quick report: push vs pull wall time ---
echo
echo "[re2e_re3] === RE3: push vs pull wall time ==="
printf "%-14s %-6s %10s %10s %8s\n" "dataset" "seed" "push_ms" "pull_ms" "ratio"
for csr in "${csrs[@]}"; do
    base="$(basename "$csr" .csr.bin)"
    for seed in $SEEDS; do
        push_json="$PUSH_DIR/${base}_seed${seed}.json"
        pull_json="$PULL_DIR/${base}_seed${seed}.json"
        if [[ -f "$push_json" && -f "$pull_json" ]]; then
            push_ms=$(python3 -c "import json; print(json.load(open('$push_json'))['wall_ms'])")
            pull_ms=$(python3 -c "import json; print(json.load(open('$pull_json'))['wall_ms'])")
            ratio=$(python3 -c "print(f'{$pull_ms/$push_ms:.2f}')")
            printf "%-14s %-6s %10.1f %10.1f %7sx\n" "$base" "$seed" "$push_ms" "$pull_ms" "$ratio"
        fi
    done
done

echo
echo "[re2e_re3] done."
