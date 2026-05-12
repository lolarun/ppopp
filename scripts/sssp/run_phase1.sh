#!/bin/bash
# Phase 1: 1000 RMAT-18 random seed stress test (W3 Task 3, paper §3.2 requirement)
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT/results/sssp}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build_gpu}"
cd "$BUILD_DIR"
mkdir -p "$RESULTS_ROOT"
> "$RESULTS_ROOT/w3_stress.jsonl"
echo "==== PHASE 1: 1000 RMAT-18 stress test (FP32 packed) ===="
fails=0
START=$(date +%s)
for seed in $(seq 1 1000); do
  out=$(./run_sssp --rmat-scale=18 --rmat-seed=$seed --rmat-edgefactor=16 --algo=delta_stepping_gpu --precision=fp32 --source=0 --reps=1 --verify=1 --emit-cert=1 --output="$RESULTS_ROOT/w3_stress.jsonl" 2>&1) || true
  verdict=$(echo "$out" | grep -oP 'verdict=\K[A-Z_]+' | head -1)
  if [ "$verdict" != "SAT" ]; then
    fails=$((fails+1))
    echo "  FAIL seed=$seed verdict=${verdict:-NONE}"
  fi
  if [ $((seed % 100)) -eq 0 ]; then
    NOW=$(date +%s)
    echo "  progress $seed/1000 fails=$fails elapsed=$((NOW-START))s"
  fi
done
echo "==== PHASE 1 DONE: $fails / 1000 UNSAT ===="
