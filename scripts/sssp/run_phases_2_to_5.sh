#!/bin/bash
# Phases 2-5: livejournal FP64 redo, E8 scaling, E9 weight stress, E1 cert extension
# Continues after Phase 1 (1000 stress test).
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT/results/sssp}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build_gpu}"
cd "$BUILD_DIR"
mkdir -p "$RESULTS_ROOT/certs"

# ────────────────────────────────────────────────────────────────────────────
echo "==== PHASE 2: livejournal FP64 redo (after d_removed buffer fix) ===="
> $RESULTS_ROOT/w3_task1_fp64redo.jsonl
./run_sssp \
  --dataset=$ROOT/data/cache/livejournal.gr --dataset-name=livejournal \
  --algo=delta_stepping_gpu --precision=fp64 \
  --source=0 --reps=1 --verify=1 --emit-cert=1 \
  --save-cert=$RESULTS_ROOT/certs/livejournal_fp64 \
  --output=$RESULTS_ROOT/w3_task1_fp64redo.jsonl 2>&1 | tail -1

# ────────────────────────────────────────────────────────────────────────────
echo ""
echo "==== PHASE 3: E8 RMAT scaling (22, 23, 24, 25) x FP32+FP64 x emit/noemit ===="
> $RESULTS_ROOT/e8_scaling.jsonl
for scale in 22 23 24 25; do
  for prec in fp32 fp64; do
    for emit in 1 0; do
      tag="rmat${scale}_${prec}_emit${emit}"
      printf "  %-30s : " "$tag"
      ./run_sssp \
        --rmat-scale=$scale --rmat-edgefactor=16 --rmat-seed=42 \
        --algo=delta_stepping_gpu --precision=$prec \
        --source=0 --reps=3 --verify=$emit --emit-cert=$emit \
        --output=$RESULTS_ROOT/e8_scaling.jsonl 2>&1 | tail -1
    done
  done
done

# ────────────────────────────────────────────────────────────────────────────
echo ""
echo "==== PHASE 4: E9 weight stress (4 distributions x 2 graphs x FP32) ===="
> $RESULTS_ROOT/e9_weights.jsonl
for ds_args in \
    "--dataset=$ROOT/data/cache/web_google.gr --dataset-name=web_google" \
    "--rmat-scale=20 --rmat-edgefactor=16 --rmat-seed=42 --dataset-name=rmat20"; do
  for wd in uniform gaussian powerlaw adversarial; do
    label=$(echo "$ds_args" | grep -oP 'dataset-name=\K\S+' | head -1)
    printf "  %-30s : " "${label}_${wd}"
    ./run_sssp $ds_args \
      --algo=delta_stepping_gpu --precision=fp32 \
      --source=0 --reps=2 --verify=1 --emit-cert=1 \
      --weight-dist=$wd \
      --output=$RESULTS_ROOT/e9_weights.jsonl 2>&1 | tail -1
  done
done

# ────────────────────────────────────────────────────────────────────────────
echo ""
echo "==== PHASE 5: E1 NVIDIA cert extension (FP weight remap on road graphs) ===="
> $RESULTS_ROOT/e1_nvidia.jsonl
# Road graphs with FP weight remap (induce drift)
for ds in ny_road usa_road; do
  for wd in uniform gaussian; do
    printf "  %-30s : " "${ds}_${wd}_fp32"
    ./run_sssp \
      --dataset=$ROOT/data/cache/$ds.gr --dataset-name=$ds \
      --algo=delta_stepping_gpu --precision=fp32 \
      --source=0 --reps=1 --verify=1 --emit-cert=1 \
      --weight-dist=$wd \
      --save-cert=$RESULTS_ROOT/certs/${ds}_${wd}_fp32 \
      --output=$RESULTS_ROOT/e1_nvidia.jsonl 2>&1 | tail -1
  done
done
# Multi-seed RMAT for cross-platform reproducibility check
for seed in 42 100 200 300 400; do
  printf "  %-30s : " "rmat20_seed${seed}_fp32"
  ./run_sssp \
    --rmat-scale=20 --rmat-edgefactor=16 --rmat-seed=$seed --dataset-name=rmat20 \
    --algo=delta_stepping_gpu --precision=fp32 \
    --source=0 --reps=1 --verify=1 --emit-cert=1 \
    --save-cert=$RESULTS_ROOT/certs/rmat20_seed${seed}_fp32 \
    --output=$RESULTS_ROOT/e1_nvidia.jsonl 2>&1 | tail -1
done

# ────────────────────────────────────────────────────────────────────────────
echo ""
echo "==== ALL PHASES DONE ===="
ls -lh $RESULTS_ROOT/
echo ""
echo "Cert files:"
ls -lh $RESULTS_ROOT/certs/
