#!/bin/bash
# Patch missing RMAT-22 entries in e8_scaling.jsonl
# Removes the 12 dummy entries (all identical config) and adds 12 proper ones (4 configs × 3 reps)
cd /root/gpu-sssp-certifying/build_gpu

# Strip the bogus RMAT-22 entries (n_v=4194304) from e8_scaling.jsonl
python3 -c "
import json
keep = []
for line in open('/root/gpu-sssp-certifying/results/e8_scaling.jsonl'):
    d = json.loads(line)
    if d['dataset']['n_v'] != 4194304:
        keep.append(line)
open('/root/gpu-sssp-certifying/results/e8_scaling.jsonl', 'w').writelines(keep)
print(f'kept {len(keep)} non-rmat22 entries')
"

echo "==== Patching: RMAT-22 with 4 configs x 3 reps ===="
for prec in fp32 fp64; do
  for emit in 1 0; do
    printf "  rmat22 prec=%s emit=%d : " "$prec" "$emit"
    ./run_sssp \
      --rmat-scale=22 --rmat-edgefactor=16 --rmat-seed=42 \
      --algo=delta_stepping_gpu --precision=$prec \
      --source=0 --reps=3 --verify=$emit --emit-cert=$emit \
      --output=/root/gpu-sssp-certifying/results/e8_scaling.jsonl 2>&1 | tail -1
  done
done

echo "==== Final line count: ===="
wc -l /root/gpu-sssp-certifying/results/e8_scaling.jsonl
