# Per-finding replication recipes

This document gives concrete reproduction commands for each numbered Finding (F17-F24) in `docs/manuscript/findings_log.md`. Each recipe lists: the GPU type required, the inputs, the exact CLI invocation, and the expected outputs (with hashes where applicable).

The reference paper version is `docs/manuscript/paper_v3.md`. All cell numbers cited below come from `docs/plans/02_data_tables.md` (auto-generated from JSONL run logs).

---

## Hardware shortcuts

| Tag | What |
|---|---|
| **NV** | NVIDIA GPU sm_70+ with 22+ GiB VRAM (tested on A10 = sm_86, 22.5 GiB) + CUDA 12.8 + nvcc |
| **AMD** | AMD CDNA3 (gfx942) with 22+ GiB VRAM (tested on MI300X VF = 1/8 SR-IOV partition, ~24 GiB) + ROCm 7.2 + amdclang++ |
| **CPU** | Any x86_64 with C++20 compiler |

Dataset preparation (one-time, ~5 GB on disk):

```bash
bash scripts/fetch_datasets.sh
# produces data/cache/{ny_road,usa_road,web_google,livejournal}.gr
```

RMAT graphs are generated in-memory by `run_sssp --rmat-scale=N --rmat-edgefactor=K --rmat-seed=S`; deterministic per (scale, edgefactor, seed).

---

## F17 — 17/17 cross-vendor reachable-`d` byte-equality + reachability-set agreement

**Hardware:** NV + AMD (run §3.3 strict-mode matrix on each vendor independently, then offline diff).

**Phase 1: Run strict-mode matrix on NV**

```bash
cmake -B build_gpu -DGPU_BACKEND=CUDA -DCMAKE_CUDA_ARCHITECTURES=86 \
                   -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build_gpu -j
bash scripts/run_strict_matrix.sh
# emits results/certs/{config}_fp{32,64}.{d,pi}.bin × 17 configs
```

**Phase 2: Run strict-mode matrix on AMD**

```bash
cmake -B build_gpu -DGPU_BACKEND=ROCM -DCMAKE_HIP_ARCHITECTURES=gfx942 \
                   -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/amdclang++ \
                   -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build build_gpu -j
bash scripts/run_strict_matrix.sh
# emits results/amd/certs/{config}_fp{32,64}.{d,pi}.bin × 17 configs
```

**Phase 3: Offline byte-equality + reachability audit**

```bash
python3 scripts/offline_a1_14_15.py
# expects results/certs/* and results/amd/certs/* on the same machine
```

**Expected output**: 17/17 `np.array_equal(d_NV, d_AMD) == True` (full vector, not reachable-only). Reachability sets agree: NV unreachable count = AMD unreachable count = (table value) for every config. See [`02_data_tables.md` cross-vendor reachable-only `d_hash` table](../plans/02_data_tables.md#cross-vendor-reachable-only-d_hash-anchor-data) for canonical hashes.

---

## F18 — π-divergence is exactly the FP-tied incoming-candidate case

**Hardware:** CPU only (numpy on existing cert binaries).

**Prerequisite:** F17 cert binaries on disk (both NV and AMD).

```bash
python3 scripts/offline_a1_17.py
```

**Expected output**: across the 5 cert pairs where `pi_NV ≠ pi_AMD` (`ny_road_fp32`, `ny_road_uniform_fp32`, `usa_road_fp32`, `usa_road_uniform_fp32`, `usa_road_gaussian_fp32` — see F17 caveat for the uniform-remap no-op), 100% of differing-π vertices have ≥2 incoming edges achieving the FP-equal minimum candidate distance `d[u] + w(u, v)`. The script prints per-vertex classification and confirms the 100% rate.

---

## F19 — Verifier-tolerance K=1 SAT on strict-mode corpus, vendor-independent

**Hardware:** CPU (numpy K-sweep on cert binaries).

**Prerequisite:** strict-mode cert binaries from F17.

**NV side:**

```bash
python3 scripts/a1_16_remote.py
# reads results/certs/*.{d,pi}.bin × 8 default-weight configs
```

**AMD side:**

```bash
python3 scripts/a1_16_amd_remote.py
# reads results/amd/certs/*.{d,pi}.bin × 6 configs (usa_road .gr unavailable on VF in our run)
```

**Expected output**: smallest-K-SAT = 1 on all 8 NV configs and 6 AMD configs. Max R3 residuals byte-identical between NV and AMD (a consequence of d-byte-equality: same `d` + same graph + same weights ⇒ same R3 residual to the bit).

---

## F20 — Multi-source within-vendor robustness, 18/18 cross-vendor byte-equal

**Hardware:** NV + AMD.

**Phase 1: Run NV multi-source matrix**

```bash
bash scripts/a1_19_remote.sh
# 3 datasets × 3 sources × 1 rep = 9 NV runs
```

**Phase 2: Run AMD multi-source matrix**

```bash
bash scripts/amd_a1_19_a1_18.sh   # contains the A1.19 + A1.18 AMD mirror
```

**Expected output**: 9/9 NV + 9/9 AMD all SAT. Each dataset produces 3 distinct `d_hash` values across the 3 sources (sanity: different sources induce different SSSPs). AMD `d_hash` byte-identical to NV at every (dataset, source) cell — 18/18 cross-vendor cells byte-equal at the per-source level.

Reference table: [F20 in findings_log.md](../manuscript/findings_log.md). NV cells: `99f897ed/7ac50fb4/8a79f089` (ny_road), `f0c9958f/a9ae3e45/d797b6d2` (web_google), `1cd2962d/ae30bcfe/dfe46c8f` (livejournal).

---

## F21 — Compiler `-O` level robustness, 12/12 cross-vendor

**Hardware:** NV + AMD.

**Phase 1: Build three optimization variants on NV**

```bash
cmake -B build_gpu_O0 -DGPU_BACKEND=CUDA -DCMAKE_BUILD_TYPE=Debug ...
cmake -B build_gpu_O3 -DGPU_BACKEND=CUDA -DCMAKE_BUILD_TYPE=Release ...
# (build_gpu = default)
cmake --build build_gpu_O0 -j ; cmake --build build_gpu_O3 -j
bash scripts/a1_18_remote.sh
```

**Phase 2: Same on AMD**

```bash
cmake -B build_gpu_O0 -DGPU_BACKEND=ROCM -DCMAKE_BUILD_TYPE=Debug ...
cmake -B build_gpu_O3 -DGPU_BACKEND=ROCM -DCMAKE_BUILD_TYPE=Release ...
bash scripts/amd_a1_19_a1_18.sh   # contains A1.18 AMD half
```

**Expected output**: 6/6 NV + 6/6 AMD all SAT, byte-identical `d_hash` per dataset across all three `-O` levels on each vendor, AND byte-equal across vendors at every (dataset, build) cell.

---

## F22 + F23 — Gunrock 2.2.0 byte-identical `d` on 6/6 datasets

**Hardware:** NV + Gunrock 2.2.0 source tree.

**Prerequisite:** Gunrock built with the cert-dump patch in `examples/algorithms/sssp/sssp.cu` (the env-var hook described in F22). Build:

```bash
git clone https://github.com/gunrock/gunrock.git /root/gunrock
cd /root/gunrock; git checkout 748f79e
# Apply the GUNROCK_DUMP_CERT env-var patch (see findings_log.md F22)
cmake -B build -DESSENTIALS_NVIDIA_BACKEND=ON -DESSENTIALS_AMD_BACKEND=OFF \
               -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build/bin/sssp -j
```

**Run the audit:**

```bash
bash scripts/a10_e1_e2_remote.sh
# converts .gr → .mtx for road / web / social graphs (gr_to_mtx.py)
# generates RMAT-20 / RMAT-22 csr via run_sssp --save-csr; converts to .mtx (csr_to_mtx.py)
# runs Gunrock SSSP on each .mtx, dumps cert via GUNROCK_DUMP_CERT
# compares against our cert binaries via np.array_equal (after FLT_MAX → +∞ normalization)
```

**Expected output**: 6/6 datasets give `BYTE-IDENTICAL` (full vector, not reachable-only) per the script's final summary. Cells: ny_road (264K vertices), web_google (875K), livejournal (4.85M), usa_road (24M), rmat-20 ef=32 (1M / 33M edges), rmat-22 ef=32 (4M / 134M edges).

---

## F24 — Δ-stepping bucket-width sensitivity, 18/18 cross-vendor

**Hardware:** NV + AMD.

**Compute average edge weight per dataset (one-time):** the script computes this offline from each `.gr` file and uses it to derive Δ ∈ {0.5×avg, 1.0×avg, 2.0×avg}.

**Phase 1: NV side**

```bash
bash scripts/a10_e1_e2_remote.sh
# section "E2 Δ-stepping bucket-width sensitivity" inside this script
# 3 datasets × 3 Δ × 2 builds × 5 reps = 90 NV GPU runs
```

**Phase 2: AMD side**

```bash
# IMPORTANT: requires d_removed buffer fix in src/sssp/delta_stepping.hip:238,347
# (4ull → 16ull) — the fix is in the committed source as of commit 3312bea.
# Without the fix, livejournal × Δ=1.001 cells fail with HIP error invalid argument.
cmake -B build_gpu -DGPU_BACKEND=ROCM ...
cmake -B build_gpu_relaxed -DGPU_BACKEND=ROCM -DRELAX_ATOMICS=ON ...
cmake --build build_gpu -j ; cmake --build build_gpu_relaxed -j
bash scripts/amd_a1_36_remote.sh
# 3 datasets × 3 Δ × 2 builds × 5 reps = 90 AMD GPU runs
```

**Expected output:**
- **NV: 90/90 runs.** Strict 9/9 cells × 5 reps: 1 unique `d_hash` per cell, all SAT, `d_hash` identical across all 3 Δ values within each dataset (ny_road=`99f897ed`, web_google=`f0c9958f`, livejournal=`1cd2962d`). Relaxed 9/9 cells: 5 unique `d_hash` per cell, all UNSAT_RELAXATION at every Δ.
- **AMD: 90/90 runs (post-fix).** Strict 9/9 cells: same 3 hashes as NV. Relaxed 9/9 cells: 5 unique `d_hash` per cell, all UNSAT_RELAXATION.
- **Cross-vendor: 18/18 strict cells byte-identical between AMD and NV.**

The summary inside each script's stdout prints the `unique_d_hashes` and `all_SAT` flags per cell for verification.

---

## Negative replication: the relaxed-atomics race-injection experiment (E12.c)

For both `build_gpu` (strict) and `build_gpu_relaxed` (RELAX_ATOMICS=ON), the §5.1 / F-experiment matrix:

```bash
bash scripts/run_e11_e12c.sh
# 6 datasets × 2 builds × 5 reps = 60 NV runs
# 4 datasets × 2 builds × 5 reps = 40 AMD runs
```

**Expected:** strict produces 1 unique `d_hash` per dataset and all SAT; relaxed produces 5 unique `d_hash` per dataset and all UNSAT_RELAXATION on Δ-stepping; both strict and relaxed produce 1 unique `d_hash` per dataset and all SAT on Bellman-Ford. This is the §5 boundary the paper headlines.

---

## Negative replication: F10 boundary case (long-diameter road FP32 with gaussian remap)

```bash
./build_gpu/run_sssp --dataset=data/cache/ny_road.gr --dataset-name=ny_road_gaussian \
    --algo=delta_stepping_gpu --precision=fp32 --weight-dist=gaussian \
    --reps=1 --verify=1 --emit-cert=1 --output=/tmp/f10_ny.jsonl
./build_gpu/run_sssp --dataset=data/cache/usa_road.gr --dataset-name=usa_road_gaussian \
    --algo=delta_stepping_gpu --precision=fp32 --weight-dist=gaussian \
    --reps=1 --verify=1 --emit-cert=1 --output=/tmp/f10_usa.jsonl
```

**Expected:** both runs emit `verdict=UNSAT_PRED_DISTANCE_MISMATCH` on both NV and AMD (same on both vendors — the verifier is conservatively-correct on long-diameter accumulation; see paper §9.4). NOT a vendor-disagreement failure; demonstrates the F10 boundary is graph + precision intrinsic.

---

## Wall-time references (for sanity-checking your reproduction)

| Workload | NV A10 SSSP | AMD MI300X VF SSSP | Verifier |
|---|---:|---:|---:|
| ny_road FP32 | ~250-500 ms | ~700-1300 ms | ~50-60 ms |
| web_google FP32 | ~40-50 ms | ~130-220 ms | ~210-280 ms |
| livejournal FP32 | ~300-1500 ms | ~1000-2000 ms | ~2400-3000 ms |
| usa_road FP32 | ~3-5 s | (slow; not in 5-rep budget on VF) | depends on R3 work |
| RMAT-22 FP32 | ~700 ms | (OOM on VF partition) | ~3800 ms |
| Gunrock SSSP (each) | comparable to ours per-dataset | (NV only in F22+F23) | n/a |

---

## Verifying the Gunrock cert-dump patch is in place

If you applied the patch correctly, the `sssp` example accepts a `GUNROCK_DUMP_CERT=<prefix>` env-var and writes `<prefix>.d.bin` + `<prefix>.pi.bin` files. Quick check:

```bash
GUNROCK_DUMP_CERT=/tmp/check /root/gunrock/build/bin/sssp \
    --market data/cache/ny_road.mtx --src 0 --validate=false 2>&1 | tail -5
ls -la /tmp/check.d.bin /tmp/check.pi.bin
```

If the files don't exist after the run, the patch isn't applied. See findings_log.md F22 for the patch text.
