# Artifact Description

**Paper:** Cross-Vendor Bit-Reproducibility in GPU Graph Algorithms: It's the Atomics, Not the Arithmetic
**Target:** PPoPP 2027

---

## Hardware Requirements

Any NVIDIA Turing or later (CUDA 12+) and any AMD CDNA3 (ROCm 6+).

Evaluated on:

| GPU | Vendor | Architecture | Compiler | VRAM |
|---|---|---|---|---|
| NVIDIA A10 | NVIDIA | Ampere (sm_86) | nvcc / CUDA 12.8 | 24 GB |
| Tesla T4 | NVIDIA | Turing (sm_75) | nvcc / CUDA 12.8 | 16 GB |
| MI300X VF | AMD | CDNA3 (gfx942) | amdclang++ / ROCm 7.2 | 24 GB (1/8 slice) |

## Software Requirements

- CMake 3.25+
- C++20 compiler
- Python 3.8+ (for dataset prep and analysis scripts)
- HIP unified source: `.hip` files compile as CUDA via `src/core/hip_cuda_compat.h` shim on NVIDIA, native HIP on AMD

## Datasets

Six graphs downloadable via `scripts/fetch_datasets.sh` (SNAP format, converted to binary CSR via `scripts/snap_to_csr.py`). RMAT-22 generated in-repo via `scripts/gen_rmat.py`.

| Dataset | |V| | |E| | Type |
|---|---|---|---|
| road-CA | 1.97M | 5.53M | Road network |
| web-Google | 916K | 5.11M | Web graph |
| LiveJournal | 4.85M | 69.0M | Social network |
| wiki-Talk | 2.39M | 5.02M | Communication |
| as-Skitter | 1.70M | 11.1M | Internet topology |
| RMAT-22 | 4.19M | 67.1M | Synthetic (Graph500) |

## One-Click Full Reproduction

On each GPU host:

```bash
# NVIDIA A10
GPU_BACKEND=CUDA CUDA_ARCH=86  DEVICE_TAG=a10_sm86    ./scripts/run_full_paper.sh

# NVIDIA T4
GPU_BACKEND=CUDA CUDA_ARCH=75  DEVICE_TAG=t4_sm75     ./scripts/run_full_paper.sh

# AMD MI300X
GPU_BACKEND=ROCM HIP_ARCH=gfx942 DEVICE_TAG=mi300x_vf ./scripts/run_full_paper.sh
```

This builds both PageRank and SSSP, runs all experiment phases, and outputs results to `results/pagerank/<tag>/` and `results/sssp/<tag>/`.

## Selective Builds

```bash
# PageRank only
cmake -B build_pr -DGPU_BACKEND=CUDA -DBUILD_PAGERANK=ON -DBUILD_SSSP=OFF \
      -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release
cmake --build build_pr -j

# SSSP only
cmake -B build_sssp -DGPU_BACKEND=CUDA -DBUILD_SSSP=ON -DBUILD_PAGERANK=OFF \
      -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release
cmake --build build_sssp -j

# CPU only (SSSP CPU algos + verifier unit tests, no GPU kernels)
cmake -B build_cpu -DGPU_BACKEND=NONE -DCMAKE_BUILD_TYPE=Release
cmake --build build_cpu -j
```

## Reproducing Each Section

### Section 5: SSSP Byte-Identity + Scheduling Purity

Strict mode (all algorithms) is included in `run_full_paper.sh`. For the relaxed-atomics scheduling purity probe (Section 5.3) separately:

```bash
cmake -B build_relaxed -DGPU_BACKEND={CUDA,ROCM} -DRELAX_ATOMICS=ON \
      -DCMAKE_BUILD_TYPE=Release
cmake --build build_relaxed -j
./build_relaxed/run_sssp --algorithm={delta_stepping,bellman_ford,async_push} \
                         --dataset=<path> --seeds=5
```

### Section 5.5: Error Injection

```bash
python3 scripts/sssp/inject_and_verify.py --dataset=<path> --seeds=100
```

### Section 6--7: PageRank Drift + Onion Peeling

Included in `run_full_paper.sh`. Cross-vendor comparison (after collecting results from all GPUs):

```bash
./scripts/compare_all.sh
```

### Section 7.6: CUDA Version Drift

Run `run_full_paper.sh` under both CUDA 12.x and CUDA 13.x on the same GPU, with different `DEVICE_TAG` (e.g., `a10_sm86` vs `a10_sm86_cuda13`), then compare CRC32 values in the output JSON files.

### SSSP Cross-Vendor Comparison

```bash
python3 scripts/sssp/cross_vendor_compare_f26ext.py
```

## Verifier Unit Tests

```bash
cmake -B build_cpu -DGPU_BACKEND=NONE -DCMAKE_BUILD_TYPE=Release
cmake --build build_cpu -j
cd build_cpu && ctest --output-on-failure
```

## Expected Outputs

### Pull v2 CRC32 Anchors (Byte-Identical Across All 3 GPUs)

| Dataset | FP32 CRC32 | FP64 CRC32 |
|---|---|---|
| as-Skitter | `dcb27f17` | `da5b9150` |
| LiveJournal | `dfeef6b0` | `9ee07b86` |
| RMAT-22 | `eef3b85c` | `611a5829` |
| road-CA | `ce22664c` | `64b45a2a` |
| web-Google | `91b78c48` | `e6f00d32` |
| wiki-Talk | `27b3fe0c` | `7814d28a` |

### SSSP Verdicts

- All strict-CAS delta-stepping runs: SAT, with identical `d_hash` across A10, T4, and MI300X
- Known UNSAT boundary: `ny_road` and `usa_road` under FP32 (both uniform and gaussian weights) due to FP tolerance on long-diameter graphs
- Stress test: 3,965/3,965 SAT (1,000 on A10, 1,000 on T4, 1,965 on MI300X VF)

## Experiment Scale

| Category | Runs |
|---|---|
| PageRank (3 variants × 3 GPUs × 6 datasets × 2 precisions × 5 seeds) | 540 |
| PageRank CUDA 13 supplementary (push × 2 GPUs × 6 datasets × 2 precisions × 5 seeds) | 120 |
| SSSP (across A10, T4, MI300X) | 4,223 |
| **Total** | **4,883** |
