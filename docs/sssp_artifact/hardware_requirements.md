# Hardware requirements

Original reproduction was performed on the following hardware. Other configurations should work but are not tested.

---

## NVIDIA path

| Component | Tested config | Minimum |
|---|---|---|
| GPU | NVIDIA A10 (sm_86, 24 GB VRAM) | sm_60+ (Pascal+) for 64-bit `atomicCAS` used by FP32 packed atomic |
| Driver | 580.126.09 | Compatible with CUDA 12.x |
| CUDA toolkit | 12.8 | 12.x recommended |
| Compiler | g++ 11.4, nvcc 12.8 | g++ ≥ 9 with C++20 support |
| OS | Ubuntu 22.04 LTS | Linux x86_64 |

VRAM requirements for tested workloads:
- ny_road / web_google: < 1 GB
- usa_road: ~5 GB (FP32 packed) / ~7 GB (FP64 emit)
- livejournal: ~3 GB / ~4 GB
- RMAT-22 to RMAT-25: ~2 GB to ~25 GB (FP64 emit at scale 25 is the upper end)

A10's 24 GB sufficed for all tested configurations including RMAT-25 FP64 emit.

---

## AMD path

| Component | Tested config | Notes |
|---|---|---|
| GPU | AMD Instinct MI300X **VF** (1/8 SR-IOV partition) | Full MI300X (not VF) should produce identical correctness; performance scales |
| Effective VRAM | ~24 GB visible (per-partition) | Full MI300X has 192 GB |
| Driver | amdgpu (kernel 6.8) | ROCm 7.x compatible |
| ROCm | 7.2.26015 (DigitalOcean ROCm 1-Click image) | 6.x also works (tested earlier) |
| Compiler | amdclang++ 22.0 (bundled with ROCm) | Required by CMake `enable_language(HIP)` |
| OS | Ubuntu 24.04 LTS (host); Ubuntu running inside Docker container `rocm` | The DigitalOcean image runs ROCm inside a container; build inside the container |

GPU detection inside the container:

```bash
docker exec rocm /opt/rocm/bin/rocm-smi --showproductname
# expected: AMD Instinct MI300X VF
```

---

## CPU-only path

| Component | Tested config | Minimum |
|---|---|---|
| CPU | various | x86_64, OpenMP-capable |
| Compiler | g++ 11+ or clang++ 14+ | C++20 (uses `std::span`, `std::is_same_v`) |
| OS | Ubuntu 22.04, Windows 11 + WSL2 | Linux x86_64 |

CPU-only build (`-DGPU_BACKEND=NONE`) is sufficient for unit tests and CPU SSSP runs (Dijkstra, Δ-stepping reference).

---

## Reviewer-pushback insurance

The submitted artifact includes data from MI300X **VF**, not the full GPU. If a reviewer requires full-GPU validation:

- Full MI300X is available as a separate DigitalOcean droplet (~$2/hour).
- Re-running a representative subset (ny_road, web_google, RMAT-25 FP32 packed) on full MI300X produces byte-identical d_hash because:
  - VF runs the same CDNA3 ISA as full GPU
  - FP arithmetic is determined by ISA, not partition size
  - Cross-vendor d-determinism (Variant E thesis) is architecture-independent
  - The only differences would be performance (which paper does not centrally claim)

Cross-vendor evidence (the central paper claim) is unaffected by VF.

---

## Findings F22-F24 hardware additions (post-cross-review)

**F22 + F23 (Gunrock cross-implementation audit) — NVIDIA only.** Requires Gunrock 2.2.0 source tree built with the cert-dump env-var hook (see `replication_recipes.md#f22-f23`). Gunrock's MatrixMarket loader uses its own memory allocator and is mostly host-side; on A10 (24 GB) the largest tested input is RMAT-22 ef=32 (4M vertices, 134M edges, 4.27 GB `.mtx` file) which loads cleanly without OOM. cuGraph cross-impl audit (separate future task) is heavier setup; not currently in scope.

**F24 (Δ-stepping bucket-width sensitivity) — NV + AMD.** The 90 NV runs + 90 AMD runs use the same hardware as the rest of §5. **AMD-side requires the `delta_stepping.hip:238,347` `d_removed` buffer fix (`4ull → 16ull`)** committed at `3312bea`. Without the fix, livejournal × Δ=1.001 on AMD wave-size 64 fails with `HIP error invalid argument` (the original 4×NV scratch allocation was insufficient under dense-graph + large-Δ + 64-wave geometry; the fix raises it to 16×NV). For livejournal NV=4.85M the d_removed buffer goes from 78 MB to 310 MB — well within the 24 GiB VF partition.
