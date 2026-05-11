# RE0 Decision Memo — GO

**Date**: 2026-05-11
**Status**: RE0 closed. Phase 0 hard gate cleared.
**Decision**: **GO** — Variant A framing (independent paper) or Variant B (merged paper) both supported.

---

## TL;DR

Cross-vendor GPU PageRank produces byte-different outputs in **every** seed pair across **all 3 datasets** tested (NVIDIA A10 ↔ AMD MI300X VF), at magnitudes well above the GO threshold (1% byte_diff_fraction). Numerical drift is small (max L∞ < 1e-8) and top-100 vertex ranks are perfectly preserved (Jaccard = Kendall τ = 1.000 across all 90 cross-vendor pairs).

The cross-vendor drift story for PageRank is **empirically confirmed**. Paper 2.2 can proceed to Phase 1.

---

## Configuration

| Item | A10 (NVIDIA) | MI300X VF (AMD) |
|---|---|---|
| Host | Aliyun, Ubuntu 22.04 | DigitalOcean AMD Developer Cloud (Atlanta), Ubuntu 24.04 |
| GPU | NVIDIA A10, sm_86 | AMD Instinct MI300X VF (1/8 core), gfx942 |
| Toolchain | CUDA 12.8.93, GCC 11.4 | ROCm 7.2.0, amdclang++ 22, GCC 13.3 |
| CMake flags | `GPU_BACKEND=CUDA -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release` | `GPU_BACKEND=ROCM -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/amdclang++ -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_BUILD_TYPE=Release` |
| Kernel source | `src/pagerank/pagerank.hip` — **identical bytes** on both hosts |
| First-build result | clean | clean (no compile errors on either platform) |

Both builds succeeded from the **same source** using the `hip_cuda_compat.h` shim pattern inherited from Paper 2.1's `src/sssp/hip_cuda_compat.h`.

## Datasets

| Name | N | E | sha256 prefix | A10/MI300X sha256 |
|---|---|---|---|---|
| web-google | 875,713 | 5,105,039 | `34b91f06...` | **identical** |
| rmat-22 | 2,396,110 (post-dedup) | 65,244,929 | `04a7468f...` | **identical** |
| livejournal | 4,846,609 | 68,475,391 | `65678da5...` | **identical** |

Notable: numpy version differed (A10: 2.2.6, MI300X: 1.26.4) but our `scripts/snap_to_csr.py` + `scripts/gen_rmat.py` produced byte-identical CSRs. The "numpy version drift" gotcha from Paper 2.1's CLAUDE.md did not materialize for our specific operations.

## Run matrix

15 runs/host (5 seeds × 3 datasets). A10 had 3 extra web-google seeds from initial scaling test — 8 total. Total: 30 PR vectors.

| Dataset | A10 wall | MI300X wall | Iters | All CRCs unique within host? |
|---|---|---|---|---|
| web-google | 18.5 ms | 27 ms | 62 | yes (8/8 on A10, 5/5 on MI300X) |
| rmat-22 | 109 ms | 305 ms | 9 | yes (5/5 / 5/5) |
| livejournal | 416 ms | 415 ms | 49 | yes (5/5 / 5/5) |

**Zero CRC32 collisions** anywhere — every PageRank run produced a byte-distinct output.

## Drift table (aggregated over all valid pairs)

| Dataset | Pair kind | N pairs | byte_diff median | byte_diff max | max L∞ max | top-100 J min | top-100 K min |
|---|---|---|---|---|---|---|---|
| livejournal | nv-nv | 10 | 33.94% | 65.39% | 2.07e-10 | 1.000 | 1.000 |
| livejournal | amd-amd | 10 | 55.82% | 81.22% | 2.84e-10 | 1.000 | 1.000 |
| livejournal | **nv-amd** | **25** | **42.37%** | **79.56%** | **4.80e-10** | **1.000** | **1.000** |
| rmat-22 | nv-nv | 10 | 77.24% | 89.49% | 3.38e-09 | 1.000 | 1.000 |
| rmat-22 | amd-amd | 10 | 76.32% | 83.80% | 7.10e-09 | 1.000 | 1.000 |
| rmat-22 | **nv-amd** | **25** | **79.03%** | **92.95%** | **9.43e-09** | **1.000** | **1.000** |
| web-google | nv-nv | 28 | 35.54% | 83.94% | 7.57e-10 | 1.000 | 1.000 |
| web-google | amd-amd | 10 | 97.77% | 99.75% | 1.75e-09 | 1.000 | 1.000 |
| web-google | **nv-amd** | **40** | **99.08%** | **99.76%** | **2.04e-09** | **1.000** | **1.000** |

**Cross-vendor totals**: 90/90 pairs GO. Min byte_diff_fraction 32.62%. Max max_Linf 9.43e-09.

## Decision logic check (per `03_experimental_design.md` §RE0)

- byte_diff_fraction > 1% → GO ✓ — *every* dataset's cross-vendor median is **>40%**, max **>92%**.
- byte_diff_fraction 0.1–1% → GO_WITH_CAVEATS — not applicable.
- byte_diff_fraction < 0.1% → STOP — not applicable.

R1 (PageRank may not drift cross-vendor) is **falsified**. Probability assigned in the design doc (30–40%) was conservative for our particular kernel + dataset choice.

## Five quick findings worth flagging for the paper

1. **Drift exists at both intra- and inter-vendor scales** (nv-nv non-zero, amd-amd non-zero, nv-amd ≥ both). This means scheduling-level non-determinism dominates within a vendor; vendor switching layers on additional ordering variance.

2. **Ranking is bulletproof.** All 90 cross-vendor pairs preserve the top-100 vertex set exactly and in the same order (Jaccard = Kendall τ = 1.000). PR drift moves values, not their semantic ranks — the gap that certificate-based verification is meant to fill.

3. **rmat-22 drifts most numerically** (max L∞ 9.4e-9) despite the *fewest* iterations (9). Cause: power-law degree distribution drives heavy atomic contention on hub vertices. This is a candidate datapoint for §3.2 of the independent-version outline ("why reduction-class drifts").

4. **web-google's cross-vendor byte_diff is near-saturated** (median 99.08%, max 99.76%). The graph is highly skewed and converges to many vertices with very small PR values where any FP perturbation flips bytes. Compare to livejournal (42%) which has more vertices in the "noise floor" of the FP32 representation. Useful as a sensitivity story.

5. **amd-amd intra-vendor drift > nv-nv intra-vendor drift** on web-google (97.77% vs 35.54%). MI300X VF appears to produce more aggressive ordering variance than A10. This may be a artifact of the VF (1/8 slice) running with different warp/wavefront scheduling than full MI300X; worth a sentence in §6.2 or noting as a §IX disclosure (matches Paper 2.1's R4 caveat).

## Cost (RE0 total)

| Resource | Wall clock | Billable |
|---|---|---|
| A10 session | ~1.5 h | minimal active compute (< 5 min sum of run wall times) |
| MI300X VF session | ~1 h | ~3 min active compute |
| Local analysis | ~10 min | $0 |
| **Total** | **~2.5 h wall, ~1 day calendar** | **< 1 GPU-hour billed** |

Design doc budgeted 2–4 GPU-hours. Actual usage was below the lower bound by 3x.

## Next: Phase 1 (RE1 — full drift matrix)

Per `04_timeline_milestones.md` §Phase 1:

- Expand to 6–8 datasets (add road networks, social networks).
- FP32 + FP64.
- Multiple NV cards (A10 baseline + try A100, L20 if available).
- 5 runs/configuration.
- Generate the per-paper-cell drift matrix.

Phase 0 closes ahead of the 2026-05-25 deadline. R1 retired. R7/R8 still active.

## Raw evidence on disk

- `results/a10_sm86/*.bin + .json` (18 files)
- `results/mi300x_vf/*.bin + .json` (15 files)
- `results/_compare/*.json` (158 pairwise drift JSONs — generated by `scripts/compare_all.sh`)
- This memo: `docs/design/08_re0_decision.md`
- Interim memo from A10-only stage: `docs/design/07_re0_a10_interim.md`

CSRs are byte-identical between hosts (see Datasets table above) and are reproducible from `scripts/snap_to_csr.py` + `scripts/gen_rmat.py`. They are gitignored locally; the canonical copies live on the A10 + MI300X hosts and should be archived to long-term storage before the GPU instances are released.
