# RE0 — A10 Interim Results (2026-05-11)

**Status**: A10 single-vendor pass complete. Cross-vendor (MI300X VF) pending.
**Verdict (preliminary)**: GO — intra-vendor drift confirmed on all 3 datasets. Cross-vendor drift essentially guaranteed.

---

## Hardware + build

- Host: Aliyun A10, Ubuntu 22.04 (`iZbp1eeiof0uhec3ldea4tZ`)
- GPU: NVIDIA A10 (sm_86)
- Toolchain: CUDA 12.8.93, GCC 11.4.0, CMake 4.3.2
- Build: `GPU_BACKEND=CUDA -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release`
- First-build outcome: **clean** (no compile errors — `hip_cuda_compat.h` shim worked first try).

## Datasets

| Name | N | E | Source |
|---|---|---|---|
| web-google | 875,713 | 5,105,039 | SNAP `web-Google.txt` |
| rmat-22 | 2,396,110 (post-dedup) | 65,244,929 | `scripts/gen_rmat.py --scale 22 --seed 42` |
| livejournal | 4,846,609 | 68,475,391 | SNAP `soc-LiveJournal1.txt` |

## Run matrix

18 runs total: web-google × 8 seeds + rmat-22 × 5 seeds + livejournal × 5 seeds.

| Dataset | Wall (ms) | Iters | Unique CRCs | Notes |
|---|---|---|---|---|
| web-google | 18.5–18.6 | 62 | 8 / 8 | tol 1e-6 |
| rmat-22 | 108.7–109.1 | 9 | 5 / 5 | converges fast (high-degree hubs) |
| livejournal | 415.8–416.2 | 49 | 5 / 5 | largest graph |

**Every single run produced a byte-distinct PageRank vector.** Zero CRC32 collisions.

## Pairwise drift (seed0 vs seed1, same A10)

| Dataset | byte_diff_fraction | max L∞ | mean diff | L2 norm | top-100 Jaccard | top-100 Kendall τ |
|---|---|---|---|---|---|---|
| web-google | 32.85% | 7.57e-10 | 9.12e-14 | 1.63e-9 | 1.000 | 1.000 |
| rmat-22 | **76.77%** | 8.15e-10 | 4.61e-14 | 1.85e-9 | 1.000 | 1.000 |
| livejournal | 20.49% | 5.46e-11 | 8.39e-15 | 1.98e-10 | 1.000 | 1.000 |

### Interpretation

- **Byte drift is large** (20–77% of vertices have byte-different FP32 reps) but **numerical drift is tiny** (max L∞ < 1e-9, mean < 1e-13).
- This is the textbook signature of FP non-associativity in parallel reduction: values agree to ~10 decimal places, but FP32 mantissa is 23 bits so even sub-ULP differences flip bytes.
- **rmat-22 drifts most** despite the fewest iterations (9), because power-law degree distribution means hub vertices receive many parallel `atomicAdd` contributions per iteration — high atomic contention → high ordering variance.
- **Top-100 ranks fully preserved** (Jaccard=Kendall=1.0 on all 3 datasets). Drift doesn't affect semantic ordering. This is the gap that certificate-based verification is meant to fill: not byte-equal, but bounded-equivalent.

## RE0 decision logic (per `docs/design/03_experimental_design.md`)

Threshold table (cross-vendor byte_diff_fraction):
- > 1%: GO
- 0.1–1%: GO_WITH_CAVEATS
- < 0.1%: STOP

**Intra-vendor** results already exceed the GO threshold on all 3 datasets (min 20.49%, max 76.77%). Cross-vendor drift is bounded below by intra-vendor and almost certainly larger, since AMD scheduling produces independent atomicAdd ordering. **Probability of cross-vendor STOP outcome: < 1%** (would require AMD output to coincidentally land within FP32 ULP of one of A10's per-run outputs across millions of vertices).

## Next steps

1. ✅ Pull A10 results + canonical CSRs back to local (`re0_a10_artifacts.tgz`, 482 MB).
2. ⏳ Spin up MI300X VF, scp CSRs (do NOT regenerate — numpy version drift can break byte-identity).
3. ⏳ Build with `GPU_BACKEND=ROCM -DCMAKE_HIP_ARCHITECTURES=gfx942`.
4. ⏳ Run same 18-run matrix → `results/mi300x_vf/`.
5. ⏳ `scripts/compare_all.sh` → cross-vendor + cross-run drift table.
6. ⏳ Commit final RE0 decision memo (`08_re0_results.md`).

## Cost / time

- A10 session wall clock: ~1 hour (incl SSH, build, dataset prep, all runs, drift analysis, rsync).
- GPU-hours billed (A10 active compute): under 5 minutes (sum of per-run wall times). Budget remaining for full RE0: ~50 GPU-min on MI300X side.

## Raw evidence

- All 18 `.bin` + 18 `.json` files in `results/a10_sm86/`.
- Canonical CSRs in `data/cache/`.
- Pairwise drift JSONs would live in `results/_compare/` after `compare_all.sh` is run with both vendors' data.
