# Paper 2.2 — Findings log

Append-only log of empirical findings, surprises, and observations that
inform the paper / future experiments. F-numbered for stable cross-reference.

Convention (inherited from Paper 2.1's `docs/manuscript/findings_log.md`):
- One entry per finding, numbered `F<N>`
- Status: `observed` (single run), `replicated` (multiple seeds/configs), `closed` (no further action), `open` (needs follow-up)
- Always cite evidence: file path + specific metric / line
- Distinguish observation from interpretation

---

## F1 — RE0 cross-vendor PageRank drift confirmed

**Date**: 2026-05-11
**Status**: replicated
**Phase gate**: Phase 0 GO

**Observation**: 90/90 cross-vendor (NVIDIA A10 ↔ AMD MI300X VF) seed pairs on 3 datasets (web-google, rmat-22, livejournal) produced byte-different PageRank outputs. Min byte_diff_fraction = 32.62%; median per-dataset 42–99%; max 99.76%.

**Evidence**:
- `results/_compare/*.json` (158 pairwise drift JSONs)
- `docs/design/08_re0_decision.md` (decision memo + full numbers)
- `scripts/re0_summary.py` (aggregator)

**Interpretation**: R1 (PageRank may not drift cross-vendor, design doc §5.R1, prior probability 30–40%) is **falsified**. The reduction-class drift hypothesis stands.

**Implications**: Paper 2.2 Phase 1 starts. Variant A (independent) and Variant B (merged with Paper 2.1) framings both supported by data.

---

## F2 — AMD intra-vendor variance > NVIDIA intra-vendor variance

**Date**: 2026-05-11
**Status**: observed (single hardware pair, 3 datasets)
**Significance**: high (paper §6.2 candidate)

**Observation**: Same-host, different-seed pairwise byte_diff_fraction medians:

| Dataset | nv-nv (A10) median | amd-amd (MI300X VF) median | Δ |
|---|---|---|---|
| web-google | 35.54% | **97.77%** | +62 pp |
| rmat-22 | 77.24% | 76.32% | ≈0 |
| livejournal | 33.94% | **55.82%** | +22 pp |

AMD intra-vendor variance is higher on 2/3 datasets, dramatically so on web-google (~3x).

**Evidence**:
- `results/_compare/web-google__a10_sm86_*__a10_sm86_*.json` (NV intra pairs)
- `results/_compare/web-google__mi300x_vf_*__mi300x_vf_*.json` (AMD intra pairs)
- Aggregated by `scripts/re0_summary.py`

**Interpretation candidates** (not yet decided):
1. **ROCm 7.2's `atomicAdd<float>` retry path is less ordered than CUDA 12.8's** — wavefront scheduling produces more order permutations per atomic destination
2. **MI300X VF (1/8 slice) has different scheduling than full MI300X** — possible artifact of virtualization
3. **Graph-structure interaction** — web-google's PR distribution has many vertices near FP32 representation boundaries; any extra ordering noise flips more bytes there than on rmat-22's hub-dominated distribution

**Implications**:
- Paper 2.1 couldn't make this comparison (SSSP is byte-exact both vendors). Paper 2.2 can — this is a novel cross-vendor empirical contribution.
- Phase 1 must replicate on a non-VF MI300X to disambiguate hypotheses 1 vs 2.
- §6.2 (mechanism attribution) gets a vendor-by-vendor breakdown, not just cross-vendor.

**Follow-up needed**: secure non-VF MI300X access for Phase 1 RE2 (currently only have VF).

---

## F3 — rmat-22 highest max L∞ despite fewest iterations

**Date**: 2026-05-11
**Status**: observed
**Significance**: high (theoretical confirmation for §3.1)

**Observation**: Maximum element-wise drift per dataset (cross-vendor pairs):

| Dataset | Iters to convergence | max L∞ (over all cross-vendor pairs) |
|---|---|---|
| web-google | 62 | 2.04e-9 |
| livejournal | 49 | 4.80e-10 |
| **rmat-22** | **9** | **9.43e-9** (largest) |

rmat-22 has the largest numerical drift despite needing 5–7x fewer iterations. Counterintuitive if one assumes drift accumulates over iterations.

**Evidence**: `results/_compare/rmat-22__*.json`, max field.

**Interpretation**: Drift magnitude is dominated by **per-iteration parallel reduce variance**, not by accumulated FP error over iterations. RMAT's power-law degree distribution concentrates many atomic contributions on a small set of hub vertices, where parallel atomic ordering produces the largest within-iteration noise.

**Implications**:
- Directly confirms design doc §3.1 thesis ("reduction order at each destination, not iteration count, is the primary FP variance source").
- §6 (drift characterization) should plot drift_magnitude vs degree_skewness, not vs iter_count.
- Predicts: as graph hub-degree grows, drift grows; this is a falsifiable claim worth a Phase 1 sweep.

---

## F4 — livejournal smallest drift despite largest graph

**Date**: 2026-05-11
**Status**: observed
**Significance**: medium

**Observation**: livejournal is the largest dataset (N=4.85M, E=68M, 49 iters to converge) yet has the smallest cross-vendor drift:

| Metric | livejournal | rmat-22 | web-google |
|---|---|---|---|
| byte_diff median (nv-amd) | **42.37%** (smallest) | 79.03% | 99.08% |
| max L∞ max (nv-amd) | **4.80e-10** (smallest) | 9.43e-9 | 2.04e-9 |
| L2 norm | smallest | mid | mid |

**Evidence**: `results/_compare/livejournal__*.json` vs rmat-22 + web-google.

**Interpretation**: livejournal is a **social network** with flatter degree distribution than rmat-22 (synthetic power-law) — atomic contention spreads more evenly across vertices, reducing the worst-case ordering variance at any single destination. web-google's `byte_diff` is near-saturated because many converged PR values are tiny (~1/N ≈ 1e-6) where any sub-ULP perturbation flips bytes.

Joint with F3, this says **graph structure (degree skewness) is the primary drift driver**, not size or iteration count.

**Implications**:
- Phase 1 RE1 dataset selection should span the degree-skewness axis, not just N or E.
- Suggest including: flat-ish degree (road networks, mesh), moderate skew (livejournal, twitter), high skew (rmat-22, kron-22).

---

## F5 — MI300X VF performance is graph-structure-dependent

**Date**: 2026-05-11
**Status**: observed (single VF instance)
**Significance**: medium (R4 generalization risk)

**Observation**: Wall-time A10 vs MI300X VF:

| Dataset | A10 (ms) | MI300X VF (ms) | VF / A10 |
|---|---|---|---|
| web-google | 18.5 | 27 | 1.46x |
| rmat-22 | 109 | 305 | **2.80x** |
| livejournal | 416 | 415 | **1.00x** |

MI300X VF (1/8 core slice) is essentially **tied with A10** on livejournal (memory-bound, bandwidth-limited) but **3x slower** on rmat-22 (atomic-bound on hub vertices).

**Evidence**: `wall_ms` field in every `results/<host>/<ds>_seed*.json`.

**Interpretation**: VF's 1/8 compute slice is enough for bandwidth-bound workloads (memory bandwidth ratio: 1/8 VF ≈ 660 GB/s vs A10 600 GB/s) but starves under high atomic contention. Full MI300X (not VF) would not show this slowdown.

**Implications**:
- MI300X VF results on hub-heavy graphs **cannot be presented as representative of full MI300X**. §IX disclosure required (analogous to Paper 2.1's MI300X VF caveat in their §IX).
- Drift findings are unaffected — drift is about atomic ordering, not throughput — but performance numbers in §8 must be carefully scoped.
- Phase 1 should secure at least one non-VF MI300X run to anchor the comparison.

---

## F6 — numpy version difference does NOT break CSR byte-identity

**Date**: 2026-05-11
**Status**: replicated (2 hosts, 3 datasets)
**Significance**: low for Paper 2.2 (operational), but worth flagging to Paper 2.1

**Observation**: `scripts/snap_to_csr.py` and `scripts/gen_rmat.py` produced byte-identical (sha256-matching) CSRs on two hosts with different numpy versions:

| Host | numpy | web-google sha256 | rmat-22 sha256 | livejournal sha256 |
|---|---|---|---|---|
| A10 (Ubuntu 22.04) | 2.2.6 | `34b91f06...` | `04a7468f...` | `65678da5...` |
| MI300X VF (Ubuntu 24.04) | 1.26.4 | **same** | **same** | **same** |

**Evidence**:
- A10 `sha256sum data/cache/*.csr.bin` output (in 08_re0_decision.md §Datasets)
- MI300X regen + sha256 (this conversation 2026-05-11)

**Interpretation**: Paper 2.1's CLAUDE.md warns "`snap_to_csr.py` ordering can drift on numpy version differences". For our specific operation sequence (`np.unique` for densification → `np.lexsort((dst, src))` for CSR sort → `np.bincount` for row counts), numpy is deterministic across versions 1.26 → 2.2. The gotcha may apply to other operation choices (e.g., `np.sort` default kind, hash-based unique, set operations).

**Implications**:
- Phase 1+ does NOT need to scp CSRs between hosts. Regeneration is byte-safe.
- Reduces operational friction (no 600 MB transfers per host setup).
- **Reverse suggestion to Paper 2.1**: their CLAUDE.md warning can be narrowed — at minimum identify *which* operation in their pipeline is version-sensitive. Or test like we did and possibly retire the warning.

---

## Index

| F# | Title | Significance | Status |
|---|---|---|---|
| F1 | RE0 cross-vendor PageRank drift confirmed | gate | replicated |
| F2 | AMD intra-vendor variance > NV | high | observed |
| F3 | rmat-22 highest L∞ despite fewest iters | high | observed |
| F4 | livejournal smallest drift despite largest graph | medium | observed |
| F5 | MI300X VF perf is graph-structure-dependent | medium | observed |
| F6 | numpy version delta does not break CSR bytes | low | replicated |
