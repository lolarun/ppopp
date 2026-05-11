# RE0 Runbook — Cross-vendor PageRank drift GO/NO-GO

**Purpose**: Hard gate for Paper 2.2. Confirm (or falsify) that GPU PageRank
produces byte-different outputs across NVIDIA and AMD platforms.

**Deadline**: 2026-05-25 (Phase 0 Week 2 close).
**Owner**: solo.
**Decision authority**: solo, based on `byte_diff_fraction` threshold (see §5).

---

## 1. Hardware

| Slot | Primary | Backups |
|---|---|---|
| NVIDIA | A10 (sm_86) | A100 (sm_80), L20 (sm_89) |
| AMD    | MI300X VF, 1/8 core slice (gfx942) | MI200 (gfx90a) |

The two hosts run **independently**. We never need both online at the same
moment. Each host produces `results/<tag>/*.bin + .json`; we copy them
together afterwards and diff offline.

---

## 2. Repo layout

Layout mirrors sibling Paper 2.1 repo (`asplos-27`) — same `src/{core,io,
pagerank,harness,analysis}` packaging, same HIP-API-with-CUDA-shim
convention, same `data/cache` + `results` top-level directories.

```
ppopp-27/
├── CLAUDE.md                       # project context entry point
├── CMakeLists.txt                  # GPU_BACKEND=CUDA|ROCM|NONE switch
├── src/
│   ├── core/
│   │   └── csr.h                   # binary CSR loader (header-only)
│   ├── pagerank/
│   │   ├── pagerank.hip            # push-based PR kernel + driver, FP32
│   │   └── hip_cuda_compat.h       # HIP→CUDA shim (only included on nvcc)
│   ├── io/                         # (placeholder — for SNAP/RMAT C++ later)
│   ├── harness/                    # (placeholder — log_writer, error_injector)
│   └── analysis/
│       └── drift_compare.py        # core comparison (Linf/L2/byte_diff/topK)
├── scripts/
│   ├── snap_to_csr.py              # SNAP edge list → binary CSR (Python prep)
│   ├── gen_rmat.py                 # Graph500 RMAT generator (Python prep)
│   ├── run_re0.sh                  # build + run all CSRs × seeds for one host
│   └── compare_all.sh              # cross product of all vendor/seed pairs
├── data/
│   └── cache/                      # binary CSRs (gitignored, ~hundreds MB)
├── results/                        # per-host PR outputs, gitignored
│   ├── cuda_86/                    # one dir per (backend,arch) tag
│   ├── rocm_gfx942/
│   └── _compare/                   # drift_compare.py JSON outputs
├── tests/                          # (placeholder)
└── docs/
    ├── design/                     # 00..05 design docs + this runbook
    └── (manuscript/, plans/, etc. — to be added in Phase 1+)
```

---

## 3. Datasets (per design doc §RE0)

| Name        | Source | Approx N / E | Notes |
|---|---|---|---|
| web-google  | SNAP `web-Google.txt.gz` | 0.9M / 5.1M | small, fast iter |
| livejournal | SNAP `soc-LiveJournal1.txt.gz` | 4.8M / 69M | medium, real-world skew |
| rmat-22     | generated locally | 4.2M / ~67M | synthetic, scale 22 |

### Download / generate (run on whichever host is most convenient)

```bash
mkdir -p data/raw data/cache

curl -L -o data/raw/web-google.txt.gz \
    https://snap.stanford.edu/data/web-Google.txt.gz
gunzip -k data/raw/web-google.txt.gz
python3 scripts/snap_to_csr.py \
    --input  data/raw/web-google.txt \
    --output data/cache/web-google.csr.bin

curl -L -o data/raw/livejournal.txt.gz \
    https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz
gunzip -k data/raw/livejournal.txt.gz
python3 scripts/snap_to_csr.py \
    --input  data/raw/livejournal.txt \
    --output data/cache/livejournal.csr.bin

python3 scripts/gen_rmat.py --scale 22 --seed 42 \
    --output data/cache/rmat-22.csr.bin
```

CSRs must be **byte-identical** on both hosts (same generator, same seed).
After copying to the other host: `sha256sum data/cache/*.csr.bin` on both
sides as a sanity check.

---

## 4. Build + run on each host

### NVIDIA A10
```bash
GPU_BACKEND=CUDA CUDA_ARCH=86 DEVICE_TAG=a10 ./scripts/run_re0.sh
```

### AMD MI300X VF
```bash
GPU_BACKEND=ROCM HIP_ARCH=gfx942 DEVICE_TAG=mi300x_vf ./scripts/run_re0.sh
```

Each writes `results/<DEVICE_TAG>/<dataset>_seed{0..4}.{bin,json}`. The bin
is the FP32 PR vector; the json is metadata + zlib CRC32 of the bin.

`SEEDS=0` (env var) for a single-seed sanity pass first; bump to `0 1 2 3 4`
once a dataset is known to converge.

### Build prerequisites

| Host    | Need |
|---|---|
| NVIDIA  | CUDA toolkit ≥ 11.8, CMake ≥ 3.21, gcc/clang. `nvcc` on PATH. |
| AMD     | ROCm ≥ 6.0 (MI300 needs ≥ 6.0), CMake ≥ 3.21, `hipcc`/`amdclang++` on PATH. `-DCMAKE_PREFIX_PATH=/opt/rocm` is set automatically by `run_re0.sh`. |

`scripts/run_re0.sh` will run `cmake -S . -B build_gpu` if `build_gpu/pagerank`
doesn't exist; otherwise it just rebuilds incrementally.

### Single-graph smoke test (do first on each host)

```bash
build_gpu/pagerank \
    --graph data/cache/web-google.csr.bin \
    --out   results/smoke/web-google_seed0 \
    --dataset web-google \
    --max-iter 50
```

Expect: ~50 iters, final L1 < 1e-4, wall < 5 s on either host. CRC32
printed to stderr should be **the same across runs on the same host** if
the kernel happens to be deterministic — but probably won't be (that is
the point of RE0).

---

## 5. Comparison + decision (after both hosts done)

Copy `results/cuda_86/` (or `a10/`) and `results/rocm_gfx942/` (or
`mi300x_vf/`) into the same checkout, then:

```bash
./scripts/compare_all.sh
```

This emits one `results/_compare/<dataset>__<tagA>_seedX__<tagB>_seedY.json`
per pair, plus a summary table on stdout.

### Decision logic (per dataset, taking median over seed pairs)

| `byte_diff_fraction` | Decision | Action |
|---|---|---|
| > 1%               | **GO**            | Variant A/B framing; proceed to Phase 1 (RE1). |
| 0.1% – 1%          | **GO_WITH_CAVEATS** | Proceed but framing must carefully quantify drift magnitude. |
| < 0.1%             | **STOP**          | Hypothesis falsified. Trigger contingency. |

### Cross-run sanity
Same-host pairs (`a10_seed0 vs a10_seed1` etc.) should **also drift** —
that confirms the drift source is intra-vendor scheduling, not just
cross-vendor. If same-host pairs are byte-identical but cross-vendor ones
are not, write that down — it's a structurally interesting finding for the
paper either way.

### Smoking-gun checklist
- [ ] All three datasets converged (`final_l1 < tol` in metadata json) on both vendors.
- [ ] Same dataset, same seed, two vendors → drift exists (byte_diff_fraction > 0.001).
- [ ] Same dataset, same vendor, two seeds → drift exists (variance source = scheduling).
- [ ] Top-100 vertex set Jaccard > 0.95 on both pairs (ranks roughly stable even when values drift).
- [ ] CRC32 of any two outputs differs whenever byte_diff_fraction > 0.

If any of the first three fail, document why before declaring STOP.

---

## 6. STOP contingencies (per design doc §R1)

If `byte_diff_fraction < 0.001` on all datasets:

1. **First**: re-check the kernel actually used `atomicAdd` and not some
   library that took a deterministic path. Inspect `results/_compare/`
   JSON — if all `crc32` values are identical across vendors, the kernel
   itself is being deterministic (unlikely with `atomicAdd<float>` but
   possible with very small graphs / single-block reductions).
2. **Second**: try a heavier reduction kernel — switch to FP32 betweenness
   centrality (the SNAP / RMAT loaders are reusable; only the kernel is
   new code).
3. **Third**: declare 2.2 STOP, advise 2.1 single submission. Update
   `01_problem_thesis.md §7` with the empirical falsification record.

---

## 7. Time budget (Week 1-2 of Phase 0)

| Day | Work |
|---|---|
| 05-12 | Repo bootstrapped (this commit). Build smoke test on whichever host is online today. |
| 05-13 | Get the OTHER vendor's host online; build smoke test there too. |
| 05-14 | Datasets generated + sha256-matched on both hosts. |
| 05-15 | First end-to-end (web-google, single seed) on both hosts. Manual `drift_compare.py` call. Initial signal. |
| 05-18 | Full matrix (3 datasets × 5 seeds) running on both hosts. |
| 05-22 | Full comparison done. Draft decision memo. |
| 05-25 | **Decision committed.** Update `00_paper22_overview.md` + `05_risk_register.md`. |

GPU-hours estimate: 2-4 total (per design doc). web-google + livejournal
are sub-second per iter on either A10 or MI300X; rmat-22 < 30 s per run.

---

## 8. Outputs to keep (whatever the decision)

After RE0:
- `results/<tag>/` archived (tar.zst) into a long-term store — these are
  the empirical record.
- `results/_compare/` JSONs: ground truth for any later claim about drift
  magnitude or top-k stability.
- `docs/design/07_re0_results.md` (write after the run): a one-pager
  capturing the decision, the numbers behind it, and any surprises.
