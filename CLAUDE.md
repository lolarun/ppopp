# Project context for Claude

**Project:** Auditable GPU Graph Reduction under Heterogeneity (Unified: SSSP + PageRank)
**Target:** PPoPP 2027 — submit ~2026-09. Backup: ASPLOS 2028 (next cycle).
**Status:** ARTIFAC prep — repo restructured (unified schema, algorithm-first results/), all experiments to be re-run on A10, T4, MI300X VF. Prior results archived in legacy/results/.

## Current state (high-level)

- SSSP code imported from asplos-27 (delta-stepping, Bellman-Ford, async-push + verifier + harness + IO)
- PageRank: push (atomic), pull (partial atomic), pull_v2 (ZERO atomic) — all tested on A10, T4, MI300X
- Key finding (F13): cross-vendor byte-identity when atomicAdd eliminated → IEEE 754 is bit-exact, drift is solely from reduction scheduling
- **ARTIFAC in progress**: repo restructured (unified JSON schema, algorithm-first results/, shared CRC32, CMake BUILD_SSSP/BUILD_PAGERANK options). Prior results archived in `legacy/results/`. All experiments to be re-run on A10, T4, MI300X VF.
- Paper draft: not started. Outline lives in `docs/design/02_paper_outline.md`.

## Doc layout

```
docs/
├── design/                    — PageRank experiment design (RE0..RE7)
├── sssp_artifact/             — SSSP artifact track (from asplos-27)
├── sssp_ts/                   — SSSP technical specs (from asplos-27)
├── sssp_experiment_status.md  — SSSP E1..E9+ status reference
├── sssp_findings_log.md       — SSSP F1..F28 findings reference
└── findings.md                — PageRank findings (F1..F13)
```

## Build commands

```bash
# NVIDIA (uses nvcc + src/core/hip_cuda_compat.h shim — no ROCm needed)
cmake -B build_gpu -DGPU_BACKEND=CUDA -DCMAKE_CUDA_ARCHITECTURES=86 \
                   -DCMAKE_BUILD_TYPE=Release
cmake --build build_gpu -j

# AMD (native hipcc / amdclang++)
cmake -B build_gpu -DGPU_BACKEND=ROCM -DCMAKE_HIP_ARCHITECTURES=gfx942 \
                   -DCMAKE_PREFIX_PATH=/opt/rocm \
                   -DCMAKE_BUILD_TYPE=Release
cmake --build build_gpu -j

# CPU only (SSSP CPU algos + tests, no GPU kernels)
cmake -B build_cpu -DGPU_BACKEND=NONE -DCMAKE_BUILD_TYPE=Release
cmake --build build_cpu -j

# Selective builds (PageRank only / SSSP only):
cmake -B build_pr   -DGPU_BACKEND=CUDA -DBUILD_PAGERANK=ON -DBUILD_SSSP=OFF ...
cmake -B build_sssp -DGPU_BACKEND=CUDA -DBUILD_SSSP=ON -DBUILD_PAGERANK=OFF ...

# One-click full paper reproduction (on each GPU host):
GPU_BACKEND=CUDA CUDA_ARCH=86  DEVICE_TAG=a10_sm86    ./scripts/run_full_paper.sh
GPU_BACKEND=CUDA CUDA_ARCH=75  DEVICE_TAG=t4_sm75     ./scripts/run_full_paper.sh
GPU_BACKEND=ROCM HIP_ARCH=gfx942 DEVICE_TAG=mi300x_vf ./scripts/run_full_paper.sh
```

## Layout

```
src/
├── common/            — shared utilities: crc32.h
├── core/              — header-only: csr_pagerank.h (unweighted), csr_sssp.h (weighted template), hip_cuda_compat.h
├── pagerank/          — PageRank kernels (.hip): push, pull, pull_v2
├── sssp/              — SSSP kernels (.hip) + CPU algos: delta_stepping, bellman_ford, async_push, dijkstra
├── io/                — C++ loaders: DIMACS, SNAP, GAP, RMAT, binary CSR I/O
├── verifier/          — CPU verifier (SSSP certificate checking)
├── harness/           — SSSP harness (main.cpp, log_writer, error_injector)
└── analysis/          — Python analysis modules

scripts/               — PageRank orchestration + dataset prep + run_full_paper.sh
scripts/sssp/          — SSSP experiment scripts (from asplos-27)
data/cache/            — binary CSRs (gitignored)
results/pagerank/      — PageRank per-host outputs (gitignored)
results/sssp/          — SSSP per-host outputs (gitignored)
results/combined/      — cross-algorithm analysis outputs
legacy/paper/          — Paper 2.1 SSSP manuscript archive
legacy/results/        — archived experiment results (pre-restructure ground truth)
tests/                 — Catch2 unit tests (SSSP)
```

## Two CSR formats (coexisting)

- **PageRank** (`src/core/csr_pagerank.h`): `CsrGraph` — unweighted, int32 row_ptr/col_idx, magic 0x52455F435352304B. Used by all PageRank .hip files.
- **SSSP** (`src/core/csr_sssp.h`): `CSR<W>` — weighted template, uint64 row_offsets (eid_t), uint32 col_indices (vid_t), W[] weights, CRC64. Used by SSSP, IO, verifier, harness.
- Both formats have separate binary loaders and separate data files. Do NOT try to unify them.

## Data locations

- `data/cache/*.csr.bin` — PageRank CSR graphs (gitignored, from `scripts/snap_to_csr.py` / `scripts/gen_rmat.py`)
- `data/raw/*.txt(.gz)` — downloaded SNAP edge lists (gitignored)
- `results/pagerank/<DEVICE_TAG>/<PRECISION>/<dataset>_seed<n>.{bin,json}` — per-run PR vector + metadata
- `results/pagerank/_compare/<PRECISION>/*.json` — pairwise drift metrics (output of `scripts/compare_all.sh`)
- `results/sssp/` — SSSP JSONL logs + cert binaries (per experiment phase)

## Workflow split

- **Local (Windows + git)**: edits, dataset prep where feasible, manuscript.
- **Cloud GPU (rented per session)**: NVIDIA A10/T4, AMD MI300X VF. Build there with scp'd source. Sync results back to local frequently.

## Operational gotchas

- **CMake HIP**: `find_package(hip CONFIG REQUIRED)` needs `-DCMAKE_PREFIX_PATH=/opt/rocm`. Use `hip::host` (not `hip::device`) for executables.
- **CSR byte-identity across hosts**: scp `data/cache/` from one host; do NOT regenerate independently.
- **Idempotent runs**: `scripts/run_re0.sh` skips outputs that already have both `.bin` and `.json`. To force rerun, delete the pair.
- **Frequent sync**: sync `results/` to local after each phase, not at the end.
- **SSSP JSONL append-only**: experiment scripts must NOT truncate JSONLs with `>` at phase start. Use `>>` append or write to dated subdirs.
- **SSSP data tables auto-generated**: `docs/plans/02_data_tables.md` (if created) is generated by `scripts/sssp/sync_data_tables.py`. Do not hand-edit.

## Multi-paper program context

This repo is the unified codebase for two layers:
- **Layer 2.1 SSSP cert** — V5 draft complete (暂存 in asplos-27), Variant E framing locked
- **Layer 2.2 PageRank reduction cert** — RE0-RE2e_v2 experiments complete, F1-F13

Sibling projects (separate repos, not merged here):
- Layer 1 DEC → ICSE 2027 (ship-ready, `C:\Users\Justin\Documents\Confs\icse-27`)
- Layer 2.3 Dynamic graph → EuroSys 2029 (~2028-Q2 launch)
- Layer 3 Verification framework → OOPSLA/CAV (gated on Layer 2 ≥ 2 ship)

## Conventions

- **HIP API in source, CUDA shim on NVIDIA**: kernel files are `.hip`; on `-DGPU_BACKEND=CUDA` they compile as CUDA via `nvcc` with `src/core/hip_cuda_compat.h` aliasing `hip*` to `cuda*`.
- **Don't editorialize when stating data**: in summaries, give the numbers and the verdict; reserve framing for the manuscript.
- **Cite by file path + line**: e.g. `src/pagerank/pagerank.hip:84` for the `atomicAdd` scatter line.
- **Don't disable checks to fix problems**: if the verifier rejects valid output at FP precision boundary, characterize the boundary — don't widen tolerance to hide it.

## Never do

- Force push to main, amend published commits, `git reset --hard` on uncommitted work
- Run experiments without syncing prior results to local first
- Edit `docs/design/0*.md` without updating any cross-referenced sections (the docs reference each other by section)
- Replace `atomicAdd<float>` in `src/pagerank/pagerank.hip` with a deterministic reduction. The non-determinism is the experiment.
- Merge the two CSR formats into one — they serve different algorithms with different binary data.
- Hand-edit auto-generated data tables (use sync_data_tables.py)
