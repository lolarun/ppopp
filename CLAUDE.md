# Project context for Claude

**Project:** Auditable GPU Graph Reduction under Heterogeneity (Paper 2.2)
**Target:** PPoPP 2027 — submit ~2026-09. Backup: ASPLOS 2028 (next cycle).
**Active phase:** Phase 0 (RE0 GO/NO-GO), Week 1 starts 2026-05-12.
**Sibling repo:** Paper 2.1 (SSSP) lives at `C:\Users\Justin\Documents\Flagship\asplos-27`. Code lineage borrowed from there but the two repos stay physically separate. Merge decision is 2026-08-01 (paper-only, not codebase).

## Current state (high-level)

- Implementation: scaffolded (CMake + HIP/CUDA kernel + CSR loaders + drift comparator). Untested on hardware.
- Data: not yet generated. Datasets fetched via `scripts/snap_to_csr.py` + `scripts/gen_rmat.py`.
- RE0 hard gate: must produce GO/NO-GO decision by 2026-05-25.
- Paper draft: not started. Outline lives in `docs/design/02_paper_outline.md`.
- See `docs/design/00_paper22_overview.md` for the full design context.

## Doc layout

```
docs/
└── design/
    ├── 00_paper22_overview.md      — top-level design + decisions
    ├── 01_problem_thesis.md        — problem statement + framing variants
    ├── 02_paper_outline.md         — independent (10pp) + merged (~2.5pp) outline
    ├── 03_experimental_design.md   — RE0..RE7 spec
    ├── 04_timeline_milestones.md   — Phase 0..3 schedule
    ├── 05_risk_register.md         — R1..R10 risks
    └── 06_re0_runbook.md           — RE0 step-by-step
```

(Manuscript / plans / artifact dirs to be added in Phase 1+.)

## Build commands

```bash
# NVIDIA (uses nvcc + src/pagerank/hip_cuda_compat.h shim — no ROCm needed)
cmake -B build_gpu -DGPU_BACKEND=CUDA -DCMAKE_CUDA_ARCHITECTURES=86 \
                   -DCMAKE_BUILD_TYPE=Release
cmake --build build_gpu -j

# AMD (native hipcc / amdclang++)
cmake -B build_gpu -DGPU_BACKEND=ROCM -DCMAKE_HIP_ARCHITECTURES=gfx942 \
                   -DCMAKE_PREFIX_PATH=/opt/rocm \
                   -DCMAKE_BUILD_TYPE=Release
cmake --build build_gpu -j

# Or use the orchestrator:
GPU_BACKEND=CUDA CUDA_ARCH=86      DEVICE_TAG=a10        ./scripts/run_re0.sh
GPU_BACKEND=ROCM HIP_ARCH=gfx942   DEVICE_TAG=mi300x_vf  ./scripts/run_re0.sh
```

## Layout (mirrors sibling Paper 2.1)

```
src/
├── core/          — header-only (csr.h)
├── pagerank/      — kernel (.hip) + hip_cuda_compat.h
├── io/            — placeholder for future C++ loaders (currently Python in scripts/)
├── harness/       — placeholder (log_writer, error_injector — Phase 1+)
└── analysis/      — Python analysis modules (drift_compare.py)

scripts/           — orchestration + dataset prep (bash + Python)
data/cache/        — binary CSRs (gitignored)
results/<tag>/     — per-host PR outputs (gitignored)
results/_compare/  — pairwise drift JSON (gitignored)
tests/             — placeholder
```

## Data locations

- `data/cache/*.csr.bin` — binary CSR graphs (gitignored, from `scripts/snap_to_csr.py` / `scripts/gen_rmat.py`)
- `data/raw/*.txt(.gz)` — downloaded SNAP edge lists (gitignored)
- `results/<DEVICE_TAG>/<dataset>_seed<n>.{bin,json}` — per-run PR vector + metadata
- `results/_compare/*.json` — pairwise drift metrics (output of `scripts/compare_all.sh`)

## Workflow split (per `docs/design/04_timeline_milestones.md`)

- **Local (Windows + git)**: edits, dataset prep where feasible, manuscript.
- **Cloud GPU (rented per session)**: NVIDIA A10 (or A100/L20 backup), AMD MI300X VF (1/8 core slice). Build there with rsync'd source. Sync results back to local frequently — cloud servers can disappear.

## Operational gotchas (inherited / expected from Paper 2.1)

- **CMake HIP**: `find_package(hip CONFIG REQUIRED)` needs `-DCMAKE_PREFIX_PATH=/opt/rocm`. Use `hip::host` (not `hip::device`) for executables, otherwise `--offload-arch=gfx942` leaks into plain g++ compile lines.
- **CSR byte-identity across hosts**: scp/rsync `data/cache/` from one host; do NOT regenerate independently. `gen_rmat.py` is seed-deterministic but `snap_to_csr.py` ordering can drift on numpy version differences.
- **Idempotent runs**: `scripts/run_re0.sh` skips outputs that already have both `.bin` and `.json`. To force rerun, delete the pair.
- **Frequent rsync**: sync `results/` to local after each phase, not at the end.

## Conventions

- **HIP API in source, CUDA shim on NVIDIA**: kernel files are `.hip`; on `-DGPU_BACKEND=CUDA` they compile as CUDA via `nvcc` with `src/pagerank/hip_cuda_compat.h` aliasing `hip*` to `cuda*`. Same convention as Paper 2.1's `src/sssp/hip_cuda_compat.h`.
- **Single source of truth for numbers**: once Phase 1 starts, drift metrics live in `results/_compare/*.json`. A future `scripts/sync_data_tables.py` (à la Paper 2.1) will roll them into `docs/plans/02_data_tables.md`.
- **Don't editorialize when stating data**: in summaries, give the numbers and the verdict; reserve framing for the manuscript.
- **Cite by file path + line**: e.g. `src/pagerank/pagerank.hip:84` for the `atomicAdd` scatter line.

## Active risks (post Phase 0)

- **R1 (Critical)**: PageRank may not drift cross-vendor (probability 30-40% per design doc). RE0 is the resolver. If `byte_diff_fraction < 0.001` on all datasets → STOP (see `docs/design/05_risk_register.md` §R1).
- **R7 (High)**: GPU access interruption — keep results synced locally, have backup vendor instance ready.
- **R8 (Medium)**: time conflict with Paper 2.1 (LaTeX conversion ongoing in `asplos-27`). Phase 0-2 must remain code-independent of 2.1.

## Never do

- Force push to main, amend published commits, `git reset --hard` on uncommitted work
- Run experiments without syncing prior results to local first
- Edit `docs/design/0*.md` without updating any cross-referenced sections (the docs reference each other by section)
- Replace `atomicAdd<float>` in `src/pagerank/pagerank.hip` with a deterministic reduction. The non-determinism is the experiment.
