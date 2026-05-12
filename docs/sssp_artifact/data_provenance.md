# Dataset provenance

All datasets are publicly available. SNAP graphs are remapped to FP weights; DIMACS road graphs use original integer weights.

---

## DIMACS 9th Implementation Challenge — Road networks

| Dataset | Source URL | Original format | Vertices | Edges | md5 (.gr) |
|---|---|---|---|---|---|
| ny_road | http://www.diag.uniroma1.it/challenge9/data/USA-road-d/USA-road-d.NY.gr.gz | DIMACS .gr (integer travel-time weights) | 264,346 | 733,846 | `c6b493b8faa0eb866d04d6725829ce69` |
| usa_road | http://www.diag.uniroma1.it/challenge9/data/USA-road-d/USA-road-d.USA.gr.gz | DIMACS .gr | 23,947,347 | 57,708,624 | `4f7e8000f40748ffed10d30035d1c69c` |

Loader: `src/io/dimacs_loader.cpp` parses standard DIMACS `p sp` and `a u v w` lines.

---

## SNAP — Web and social graphs

| Dataset | Source URL | Original format | Vertices | Edges | md5 (.gr post-conversion) |
|---|---|---|---|---|---|
| web_google | https://snap.stanford.edu/data/web-Google.txt.gz | SNAP edge list (unweighted) | 875,713 | 5,105,039 | `65343c5c336cf916c5ea23af68003caf` |
| livejournal | https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz | SNAP edge list (unweighted) | 4,846,609 | 68,475,391 | `accd0eefdcf12af791e458e8d6062890` |

**Conversion:** `scripts/snap_to_dimacs.py` reads the SNAP edge list and emits DIMACS .gr with FP weights uniform in [0.001, 1.0], seeded by `--seed`. Two-pass streaming (no full edge list in memory):
1. Pass 1: collect node IDs, count edges
2. Pass 2: emit edges with deterministic per-edge weight based on input order + global seed

Default seed: 42. Same seed → byte-identical .gr output across runs.

---

## RMAT (Graph500) — Synthetic

Generated in-memory via `src/io/rmat_generator.cpp` using Graph500 spec parameters (a=0.57, b=0.19, c=0.19, d=0.05). FP weights uniform in [0.001, 1.0] derived from `std::mt19937_64` seeded by `--rmat-seed`.

| Scale | Vertices (2^scale) | Default edges (16×n_v) | F22+F23 ef=32 (32×n_v) |
|---|---|---|---|
| 18 | 262,144 | 4,194,304 | 8,388,608 |
| 20 | 1,048,576 | 16,777,216 | 33,554,432 |
| 22 | 4,194,304 | 67,108,864 | 134,217,728 |
| 23 | 8,388,608 | 134,217,728 | 268,435,456 |
| 24 | 16,777,216 | 268,435,456 | 536,870,912 |
| 25 | 33,554,432 | 536,870,912 | 1,073,741,824 |

`std::mt19937_64` is platform-independent; same `(scale, edgefactor, seed)` → byte-identical CSR on any IEEE-754-compliant host.

The F22 + F23 Gunrock cross-implementation audit uses RMAT-20 and RMAT-22 with **edgefactor 32** (default harness setting). The §3.3 Table 1 RMAT-20 cells use **edgefactor 16** per the original §3 corpus. These are *different* graphs by design (different edge counts, different cert binaries, different `d_hash`); the F22 + F23 evidence is over freshly-generated graphs at ef=32 with our SSSP and Gunrock both consuming the same `.csr → .mtx` derived input, ensuring byte-comparability.

---

## CSR ↔ MatrixMarket conversion (for Gunrock audits)

Gunrock 2.2.0's SSSP example consumes MatrixMarket (`.mtx`). Our harness emits binary CSR (`.csr`). For the F22 + F23 audit:

- **Road / web / social graphs**: `scripts/gr_to_mtx.py data/cache/<name>.gr data/cache/<name>.mtx` (1-indexed in both formats; header swap).
- **RMAT graphs**: first save the in-memory CSR with `run_sssp --rmat-scale=N --rmat-edgefactor=K --rmat-seed=S --save-csr=data/cache/<name>.csr --reps=1 --emit-cert=0 --verify=0`, then `scripts/csr_to_mtx.py data/cache/<name>.csr data/cache/<name>.mtx`.

The `.csr` binary format is documented in `src/io/csr_io.cpp` (8-byte magic, FP32/FP64 tag, nv+ne, row_offsets[nv+1], col_indices[ne], weights[ne], CRC64 trailer).

---

## Verification

Sanity-check your downloaded datasets:

```bash
md5sum data/cache/*.gr
```

Should produce the hashes listed above. If different, re-download.
