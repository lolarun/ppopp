# Experiment status inventory

Living document. Each experiment: ID, status, what's done, what's missing, where data lives.

**Last updated:** 2026-05-02 (post E4 extension + E12.c relaxed-atomics A10 session).

---

## Status legend

- ✅ **Done** — sufficient data for paper inclusion
- 🟡 **Partial** — data exists but incomplete; further runs or analysis needed
- ❌ **Not started** — no data yet
- 🔵 **Skipped** — out-of-scope under Variant E framing

---

## E-series experiment matrix

| ID | Experiment | Status | Data location | What's missing |
|---|---|---|---|---|
| E1 | Cross-platform reproducibility characterization | 🟡 | `results/certs`, `results/amd/certs`, `results/{e1_nvidia,amd/e1_amd_nvidia}.jsonl` | 17 cert pairs verified byte-exact; Variant E may want extended dataset/precision matrix |
| E2 | Drift mechanism attribution (profiler) | 🔵 | — | Skipped — drift = 0 makes mechanism breakdown less interesting |
| E3 | Verifier soundness validation (vs Boost.Graph CPU truth) | 🟡 | 1085 SAT runs implicitly demonstrate non-false-rejection | Explicit comparison with CPU reference not run |
| E4 | Verifier coverage via error injection | ✅ | `results/e4_*.jsonl` | Extended to 4 datasets (ny_road, web_google, livejournal, usa_road); 100 seeds on small graphs (ny_road, web_google), 10 seeds on large graphs (livejournal, usa_road); 5 error kinds × magnitudes; 100% verifier UNSAT detection on all error kinds where signal is possible. MISSED_UNREACHABLE = 0/0 on fully-connected road graphs (graph-property, not verifier-deficient) |
| E5 | Real-world bug discovery on external libraries | ❌ | — | Tier-1 sharpener. Need Gunrock or cuGraph integration |
| E6 | Certificate emission overhead | 🟡 | `results/{e8,e9,w3_task1}.jsonl`, `results/amd/*` | NVIDIA 4 datasets + AMD 4 datasets done; need systematic table presentation |
| E7 | Verifier cost characterization (verifier_ms vs sssp_ms) | 🟡 | embedded in all JSONL `verifier_ms` field | Need explicit comparison tables / scaling plots |
| E8 | RMAT scaling study | 🟡 | `results/e8_scaling.jsonl` (NVIDIA), `results/amd/e8_amd_scaling.jsonl` (AMD) | RMAT-22..25 done × FP32+FP64 × emit/noemit × 3 reps; RMAT-26 requires full GPU (out of scope for VF) |
| E9 | Heterogeneity stress (weight × precision × structure) | 🟡 | `results/e9_weights.jsonl`, `results/amd/e9_amd_weights.jsonl` | 4 dist × 2 graphs × 2 reps done both platforms; could expand to more graphs |
| E10 | Cross-CUDA-version drift | 🔵 | — | Optional, deprioritised |
| E11 | Bellman-Ford companion (algorithm-vs-discipline test) | ❌ | — | Tier-1 sharpener for §II.E argument |
| E12 | Reproducibility breaking conditions (FP16, non-strict atomics, mixed precision) | 🟡 | `results/e12c/{strict,relaxed}.jsonl` | E12.c relaxed atomics ✅ (commit `e6d0bc4`): strict 30/30 SAT + 1 unique d_hash, relaxed 30/30 UNSAT_RELAXATION + 5 unique, across 6 datasets (ny_road, web_google, livejournal, usa_road, rmat-20, rmat-22), FP32. E12.a TF32 reframed out (inapplicable to scalar SSSP — no tensor-core path). E12.b FP16 weights and E12.d mixed precision still pending. |
| E13 | Theoretical analysis validation (controlled experiments) | ❌ | — | Tier-2 — confirms §II.E claim with constructed examples |

---

## W-series milestones (engineering phases)

| ID | Phase | Status |
|---|---|---|
| W1 | CPU Dijkstra reference | ✅ |
| W2 | CPU Δ-stepping reference | ✅ |
| W3 | GPU Δ-stepping correctness (NVIDIA) | ✅ — 1000/1000 stress + 12 dataset×path SAT |
| W3' | GPU Δ-stepping correctness (AMD MI300X) | ✅ — 1000/1000 stress + 12 dataset×path SAT |
| W4 | Performance tuning | 🟡 — emission overhead measured; not aggressively tuned (paper does not depend on TEPS) |
| W5 | Cross-platform parity | ✅ — 17/17 reachable-only d_hash byte-exact |
| W6 | Certificate emission integration | ✅ — packed atomic FP32 + post-hoc FP64 |
| W7 | Soundness fuzzer | ✅ — 1000/1000 stress on each platform |

---

## Counts (audit, 2026-05-02)

**Per platform:**
- NVIDIA A10: 1085 runs, 1055 verified SAT, 2 expected UNSAT (F10 boundary), 0 unexplained UNSAT
- AMD MI300X VF: 1085 runs, 1055 verified SAT, 2 expected UNSAT (F10 boundary), 0 unexplained UNSAT

**Cross-vendor:**
- 17/17 reachable-only d_hash match
- 12/17 .pi.bin byte-exact match; 5/17 differ (FP32 road, race-determined)

**Cert binaries:** 34 per platform = 68 total, all sizes match `n_vertices × element_size`.

**E4 error injection (2026-05-02 extension):** 2640 new injection cases on top of 1080 from commit `2f04f1a`. Coverage: 4 datasets × 5 error kinds × seeds (100 small / 10 large) × magnitudes. See `results/e4_*.jsonl` and [02_data_tables.md](02_data_tables.md).

**E12.c relaxed atomics (2026-05-02):** 6 datasets × 5 reps × 2 build modes = 60 runs total. Strict: 30/30 SAT, 1 unique d_hash per dataset. Relaxed: 30/30 UNSAT_RELAXATION, 5 unique d_hash per dataset. See `results/e12c/{strict,relaxed}.jsonl`.

---

## Variant E new requirements (added experiments)

Three new experiments needed for full Variant E backing:

- **E11** Bellman-Ford companion (validates §II.E premise on a different algorithm)
- **E12** Reproducibility breaking conditions (gives controlled drift evidence)
- **E13** Theoretical analysis validation (confirms §II.E formal argument)

See [03_action_items.md](03_action_items.md) for priority and time estimates.
