#!/usr/bin/env python3
"""A1.20 — CPU Dijkstra baseline for cross-vendor d_hash anchor.

Per §3 + §6.2 the paper establishes that NVIDIA and AMD GPU strict-mode
outputs are byte-equal.  This addresses *cross-vendor consistency*.
A1.20 asks the orthogonal *correctness anchor* question: is the GPU
strict output also byte-equal to a deterministic CPU sequential
Dijkstra?  If yes, the GPU result is *equal to* the canonical answer
(not just stable across GPU implementations) — a stronger §3 claim.

Implementation: pure Python heapq Dijkstra with explicit np.float32
arithmetic.  Because IEEE 754 single-add is operand-determined, a
sequential Dijkstra with FP32 arithmetic should produce a byte-pattern
that matches GPU strict if and only if the §4.1 algebraic argument
("min-plus + atomic CAS makes d a function of (graph, weights, source,
precision) only, not of the schedule") holds end-to-end.

Run from project root:
    python scripts/offline_a1_20.py
"""
from __future__ import annotations
import heapq
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Worktree's data/cache is gitignored and not symlinked; use main repo's data dir.
MAIN_REPO = Path(r"C:/Users/Justin/Documents/Confs/asplos-27")


def parse_gr(path: Path) -> tuple[int, list[list[tuple[int, float]]]]:
    """Parse DIMACS .gr file -> (n_vertices, adjacency list with float weights).

    DIMACS format:
      c <comment>
      p sp <V> <E>
      a <u> <v> <w>

    Weights may be int (DIMACS travel times) or float (FP-remapped graphs).
    Stored as Python floats; cast to np.float32 at relaxation time.
    Returns 0-indexed adjacency.
    """
    n_v = 0
    adj: list[list[tuple[int, float]]] = []
    with open(path, "rb") as f:
        for raw in f:
            if not raw or raw[0:1] in (b"c", b"\n"):
                continue
            line = raw.decode("ascii", errors="replace").strip()
            if line.startswith("p "):
                _, _, nvs, _ = line.split()
                n_v = int(nvs)
                adj = [[] for _ in range(n_v)]
            elif line.startswith("a "):
                _, us, vs, ws = line.split()
                u = int(us) - 1   # DIMACS is 1-indexed
                v = int(vs) - 1
                w = float(ws)
                adj[u].append((v, w))
    return n_v, adj


def dijkstra_fp32(adj: list[list[tuple[int, int]]], source: int, n_v: int) -> np.ndarray:
    """Sequential Dijkstra with strict FP32 single-add semantics.

    d is np.float32 throughout; every relaxation is a single np.float32 +
    which is IEEE 754 operand-determined.  No reduction, no fused multiply-
    add, no other rounding modes.
    """
    INF32 = np.float32("inf")
    d = np.full(n_v, INF32, dtype=np.float32)
    d[source] = np.float32(0.0)

    heap = [(np.float32(0.0), source)]
    while heap:
        du, u = heapq.heappop(heap)
        if du > d[u]:
            continue
        for v, w in adj[u]:
            nd = np.float32(du) + np.float32(w)
            if nd < d[v]:
                d[v] = nd
                heapq.heappush(heap, (nd, v))
    return d


def reachable_crc32(d: np.ndarray) -> str:
    """CRC32 over the reachable (non-sentinel) subset, big-endian byte order
    matching the C++ harness.  Empty reachable set -> '00000000'."""
    INF = np.float32("inf")
    reach = (d != INF)
    if not reach.any():
        return "00000000"
    blob = d[reach].tobytes()
    return f"{zlib.crc32(blob) & 0xFFFFFFFF:08x}"


def load_gpu_cert_d(prefix: Path) -> np.ndarray:
    return np.fromfile(prefix.with_suffix(".d.bin"), dtype=np.float32)


def main() -> int:
    # Config: (graph file, GPU cert prefix, source vertex)
    # Source vertex is 0 across the paper; weights are DIMACS integers.
    # Limit to FP32 datasets where Python sequential Dijkstra is tractable.
    # usa_road has 24M vertices -> Dijkstra ~ minutes in pure Python; skip
    # unless the user wants to wait.  Address the smaller graphs first.
    configs = [
        ("ny_road",     MAIN_REPO / "data" / "cache" / "ny_road.gr",
                        MAIN_REPO / "results" / "certs" / "ny_road_fp32",
                        MAIN_REPO / "results" / "amd" / "certs" / "ny_road_fp32",
                        0),
        ("web_google",  MAIN_REPO / "data" / "cache" / "web_google.gr",
                        MAIN_REPO / "results" / "certs" / "web_google_fp32",
                        MAIN_REPO / "results" / "amd" / "certs" / "web_google_fp32",
                        0),
        ("livejournal", MAIN_REPO / "data" / "cache" / "livejournal.gr",
                        MAIN_REPO / "results" / "certs" / "livejournal_fp32",
                        MAIN_REPO / "results" / "amd" / "certs" / "livejournal_fp32",
                        0),
    ]

    print("=" * 78)
    print("A1.20 -- CPU Dijkstra (FP32) baseline vs GPU strict-mode d_hash")
    print("=" * 78)
    print(f"\n{'dataset':<14}  {'n_v':>10}  "
          f"{'CPU d_hash':>12}  {'NV d_hash':>12}  {'AMD d_hash':>12}  "
          f"{'CPU=NV?':>8}  {'CPU=AMD?':>9}  {'NV=AMD?':>9}")
    print("-" * 100)

    for name, gr_path, nv_pref, amd_pref, src in configs:
        if not gr_path.exists():
            print(f"{name:<14}  MISSING: {gr_path}")
            continue
        if not nv_pref.with_suffix(".d.bin").exists():
            print(f"{name:<14}  MISSING NV cert: {nv_pref}")
            continue

        # Parse graph (this is the slow part; ~30s for ny_road, ~few min for livejournal)
        print(f"  [{name}] parsing .gr ...", end="", flush=True)
        import time
        t0 = time.time()
        n_v, adj = parse_gr(gr_path)
        print(f" done ({time.time()-t0:.1f}s, n_v={n_v}).  Running Dijkstra...",
              end="", flush=True)

        t0 = time.time()
        d_cpu = dijkstra_fp32(adj, src, n_v)
        print(f" done ({time.time()-t0:.1f}s).")

        cpu_h = reachable_crc32(d_cpu)
        nv_d = load_gpu_cert_d(nv_pref)
        amd_d = load_gpu_cert_d(amd_pref) if amd_pref.with_suffix(".d.bin").exists() else None
        nv_h = reachable_crc32(nv_d)
        amd_h = reachable_crc32(amd_d) if amd_d is not None else "n/a"

        cpu_eq_nv = "Y" if cpu_h == nv_h else "N"
        cpu_eq_amd = "Y" if (amd_d is not None and cpu_h == amd_h) else ("-" if amd_d is None else "N")
        nv_eq_amd = "Y" if (amd_d is not None and nv_h == amd_h) else ("-" if amd_d is None else "N")

        print(f"{name:<14}  {n_v:>10d}  "
              f"{cpu_h:>12}  {nv_h:>12}  {amd_h:>12}  "
              f"{cpu_eq_nv:>8}  {cpu_eq_amd:>9}  {nv_eq_amd:>9}")

        # Also full-vector byte-equality (including sentinel encoding)
        full_cpu_eq_nv = bool(np.array_equal(d_cpu, nv_d))
        full_cpu_eq_amd = bool(np.array_equal(d_cpu, amd_d)) if amd_d is not None else None
        print(f"  -> full-vector bit-equal: CPU=NV {full_cpu_eq_nv}, "
              f"CPU=AMD {full_cpu_eq_amd}")

    print()
    print("Interpretation:")
    print("  - If CPU=NV=AMD on every dataset, sec 3 cross-vendor claim gains a")
    print("    correctness anchor: the byte-exact GPU result IS the canonical")
    print("    answer (not just stable across GPUs).")
    print("  - If CPU != GPU, the GPU result is internally consistent but")
    print("    diverges from sequential Dijkstra; this is itself a finding")
    print("    (FP rounding under different summation orders -- sec 4.4 pi-divergence")
    print("    analog for d).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
