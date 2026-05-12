#!/usr/bin/env python3
"""A1.17 -- pi-divergence vertex-level pattern classification.

Sec 4.4 / Sec 6.2 claim that pi differs across vendors precisely on
vertices where multiple incoming edges yield FP-equal candidate
distances (race-determined ties).  A1.14 confirmed 5/17 cert pairs
have differing pi: ny_road_fp32 (default), ny_road_uniform_fp32,
usa_road_fp32, usa_road_gaussian_fp32, usa_road_uniform_fp32.

For each differing pair, this script:
  1. Identifies vertices v where pi_NV[v] != pi_AMD[v]
  2. For each such v, computes incoming candidate distances
     {d[u] + w(u,v) : (u,v) in E} using the graph + d vector
  3. Reports whether at least two candidates are FP32-equal
     (i.e., the vertex is in a tie state, which is the predicted
     mechanism for pi divergence)

If 100% of differing-pi vertices have FP-tied candidates -> sec 4.4
is empirically validated as the structural cause of pi divergence;
the paper can promote sec 4.4 from hypothesis to measurement.

Run from project root:
    python scripts/offline_a1_17.py
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAIN_REPO = Path(r"C:/Users/Justin/Documents/Confs/asplos-27")

# 5 cert pairs identified by A1.14 as having differing pi.
# (config_name, source_path, weight-remap-or-default)
#
# For "default" configs the source_path is a DIMACS .gr file (parsed below).
# For "uniform"/"gaussian" remap configs, source_path is a binary .csr file
# produced by `run_sssp --weight-dist=<dist> --save-csr=<path>` — i.e., the
# graph with weight remap already applied.  Replaying the harness PRNG in
# Python would diverge from the actual remapped weights; reading the
# harness-saved CSR ensures we analyze the *exact* graph the cert came from.
DIFFERING_PI_CONFIGS = [
    ("ny_road_fp32",            "ny_road.gr",                "default"),
    ("ny_road_uniform_fp32",    "ny_road_uniform_fp32.csr",  "uniform"),
    ("usa_road_fp32",           "usa_road.gr",               "default"),
    ("usa_road_gaussian_fp32",  "usa_road_gaussian_fp32.csr","gaussian"),
    ("usa_road_uniform_fp32",   "usa_road_uniform_fp32.csr", "uniform"),
]


def parse_gr_csr_in(path: Path) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Parse DIMACS .gr -> CSR by IN-edges (so we can list in-edges of v).

    Returns: (n_v, row_off, col_idx, weights_fp32)
      row_off[v] .. row_off[v+1]  index into col_idx / weights for in-edges of v
      col_idx[k] = source vertex u
      weights[k] = w(u,v) as float32
    """
    n_v = 0
    edges_in: list[list[tuple[int, float]]] = []  # edges_in[v] = [(u, w), ...]
    with open(path, "rb") as f:
        for raw in f:
            if not raw or raw[0:1] in (b"c", b"\n"):
                continue
            line = raw.decode("ascii", errors="replace").strip()
            if line.startswith("p "):
                _, _, nvs, _ = line.split()
                n_v = int(nvs)
                edges_in = [[] for _ in range(n_v)]
            elif line.startswith("a "):
                _, us, vs, ws = line.split()
                u = int(us) - 1
                v = int(vs) - 1
                w = float(ws)
                edges_in[v].append((u, w))

    # Flatten to CSR
    counts = np.array([len(es) for es in edges_in], dtype=np.int64)
    row_off = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    n_e = int(row_off[-1])
    col_idx = np.empty(n_e, dtype=np.int64)
    weights = np.empty(n_e, dtype=np.float32)
    for v in range(n_v):
        s, e = row_off[v], row_off[v+1]
        for k, (u, w) in enumerate(edges_in[v]):
            col_idx[s+k] = u
            weights[s+k] = np.float32(w)
    return n_v, row_off, col_idx, weights


def parse_csr_in(path: Path) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Read the binary CSR format produced by harness `--save-csr=` and
    transpose to an IN-edge CSR for FP-tied incoming-candidate analysis.

    Binary CSR layout (src/io/csr_io.cpp):
        8B magic 0x4353520000000001
        4B precision tag (FP32=0x46503332, FP64=0x46503634)
        8B nv, 8B ne
        (nv+1) * 8B row_offsets (uint64)        OUT-edge offsets
        ne * 4B col_indices (uint32)
        ne * sizeof(W) weights
        8B CRC64 trailer
    """
    import struct
    MAGIC = 0x4353520000000001
    TAG_FP32 = 0x46503332
    TAG_FP64 = 0x46503634
    with open(path, "rb") as f:
        magic = struct.unpack("<Q", f.read(8))[0]
        if magic != MAGIC:
            raise SystemExit(f"  parse_csr_in: bad magic 0x{magic:016x} in {path}")
        tag = struct.unpack("<I", f.read(4))[0]
        if tag == TAG_FP32:
            wdtype = np.float32
        elif tag == TAG_FP64:
            wdtype = np.float64
        else:
            raise SystemExit(f"  parse_csr_in: bad precision tag 0x{tag:08x} in {path}")
        nv, ne = struct.unpack("<QQ", f.read(16))
        out_row = np.frombuffer(f.read((nv + 1) * 8), dtype=np.uint64).copy()
        out_col = np.frombuffer(f.read(ne * 4), dtype=np.uint32).copy()
        out_w   = np.frombuffer(f.read(ne * np.dtype(wdtype).itemsize), dtype=wdtype).copy()
    # CSR is stored as out-edges; transpose to in-edges so we can answer
    # "for vertex v, list all (u, w(u, v))".  Use bucket sort by destination:
    #   - count in-degree per v
    #   - prefix-sum to in_row_off
    #   - scatter (u, w) to in_col, in_weights at in_row_off[col[k]] cursors
    in_count = np.bincount(out_col, minlength=int(nv)).astype(np.int64)
    in_row = np.concatenate(([0], np.cumsum(in_count))).astype(np.int64)
    in_col = np.empty(int(ne), dtype=np.int64)
    in_w   = np.empty(int(ne), dtype=np.float32)  # cast to fp32 for tied-candidate test
    cursor = in_row[:-1].copy()
    # Build u-array per OUT edge: u_for_edge[k] = source of edge k
    u_for_edge = np.repeat(np.arange(int(nv), dtype=np.int64),
                           np.diff(out_row).astype(np.int64))
    for k in range(int(ne)):
        v = int(out_col[k])
        u = int(u_for_edge[k])
        w = float(out_w[k])
        idx = cursor[v]
        in_col[idx] = u
        in_w[idx]   = np.float32(w)
        cursor[v]  = idx + 1
    return int(nv), in_row, in_col, in_w


def load_cert(prefix: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.fromfile(prefix.with_suffix(".d.bin"), dtype=np.float32)
    pi = np.fromfile(prefix.with_suffix(".pi.bin"), dtype=np.uint32)
    return d, pi


def classify_vertex(v: int, d: np.ndarray, row_off: np.ndarray,
                    col_idx: np.ndarray, weights: np.ndarray
                   ) -> tuple[float, list[tuple[int, float]], int]:
    """For vertex v, return
      (min_candidate_d, all candidates achieving min, n_distinct_candidate_values)
    where each candidate is (u, d[u] + w(u,v)) computed in fp32.
    """
    s, e = int(row_off[v]), int(row_off[v+1])
    INF = np.float32("inf")
    cands: list[tuple[int, float]] = []
    for k in range(s, e):
        u = int(col_idx[k])
        if d[u] == INF:
            continue
        nd = np.float32(d[u]) + np.float32(weights[k])
        cands.append((u, float(nd)))
    if not cands:
        return float("inf"), [], 0
    min_d = min(c[1] for c in cands)
    tied = [(u, nd) for (u, nd) in cands if nd == min_d]
    distinct = len(set(c[1] for c in cands))
    return min_d, tied, distinct


def analyze_one(name: str, source_filename: str) -> None:
    print(f"\n=== {name}  ({source_filename}) ===")
    nv_pref = MAIN_REPO / "results" / "certs" / name
    amd_pref = MAIN_REPO / "results" / "amd" / "certs" / name
    if not (nv_pref.with_suffix(".d.bin").exists()
            and amd_pref.with_suffix(".d.bin").exists()):
        print(f"  MISSING cert files for {name}, skip")
        return

    nv_d, nv_pi = load_cert(nv_pref)
    amd_d, amd_pi = load_cert(amd_pref)

    if not np.array_equal(nv_d, amd_d):
        print(f"  NOTE: d differs across vendors for {name} -- unexpected")
    diff_v = np.where(nv_pi != amd_pi)[0]
    print(f"  pi differs at {len(diff_v)} vertices "
          f"(out of {len(nv_pi)} total, {100.0*len(diff_v)/len(nv_pi):.4f}%)")

    if len(diff_v) == 0:
        print(f"  no differing-pi vertices -- nothing to classify (unexpected for {name})")
        return

    # Source path is either a DIMACS .gr (default-weight configs) or a
    # harness-saved binary .csr (post-remap; produced by running
    # `run_sssp --weight-dist=<dist> --save-csr=...` once).
    source_path = MAIN_REPO / "data" / "cache" / source_filename
    if not source_path.exists():
        print(f"  MISSING source: {source_path}")
        if source_filename.endswith(".csr"):
            print(f"    (regenerate via `run_sssp --weight-dist={'gaussian' if 'gaussian' in name else 'uniform'} "
                  f"--dataset=<base.gr> --save-csr={source_path}`)")
        return

    print(f"  parsing graph (in-edge CSR) ...", end="", flush=True)
    t0 = time.time()
    if source_filename.endswith(".csr"):
        n_v, row_off, col_idx, weights = parse_csr_in(source_path)
    else:
        n_v, row_off, col_idx, weights = parse_gr_csr_in(source_path)
    print(f" done ({time.time()-t0:.1f}s, n_v={n_v}, n_e={len(col_idx)})")

    if len(nv_d) != n_v:
        print(f"  SHAPE MISMATCH: cert n_v={len(nv_d)}, graph n_v={n_v}")
        return

    # For each differing-pi vertex, classify whether it's a tie case.
    print(f"  classifying {len(diff_v)} differing-pi vertices ...", end="", flush=True)
    t0 = time.time()
    n_tied = 0
    n_lone_min = 0  # only one candidate achieves min (paper hypothesis: this should be 0)
    n_pi_in_tied_set_NV = 0  # pi_NV is in the tied set
    n_pi_in_tied_set_AMD = 0
    n_pi_NV_unreach = 0
    n_pi_AMD_unreach = 0

    INVALID = np.uint32(0xFFFFFFFF)
    for v_int in diff_v:
        v = int(v_int)
        # If v is unreachable (d[v] == inf), pi is INVALID; skip.
        if nv_d[v] == np.float32("inf"):
            continue
        min_d, tied, distinct = classify_vertex(v, nv_d, row_off, col_idx, weights)
        if not tied:
            continue
        if len(tied) >= 2:
            n_tied += 1
        else:
            n_lone_min += 1

        tied_us = {u for (u, _) in tied}
        if int(nv_pi[v]) in tied_us:
            n_pi_in_tied_set_NV += 1
        else:
            if nv_pi[v] == INVALID:
                n_pi_NV_unreach += 1

        if int(amd_pi[v]) in tied_us:
            n_pi_in_tied_set_AMD += 1
        else:
            if amd_pi[v] == INVALID:
                n_pi_AMD_unreach += 1

    print(f" done ({time.time()-t0:.1f}s)")

    total_classified = n_tied + n_lone_min
    print(f"\n  Of {len(diff_v)} differing-pi vertices:")
    print(f"    classified (had >=1 reachable in-edge): {total_classified}")
    print(f"      -> >=2 in-edges produce FP-equal candidate min: {n_tied}  "
          f"({100.0*n_tied/max(total_classified,1):.1f}%)")
    print(f"      -> only one candidate achieves min (no tie): {n_lone_min}  "
          f"({100.0*n_lone_min/max(total_classified,1):.1f}%)")
    print(f"    pi_NV is in the tied-candidate set: {n_pi_in_tied_set_NV}/{total_classified}")
    print(f"    pi_AMD is in the tied-candidate set: {n_pi_in_tied_set_AMD}/{total_classified}")
    if n_pi_NV_unreach or n_pi_AMD_unreach:
        print(f"    pi was INVALID despite differing: NV={n_pi_NV_unreach}, AMD={n_pi_AMD_unreach}")

    if n_lone_min == 0:
        print(f"  RESULT: 100% of differing-pi vertices on {name} have >=2 FP-tied "
              f"incoming candidates.  The sec 4.4 mechanism is empirically validated.")
    else:
        print(f"  WARNING: {n_lone_min} differing-pi vertices have a unique-min "
              f"candidate.  sec 4.4 mechanism is the dominant but not the only cause.")


def main() -> int:
    print("=" * 78)
    print("A1.17 -- pi-divergence vertex-level pattern classification")
    print("=" * 78)
    print()
    print("For each cert pair where pi differs across vendors, classify whether")
    print("the differing vertices have FP-tied incoming candidates (the sec 4.4")
    print("predicted mechanism).  Default-weight configs read DIMACS .gr files;")
    print("uniform/gaussian remap configs read harness-saved binary .csr files")
    print("(post-remap topology, ensuring we analyze the exact graph the cert")
    print("came from).")

    for name, source_filename, remap in DIFFERING_PI_CONFIGS:
        analyze_one(name, source_filename)

    print()
    print("Interpretation:")
    print("  100% tied -> sec 4.4 hypothesis validated (pi divergence is exactly")
    print("    the FP-tie race-resolution case).  Paper can graduate sec 4.4 from")
    print("    hypothesis to measurement.")
    print("  Some lone-min -> there's a second source of pi divergence; sec 4.4")
    print("    needs revision.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
