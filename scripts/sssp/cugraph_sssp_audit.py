#!/usr/bin/env python3
"""cuGraph SSSP cross-implementation audit (companion to F22+F23 Gunrock audit).

Loads a graph (DIMACS .gr or MatrixMarket .mtx), runs cuGraph SSSP from
source=0, and dumps the resulting d (distance) and pi (predecessor) vectors
to binary files in the same layout as our cert binaries (.d.bin float32,
.pi.bin uint32). Then compares against our reference cert if present.

Usage:
    python3 scripts/cugraph_sssp_audit.py \\
        --input data/cache/ny_road.gr \\
        --output results/a2_1_cugraph/ny_road_fp32 \\
        [--reference results/certs/ny_road_fp32]

Requires NVIDIA RAPIDS (cugraph + cudf). Install:
    pip install --extra-index-url=https://pypi.nvidia.com \\
        cugraph-cu12 cudf-cu12 dask-cudf-cu12

Run on a CUDA-capable machine (A10 + CUDA 12.x).
"""
from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np


def load_dimacs_gr(path: Path) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Parse DIMACS .gr — return (n_v, src, dst, weight) edge arrays (0-indexed)."""
    n_v = 0
    src_list: list[int] = []
    dst_list: list[int] = []
    w_list: list[float] = []
    with open(path, "rb") as f:
        for raw in f:
            if not raw or raw[0:1] in (b"c", b"\n"):
                continue
            line = raw.decode("ascii", errors="replace").strip()
            if line.startswith("p "):
                _, _, nvs, _ = line.split()
                n_v = int(nvs)
            elif line.startswith("a "):
                _, us, vs, ws = line.split()
                src_list.append(int(us) - 1)
                dst_list.append(int(vs) - 1)
                w_list.append(float(ws))
    return (n_v,
            np.asarray(src_list, dtype=np.int32),
            np.asarray(dst_list, dtype=np.int32),
            np.asarray(w_list, dtype=np.float32))


def load_mtx(path: Path) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Parse MatrixMarket coordinate file — return (n_v, src, dst, weight) (0-indexed)."""
    src_list: list[int] = []
    dst_list: list[int] = []
    w_list: list[float] = []
    n_v = 0
    with open(path, "r") as f:
        header = f.readline()
        if not header.startswith("%%MatrixMarket"):
            raise SystemExit(f"  load_mtx: bad header in {path}")
        # skip remaining comment lines
        for line in f:
            if line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) >= 3 and n_v == 0:
                # First non-comment line is the size header
                n_v = int(parts[0])
                continue
            if len(parts) >= 3:
                u = int(parts[0]) - 1
                v = int(parts[1]) - 1
                w = float(parts[2])
                src_list.append(u)
                dst_list.append(v)
                w_list.append(w)
    return (n_v,
            np.asarray(src_list, dtype=np.int32),
            np.asarray(dst_list, dtype=np.int32),
            np.asarray(w_list, dtype=np.float32))


def load_csr_bin(path: Path) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Read harness binary .csr (post-remap) — return (n_v, src, dst, weight)."""
    MAGIC = 0x4353520000000001
    TAG_FP32 = 0x46503332
    TAG_FP64 = 0x46503634
    with open(path, "rb") as f:
        magic = struct.unpack("<Q", f.read(8))[0]
        if magic != MAGIC:
            raise SystemExit(f"  load_csr_bin: bad magic 0x{magic:016x}")
        tag = struct.unpack("<I", f.read(4))[0]
        wdtype = np.float32 if tag == TAG_FP32 else np.float64
        nv, ne = struct.unpack("<QQ", f.read(16))
        out_row = np.frombuffer(f.read((nv + 1) * 8), dtype=np.uint64).copy()
        out_col = np.frombuffer(f.read(ne * 4), dtype=np.uint32).copy()
        out_w = np.frombuffer(f.read(ne * np.dtype(wdtype).itemsize), dtype=wdtype).astype(np.float32)
    src = np.repeat(np.arange(int(nv), dtype=np.int32), np.diff(out_row).astype(np.int64))
    dst = out_col.astype(np.int32)
    return int(nv), src, dst, out_w.copy()


def run_cugraph_sssp(n_v: int, src: np.ndarray, dst: np.ndarray,
                     weight: np.ndarray, source: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Build cuGraph and run SSSP. Returns (d_fp32[n_v], pi_uint32[n_v], wall_ms)."""
    import cudf
    import cugraph

    df = cudf.DataFrame({
        "src": cudf.Series(src),
        "dst": cudf.Series(dst),
        "weight": cudf.Series(weight),
    })
    g = cugraph.Graph(directed=True)
    g.from_cudf_edgelist(df, source="src", destination="dst", edge_attr="weight",
                         renumber=False)
    t0 = time.time()
    out = cugraph.sssp(g, source=source)
    wall_ms = (time.time() - t0) * 1000.0

    # Output DataFrame columns: vertex, distance, predecessor
    out = out.sort_values("vertex").to_pandas()
    d = np.full(n_v, np.float32("inf"), dtype=np.float32)
    pi = np.full(n_v, np.uint32(0xFFFFFFFF), dtype=np.uint32)
    for _, row in out.iterrows():
        v = int(row["vertex"])
        if v < 0 or v >= n_v:
            continue
        dist = row["distance"]
        pred = row["predecessor"]
        # cuGraph uses np.float32(inf) or sometimes np.finfo(np.float32).max for unreachable
        if not np.isfinite(dist) or dist >= 1e38:
            d[v] = np.float32("inf")
            pi[v] = np.uint32(0xFFFFFFFF)
        else:
            d[v] = np.float32(dist)
            pi[v] = np.uint32(pred) if pred >= 0 else np.uint32(0xFFFFFFFF)
    return d, pi, wall_ms


def compare_with_reference(d_cu: np.ndarray, ref_prefix: Path) -> None:
    ref_d_path = ref_prefix.with_suffix(".d.bin")
    if not ref_d_path.exists():
        print(f"  no reference at {ref_d_path}; skipping byte-equality check")
        return
    ref_d = np.fromfile(ref_d_path, dtype=np.float32)
    if len(ref_d) != len(d_cu):
        print(f"  LEN MISMATCH: cugraph={len(d_cu)} vs reference={len(ref_d)}")
        return
    n_unreach_cu = int(np.isinf(d_cu).sum())
    n_unreach_ref = int(np.isinf(ref_d).sum())
    eq = np.array_equal(d_cu, ref_d)
    tag = "BYTE-IDENTICAL" if eq else "DIFFERENT"
    print(f"  np.array_equal vs reference: {tag}")
    print(f"    nv={len(d_cu)} unreach_cugraph={n_unreach_cu} unreach_reference={n_unreach_ref}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to .gr / .mtx / .csr")
    ap.add_argument("--output", required=True, help="cert prefix for .d.bin / .pi.bin")
    ap.add_argument("--reference", default=None, help="(optional) prefix to our cert for byte-comparison")
    ap.add_argument("--source", type=int, default=0, help="SSSP source vertex (default 0)")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_pref = Path(args.output)
    out_pref.parent.mkdir(parents=True, exist_ok=True)

    print(f"=== cuGraph SSSP audit on {in_path.name} ===")
    print(f"  loading graph ...")
    t0 = time.time()
    if in_path.suffix == ".gr":
        n_v, src, dst, weight = load_dimacs_gr(in_path)
    elif in_path.suffix == ".mtx":
        n_v, src, dst, weight = load_mtx(in_path)
    elif in_path.suffix == ".csr":
        n_v, src, dst, weight = load_csr_bin(in_path)
    else:
        raise SystemExit(f"  unknown input format: {in_path.suffix}")
    print(f"  loaded n_v={n_v} n_e={len(src)} ({time.time()-t0:.1f}s)")

    print(f"  running cuGraph SSSP from source={args.source} ...")
    d, pi, wall_ms = run_cugraph_sssp(n_v, src, dst, weight, args.source)
    print(f"  cuGraph SSSP wall: {wall_ms:.1f} ms")

    d.tofile(str(out_pref) + ".d.bin")
    pi.tofile(str(out_pref) + ".pi.bin")
    print(f"  wrote {out_pref}.d.bin ({d.nbytes / 1024 / 1024:.1f} MB)")
    print(f"  wrote {out_pref}.pi.bin ({pi.nbytes / 1024 / 1024:.1f} MB)")

    if args.reference:
        ref_pref = Path(args.reference)
        print(f"  comparing against reference {ref_pref} ...")
        compare_with_reference(d, ref_pref)

    return 0


if __name__ == "__main__":
    sys.exit(main())
