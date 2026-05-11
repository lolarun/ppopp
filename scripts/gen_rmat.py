#!/usr/bin/env python3
"""Generate an R-MAT (Recursive MATrix) graph in CSR binary format.

Default parameters match Graph500 / RMAT-22:
    scale=22 (N = 2^22 = 4,194,304 vertices)
    edge_factor=16 (E = 16 * N)
    a=0.57, b=0.19, c=0.19, d=0.05  (Graph500 spec)

Output format matches include/csr.hpp / scripts/snap_to_csr.py.

Self-loops dropped, duplicate directed edges deduplicated. Reproducible via --seed.
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np

MAGIC = 0x52455F435352304B  # 'RE_CSR0K'


def gen_rmat_edges(scale: int, edge_factor: int,
                   a: float, b: float, c: float, d: float,
                   seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    N = 1 << scale
    E = edge_factor * N
    # Graph500 RMAT: at each level, pick a quadrant by (a, b, c, d).
    # Vectorise across all E edges and all `scale` levels.
    src = np.zeros(E, dtype=np.int64)
    dst = np.zeros(E, dtype=np.int64)
    ab = a + b
    abc = a + b + c
    chunk = 1 << 22  # 4M edges/chunk to keep memory bounded
    pos = 0
    while pos < E:
        m = min(chunk, E - pos)
        s = np.zeros(m, dtype=np.int64)
        t = np.zeros(m, dtype=np.int64)
        for level in range(scale):
            r = rng.random(m)
            top = r < ab            # top half (quadrants a, b)
            left = np.where(r < a, True,
                            np.where(r < ab, False,
                                     np.where(r < abc, True, False)))
            bit = 1 << (scale - 1 - level)
            s += np.where(top, 0, bit)
            t += np.where(left, 0, bit)
        src[pos:pos + m] = s
        dst[pos:pos + m] = t
        pos += m
    return np.stack([src, dst], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, default=22)
    ap.add_argument("--edge-factor", type=int, default=16)
    ap.add_argument("--a", type=float, default=0.57)
    ap.add_argument("--b", type=float, default=0.19)
    ap.add_argument("--c", type=float, default=0.19)
    ap.add_argument("--d", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    abcd = args.a + args.b + args.c + args.d
    if abs(abcd - 1.0) > 1e-6:
        print(f"a+b+c+d must sum to 1.0, got {abcd}", file=sys.stderr)
        sys.exit(1)

    N_full = 1 << args.scale
    print(f"[gen_rmat] scale={args.scale} N_full={N_full} "
          f"E_target={args.edge_factor * N_full}", file=sys.stderr)
    edges = gen_rmat_edges(args.scale, args.edge_factor,
                           args.a, args.b, args.c, args.d, args.seed)

    # Dedup + drop self-loops
    edges = edges[edges[:, 0] != edges[:, 1]]
    edges = np.unique(edges, axis=0)

    # Densify ids (R-MAT can leave isolated vertices; drop them so N matches actual)
    unique_ids, inv = np.unique(edges.reshape(-1), return_inverse=True)
    inv = inv.reshape(edges.shape).astype(np.int32)
    N = len(unique_ids)
    E = len(inv)
    print(f"[gen_rmat] after dedup: N={N} E={E}", file=sys.stderr)

    src = inv[:, 0]
    dst = inv[:, 1]
    order = np.lexsort((dst, src))
    src = src[order]
    dst = dst[order]

    row_ptr = np.zeros(N + 1, dtype=np.int32)
    counts = np.bincount(src, minlength=N).astype(np.int32)
    row_ptr[1:] = np.cumsum(counts)
    col_idx = dst.astype(np.int32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as f:
        f.write(struct.pack("<QQQ", MAGIC, N, E))
        f.write(row_ptr.tobytes())
        f.write(col_idx.tobytes())
    print(f"[gen_rmat] wrote {args.output} "
          f"({args.output.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)


if __name__ == "__main__":
    main()
