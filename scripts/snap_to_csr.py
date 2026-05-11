#!/usr/bin/env python3
"""Convert a SNAP edge-list (web-Google.txt, soc-LiveJournal1.txt, ...) to
the binary CSR format consumed by src/pagerank.hip.

Format spec (matches include/csr.hpp):
    uint64 magic = 0x52455F435352304B   ('RE_CSR0K' little-endian)
    uint64 N
    uint64 E
    int32  row_ptr[N+1]
    int32  col_idx[E]

CSR rows index OUTGOING edges. Vertex IDs are densified to [0, N).
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np

MAGIC = 0x52455F435352304B  # 'RE_CSR0K'


def read_edges(path: Path):
    """Yield (u, v) pairs from a SNAP-style edge list, skipping comments."""
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            yield int(parts[0]), int(parts[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path,
                    help="SNAP edge list (text)")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output binary CSR")
    ap.add_argument("--symmetric", action="store_true",
                    help="Add reverse edges (treat as undirected)")
    ap.add_argument("--no-self-loops", action="store_true", default=True,
                    help="Drop self loops (default on)")
    ap.add_argument("--keep-self-loops", dest="no_self_loops",
                    action="store_false")
    args = ap.parse_args()

    print(f"[snap_to_csr] reading {args.input}", file=sys.stderr)
    edges = []
    for u, v in read_edges(args.input):
        if args.no_self_loops and u == v:
            continue
        edges.append((u, v))
        if args.symmetric and u != v:
            edges.append((v, u))
    if not edges:
        print("no edges read", file=sys.stderr)
        sys.exit(1)
    edges = np.asarray(edges, dtype=np.int64)
    print(f"[snap_to_csr] read {len(edges)} edges", file=sys.stderr)

    # Densify ids — sort unique, map to [0, N)
    unique_ids, inv = np.unique(edges.reshape(-1), return_inverse=True)
    inv = inv.reshape(edges.shape).astype(np.int32)
    N = len(unique_ids)
    E = len(inv)
    print(f"[snap_to_csr] N={N} E={E}", file=sys.stderr)
    if N > np.iinfo(np.int32).max:
        print("N exceeds int32 range; CSR format needs int64", file=sys.stderr)
        sys.exit(1)

    # Sort by source for CSR row_ptr; secondary sort by dest for stable layout
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
    print(f"[snap_to_csr] wrote {args.output} "
          f"({args.output.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)


if __name__ == "__main__":
    main()
