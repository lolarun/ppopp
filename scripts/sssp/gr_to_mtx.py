#!/usr/bin/env python3
"""Convert DIMACS .gr file to MatrixMarket .mtx (real general).

Both formats are 1-indexed; the conversion is essentially a header swap.

Usage:
    python3 gr_to_mtx.py <input.gr> <output.mtx>
"""
import sys
from pathlib import Path


def convert(gr_path: Path, mtx_path: Path):
    n_v = 0
    n_e = 0
    edges = []
    with open(gr_path, "rb") as f:
        for raw in f:
            if not raw or raw[0:1] in (b"c", b"\n"):
                continue
            line = raw.decode("ascii", errors="replace").strip()
            if line.startswith("p "):
                parts = line.split()
                n_v = int(parts[2])
                n_e = int(parts[3])
            elif line.startswith("a "):
                _, us, vs, ws = line.split()
                edges.append(f"{us} {vs} {ws}")
    print(f"  parsed n_v={n_v} n_e={n_e} (declared)  actual_edges={len(edges)}")
    with open(mtx_path, "w") as f:
        f.write("%%MatrixMarket matrix coordinate real general\n")
        f.write(f"% Converted from {gr_path.name}\n")
        f.write(f"{n_v} {n_v} {len(edges)}\n")
        for e in edges:
            f.write(e + "\n")
    print(f"  wrote {mtx_path}  ({mtx_path.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python3 gr_to_mtx.py <in.gr> <out.mtx>", file=sys.stderr)
        sys.exit(2)
    convert(Path(sys.argv[1]), Path(sys.argv[2]))
