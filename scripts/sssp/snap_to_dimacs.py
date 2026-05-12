#!/usr/bin/env python3
"""
Convert SNAP edge-list format to DIMACS .gr with random FP weights.

SNAP format:
  # comment lines
  <src> <dst>   (0-indexed or 1-indexed, tab/space separated)

Output DIMACS .gr:
  p sp <n_v> <n_e>
  a <u+1> <v+1> <w>   (1-indexed, weight as decimal float)

Weight distribution: uniform float in [0.001, 1.000] stored as 6-decimal float.
dimacs_loader reads weights as double, so both integer (road) and float (SNAP) work.

Usage:
  python3 snap_to_dimacs.py <input.txt> <output.gr> [--seed N]
"""

import sys
import random
import argparse
import tempfile
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Pass 1: build node remap and count edges (never store edge list)
    nodes = set()
    n_e = 0
    with open(args.input) as f:
        for line in f:
            if not line or line[0] == '#': continue
            parts = line.split()
            if len(parts) < 2: continue
            u, v = int(parts[0]), int(parts[1])
            if u == v: continue
            nodes.add(u); nodes.add(v)
            n_e += 1

    node_list = sorted(nodes)
    remap = {n: i + 1 for i, n in enumerate(node_list)}  # 1-indexed
    n_v = len(node_list)
    del node_list  # free memory

    # Pass 2: stream edges directly to a temp file (avoids holding all in RAM)
    tmp_path = args.output + ".tmp"
    with open(args.input) as fin, open(tmp_path, 'w') as ftmp:
        for line in fin:
            if not line or line[0] == '#': continue
            parts = line.split()
            if len(parts) < 2: continue
            u, v = int(parts[0]), int(parts[1])
            if u == v: continue
            w = rng.uniform(0.001, 1.000)
            ftmp.write(f"a {remap[u]} {remap[v]} {w:.6f}\n")

    # Write final file: header + temp body
    with open(args.output, 'w') as fout:
        fout.write(f"c Converted from SNAP format. Weights: uniform FP [0.001,1.0]\n")
        fout.write(f"c seed={args.seed}\n")
        fout.write(f"p sp {n_v} {n_e}\n")
        with open(tmp_path) as ftmp:
            for chunk in iter(lambda: ftmp.read(1 << 20), ''):
                fout.write(chunk)
    os.remove(tmp_path)

    print(f"Written {n_v} vertices, {n_e} edges → {args.output}")

if __name__ == "__main__":
    main()
