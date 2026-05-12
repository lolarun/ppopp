#!/usr/bin/env python3
"""Convert SNAP edge-list (tab/space separated, # comments) to DIMACS .gr format.

Adds random integer weights in [1, 1000] (seed=42 for reproducibility).

Usage:
  python3 snap_to_dimacs.py --input web-Google.txt --output web_google.gr
  python3 snap_to_dimacs.py --input soc-LiveJournal1.txt --output livejournal.gr
  python3 snap_to_dimacs.py --download livejournal --output livejournal.gr
"""
import argparse, gzip, os, random, struct, sys
from collections import defaultdict

SNAP_URLS = {
    "livejournal":  "https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz",
    "web_google":   "https://snap.stanford.edu/data/web-Google.txt.gz",
}

# DIMACS road graphs are already in .gr format — download directly
DIMACS_URLS = {
    "ny_road":  "http://www.diag.uniroma1.it/challenge9/data/USA-road-d/USA-road-d.NY.gr.gz",
    "usa_road": "http://www.diag.uniroma1.it/challenge9/data/USA-road-d/USA-road-d.USA.gr.gz",
}


def download(url, dest):
    """Download file using urllib (no deps)."""
    import urllib.request
    print(f"  downloading {url} ...", file=sys.stderr)
    urllib.request.urlretrieve(url, dest)
    print(f"  saved to {dest}", file=sys.stderr)


def read_snap_edges(path):
    """Read SNAP edge list, return list of (u, v) tuples (0-indexed)."""
    opener = gzip.open if path.endswith(".gz") else open
    edges = []
    max_node = 0
    with opener(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("%"):
                continue
            parts = line.split()
            u, v = int(parts[0]), int(parts[1])
            edges.append((u, v))
            max_node = max(max_node, u, v)
    return edges, max_node + 1


def write_dimacs(edges, n_v, outpath, seed=42):
    """Write DIMACS .gr format with FP weights uniform in [0.001, 1.0]."""
    rng = random.Random(seed)
    with open(outpath, "w") as f:
        f.write(f"c SNAP->DIMACS conversion (seed={seed}, weights uniform [0.001, 1.0])\n")
        f.write(f"p sp {n_v} {len(edges)}\n")
        for u, v in edges:
            w = rng.uniform(0.001, 1.0)
            f.write(f"a {u+1} {v+1} {w:.6f}\n")  # DIMACS is 1-indexed
    print(f"  wrote {outpath}: {n_v} vertices, {len(edges)} edges", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Path to SNAP edge-list file (.txt or .txt.gz)")
    parser.add_argument("--download", help="Dataset name to download: " + ", ".join(
        list(SNAP_URLS.keys()) + list(DIMACS_URLS.keys())))
    parser.add_argument("--output", required=True, help="Output .gr file path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.download:
        name = args.download
        if name in DIMACS_URLS:
            # Already DIMACS format, just download and decompress
            gz_path = args.output + ".gz"
            download(DIMACS_URLS[name], gz_path)
            print(f"  decompressing...", file=sys.stderr)
            with gzip.open(gz_path, "rb") as fin, open(args.output, "wb") as fout:
                fout.write(fin.read())
            os.remove(gz_path)
            print(f"  done: {args.output}", file=sys.stderr)
            return
        elif name in SNAP_URLS:
            gz_path = args.output + ".snap.gz"
            download(SNAP_URLS[name], gz_path)
            args.input = gz_path
        else:
            print(f"Unknown dataset: {name}. Options: {list(SNAP_URLS.keys()) + list(DIMACS_URLS.keys())}", file=sys.stderr)
            sys.exit(1)

    if not args.input:
        print("Error: --input or --download required", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {args.input}...", file=sys.stderr)
    edges, n_v = read_snap_edges(args.input)
    print(f"  {n_v} vertices, {len(edges)} edges", file=sys.stderr)
    write_dimacs(edges, n_v, args.output, seed=args.seed)


if __name__ == "__main__":
    main()
