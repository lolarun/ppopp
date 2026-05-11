#!/usr/bin/env python3
"""Compare two PageRank output binaries (FP32 or FP64) and emit drift metrics.

Inputs:  two `<prefix>.bin` files plus their sidecar `<prefix>.json` produced
         by src/pagerank/pagerank.hip. The `.json` records `"precision"` and
         that drives the dtype used to load the `.bin`.

Metrics (per the RE0 spec in docs/design/03_experimental_design.md):
    byte_diff_fraction : count(a_bytes != b_bytes) at element granularity / N
    max_Linf           : max |a[i] - b[i]|
    mean_diff          : mean |a[i] - b[i]|
    L2_norm            : ||a - b||_2
    rank_top100_jaccard: Jaccard overlap of top-100 indices
    rank_top100_kendall: Kendall tau of top-100 ordering (only on overlap)

Decision logic from RE0:
    byte_diff_fraction > 0.01  -> GO (drift confirmed)
    0.001 <= ... <= 0.01       -> GO with caveats
    < 0.001                    -> STOP (hypothesis falsified)

Fails fast if A and B were generated at different precisions — drift at
fp32 vs fp64 is a separate (also interesting) experiment, not the
within-precision drift this script reports.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

_DTYPE_BY_PRECISION = {
    "fp32": np.float32,
    "fp64": np.float64,
}
_BYTE_VIEW_BY_PRECISION = {
    "fp32": np.uint32,
    "fp64": np.uint64,
}


def load_pair(prefix: Path):
    bin_path = prefix.with_suffix(".bin")
    json_path = prefix.with_suffix(".json")
    meta = json.loads(json_path.read_text())
    prec = meta.get("precision", "fp32")
    if prec not in _DTYPE_BY_PRECISION:
        raise SystemExit(f"unknown precision {prec!r} in {json_path}")
    arr = np.fromfile(bin_path, dtype=_DTYPE_BY_PRECISION[prec])
    return arr, meta, prec


def kendall_tau_top(a_idx, b_idx):
    common = list(set(a_idx) & set(b_idx))
    if len(common) < 2:
        return float("nan")
    rank_a = {v: i for i, v in enumerate(a_idx)}
    rank_b = {v: i for i, v in enumerate(b_idx)}
    n = len(common)
    concord = discord = 0
    for i in range(n):
        for j in range(i + 1, n):
            x, y = common[i], common[j]
            sa = np.sign(rank_a[x] - rank_a[y])
            sb = np.sign(rank_b[x] - rank_b[y])
            if sa * sb > 0: concord += 1
            elif sa * sb < 0: discord += 1
    pairs = n * (n - 1) // 2
    return (concord - discord) / pairs if pairs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, type=Path,
                    help="Output prefix A (without .bin/.json)")
    ap.add_argument("--b", required=True, type=Path,
                    help="Output prefix B (without .bin/.json)")
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--out", type=Path, default=None,
                    help="Write JSON result here; else print to stdout")
    args = ap.parse_args()

    a, meta_a, prec_a = load_pair(args.a)
    b, meta_b, prec_b = load_pair(args.b)
    if prec_a != prec_b:
        print(f"ERROR: precision mismatch a={prec_a} b={prec_b} — "
              f"this script compares within-precision drift only",
              file=sys.stderr)
        sys.exit(2)
    if a.size != b.size:
        print(f"ERROR: vector size mismatch a={a.size} b={b.size}", file=sys.stderr)
        sys.exit(2)
    N = a.size

    byte_view = _BYTE_VIEW_BY_PRECISION[prec_a]
    diff_mask = np.frombuffer(a.tobytes(), dtype=byte_view) != \
                np.frombuffer(b.tobytes(), dtype=byte_view)
    byte_diff_fraction = float(diff_mask.mean())

    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    max_linf = float(diff.max())
    mean_diff = float(diff.mean())
    l2 = float(np.linalg.norm(a.astype(np.float64) - b.astype(np.float64)))

    k = min(args.top_k, N)
    top_a = np.argpartition(-a, k - 1)[:k]
    top_a = top_a[np.argsort(-a[top_a])].tolist()
    top_b = np.argpartition(-b, k - 1)[:k]
    top_b = top_b[np.argsort(-b[top_b])].tolist()
    jaccard = len(set(top_a) & set(top_b)) / len(set(top_a) | set(top_b))
    kendall = kendall_tau_top(top_a, top_b)

    if byte_diff_fraction > 0.01:
        decision = "GO"
        decision_reason = "byte_diff_fraction > 1% -> drift confirmed"
    elif byte_diff_fraction >= 0.001:
        decision = "GO_WITH_CAVEATS"
        decision_reason = "0.1%-1% byte-different -> framing needs careful quantification"
    else:
        decision = "STOP"
        decision_reason = "byte_diff_fraction < 0.1% -> drift hypothesis falsified"

    result = {
        "a": {"prefix": str(args.a), "vendor": meta_a.get("vendor"),
              "device": meta_a.get("device"), "crc32": meta_a.get("output_crc32"),
              "iters_run": meta_a.get("iters_run")},
        "b": {"prefix": str(args.b), "vendor": meta_b.get("vendor"),
              "device": meta_b.get("device"), "crc32": meta_b.get("output_crc32"),
              "iters_run": meta_b.get("iters_run")},
        "dataset": meta_a.get("dataset"),
        "precision": prec_a,
        "N": int(N),
        "byte_diff_fraction": byte_diff_fraction,
        "max_Linf": max_linf,
        "mean_diff": mean_diff,
        "L2_norm": l2,
        "rank_top100_jaccard": jaccard,
        "rank_top100_kendall": kendall,
        "decision": decision,
        "decision_reason": decision_reason,
    }
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
