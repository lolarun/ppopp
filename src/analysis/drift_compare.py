#!/usr/bin/env python3
"""
drift_compare.py — Compare SSSP certificates from two GPU platforms.

Reads JSONL run logs (one line per run), extracts d[] and pi[] arrays,
and computes byte-level and magnitude drift statistics.

Usage:
    python3 -m analysis.drift_compare \
        --jsonl-a results/run_a100_livejournal_fp32.jsonl \
        --jsonl-b results/run_mi250_livejournal_fp32.jsonl \
        --output  results/drift_livejournal_fp32.csv

The JSONL lines must include "cert_d" (list of floats) and "cert_pi"
(list of ints) fields.  These are only present when run with --emit-cert=1.
For large graphs, use --cert-dir to read cert arrays from separate binary
files referenced in the JSONL.
"""

import argparse
import json
import math
import pathlib
import struct
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class DriftReport:
    dataset:        str
    gpu_a:          str
    gpu_b:          str
    precision:      str
    seed:           int
    n_vertices:     int

    # Distance drift
    n_d_diff:       int           # bytes differ
    d_diff_p50:     float
    d_diff_p99:     float
    d_diff_max:     float
    d_rel_p50:      float         # relative: |Δd| / d_a
    d_rel_p99:      float
    d_rel_max:      float

    # Predecessor drift
    n_pi_diff:      int

    # Verifier outcome
    verdict_a:      str
    verdict_b:      str
    both_sat:       bool

    # Metadata
    algo:           str
    rep:            int


# ── JSONL helpers ─────────────────────────────────────────────────────────────

def _load_run(path: pathlib.Path) -> dict:
    """Return the first valid JSONL line from path."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    raise ValueError(f"No valid JSON line in {path}")


def _load_cert_arrays(run: dict, cert_dir: Optional[pathlib.Path]):
    """
    Return (d_array, pi_array) as numpy arrays.
    Priority: inline 'cert_d'/'cert_pi' → external binary file.
    """
    if "cert_d" in run and run["cert_d"] is not None:
        d  = np.array(run["cert_d"],  dtype=np.float64)
        pi = np.array(run["cert_pi"], dtype=np.int64) if "cert_pi" in run else None
        return d, pi

    if cert_dir is not None:
        run_id = run.get("run_id", "")
        d_path  = cert_dir / f"{run_id}_d.bin"
        pi_path = cert_dir / f"{run_id}_pi.bin"
        if d_path.exists():
            d  = np.fromfile(d_path,  dtype=np.float32)
            pi = np.fromfile(pi_path, dtype=np.uint32) if pi_path.exists() else None
            return d.astype(np.float64), pi
    raise ValueError("No cert data in run log and --cert-dir not provided")


# ── Core comparison ───────────────────────────────────────────────────────────

def compare_runs(run_a: dict, run_b: dict,
                 cert_dir: Optional[pathlib.Path] = None) -> DriftReport:
    d_a, pi_a = _load_cert_arrays(run_a, cert_dir)
    d_b, pi_b = _load_cert_arrays(run_b, cert_dir)

    n_v = min(len(d_a), len(d_b))
    d_a = d_a[:n_v]
    d_b = d_b[:n_v]

    INF = 1e30

    # Mask to finite (reachable by both)
    finite = (d_a < INF) & (d_b < INF)
    diff_d = np.abs(d_a - d_b)

    n_d_diff = int(np.sum(diff_d > 0))

    # Magnitude stats (only finite vertices)
    mags = diff_d[finite]
    if len(mags) > 0:
        d_diff_p50 = float(np.percentile(mags, 50))
        d_diff_p99 = float(np.percentile(mags, 99))
        d_diff_max = float(np.max(mags))
        denom = np.maximum(d_a[finite], 1e-12)
        rel   = mags / denom
        d_rel_p50 = float(np.percentile(rel, 50))
        d_rel_p99 = float(np.percentile(rel, 99))
        d_rel_max = float(np.max(rel))
    else:
        d_diff_p50 = d_diff_p99 = d_diff_max = 0.0
        d_rel_p50 = d_rel_p99 = d_rel_max = 0.0

    n_pi_diff = 0
    if pi_a is not None and pi_b is not None:
        pi_a = pi_a[:n_v]
        pi_b = pi_b[:n_v]
        n_pi_diff = int(np.sum(pi_a != pi_b))

    verdict_a = run_a.get("verifier_verdict", "unknown")
    verdict_b = run_b.get("verifier_verdict", "unknown")
    both_sat  = (verdict_a == "SAT" and verdict_b == "SAT")

    cfg_a = run_a.get("config", {})
    cfg_b = run_b.get("config", {})

    return DriftReport(
        dataset    = run_a.get("dataset", {}).get("name", "unknown"),
        gpu_a      = run_a.get("hardware", {}).get("gpu", "gpu_a"),
        gpu_b      = run_b.get("hardware", {}).get("gpu", "gpu_b"),
        precision  = cfg_a.get("precision", "fp32"),
        seed       = cfg_a.get("seed", 42),
        n_vertices = n_v,
        n_d_diff   = n_d_diff,
        d_diff_p50 = d_diff_p50,
        d_diff_p99 = d_diff_p99,
        d_diff_max = d_diff_max,
        d_rel_p50  = d_rel_p50,
        d_rel_p99  = d_rel_p99,
        d_rel_max  = d_rel_max,
        n_pi_diff  = n_pi_diff,
        verdict_a  = verdict_a,
        verdict_b  = verdict_b,
        both_sat   = both_sat,
        algo       = cfg_a.get("algo", "unknown"),
        rep        = run_a.get("rep", 0),
    )


# ── CSV output ────────────────────────────────────────────────────────────────

CSV_HEADER = (
    "dataset,gpu_a,gpu_b,precision,seed,n_vertices,"
    "n_d_diff,d_diff_p50,d_diff_p99,d_diff_max,"
    "d_rel_p50,d_rel_p99,d_rel_max,"
    "n_pi_diff,verdict_a,verdict_b,both_sat,algo,rep\n"
)

def report_to_csv_row(r: DriftReport) -> str:
    return (
        f"{r.dataset},{r.gpu_a},{r.gpu_b},{r.precision},{r.seed},{r.n_vertices},"
        f"{r.n_d_diff},{r.d_diff_p50:.6e},{r.d_diff_p99:.6e},{r.d_diff_max:.6e},"
        f"{r.d_rel_p50:.6e},{r.d_rel_p99:.6e},{r.d_rel_max:.6e},"
        f"{r.n_pi_diff},{r.verdict_a},{r.verdict_b},{r.both_sat},"
        f"{r.algo},{r.rep}\n"
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Compare SSSP certificates from two platforms")
    ap.add_argument("--jsonl-a",  required=True,  help="JSONL run log from platform A")
    ap.add_argument("--jsonl-b",  required=True,  help="JSONL run log from platform B")
    ap.add_argument("--output",   required=True,  help="Output CSV path")
    ap.add_argument("--cert-dir", default=None,   help="Directory with cert binary files")
    ap.add_argument("--append",   action="store_true",
                    help="Append to output CSV instead of overwriting")
    args = ap.parse_args()

    cert_dir = pathlib.Path(args.cert_dir) if args.cert_dir else None

    run_a = _load_run(pathlib.Path(args.jsonl_a))
    run_b = _load_run(pathlib.Path(args.jsonl_b))

    report = compare_runs(run_a, run_b, cert_dir)

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    with open(out_path, mode) as f:
        if mode == "w":
            f.write(CSV_HEADER)
        f.write(report_to_csv_row(report))

    # Summary to stdout
    print(f"[drift_compare] {report.dataset}  {report.gpu_a} vs {report.gpu_b}")
    print(f"  d diff: {report.n_d_diff}/{report.n_vertices} vertices "
          f"({100*report.n_d_diff/max(1, report.n_vertices):.2f}%)")
    print(f"  d_diff p50={report.d_diff_p50:.3e}  p99={report.d_diff_p99:.3e}  "
          f"max={report.d_diff_max:.3e}")
    print(f"  pi diff: {report.n_pi_diff}")
    print(f"  verdicts: {report.verdict_a} / {report.verdict_b}  both_sat={report.both_sat}")


if __name__ == "__main__":
    main()
