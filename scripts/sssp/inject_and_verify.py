#!/usr/bin/env python3
"""
inject_and_verify.py — E4 error-injection coverage harness.

Reads a clean cert binary (.d.bin + .pi.bin), injects errors of various
kinds × magnitudes × seeds, runs run_sssp --verify-only on each
corrupted cert, and records:
  - verifier verdict (SAT / specific UNSAT kind)
  - golden-output detection (would simple d-array comparison catch it?)
  - detection rate per (error_kind, magnitude)

The contrast between verifier-detected and golden-output-detected gives
the §VII paper claim: "verifier catches errors that golden-output misses".

Usage:
    python3 scripts/inject_and_verify.py \\
      --graph data/cache/ny_road.gr \\
      --dataset-name ny_road \\
      --clean-prefix results/certs/ny_road_fp32 \\
      --precision fp32 \\
      --binary build/run_sssp \\
      --output results/e4_ny_road_fp32.jsonl \\
      --n-seeds 30
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

INVALID_VID = np.uint32(0xFFFFFFFF)

# Error kinds and magnitudes (n_errors per cert) injected for each.
ERROR_KINDS = {
    "DISTANCE_PERTURB":   [1, 5, 25],
    "PREDECESSOR_RANDOM": [1, 5, 25],
    "INCONSISTENT":       [1, 5, 25],
    "MISSED_UNREACHABLE": [1, 5],
    "CYCLE":              [1],
}


def load_clean(prefix, precision):
    dtype = np.float32 if precision == "fp32" else np.float64
    d = np.fromfile(prefix + ".d.bin", dtype=dtype)
    pi = np.fromfile(prefix + ".pi.bin", dtype=np.uint32)
    return d, pi


def save_cert(prefix, d, pi):
    d.tofile(prefix + ".d.bin")
    pi.tofile(prefix + ".pi.bin")


def inject_distance_perturb(d, pi, n_errors, rng, sentinel):
    d2 = d.copy(); pi2 = pi.copy()
    reachable = np.where(d2 != sentinel)[0]
    reachable = reachable[reachable != 0]
    if len(reachable) == 0: return d2, pi2
    pick = rng.choice(reachable, size=min(n_errors, len(reachable)), replace=False)
    factors = rng.uniform(-0.10, 0.10, size=len(pick))
    d2[pick] = d2[pick] * (1.0 + factors)
    return d2, pi2


def inject_predecessor_random(d, pi, n_errors, rng, sentinel):
    d2 = d.copy(); pi2 = pi.copy()
    reachable = np.where(d2 != sentinel)[0]
    reachable = reachable[reachable != 0]
    if len(reachable) == 0: return d2, pi2
    pick = rng.choice(reachable, size=min(n_errors, len(reachable)), replace=False)
    new_preds = rng.integers(0, len(d2), size=len(pick), dtype=np.uint32)
    pi2[pick] = new_preds
    return d2, pi2


def inject_inconsistent(d, pi, n_errors, rng, sentinel):
    """Both d and pi are corrupted at the same vertices (chaotic corruption).
    Distinct from DISTANCE_PERTURB (only d) and PREDECESSOR_RANDOM (only pi).
    Multiplicative d perturbation guarantees FP32 visibility on large d values."""
    d2 = d.copy(); pi2 = pi.copy()
    reachable = np.where(d2 != sentinel)[0]
    reachable = reachable[reachable != 0]
    if len(reachable) == 0: return d2, pi2
    pick = rng.choice(reachable, size=min(n_errors, len(reachable)), replace=False)
    factors = rng.uniform(-0.05, 0.05, size=len(pick))   # ±5% multiplicative
    d2[pick] = d2[pick] * (1.0 + factors)
    pi2[pick] = rng.integers(0, len(d2), size=len(pick), dtype=np.uint32)
    return d2, pi2


def inject_missed_unreachable(d, pi, n_errors, rng, sentinel):
    d2 = d.copy(); pi2 = pi.copy()
    unreachable = np.where(d2 == sentinel)[0]
    if len(unreachable) == 0: return d2, pi2
    pick = rng.choice(unreachable, size=min(n_errors, len(unreachable)), replace=False)
    finite_d = d2[d2 != sentinel]
    plausible = float(np.median(finite_d)) if len(finite_d) > 0 else 1.0
    d2[pick] = plausible
    return d2, pi2


def inject_cycle(d, pi, n_errors, rng, sentinel):
    d2 = d.copy(); pi2 = pi.copy()
    reachable = np.where((d2 != sentinel) & (pi2 != INVALID_VID))[0]
    reachable = reachable[reachable != 0]
    if len(reachable) == 0: return d2, pi2
    v = int(rng.choice(reachable))
    u = int(pi2[v])
    if u >= len(pi2): return d2, pi2
    pi2[u] = np.uint32(v)
    return d2, pi2


INJECTORS = {
    "DISTANCE_PERTURB":   inject_distance_perturb,
    "PREDECESSOR_RANDOM": inject_predecessor_random,
    "INCONSISTENT":       inject_inconsistent,
    "MISSED_UNREACHABLE": inject_missed_unreachable,
    "CYCLE":              inject_cycle,
}


def run_verifier_batch(binary, graph_path, dataset_name, cert_prefixes,
                       precision, output_jsonl, list_path):
    """Batch verify: write cert prefixes to a file, run run_sssp once.
    Returns dict {prefix: verdict} parsed from the output JSONL."""
    with open(list_path, "w") as f:
        for p in cert_prefixes:
            f.write(p + "\n")
    cmd = [
        binary,
        f"--dataset={graph_path}",
        f"--dataset-name={dataset_name}",
        f"--precision={precision}",
        "--source=0",
        f"--cert-prefix-list={list_path}",
        f"--output={output_jsonl}",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        sys.stderr.write(f"run_sssp failed: {r.stderr[-2000:]}\n")
    # Parse JSONL: each line has cert prefix in config.cert_prefix-style field?
    # Our log_writer writes the prefix into config.verify_cert_prefix (NOT in JSONL by default).
    # Workaround: parse output lines in order — they correspond to cert_prefixes order.
    results = {}
    if os.path.exists(output_jsonl):
        with open(output_jsonl) as f:
            lines = [json.loads(l) for l in f]
        # Map by file order (run_sssp processes prefixes in input order)
        for i, line_obj in enumerate(lines):
            if i < len(cert_prefixes):
                results[cert_prefixes[i]] = line_obj.get("verifier_verdict")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--clean-prefix", required=True)
    ap.add_argument("--precision", choices=["fp32", "fp64"], default="fp32")
    ap.add_argument("--binary", default="build/run_sssp")
    ap.add_argument("--output", required=True)
    ap.add_argument("--n-seeds", type=int, default=30)
    ap.add_argument("--kinds", default="all")
    args = ap.parse_args()

    sentinel = np.float32("inf") if args.precision == "fp32" else np.float64("inf")
    clean_d, clean_pi = load_clean(args.clean_prefix, args.precision)
    print(f"Loaded clean cert: {len(clean_d)} vertices, "
          f"{int((clean_d == sentinel).sum())} unreachable")

    kinds = list(ERROR_KINDS.keys()) if args.kinds == "all" else args.kinds.split(",")

    out_lines = []
    tmp_dir = tempfile.mkdtemp(prefix="e4_inject_")
    print(f"Working dir: {tmp_dir}")

    # Phase 1: inject all errors, save corrupted certs, record metadata
    print("Phase 1: injecting + saving corrupted certs...")
    cert_meta = []   # (prefix, kind, n_errors, seed, golden_detects)
    for kind in kinds:
        injector = INJECTORS[kind]
        for n_errors in ERROR_KINDS[kind]:
            for seed in range(args.n_seeds):
                rng = np.random.default_rng(seed)
                d_inj, pi_inj = injector(clean_d, clean_pi, n_errors, rng, sentinel)
                cprefix = os.path.join(tmp_dir, f"{kind}_n{n_errors}_s{seed}")
                save_cert(cprefix, d_inj, pi_inj)
                gold = not np.array_equal(clean_d, d_inj)
                cert_meta.append((cprefix, kind, n_errors, seed, gold))
    print(f"  injected {len(cert_meta)} corrupted certs")

    # Phase 2: batch verify (graph loaded once)
    print(f"Phase 2: batch verifying via {args.binary} --cert-prefix-list ...")
    list_path = os.path.join(tmp_dir, "cert_list.txt")
    output_jsonl = os.path.join(tmp_dir, "e4.jsonl")
    prefixes = [m[0] for m in cert_meta]
    results = run_verifier_batch(
        args.binary, args.graph, args.dataset_name,
        prefixes, args.precision, output_jsonl, list_path,
    )

    # Phase 3: aggregate
    summary = {}
    for cprefix, kind, n_errors, seed, gold in cert_meta:
        key = (kind, n_errors)
        if key not in summary:
            summary[key] = dict(total=0, verifier_unsat=0, golden_detects=0, verifier_only=0)
        verdict = results.get(cprefix)
        ver = (verdict is not None and verdict != "SAT")
        summary[key]["total"] += 1
        summary[key]["verifier_unsat"] += int(ver)
        summary[key]["golden_detects"] += int(gold)
        if ver and not gold:
            summary[key]["verifier_only"] += 1
        out_lines.append({
            "dataset": args.dataset_name,
            "precision": args.precision,
            "error_kind": kind,
            "n_errors": n_errors,
            "seed": seed,
            "verdict": verdict,
            "golden_detects": bool(gold),
        })

    with open(args.output, "w") as f:
        for r in out_lines:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(out_lines)} rows -> {args.output}\n")

    print(f"{'kind':<22}{'mag':>6}{'total':>8}{'ver_UNSAT':>14}{'gold_diff':>14}{'ver_only':>10}")
    for (kind, mag), d in sorted(summary.items()):
        v_pct = 100.0 * d["verifier_unsat"] / d["total"] if d["total"] else 0
        g_pct = 100.0 * d["golden_detects"] / d["total"] if d["total"] else 0
        print(f"{kind:<22}{mag:>6}{d['total']:>8}"
              f"  {d['verifier_unsat']:>3} ({v_pct:>4.0f}%)"
              f"  {d['golden_detects']:>3} ({g_pct:>4.0f}%)"
              f"{d['verifier_only']:>10}")

    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
