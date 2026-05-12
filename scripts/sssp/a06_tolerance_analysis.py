#!/usr/bin/env python3
"""A0.6 — verify the v2/§9.3 explanation that the missed E4 ny_road FP32
DISTANCE_PERTURB n=1 cases are sub-tolerance injections.

Replays the injection RNG for each seed reported in
results/e4/e4_ny_road_fp32{,_n100}.jsonl, computes the actual perturbation
factor delta, and checks whether |d[picked] * delta| was below the
verifier tolerance 4096 * eps_fp32 * max(|d|, 1).

If the missed-seed cases have |delta * d| < tolerance, §9.3's claim
("sub-eps DISTANCE_PERTURB injection — verifier soundly accepted")
holds and no verifier fix is needed.

Run from project root:
    python scripts/a06_tolerance_analysis.py
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

EPS_FP32 = float(np.finfo(np.float32).eps)
TOL_K = 4096.0  # cpu_verifier.cpp::fp_ne(): tolerance = K * eps * max(|d|, 1)


def load_clean_fp32(prefix: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.fromfile(prefix.with_suffix(".d.bin"), dtype=np.float32)
    pi = np.fromfile(prefix.with_suffix(".pi.bin"), dtype=np.uint32)
    return d, pi


def replay_distance_perturb(
    d: np.ndarray, pi: np.ndarray, n_errors: int, seed: int, sentinel: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror inject_distance_perturb deterministically."""
    rng = np.random.default_rng(seed)
    reachable = np.where(d != sentinel)[0]
    reachable = reachable[reachable != 0]
    if len(reachable) == 0:
        return np.array([], dtype=np.uint32), np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    pick = rng.choice(reachable, size=min(n_errors, len(reachable)), replace=False)
    factors = rng.uniform(-0.10, 0.10, size=len(pick))
    d_orig = d[pick].astype(np.float64)
    d_inj = (d_orig * (1.0 + factors)).astype(np.float64)
    return pick, factors, d_inj - d_orig  # picked indices, delta factors, injection magnitude


def analyze_jsonl(path: Path, clean_d: np.ndarray, clean_pi: np.ndarray) -> None:
    print(f"\n=== {path.relative_to(ROOT)} ===")
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    sentinel = float(np.float32("inf"))

    # Filter DISTANCE_PERTURB n=1 cases
    dp1 = [r for r in rows if r["error_kind"] == "DISTANCE_PERTURB" and r["n_errors"] == 1]
    print(f"  DISTANCE_PERTURB n=1: {len(dp1)} cases")

    sat_seeds = [r["seed"] for r in dp1 if r["verdict"] == "SAT"]
    unsat_seeds = [r["seed"] for r in dp1 if r["verdict"] != "SAT"]
    print(f"    verifier SAT (missed): {len(sat_seeds)} -> seeds {sat_seeds}")
    print(f"    verifier UNSAT (caught): {len(unsat_seeds)}")

    # Replay each missed seed and compute the actual injection magnitude vs tolerance
    if not sat_seeds:
        print("  no missed seeds in this file; nothing to analyze.")
        return

    print(f"\n  Replay analysis (k=4096, eps_fp32 = {EPS_FP32:.6e}):")
    print(f"  {'seed':>6}  {'picked vid':>12}  {'d[v]':>14}  {'delta':>14}  "
          f"{'|inj|':>14}  {'tolerance':>14}  {'sub-tol?':>10}")
    for seed in sat_seeds:
        pick, factors, magnitudes = replay_distance_perturb(
            clean_d, clean_pi, n_errors=1, seed=seed, sentinel=sentinel
        )
        if len(pick) == 0:
            print(f"  {seed:>6}  (no reachable picked)")
            continue
        v = int(pick[0])
        delta = float(factors[0])
        d_v = float(clean_d[v])
        inj_mag = abs(float(magnitudes[0]))
        tol = TOL_K * EPS_FP32 * max(abs(d_v), 1.0)
        sub_tol = inj_mag <= tol
        print(f"  {seed:>6}  {v:>12d}  {d_v:>14.6e}  {delta:>14.6e}  "
              f"{inj_mag:>14.6e}  {tol:>14.6e}  {'YES' if sub_tol else 'no':>10}")

    # Sanity-check the caught cases: do they all have inj_mag > tol?
    print(f"\n  Sanity check on UNSAT (caught) cases — should ALL be > tolerance:")
    above_count = 0
    below_count = 0
    for seed in unsat_seeds:
        pick, factors, magnitudes = replay_distance_perturb(
            clean_d, clean_pi, n_errors=1, seed=seed, sentinel=sentinel
        )
        if len(pick) == 0:
            continue
        v = int(pick[0])
        d_v = float(clean_d[v])
        inj_mag = abs(float(magnitudes[0]))
        tol = TOL_K * EPS_FP32 * max(abs(d_v), 1.0)
        if inj_mag > tol:
            above_count += 1
        else:
            below_count += 1
    total = above_count + below_count
    print(f"    of {total} caught UNSAT seeds: {above_count} above tolerance, {below_count} below tolerance")
    if below_count > 0:
        print(f"    NOTE: {below_count} UNSAT cases are below tolerance — would mean verifier "
              f"is detecting them by mechanism other than the |d - (d_pi+w)| inequality "
              f"(e.g., R3 inequality on outgoing edge). Worth investigating separately.")


def main() -> int:
    cert_prefix = RESULTS / "certs" / "ny_road_fp32"
    clean_d, clean_pi = load_clean_fp32(cert_prefix)
    n_v = len(clean_d)
    n_unreach = int((clean_d == np.float32("inf")).sum())
    print(f"Loaded ny_road FP32 cert: {n_v} vertices, {n_unreach} unreachable")
    print(f"  d range: [{clean_d.min():.6e}, {clean_d.max():.6e}] (excluding sentinel)")

    for fname in ("e4_ny_road_fp32.jsonl", "e4_ny_road_fp32_n100.jsonl"):
        path = RESULTS / "e4" / fname
        if not path.exists():
            print(f"\n!! missing: {path}")
            continue
        analyze_jsonl(path, clean_d, clean_pi)

    return 0


if __name__ == "__main__":
    sys.exit(main())
