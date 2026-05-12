#!/usr/bin/env python3
"""A1.14 + A1.15 — direct cert-binary comparison + unreachable-set agreement.

Closes two reviewer-attack vectors with offline analysis on the existing
cert binaries (no graph load, no GPU re-runs):

  A1.14  Replace the sec 3.3 CRC32-equality claim with bit-equality on the
         reachable subset.  Astronomically unlikely collision risk
         (~2^-32) is removed.

  A1.15  Confirm NVIDIA and AMD agree on which vertices are reachable
         (i.e., d == sentinel ↔ d == sentinel for the same v).  If they
         disagree, the sec 3.3 hashes are over different subsets and the
         comparison is hollow.

Run from project root:
    python scripts/offline_a1_14_15.py
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Map of "config" name -> precision tag.  17 configs comprise the cross-vendor
# claim in sec 3.3 / Table 1.  Canonical AMD cert files lack the _repN suffix.
CONFIGS = [
    ("livejournal_fp32", "fp32"),
    ("livejournal_fp64", "fp64"),
    ("ny_road_fp32",     "fp32"),
    ("ny_road_fp64",     "fp64"),
    ("ny_road_gaussian_fp32", "fp32"),
    ("ny_road_uniform_fp32",  "fp32"),
    ("rmat20_seed100_fp32",  "fp32"),
    ("rmat20_seed200_fp32",  "fp32"),
    ("rmat20_seed300_fp32",  "fp32"),
    ("rmat20_seed400_fp32",  "fp32"),
    ("rmat20_seed42_fp32",   "fp32"),
    ("usa_road_fp32",        "fp32"),
    ("usa_road_fp64",        "fp64"),
    ("usa_road_gaussian_fp32", "fp32"),
    ("usa_road_uniform_fp32",  "fp32"),
    ("web_google_fp32",      "fp32"),
    ("web_google_fp64",      "fp64"),
]


def load_cert(prefix: Path, prec: str):
    dt = np.float32 if prec == "fp32" else np.float64
    d = np.fromfile(prefix.with_suffix(".d.bin"), dtype=dt)
    pi = np.fromfile(prefix.with_suffix(".pi.bin"), dtype=np.uint32)
    return d, pi


def is_inf_mask(d: np.ndarray, prec: str) -> np.ndarray:
    """Identify sentinel (unreachable) entries by IEEE 754 +inf bit-pattern."""
    sentinel = np.float32(np.inf) if prec == "fp32" else np.float64(np.inf)
    return d == sentinel


def main() -> int:
    nv_cert_dir = ROOT / "results" / "certs"
    amd_cert_dir = ROOT / "results" / "amd" / "certs"

    print("=" * 78)
    print("A1.14 + A1.15 — direct cert byte comparison + unreachable-set audit")
    print(f"NVIDIA dir: {nv_cert_dir}")
    print(f"AMD dir   : {amd_cert_dir}")
    print("=" * 78)

    n_configs = 0
    n_d_bit_equal = 0
    n_d_reach_bit_equal = 0  # equal on reachable subset (the sec 3.3 claim refined)
    n_pi_bit_equal = 0
    n_reachable_set_match = 0
    failures: list[str] = []

    print(f"\n{'config':<28}  {'n_v':>10}  {'n_unreach':>10}  "
          f"{'unreach=':>9}  {'d-reach=':>9}  {'d-bit=':>7}  {'pi-bit=':>7}")
    print("-" * 95)

    for name, prec in CONFIGS:
        nv_pref = nv_cert_dir / name
        amd_pref = amd_cert_dir / name
        if not (nv_pref.with_suffix(".d.bin").exists()
                and amd_pref.with_suffix(".d.bin").exists()):
            print(f"{name:<28}  MISSING ({nv_pref.with_suffix('.d.bin').exists()=}, "
                  f"{amd_pref.with_suffix('.d.bin').exists()=})")
            failures.append(name)
            continue

        nv_d, nv_pi = load_cert(nv_pref, prec)
        amd_d, amd_pi = load_cert(amd_pref, prec)

        if nv_d.shape != amd_d.shape:
            print(f"{name:<28}  SHAPE MISMATCH: NV {nv_d.shape} AMD {amd_d.shape}")
            failures.append(name)
            continue

        n_v = len(nv_d)
        nv_inf = is_inf_mask(nv_d, prec)
        amd_inf = is_inf_mask(amd_d, prec)

        # A1.15: unreachable-set agreement -- both vendors mark the same vertices as inf.
        unreach_match = bool(np.array_equal(nv_inf, amd_inf))

        # A1.14: bit equality (the strong form of the sec 3.3 claim).
        # Use np.array_equal which compares bit-for-bit on float bit patterns.
        d_full_match = bool(np.array_equal(nv_d, amd_d))

        # A1.14b: bit equality on reachable subset only (mirroring sec 3.3's hash scope).
        # If unreach_match is True, the reachable subset is well-defined.
        if unreach_match:
            reach = ~nv_inf
            d_reach_match = bool(np.array_equal(nv_d[reach], amd_d[reach]))
        else:
            d_reach_match = False  # ill-defined if reachability sets differ

        pi_full_match = bool(np.array_equal(nv_pi, amd_pi))

        n_unreach = int(nv_inf.sum())
        ucheck = "Y" if unreach_match else "N"
        rcheck = "Y" if d_reach_match else "N"
        dcheck = "Y" if d_full_match else "N"
        pcheck = "Y" if pi_full_match else "N"
        print(f"{name:<28}  {n_v:>10d}  {n_unreach:>10d}  "
              f"{ucheck:>9}  {rcheck:>9}  {dcheck:>7}  {pcheck:>7}")

        n_configs += 1
        if unreach_match:
            n_reachable_set_match += 1
        if d_reach_match:
            n_d_reach_bit_equal += 1
        if d_full_match:
            n_d_bit_equal += 1
        if pi_full_match:
            n_pi_bit_equal += 1

    print("-" * 95)
    print(f"\n  Configs compared:                     {n_configs}/17 "
          f"({len(failures)} missing/error)")
    print(f"  A1.15  Reachability sets agree:       {n_reachable_set_match}/{n_configs}")
    print(f"  A1.14  d byte-equal on reachable:     {n_d_reach_bit_equal}/{n_configs}  "
          f"(this strengthens sec 3.3 from CRC32-equal to bit-equal)")
    print(f"  A1.14  d byte-equal full vector:      {n_d_bit_equal}/{n_configs}  "
          f"(includes sentinel encoding)")
    print(f"  Side-finding: pi byte-equal:            {n_pi_bit_equal}/{n_configs}  "
          f"(complement: pi differs on the rest -- sec 6.2 cross-vendor pi-divergence claim)")

    if failures:
        print(f"\n  Configs with missing/error files: {failures}")
        return 1

    if n_d_reach_bit_equal == n_configs and n_reachable_set_match == n_configs:
        print("\n  RESULT: sec 3.3 claim is BIT-EQUAL on reachable subset for all 17 configs.")
        print("          sec 3.3 caption can be promoted from 'CRC32 byte-exact' to 'bit-equal'")
        print("          on the reachable subset, with reachable-set membership identical")
        print("          across vendors'.  CRC32 collision footnote can be removed.")
        return 0
    else:
        print("\n  WARNING: at least one config has either disagreeing reachability sets "
              "or non-bit-equal reachable d.  sec 3.3 needs a footnote.")
        return 2


if __name__ == "__main__":
    sys.exit(main())
