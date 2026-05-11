#!/usr/bin/env python3
"""Aggregate results/_compare/*.json into the RE0 decision table.

Buckets pairs by (dataset, pair_kind) where pair_kind is one of:
    nv-nv     — both runs on NVIDIA (intra-vendor variance)
    amd-amd   — both runs on AMD     (intra-vendor variance)
    nv-amd    — one NV, one AMD      (cross-vendor drift — the GO/NO-GO target)
"""
import json
import glob
import statistics as S
from collections import defaultdict

buckets = defaultdict(list)
# Scan both flat (legacy) and precision-subdir layouts.
for f in sorted(glob.glob("results/_compare/**/*.json", recursive=True)):
    d = json.loads(open(f).read())
    pair = tuple(sorted([d["a"]["vendor"], d["b"]["vendor"]]))
    if pair == ("nvidia", "nvidia"):
        kind = "nv-nv"
    elif pair == ("amd", "amd"):
        kind = "amd-amd"
    else:
        kind = "nv-amd"
    prec = d.get("precision", "fp32")
    buckets[(prec, d["dataset"], kind)].append(d)

hdr = ("prec", "dataset", "pair", "N", "byte_diff_med",
       "byte_diff_max", "max_Linf_max", "top100_J_min", "top100_K_min")
print(f"{hdr[0]:<6}{hdr[1]:<14}{hdr[2]:<10}{hdr[3]:>4}{hdr[4]:>16}{hdr[5]:>16}{hdr[6]:>16}{hdr[7]:>14}{hdr[8]:>14}")
print("-" * 110)
for key in sorted(buckets, key=lambda k: (k[0], k[1], {"nv-nv": 0, "amd-amd": 1, "nv-amd": 2}[k[2]])):
    rows = buckets[key]
    bd = [r["byte_diff_fraction"] for r in rows]
    mi = [r["max_Linf"] for r in rows]
    jc = [r["rank_top100_jaccard"] for r in rows]
    ke = [r["rank_top100_kendall"] for r in rows]
    prec, ds, pk = key
    print(f"{prec:<6}{ds:<14}{pk:<10}{len(rows):>4}{S.median(bd):>15.2%}"
          f"{max(bd):>15.2%}{max(mi):>16.3e}{min(jc):>14.3f}{min(ke):>14.3f}")

# Verdict
print()
nvamd = [r for k, rs in buckets.items() if k[2] == "nv-amd" for r in rs]
if nvamd:
    all_go = all(r["decision"] == "GO" for r in nvamd)
    print(f"Cross-vendor pairs total: {len(nvamd)}")
    print(f"All cross-vendor pairs GO: {all_go}")
    print(f"Min cross-vendor byte_diff_fraction: {min(r['byte_diff_fraction'] for r in nvamd):.2%}")
    print(f"Max cross-vendor max_Linf:           {max(r['max_Linf'] for r in nvamd):.3e}")
else:
    print("No cross-vendor pairs found.")
