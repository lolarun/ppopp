#!/usr/bin/env python3
"""Cross-vendor d_hash compare for F26-extension cells.
AMD MI300X SPX-VF (host 165.245.128.59) vs NVIDIA A10 (existing baseline).
"""
import json, sys
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parent.parent

# AMD F26-extension d_hash (rep0 strict, from new batch)
AMD_NEW = {
    "usa_road": {"async_push": "84ba4c3c", "bellman_ford": "84ba4c3c", "delta_stepping": "84ba4c3c"},
    "rmat-22":  {"async_push": "1405bbe6", "bellman_ford": "1405bbe6", "delta_stepping": "1405bbe6"},
}

# Search NV JSONLs (top-level results/) for usa_road and rmat-22 FP32 strict d_hash
nv_hashes = defaultdict(set)
for jf in (REPO / "results").glob("*.jsonl"):
    try:
        for line in jf.read_text().splitlines():
            r = json.loads(line)
            n = r.get("dataset", {}).get("name", "")
            cfg = r.get("config", {})
            if cfg.get("precision") != "fp32": continue
            wd = cfg.get("weight_dist", "")
            if wd not in ("", "native"): continue
            algo = cfg.get("algo", "")
            verdict = r.get("verifier_verdict", "")
            if verdict != "SAT": continue
            h = r.get("cert_summary", {}).get("d_hash", "")[:10]
            for ds in ("usa_road", "rmat-22", "rmat22", "rmat_22"):
                if ds in n.lower():
                    key = "rmat-22" if "rmat" in ds else "usa_road"
                    nv_hashes[(key, algo)].add(h)
    except Exception as e:
        pass

# Also check subdirs (a1_*, e1, e2 batches)
for jf in (REPO / "results").rglob("*.jsonl"):
    if "amd" in str(jf): continue
    try:
        for line in jf.read_text().splitlines():
            r = json.loads(line)
            n = r.get("dataset", {}).get("name", "")
            cfg = r.get("config", {})
            if cfg.get("precision") != "fp32": continue
            wd = cfg.get("weight_dist", "")
            if wd not in ("", "native"): continue
            algo = cfg.get("algo", "")
            verdict = r.get("verifier_verdict", "")
            if verdict != "SAT": continue
            h = r.get("cert_summary", {}).get("d_hash", "")[:10]
            for ds in ("usa_road", "rmat-22", "rmat22", "rmat_22"):
                if ds in n.lower():
                    key = "rmat-22" if "rmat" in ds else "usa_road"
                    nv_hashes[(key, algo)].add(h)
    except Exception:
        pass

print("# AMD MI300X SPX-VF (new batch, rep0 strict d_hash):")
for ds, m in AMD_NEW.items():
    print(f"  {ds}:")
    for a, h in m.items():
        print(f"    {a:<18} {h}")

print("\n# NVIDIA A10 (existing baseline, FP32 strict SAT d_hash):")
if not nv_hashes:
    print("  (no NV usa_road/rmat-22 FP32 strict SAT runs found in results/)")
for (ds, algo), hashes in sorted(nv_hashes.items()):
    print(f"  {ds:<10} {algo:<22} d_hashes={sorted(hashes)}")

print("\n# Cross-vendor verdict:")
for ds in AMD_NEW:
    for algo_short, algo_full in [("async_push","async_push_sssp_gpu"), ("bellman_ford","bellman_ford_gpu"), ("delta_stepping","delta_stepping_gpu")]:
        amd_h = AMD_NEW[ds][algo_short]
        nv_set = nv_hashes.get((ds, algo_full), set())
        if not nv_set:
            print(f"  {ds:<10} {algo_short:<18} AMD={amd_h}  NV=NONE  → cannot compare (no NV data)")
        elif amd_h in nv_set:
            print(f"  {ds:<10} {algo_short:<18} AMD={amd_h}  NV matches  ✅ BYTE-EQUAL")
        else:
            print(f"  {ds:<10} {algo_short:<18} AMD={amd_h}  NV={sorted(nv_set)}  ❌ MISMATCH")
