#!/usr/bin/env python3
"""Summarize F26+E11+E12c extension batch on AMD MI300X SPX-VF.

Reads three JSONLs from results/amd/{a1_38_extension,a1_e11_extension,a1_e12c_extension}/
Reports per-cell unique d_hash count + verdict distribution.
Replaces the broken inline python summarizer (used wrong field name).
"""
from __future__ import annotations
import json, sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PHASES = [
    ("F26 async push",        REPO / "results/amd/a1_38_extension/async_push_extension.jsonl"),
    ("E11 Bellman-Ford",      REPO / "results/amd/a1_e11_extension/bellman_ford_extension.jsonl"),
    ("E12.c Δ-stepping",      REPO / "results/amd/a1_e12c_extension/delta_stepping_extension.jsonl"),
]

def parse_cell(name: str) -> tuple[str, str]:
    """name = 'build_gpu__usa_road__rep0' → ('build_gpu', 'usa_road')."""
    parts = name.split("__")
    if len(parts) >= 3:
        return parts[0], parts[1]
    return ("?", name)

def main():
    print("# F26 extension batch summary (AMD MI300X SPX-VF, host 165.245.128.59)")
    print()
    grand = 0
    for label, path in PHASES:
        if not path.exists():
            print(f"## {label}: MISSING {path}"); continue
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        cells: dict[tuple[str,str], list[dict]] = defaultdict(list)
        for r in rows:
            name = r.get("dataset", {}).get("name", "")
            cells[parse_cell(name)].append(r)
        print(f"## {label}  ({len(rows)} runs)")
        print()
        print(f"| Build | Dataset | Reps | Unique d_hash | Verdicts |")
        print(f"|---|---|---:|---:|---|")
        for (b, c), rs in sorted(cells.items()):
            hashes = sorted({r.get("cert_summary", {}).get("d_hash", "")[:10] for r in rs})
            verdicts = sorted({r.get("verifier_verdict", "?") for r in rs})
            print(f"| `{b}` | {c} | {len(rs)} | {len(hashes)} | {','.join(verdicts)} |")
        print()
        # First-line d_hash per (build, cell) for cross-vendor diff
        print(f"  d_hash (rep0):")
        for (b, c), rs in sorted(cells.items()):
            r0 = next((r for r in rs if "rep0" in r.get("dataset",{}).get("name","")), rs[0])
            h = r0.get("cert_summary", {}).get("d_hash", "")[:10]
            print(f"    {b:<18} {c:<10} d_hash={h}")
        print()
        grand += len(rows)
    print(f"---\n**Total: {grand} runs**")

if __name__ == "__main__":
    main()
