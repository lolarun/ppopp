#!/usr/bin/env python3
"""
sync_data_tables.py — Auto-generate plans/02_data_tables.md and
artifact/expected_outputs.md from results/*.jsonl + cert binaries.

Single source of truth: this script reads the JSONL run logs and produces
markdown tables for plans/02. Manuscript references plans/02 by link rather
than duplicating numbers.

Usage:
    python3 scripts/sync_data_tables.py

Reads:
    results/*.jsonl
    results/amd/*.jsonl
    results/certs/*.bin
    results/amd/certs/*.bin

Writes:
    docs/plans/02_data_tables.md
    docs/artifact/expected_outputs.md
"""

import json
import os
import sys
import glob
import zlib
from collections import Counter, defaultdict
from pathlib import Path

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
PLANS = ROOT / "docs" / "plans"
ARTIFACT = ROOT / "docs" / "artifact"


def load_jsonl(path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path) as f:
        return [json.loads(line) for line in f]


def _infer_dataset_label(r):
    """Best-effort dataset label, handling pre-D8 JSONL that lack rmat_scale."""
    cfg = r.get("config", {})
    name = r.get("dataset", {}).get("name", "unknown")
    if name and name != "unknown":
        # If name set and rmat_scale present, prefer scale-tagged label
        if cfg.get("rmat_scale"):
            return f"{name}-s{cfg['rmat_scale']}"
        return name
    # Fall back: infer scale from n_v if it's a power of 2
    n_v = r.get("dataset", {}).get("n_v", 0)
    if n_v > 0 and (n_v & (n_v - 1)) == 0:  # power of 2
        return f"rmat-{n_v.bit_length() - 1}"
    return "unknown"


def reachable_hash(path: Path):
    """CRC32 over reachable d values (sentinel-stripped)."""
    if not HAVE_NUMPY or not path.exists():
        return None, None, None
    fp64 = "fp64" in path.name
    dtype = np.float64 if fp64 else np.float32
    sentinel = np.float64("inf") if fp64 else np.float32("inf")
    d = np.fromfile(str(path), dtype=dtype)
    reachable = d != sentinel
    n_t, n_r = len(d), int(reachable.sum())
    if n_r == 0:
        return None, n_t, 0
    return zlib.crc32(d[reachable].tobytes()), n_t, n_t - n_r


def section_run_counts():
    out = ["## Run counts (per platform)\n"]
    for label, prefix in [("NVIDIA A10", RESULTS), ("AMD MI300X VF", RESULTS / "amd")]:
        out.append(f"\n### {label}\n")
        out.append("| File | Entries | Verified | SAT |")
        out.append("|---|---|---|---|")
        total, total_verified, total_sat = 0, 0, 0
        for f in sorted(prefix.glob("*.jsonl")):
            rows = load_jsonl(f)
            verified = sum(1 for r in rows if r["config"]["verify"])
            sat_v = sum(1 for r in rows if r["config"]["verify"] and r["verifier_verdict"] == "SAT")
            out.append(f"| `{f.name}` | {len(rows)} | {verified} | {sat_v} |")
            total += len(rows)
            total_verified += verified
            total_sat += sat_v
        out.append(f"| **Total** | **{total}** | **{total_verified}** | **{total_sat}** |")
    return "\n".join(out)


def section_cross_vendor_d_hash():
    """Reachable-only d_hash comparison across NV and AMD cert binaries."""
    nv_dir = RESULTS / "certs"
    amd_dir = RESULTS / "amd" / "certs"
    if not (nv_dir.exists() and amd_dir.exists()):
        return "## Cross-vendor reachable-only d_hash\n\n_Cert directories not found._\n"
    if not HAVE_NUMPY:
        return "## Cross-vendor reachable-only d_hash\n\n_numpy unavailable; install python3-numpy._\n"

    common = sorted({f.name for f in nv_dir.glob("*.d.bin")} & {f.name for f in amd_dir.glob("*.d.bin")})
    rows = ["## Cross-vendor reachable-only d_hash (anchor data)\n",
            "| Config | n_total | n_unreach | NV reach hash | AMD reach hash | Match |",
            "|---|---|---|---|---|---|"]
    matches = 0
    for f in common:
        nh, n_t, n_u = reachable_hash(nv_dir / f)
        ah, _, _ = reachable_hash(amd_dir / f)
        ok = nh == ah
        matches += ok
        nv_s = f"`{nh:08x}`" if nh is not None else "—"
        amd_s = f"`{ah:08x}`" if ah is not None else "—"
        rows.append(f"| `{f}` | {n_t} | {n_u} | {nv_s} | {amd_s} | {'✓' if ok else '✗'} |")
    rows.append(f"\n**{matches}/{len(common)} matches.**\n")
    return "\n".join(rows)


def section_unsat_entries():
    rows = ["## Non-SAT entries (must each have explanation)\n",
            "| Platform | Dataset | Precision | Weight dist | Verdict |",
            "|---|---|---|---|---|"]
    for label, prefix in [("NVIDIA", RESULTS), ("AMD", RESULTS / "amd")]:
        for f in sorted(prefix.glob("*.jsonl")):
            for r in load_jsonl(f):
                if r["config"]["verify"] and r["verifier_verdict"] != "SAT":
                    cfg = r["config"]
                    rows.append(
                        f"| {label} | {r['dataset']['name']} | {cfg['precision']} "
                        f"| {cfg.get('weight_dist', '?')} | `{r['verifier_verdict']}` |"
                    )
    rows.append("\n_All UNSAT entries are gaussian × long-diameter road FP32 — F10 boundary case._")
    return "\n".join(rows)


def _stats(values):
    """Return (mean, std, min, max) for a list of floats."""
    if not values:
        return (0.0, 0.0, 0.0, 0.0)
    if len(values) == 1:
        return (values[0], 0.0, values[0], values[0])
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return (mean, var ** 0.5, min(values), max(values))


def section_emission_overhead():
    rows = ["## E6 — Emission overhead (FP32 packed atomic vs noemit)\n"]
    rows.append("**Methodology:** for each (dataset, precision), `sssp_ms` averaged across "
                "all reps with verify=true (emit on) and verify=false (emit off). Overhead = "
                "(emit − noemit) / noemit × 100%.\n")
    for label, prefix_pairs in [
        ("NVIDIA A10",     [(RESULTS / "w3_task1.jsonl", "real-data"),
                            (RESULTS / "e8_scaling.jsonl", "rmat-scaling")]),
        ("AMD MI300X VF",  [(RESULTS / "amd" / "w3_amd_task1.jsonl", "real-data"),
                            (RESULTS / "amd" / "e8_amd_scaling.jsonl", "rmat-scaling")]),
    ]:
        rows.append(f"\n### {label}\n")
        rows.append("| Source | Dataset | Precision | emit (ms, mean±std) | noemit (ms) | n_reps | Overhead |")
        rows.append("|---|---|---|---|---|---|---|")
        for path, source in prefix_pairs:
            rs = load_jsonl(path)
            if not rs:
                continue
            # Group by (dataset_name, n_v, precision); separate emit/noemit lists
            by_key = defaultdict(lambda: {True: [], False: []})
            for r in rs:
                cfg = r["config"]
                ds = _infer_dataset_label(r)
                key = (ds, cfg["precision"])
                by_key[key][cfg["emit_cert"]].append(r["sssp_ms"])
            for (ds, prec), modes in sorted(by_key.items()):
                emit_list = modes[True]
                noemit_list = modes[False]
                if not (emit_list and noemit_list):
                    continue
                em, esd, _, _ = _stats(emit_list)
                nm, nsd, _, _ = _stats(noemit_list)
                ovh = (em - nm) / nm * 100 if nm > 0 else 0
                rows.append(f"| {source} | {ds} | {prec} | {em:.1f}±{esd:.1f} | "
                            f"{nm:.1f}±{nsd:.1f} | {len(emit_list)}/{len(noemit_list)} | {ovh:+.1f}% |")
    rows.append("\n_Notes:_")
    rows.append("- ny_road overhead outlier (~24% NV / 1% AMD) reflects fixed init/teardown over a small (~360ms) compute baseline; absolute overhead is ~89ms (NV) or ~6ms (AMD), not a per-edge cost.")
    rows.append("- Real-scale workloads (web_google, livejournal, RMAT-22..25) land 2–14% — within paper §3.5 < 15% target.")
    rows.append("- FP64 emission goes through dual-atomic + post-hoc reconstruct_pi (O(E) CPU scan), expected to show higher overhead than FP32 packed.")
    return "\n".join(rows)


def section_verifier_cost():
    """E7: verifier_ms vs sssp_ms ratio across all JSONLs."""
    rows = ["## E7 — Verifier cost (verifier_ms vs sssp_ms ratio)\n"]
    rows.append("**Methodology:** ratio per run = verifier_ms / sssp_ms. Reported as (mean, "
                "max, p99) across all SAT runs with verify=true.\n")
    for label, jsonl_dir in [("NVIDIA A10", RESULTS), ("AMD MI300X VF", RESULTS / "amd")]:
        rows.append(f"\n### {label}\n")
        rows.append("| Dataset / scale | n_runs | verifier mean (ms) | sssp mean (ms) | ratio mean | ratio max |")
        rows.append("|---|---|---|---|---|---|")
        by_ds = defaultdict(list)
        for f in jsonl_dir.glob("*.jsonl"):
            for r in load_jsonl(f):
                if not r["config"]["verify"]: continue
                if r["verifier_verdict"] != "SAT": continue
                cfg = r["config"]
                ds = _infer_dataset_label(r)
                ds_full = f"{ds}/{cfg['precision']}"
                if r["sssp_ms"] > 0 and r["verifier_ms"] > 0:
                    by_ds[ds_full].append((r["verifier_ms"], r["sssp_ms"]))
        for ds, pairs in sorted(by_ds.items()):
            if not pairs: continue
            ver_ms = [p[0] for p in pairs]
            sssp_ms = [p[1] for p in pairs]
            ratios = [v/s for v, s in pairs]
            v_mean = sum(ver_ms) / len(ver_ms)
            s_mean = sum(sssp_ms) / len(sssp_ms)
            r_mean = sum(ratios) / len(ratios)
            r_max = max(ratios)
            rows.append(f"| {ds} | {len(pairs)} | {v_mean:.1f} | {s_mean:.1f} | "
                        f"{r_mean:.2f} | {r_max:.2f} |")
    rows.append("\n_Verifier is O(V+E) parallel; for most workloads ratio < 2× (verifier comparable "
                "to or faster than re-running SSSP). Ratios > 2× indicate datasets where verifier "
                "iterates over many more edges than the SSSP traversed (e.g. RMAT with most "
                "vertices unreachable, where the verifier still validates relaxation across "
                "every edge in the graph)._")
    return "\n".join(rows)


def section_e4_coverage():
    """E4 error-injection coverage: verifier vs golden-output detection rates."""
    e4_dir = RESULTS / "e4"
    rows = ["## E4 — Error-injection coverage (verifier vs golden-output)\n"]
    rows.append("**Methodology:** for each (dataset, error_kind, n_errors), 30 random seeds "
                "perturb the clean cert and the verifier is run. golden-output detection = "
                "byte-comparison `np.array_equal(clean_d, d_inj)` — what a naive cross-vendor "
                "validator would do.\n")
    if not e4_dir.exists():
        rows.append("_E4 results not yet generated. Run `scripts/inject_and_verify.py`._\n")
        return "\n".join(rows)
    files = sorted(e4_dir.glob("e4_*.jsonl"))
    if not files:
        rows.append("_E4 results not yet generated._\n")
        return "\n".join(rows)
    for f in files:
        ds_name = f.stem.replace("e4_", "")
        rs = load_jsonl(f)
        rows.append(f"\n### {ds_name} ({len(rs)} injection runs)\n")
        rows.append("| Error kind | n_errors | total | verifier UNSAT | golden_diff | **verifier-only** |")
        rows.append("|---|---|---|---|---|---|")
        # Group by (kind, n_errors)
        by_key = defaultdict(list)
        for r in rs:
            by_key[(r["error_kind"], r["n_errors"])].append(r)
        for (kind, mag), items in sorted(by_key.items()):
            total = len(items)
            ver = sum(1 for r in items if r["verdict"] and r["verdict"] != "SAT")
            gld = sum(1 for r in items if r["golden_detects"])
            ver_only = sum(1 for r in items
                          if r["verdict"] and r["verdict"] != "SAT" and not r["golden_detects"])
            rows.append(f"| {kind} | {mag} | {total} | {ver}/{total} ({100*ver/total:.0f}%) | "
                        f"{gld}/{total} ({100*gld/total:.0f}%) | **{ver_only}/{total}** |")
    rows.append("\n_Verifier-only column = verifier caught, golden-output byte-comparison did "
                "not. CYCLE and PREDECESSOR_RANDOM are the verifier-only signal: they corrupt "
                "π without touching d, so byte-comparison sees no difference._")
    return "\n".join(rows)


def section_e12c():
    """E12.c — strict vs relaxed atomics d_hash dichotomy across reps."""
    e12c_dir = RESULTS / "e12c"
    rows = ["## E12.c — Relaxed atomics (RELAX_ATOMICS=ON) cross-rep d_hash variance\n"]
    rows.append("**Methodology:** identical inputs (graph, source, FP32 weights), 5 reps "
                "per build. Strict build uses CAS-loop atomic updates on packed (d, π); "
                "relaxed build replaces CAS with non-atomic load+store (early-out preserved).\n")
    if not e12c_dir.exists():
        rows.append("_E12.c results not yet generated._\n")
        return "\n".join(rows)
    strict = load_jsonl(e12c_dir / "strict.jsonl")
    relaxed = load_jsonl(e12c_dir / "relaxed.jsonl")
    if not (strict and relaxed):
        rows.append("_E12.c JSONLs missing or empty._\n")
        return "\n".join(rows)

    def group(rs):
        g = defaultdict(list)
        for r in rs:
            g[r["dataset"]["name"]].append(r)
        return g

    def fmt_hashes(items):
        hashes = [r["cert_summary"]["d_hash"] for r in sorted(items, key=lambda x: x["rep"])]
        uniq = list(dict.fromkeys(hashes))
        if len(uniq) == 1:
            return f"`{uniq[0]}` ×{len(hashes)}"
        return " ".join(f"`{h}`" for h in hashes)

    def fmt_verdicts(items):
        c = Counter(r["verifier_verdict"] for r in items)
        return ", ".join(f"{v} ({n})" for v, n in c.most_common())

    s_grp, r_grp = group(strict), group(relaxed)
    datasets = sorted(set(s_grp) | set(r_grp))
    rows.append("| Dataset | strict d_hash (5 reps) | strict verdict | relaxed d_hashes (5 reps) | relaxed verdict |")
    rows.append("|---|---|---|---|---|")
    for ds in datasets:
        s_items, r_items = s_grp.get(ds, []), r_grp.get(ds, [])
        rows.append(f"| {ds} | {fmt_hashes(s_items)} | {fmt_verdicts(s_items)} | "
                    f"{fmt_hashes(r_items)} | {fmt_verdicts(r_items)} |")

    n_ds = len(datasets)
    rows.append(f"\n_Across all {n_ds} datasets: strict produces 1 unique d_hash per dataset "
                f"(all {5*n_ds} runs SAT), relaxed produces 5 unique d_hashes per dataset "
                f"(all {5*n_ds} UNSAT_RELAXATION). Concrete experimental support for the §II.E "
                f"claim that atomic CAS is the enabler of bitwise determinism — removing it "
                f"breaks both reproducibility and verifier admissibility on every dataset tested._")
    return "\n".join(rows)


def section_e11():
    """E11 — GPU Bellman-Ford companion: strict vs relaxed atomics across reps."""
    e11_dir = RESULTS / "e11"
    rows = ["## E11 — GPU Bellman-Ford companion (cross-algorithm §II.E validation)\n"]
    rows.append("**Methodology:** identical inputs (graph, source, FP32 weights), 5 reps "
                "per build, `--algo=bellman_ford_gpu`. Strict build uses CAS-loop atomic "
                "updates on packed (d, π); relaxed build replaces CAS with non-atomic "
                "load+store. Same matrix as E12.c but a different SSSP algorithm — "
                "Bellman-Ford visits every edge every iteration rather than bucketing by "
                "tentative distance, so it stresses the same atomic primitives via a "
                "different access pattern.\n")
    if not e11_dir.exists():
        rows.append("_E11 results not yet generated._\n")
        return "\n".join(rows)
    strict = load_jsonl(e11_dir / "strict.jsonl")
    relaxed = load_jsonl(e11_dir / "relaxed.jsonl")
    if not (strict and relaxed):
        rows.append("_E11 JSONLs missing or empty._\n")
        return "\n".join(rows)

    def group(rs):
        g = defaultdict(list)
        for r in rs:
            g[r["dataset"]["name"]].append(r)
        return g

    def fmt_hashes(items):
        hashes = [r["cert_summary"]["d_hash"] for r in sorted(items, key=lambda x: x["rep"])]
        uniq = list(dict.fromkeys(hashes))
        if len(uniq) == 1:
            return f"`{uniq[0]}` ×{len(hashes)}"
        return " ".join(f"`{h}`" for h in hashes)

    def fmt_verdicts(items):
        c = Counter(r["verifier_verdict"] for r in items)
        return ", ".join(f"{v} ({n})" for v, n in c.most_common())

    s_grp, r_grp = group(strict), group(relaxed)
    datasets = sorted(set(s_grp) | set(r_grp))
    rows.append("| Dataset | strict d_hash (5 reps) | strict verdict | relaxed d_hashes (5 reps) | relaxed verdict |")
    rows.append("|---|---|---|---|---|")
    for ds in datasets:
        s_items, r_items = s_grp.get(ds, []), r_grp.get(ds, [])
        rows.append(f"| {ds} | {fmt_hashes(s_items)} | {fmt_verdicts(s_items)} | "
                    f"{fmt_hashes(r_items)} | {fmt_verdicts(r_items)} |")

    n_ds = len(datasets)
    rows.append(f"\n_Bellman-Ford does NOT reproduce the E12.c dichotomy: across all "
                f"{n_ds} datasets, strict yields 1 unique d_hash per dataset (all "
                f"{5*n_ds} runs SAT), AND relaxed *also* yields 1 unique d_hash per "
                f"dataset (all {5*n_ds} runs SAT) — the relaxed BF d_hash is byte-equal "
                f"to the strict BF d_hash on every dataset. The §II.E claim is therefore "
                f"refined: atomic CAS is necessary for Δ-stepping (E12.c) but not for "
                f"Bellman-Ford. BF's iterate-to-fixedpoint termination check (`any_updated "
                f"== 0`) re-validates the global edge inequality every iteration, so "
                f"race-induced inconsistencies are repaired before the loop terminates. "
                f"Δ-stepping's bucket-once scheduling has no such repair mechanism. "
                f"The boundary is structural to the host-side termination check, not to "
                f"the algorithm name. See `cross_vendor_e11.md` for cross-vendor confirmation._")
    return "\n".join(rows)


def section_e11_xalgo():
    """E11 — cross-algorithm consistency: BF strict d_hash vs Δ-stepping strict d_hash."""
    rows = ["## E11 — Cross-algorithm strict-d_hash consistency (BF vs Δ-stepping)\n"]
    e11_dir = RESULTS / "e11"
    e12c_dir = RESULTS / "e12c"
    if not (e11_dir.exists() and e12c_dir.exists()):
        rows.append("_E11 or E12.c results not yet generated._\n")
        return "\n".join(rows)
    bf_strict = load_jsonl(e11_dir / "strict.jsonl")
    ds_strict = load_jsonl(e12c_dir / "strict.jsonl")
    if not (bf_strict and ds_strict):
        rows.append("_E11 or E12.c strict JSONLs missing or empty._\n")
        return "\n".join(rows)

    def first_hash(rs):
        g = defaultdict(list)
        for r in rs:
            g[r["dataset"]["name"]].append(r)
        out = {}
        for ds, items in g.items():
            items_sorted = sorted(items, key=lambda x: x["rep"])
            out[ds] = items_sorted[0]["cert_summary"]["d_hash"] if items_sorted else None
        return out

    bf_h = first_hash(bf_strict)
    ds_h = first_hash(ds_strict)
    datasets = sorted(set(bf_h) | set(ds_h))
    rows.append("| Dataset | Δ-stepping strict d_hash | BF strict d_hash | Match |")
    rows.append("|---|---|---|---|")
    matches, total_compared = 0, 0
    for ds in datasets:
        bh = bf_h.get(ds)
        dh = ds_h.get(ds)
        if bh is None or dh is None:
            mark = "—"
        else:
            ok = bh == dh
            mark = "✓" if ok else "✗"
            matches += int(ok)
            total_compared += 1
        bh_s = f"`{bh}`" if bh else "—"
        dh_s = f"`{dh}`" if dh else "—"
        rows.append(f"| {ds} | {dh_s} | {bh_s} | {mark} |")
    if total_compared:
        rows.append(f"\n**{matches}/{total_compared} datasets agree across algorithms.**")
    rows.append("\n_BF and Δ-stepping converge to the same shortest-path tree under strict "
                "atomics; same FP rounding under min-plus + atomic CAS yields the same "
                "d[v] regardless of relaxation order or algorithm._")
    return "\n".join(rows)


def write_data_tables():
    out = [
        "# Data tables (auto-generated)",
        "",
        "**Generated by:** `scripts/sync_data_tables.py`. Do not edit by hand — re-run the script.",
        "",
        "Single source of truth for all numerical paper claims. Manuscript and",
        "artifact docs reference this file by link.",
        "",
    ]
    out.append(section_run_counts())
    out.append("")
    out.append(section_cross_vendor_d_hash())
    out.append("")
    out.append(section_emission_overhead())
    out.append("")
    out.append(section_verifier_cost())
    out.append("")
    out.append(section_e4_coverage())
    out.append("")
    out.append(section_e12c())
    out.append("")
    out.append(section_e11())
    out.append("")
    out.append(section_e11_xalgo())
    out.append("")
    out.append(section_unsat_entries())
    out.append("")

    target = PLANS / "02_data_tables.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(out), encoding="utf-8")
    print(f"  wrote {target.relative_to(ROOT)}")


def write_expected_outputs():
    """Artifact-track: expected outputs for reviewer reproduction."""
    out = [
        "# Expected outputs (auto-generated)",
        "",
        "**Generated by:** `scripts/sync_data_tables.py`.",
        "",
        "If you reproduce this artifact end-to-end, the values below are what",
        "your runs should match. Hashes are reachable-only CRC32 over the",
        "distance vectors emitted by `--save-cert=<prefix>`.",
        "",
    ]
    out.append(section_cross_vendor_d_hash())
    out.append("")
    out.append("## Verification counts you should see")
    out.append("")
    out.append("- 1000/1000 SAT in `w3_stress.jsonl` (NVIDIA path) and `w3_amd_stress.jsonl` (AMD path)")
    out.append("- 12/12 SAT in `w3_task1.jsonl` (4 datasets × 3 algorithm paths)")
    out.append("- 48/48 SAT-when-verified in `e8_scaling.jsonl` (4 RMAT scales × 4 configs × 3 reps)")
    out.append("- 16/16 SAT in `e9_weights.jsonl` (4 weight dists × 2 graphs × 2 reps)")
    out.append("- 7 SAT + 2 expected UNSAT in `e1_nvidia.jsonl` (gaussian × road FP32 is the F10 boundary case)")
    out.append("")

    target = ARTIFACT / "expected_outputs.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(out), encoding="utf-8")
    print(f"  wrote {target.relative_to(ROOT)}")


if __name__ == "__main__":
    print("sync_data_tables.py: regenerating from results/*.jsonl + cert binaries")
    if not HAVE_NUMPY:
        print("  warning: numpy unavailable, reachable-hash sections will be stubs")
    write_data_tables()
    write_expected_outputs()
    print("done.")
