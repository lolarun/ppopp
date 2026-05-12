#!/usr/bin/env python3
"""
overhead_summary.py — Summarise certificate emission and verifier overhead (E6, E7).

Reads JSONL run logs and computes:
  E6: overhead of cert emission (augmented TEPS vs baseline TEPS)
  E7: verifier cost vs SSSP recompute time

Usage:
    python3 -m analysis.overhead_summary \
        --e6-dir results/e6 \
        --e7-dir results/e7 \
        --output results/overhead.csv \
        [--print-table]
"""

import argparse
import json
import pathlib
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class E6Row:
    """Emission overhead: baseline vs augmented."""
    dataset:      str
    gpu:          str
    precision:    str
    teps_base:    float   # TEPS without cert
    teps_aug:     float   # TEPS with cert
    overhead_pct: float   # (teps_base / teps_aug - 1) * 100

    @classmethod
    def from_pair(cls, base: dict, aug: dict) -> "E6Row":
        n_e = base.get("dataset", {}).get("n_e", 1)
        t_b = base.get("sssp_ms", 1.0)
        t_a = aug.get("sssp_ms",  1.0)
        teps_b = n_e / (t_b / 1000.0) if t_b > 0 else 0
        teps_a = n_e / (t_a / 1000.0) if t_a > 0 else 0
        oh = (t_a / t_b - 1.0) * 100.0 if t_b > 0 else 0
        return cls(
            dataset=base.get("dataset", {}).get("name", "?"),
            gpu=base.get("hardware", {}).get("gpu", "?"),
            precision=base.get("config", {}).get("precision", "fp32"),
            teps_base=teps_b,
            teps_aug=teps_a,
            overhead_pct=oh,
        )


@dataclass
class E7Row:
    """Verifier cost vs SSSP recompute."""
    dataset:         str
    gpu:             str
    precision:       str
    sssp_ms:         float
    verifier_ms:     float
    ratio:           float   # verifier_ms / sssp_ms

    @classmethod
    def from_run(cls, r: dict) -> "E7Row":
        sssp_ms = r.get("sssp_ms", 0.0)
        ver_ms  = r.get("verifier_ms", 0.0)
        ratio   = ver_ms / sssp_ms if sssp_ms > 0 else 0.0
        return cls(
            dataset=r.get("dataset", {}).get("name", "?"),
            gpu=r.get("hardware", {}).get("gpu", "cpu"),
            precision=r.get("config", {}).get("precision", "fp32"),
            sssp_ms=sssp_ms,
            verifier_ms=ver_ms,
            ratio=ratio,
        )


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_jsonl_dir(d: pathlib.Path) -> List[dict]:
    rows = []
    for f in sorted(d.glob("*.jsonl")):
        with open(f) as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def _pair_runs(rows: List[dict]) -> List[E6Row]:
    """
    For E6, pair runs with emit_cert=false and emit_cert=true
    that share (dataset, gpu, precision, seed, rep).
    """
    by_key: Dict[tuple, Dict[bool, dict]] = defaultdict(dict)
    for r in rows:
        cfg = r.get("config", {})
        key = (
            r.get("dataset", {}).get("name", "?"),
            r.get("hardware", {}).get("gpu", "?"),
            cfg.get("precision", "fp32"),
            cfg.get("seed", 42),
            r.get("rep", 0),
        )
        emit = bool(cfg.get("emit_cert", False))
        by_key[key][emit] = r

    result = []
    for key, pair in by_key.items():
        if False in pair and True in pair:
            result.append(E6Row.from_pair(pair[False], pair[True]))
    return result


# ── CSV output ────────────────────────────────────────────────────────────────

E6_HEADER = "dataset,gpu,precision,teps_base,teps_aug,overhead_pct\n"
E7_HEADER = "dataset,gpu,precision,sssp_ms,verifier_ms,ratio\n"


def e6_row_csv(r: E6Row) -> str:
    return (f"{r.dataset},{r.gpu},{r.precision},"
            f"{r.teps_base:.4e},{r.teps_aug:.4e},{r.overhead_pct:.2f}\n")


def e7_row_csv(r: E7Row) -> str:
    return (f"{r.dataset},{r.gpu},{r.precision},"
            f"{r.sssp_ms:.2f},{r.verifier_ms:.2f},{r.ratio:.4f}\n")


def print_table_e6(rows: List[E6Row]):
    print(f"{'Dataset':>14}  {'GPU':>12}  {'Prec':>5}  "
          f"{'TEPS_base':>10}  {'TEPS_aug':>10}  {'OH%':>6}")
    for r in sorted(rows, key=lambda x: (x.dataset, x.gpu)):
        print(f"{r.dataset:>14}  {r.gpu:>12}  {r.precision:>5}  "
              f"{r.teps_base:>10.3e}  {r.teps_aug:>10.3e}  {r.overhead_pct:>+5.1f}%")


def print_table_e7(rows: List[E7Row]):
    print(f"{'Dataset':>14}  {'GPU':>12}  {'Prec':>5}  "
          f"{'SSSP ms':>9}  {'Verif ms':>9}  {'Ratio':>6}")
    for r in sorted(rows, key=lambda x: (x.dataset, x.gpu)):
        print(f"{r.dataset:>14}  {r.gpu:>12}  {r.precision:>5}  "
              f"{r.sssp_ms:>9.1f}  {r.verifier_ms:>9.1f}  {r.ratio:>6.3f}x")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Summarise emission and verifier overhead")
    ap.add_argument("--e6-dir", default=None)
    ap.add_argument("--e7-dir", default=None)
    ap.add_argument("--output", required=True, help="Base path for output CSVs")
    ap.add_argument("--print-table", action="store_true")
    args = ap.parse_args()

    out_base = pathlib.Path(args.output)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    if args.e6_dir:
        rows_e6_raw = _load_jsonl_dir(pathlib.Path(args.e6_dir))
        rows_e6 = _pair_runs(rows_e6_raw)
        with open(str(out_base).replace(".csv", "_e6.csv"), "w") as f:
            f.write(E6_HEADER)
            for r in rows_e6:
                f.write(e6_row_csv(r))
        print(f"[overhead] E6: {len(rows_e6)} pairs written")
        if args.print_table:
            print_table_e6(rows_e6)

    if args.e7_dir:
        rows_e7_raw = _load_jsonl_dir(pathlib.Path(args.e7_dir))
        rows_e7 = [E7Row.from_run(r) for r in rows_e7_raw if r.get("verifier_ms")]
        with open(str(out_base).replace(".csv", "_e7.csv"), "w") as f:
            f.write(E7_HEADER)
            for r in rows_e7:
                f.write(e7_row_csv(r))
        print(f"[overhead] E7: {len(rows_e7)} rows written")
        if args.print_table:
            print_table_e7(rows_e7)


if __name__ == "__main__":
    main()
