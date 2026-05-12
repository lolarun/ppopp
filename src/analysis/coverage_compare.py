#!/usr/bin/env python3
"""
coverage_compare.py — Aggregate E4 error-injection results into detection rates.

Reads JSONL files produced by scripts/inject_and_verify.py and computes:
  - Detection rate per (error_kind, n_errors, dataset)
  - Comparison: certificate verifier vs. golden-output comparison (byte diff)
  - False-negative rate breakdown

Usage:
    python3 -m analysis.coverage_compare \
        --e4-dir results/e4 \
        --output results/coverage.csv \
        [--print-table]
"""

import argparse
import json
import pathlib
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class CoverageRow:
    dataset:         str
    error_kind:      str
    n_errors:        int
    n_trials:        int
    detected_cert:   int   # verifier caught it (verdict != SAT)
    detected_golden: int   # golden comparison caught it (d[] != d_ref[])
    false_neg_cert:  int   # missed by verifier
    false_neg_both:  int   # missed by both

    @property
    def rate_cert(self) -> float:
        return self.detected_cert / max(1, self.n_trials)

    @property
    def rate_golden(self) -> float:
        return self.detected_golden / max(1, self.n_trials)

    @property
    def advantage(self) -> float:
        """Certificate detection rate minus golden detection rate."""
        return self.rate_cert - self.rate_golden


# ── Load E4 JSONL ──────────────────────────────────────────────────────────────

def _load_e4_dir(e4_dir: pathlib.Path) -> List[dict]:
    rows = []
    for f in sorted(e4_dir.glob("*.jsonl")):
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


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate(rows: List[dict]) -> List[CoverageRow]:
    """
    Group by (dataset, error_kind, n_errors) and compute detection rates.

    Expected JSONL fields per row (from inject_and_verify.py):
      dataset, error_kind, n_errors, seed,
      verdict         (verifier result on injected cert: SAT / UNSAT_*)
      golden_mismatch (bool: d[] != d_ref[])
    """
    Key = Tuple[str, str, int]
    buckets: Dict[Key, List[dict]] = defaultdict(list)

    for r in rows:
        k = (r.get("dataset", "?"),
             r.get("error_kind", "?"),
             int(r.get("n_errors", 0)))
        buckets[k].append(r)

    result = []
    for (dataset, ekind, n_err), trials in sorted(buckets.items()):
        n_trials = len(trials)
        detected_cert   = sum(1 for t in trials if t.get("verdict", "SAT") != "SAT")
        detected_golden = sum(1 for t in trials if t.get("golden_mismatch", False))
        false_neg_cert  = n_trials - detected_cert
        false_neg_both  = sum(
            1 for t in trials
            if t.get("verdict", "SAT") == "SAT" and not t.get("golden_mismatch", False)
        )
        result.append(CoverageRow(
            dataset=dataset, error_kind=ekind, n_errors=n_err,
            n_trials=n_trials,
            detected_cert=detected_cert, detected_golden=detected_golden,
            false_neg_cert=false_neg_cert, false_neg_both=false_neg_both,
        ))
    return result


# ── Output ────────────────────────────────────────────────────────────────────

CSV_HEADER = (
    "dataset,error_kind,n_errors,n_trials,"
    "detected_cert,detected_golden,"
    "rate_cert,rate_golden,advantage,"
    "false_neg_cert,false_neg_both\n"
)

def to_csv_row(r: CoverageRow) -> str:
    return (
        f"{r.dataset},{r.error_kind},{r.n_errors},{r.n_trials},"
        f"{r.detected_cert},{r.detected_golden},"
        f"{r.rate_cert:.4f},{r.rate_golden:.4f},{r.advantage:.4f},"
        f"{r.false_neg_cert},{r.false_neg_both}\n"
    )


def print_table(rows: List[CoverageRow]):
    header = (
        f"{'Dataset':>12}  {'ErrorKind':>20}  {'n_err':>5}  "
        f"{'N':>5}  {'cert%':>6}  {'gold%':>6}  {'Δ':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r.dataset:>12}  {r.error_kind:>20}  {r.n_errors:>5}  "
            f"{r.n_trials:>5}  {100*r.rate_cert:>5.1f}%  "
            f"{100*r.rate_golden:>5.1f}%  {100*r.advantage:>+5.1f}%"
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Aggregate E4 coverage results")
    ap.add_argument("--e4-dir",     required=True, help="Directory with E4 JSONL files")
    ap.add_argument("--output",     required=True, help="Output CSV path")
    ap.add_argument("--print-table", action="store_true",
                    help="Print ASCII summary table")
    args = ap.parse_args()

    e4_dir = pathlib.Path(args.e4_dir)
    rows_raw = _load_e4_dir(e4_dir)
    if not rows_raw:
        print(f"[coverage_compare] No data found in {e4_dir}", file=sys.stderr)
        sys.exit(1)

    agg = aggregate(rows_raw)

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(CSV_HEADER)
        for r in agg:
            f.write(to_csv_row(r))

    print(f"[coverage_compare] Wrote {len(agg)} rows to {out}")

    if args.print_table:
        print()
        print_table(agg)


if __name__ == "__main__":
    main()
