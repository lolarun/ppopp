#!/usr/bin/env python3
"""
make_figures.py — Generate all paper figures from experiment JSONL/CSV outputs.

Figure ↔ Experiment mapping (per docs/ts/03_infrastructure.md §9):
  fig1_drift_heatmap        ← E1  (drift baseline)
  fig2_drift_mechanism_pie  ← E2  (mechanism attribution)
  fig3_correctness_split    ← E1 + E3 (verifier soundness + drift)
  fig4_coverage_compare     ← E4  (error injection coverage)
  fig5_emission_overhead    ← E6  (cert emission overhead)
  fig6_verifier_recompute   ← E7  (verifier vs recompute)
  fig7_scaling              ← E8  (RMAT scaling)
  fig8_stress_heatmap       ← E9  (weight × precision × structure)

Usage:
    python3 -m analysis.make_figures \
        --results-dir results/ \
        --output      paper/figures/

All figures are saved as .pdf (vector) suitable for LaTeX \includegraphics.
"""

import argparse
import json
import pathlib
import sys
from typing import Any, Dict, List, Optional

import numpy as np

# Matplotlib import — fail gracefully with instructions if not installed.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.ticker import FuncFormatter
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ── Style ─────────────────────────────────────────────────────────────────────

PAPER_RC = {
    "font.size":         9,
    "axes.titlesize":    9,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "figure.dpi":        200,
    "figure.figsize":    (3.5, 2.6),   # single-column ACM
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
    "lines.linewidth":   1.2,
    "lines.markersize":  4,
}

DATASETS_ORDER = [
    "ny_road", "usa_road", "road_usa",
    "livejournal", "web_google", "twitter", "friendster",
    "kron_25",
]

GPU_COLORS = {
    "A100": "#4A90D9",
    "MI250X": "#E8854A",
    "RTX4090": "#6BBF6B",
    "RX7900XTX": "#CC6699",
}


# ── Data loaders ──────────────────────────────────────────────────────────────

def _load_csv(path: pathlib.Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    if not lines:
        return []
    header = lines[0].split(",")
    return [dict(zip(header, l.split(","))) for l in lines[1:]]


def _load_jsonl(path: pathlib.Path) -> List[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


# ── Figure 1: Drift heatmap ───────────────────────────────────────────────────

def fig1_drift_heatmap(results_dir: pathlib.Path, out: pathlib.Path):
    csv_path = results_dir / "e1" / "drift_summary.csv"
    rows = _load_csv(csv_path)
    if not rows:
        print(f"[fig1] No data at {csv_path}; skipping")
        return

    datasets = DATASETS_ORDER
    gpus = sorted({r["gpu_a"] + " vs " + r["gpu_b"] for r in rows})

    matrix = np.zeros((len(datasets), len(gpus)))
    for r in rows:
        pair = r["gpu_a"] + " vs " + r["gpu_b"]
        ds   = r["dataset"]
        try:
            i = datasets.index(ds)
            j = gpus.index(pair)
        except ValueError:
            continue
        pct = float(r.get("n_d_diff", 0)) / max(1, float(r.get("n_vertices", 1)))
        matrix[i, j] = pct * 100

    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots()
        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0)
        ax.set_xticks(range(len(gpus)))
        ax.set_xticklabels(gpus, rotation=30, ha="right")
        ax.set_yticks(range(len(datasets)))
        ax.set_yticklabels(datasets)
        ax.set_title("Distance drift (% vertices)")
        plt.colorbar(im, ax=ax, label="%")
        fig.tight_layout()
        fig.savefig(out / "fig1_drift_heatmap.pdf")
        plt.close(fig)
    print(f"[fig1] saved")


# ── Figure 2: Drift mechanism pie ────────────────────────────────────────────

def fig2_drift_mechanism_pie(results_dir: pathlib.Path, out: pathlib.Path):
    csv_path = results_dir / "e2" / "mechanism.csv"
    rows = _load_csv(csv_path)
    if not rows:
        print(f"[fig2] No data at {csv_path}; skipping")
        return

    labels  = [r["mechanism"] for r in rows]
    sizes   = [float(r["fraction"]) for r in rows]

    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(2.8, 2.8))
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, autopct="%1.0f%%",
            startangle=90, counterclock=False)
        for t in autotexts:
            t.set_fontsize(7)
        ax.set_title("Drift mechanism attribution")
        fig.tight_layout()
        fig.savefig(out / "fig2_drift_mechanism_pie.pdf")
        plt.close(fig)
    print(f"[fig2] saved")


# ── Figure 3: Correctness / drift split ──────────────────────────────────────

def fig3_correctness_split(results_dir: pathlib.Path, out: pathlib.Path):
    csv_e1 = results_dir / "e1" / "drift_summary.csv"
    csv_e3 = results_dir / "e3" / "verifier_soundness.csv"
    rows_e1 = _load_csv(csv_e1)
    rows_e3 = _load_csv(csv_e3)
    if not rows_e1:
        print(f"[fig3] No E1 data; skipping")
        return

    datasets = list({r["dataset"] for r in rows_e1})
    x = np.arange(len(datasets))

    n_drift     = {r["dataset"]: int(r["n_d_diff"]) for r in rows_e1}
    n_unsat     = {r["dataset"]: int(r.get("n_unsat", 0)) for r in rows_e3} if rows_e3 else {}

    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots()
        w = 0.35
        ax.bar(x - w/2, [n_drift.get(d, 0) for d in datasets],
               width=w, label="byte-diff d[]", color="#4A90D9")
        ax.bar(x + w/2, [n_unsat.get(d, 0) for d in datasets],
               width=w, label="verifier UNSAT", color="#E8854A")
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=30, ha="right")
        ax.set_ylabel("# vertices")
        ax.set_title("Cross-platform drift vs. verifier catch")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "fig3_correctness_split.pdf")
        plt.close(fig)
    print(f"[fig3] saved")


# ── Figure 4: Coverage compare ───────────────────────────────────────────────

def fig4_coverage_compare(results_dir: pathlib.Path, out: pathlib.Path):
    csv_path = results_dir / "coverage.csv"
    rows = _load_csv(csv_path)
    if not rows:
        print(f"[fig4] No data at {csv_path}; skipping")
        return

    kinds = sorted({r["error_kind"] for r in rows})
    rate_cert   = [np.mean([float(r["rate_cert"]) for r in rows if r["error_kind"] == k])
                   for k in kinds]
    rate_golden = [np.mean([float(r["rate_golden"]) for r in rows if r["error_kind"] == k])
                   for k in kinds]

    x = np.arange(len(kinds))
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots()
        w = 0.35
        ax.bar(x - w/2, rate_cert,   width=w, label="Certificate verifier", color="#4A90D9")
        ax.bar(x + w/2, rate_golden, width=w, label="Golden comparison",    color="#BBBBBB")
        ax.set_xticks(x)
        ax.set_xticklabels([k.replace("_", "\n") for k in kinds], fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_ylabel("Detection rate")
        ax.set_title("Error detection coverage (E4)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "fig4_coverage_compare.pdf")
        plt.close(fig)
    print(f"[fig4] saved")


# ── Figure 5: Emission overhead ───────────────────────────────────────────────

def fig5_emission_overhead(results_dir: pathlib.Path, out: pathlib.Path):
    csv_path = results_dir / "overhead_e6.csv"
    rows = _load_csv(csv_path)
    if not rows:
        print(f"[fig5] No data at {csv_path}; skipping")
        return

    datasets = list({r["dataset"] for r in rows})
    gpus     = sorted({r["gpu"] for r in rows})
    x = np.arange(len(datasets))
    w = 0.8 / max(len(gpus), 1)

    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots()
        for i, gpu in enumerate(gpus):
            oh = [float(next((r["overhead_pct"] for r in rows
                              if r["dataset"] == d and r["gpu"] == gpu), 0))
                  for d in datasets]
            ax.bar(x + i * w - w * len(gpus) / 2, oh, width=w,
                   label=gpu, color=list(GPU_COLORS.values())[i % len(GPU_COLORS)])
        ax.axhline(15, color="red", linestyle="--", linewidth=0.8, label="15% target")
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=30, ha="right")
        ax.set_ylabel("Overhead (%)")
        ax.set_title("Certificate emission overhead (E6)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "fig5_emission_overhead.pdf")
        plt.close(fig)
    print(f"[fig5] saved")


# ── Figure 6: Verifier vs recompute ──────────────────────────────────────────

def fig6_verifier_recompute(results_dir: pathlib.Path, out: pathlib.Path):
    csv_path = results_dir / "overhead_e7.csv"
    rows = _load_csv(csv_path)
    if not rows:
        print(f"[fig6] No data at {csv_path}; skipping")
        return

    datasets = list({r["dataset"] for r in rows})
    x = np.arange(len(datasets))

    sssp_ms = [float(next((r["sssp_ms"] for r in rows if r["dataset"] == d), 0))
               for d in datasets]
    ver_ms  = [float(next((r["verifier_ms"] for r in rows if r["dataset"] == d), 0))
               for d in datasets]

    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots()
        w = 0.35
        ax.bar(x - w/2, sssp_ms, width=w, label="SSSP (recompute)", color="#4A90D9")
        ax.bar(x + w/2, ver_ms,  width=w, label="Verifier",         color="#E8854A")
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=30, ha="right")
        ax.set_ylabel("Wall time (ms)")
        ax.set_title("Verifier cost vs SSSP recompute (E7)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "fig6_verifier_recompute.pdf")
        plt.close(fig)
    print(f"[fig6] saved")


# ── Figure 7: Scaling ─────────────────────────────────────────────────────────

def fig7_scaling(results_dir: pathlib.Path, out: pathlib.Path):
    csv_path = results_dir / "e8" / "scaling.csv"
    rows = _load_csv(csv_path)
    if not rows:
        print(f"[fig7] No data at {csv_path}; skipping")
        return

    scales = sorted({int(r["scale"]) for r in rows})
    gpus   = sorted({r["gpu"] for r in rows})

    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots()
        for i, gpu in enumerate(gpus):
            teps = [float(next((r["teps"] for r in rows
                                if int(r["scale"]) == s and r["gpu"] == gpu), 0))
                    for s in scales]
            ax.plot(scales, teps, marker="o",
                    label=gpu,
                    color=list(GPU_COLORS.values())[i % len(GPU_COLORS)])
        ax.set_xlabel("RMAT scale (log₂ n_v)")
        ax.set_ylabel("TEPS")
        ax.set_title("SSSP scaling (E8)")
        ax.legend()
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1e}"))
        fig.tight_layout()
        fig.savefig(out / "fig7_scaling.pdf")
        plt.close(fig)
    print(f"[fig7] saved")


# ── Figure 8: Stress heatmap ──────────────────────────────────────────────────

def fig8_stress_heatmap(results_dir: pathlib.Path, out: pathlib.Path):
    csv_path = results_dir / "e9" / "stress.csv"
    rows = _load_csv(csv_path)
    if not rows:
        print(f"[fig8] No data at {csv_path}; skipping")
        return

    precisions = sorted({r["precision"] for r in rows})
    weight_dists = sorted({r["weight_dist"] for r in rows})

    matrix = np.zeros((len(precisions), len(weight_dists)))
    for r in rows:
        try:
            i = precisions.index(r["precision"])
            j = weight_dists.index(r["weight_dist"])
        except ValueError:
            continue
        matrix[i, j] = float(r.get("n_d_diff_pct", 0))

    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots()
        im = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0)
        ax.set_xticks(range(len(weight_dists)))
        ax.set_xticklabels(weight_dists, rotation=30, ha="right")
        ax.set_yticks(range(len(precisions)))
        ax.set_yticklabels(precisions)
        ax.set_title("Drift % under weight × precision stress (E9)")
        plt.colorbar(im, ax=ax, label="% vertices differ")
        fig.tight_layout()
        fig.savefig(out / "fig8_stress_heatmap.pdf")
        plt.close(fig)
    print(f"[fig8] saved")


# ── Main ──────────────────────────────────────────────────────────────────────

FIGURES = {
    "fig1": fig1_drift_heatmap,
    "fig2": fig2_drift_mechanism_pie,
    "fig3": fig3_correctness_split,
    "fig4": fig4_coverage_compare,
    "fig5": fig5_emission_overhead,
    "fig6": fig6_verifier_recompute,
    "fig7": fig7_scaling,
    "fig8": fig8_stress_heatmap,
}


def main():
    if not HAS_MPL:
        print("ERROR: matplotlib not installed.  Run: pip install matplotlib numpy",
              file=sys.stderr)
        sys.exit(1)

    ap = argparse.ArgumentParser(description="Generate all paper figures")
    ap.add_argument("--results-dir", default="results",
                    help="Root results directory")
    ap.add_argument("--output", default="paper/figures",
                    help="Output directory for PDF figures")
    ap.add_argument("--figs", default="all",
                    help="Comma-separated list of figures to generate, or 'all'")
    args = ap.parse_args()

    results_dir = pathlib.Path(args.results_dir)
    out_dir     = pathlib.Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    to_run = FIGURES if args.figs == "all" else {
        k: v for k, v in FIGURES.items()
        if k in args.figs.split(",")
    }

    for name, fn in to_run.items():
        try:
            fn(results_dir, out_dir)
        except Exception as exc:
            print(f"[{name}] ERROR: {exc}", file=sys.stderr)

    print(f"\nFigures written to {out_dir}/")


if __name__ == "__main__":
    main()
