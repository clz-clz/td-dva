"""
plot_sweep.py — 5 figures, one per dataset.
Each figure: left=F1, right=SER, 6 method curves per panel.

Usage:
    python plot_sweep.py
    python plot_sweep.py --input sweep_results.csv --outdir ./plots
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.size":            9,
    "axes.titlesize":       10,
    "axes.labelsize":       9,
    "legend.fontsize":      7.5,
    "xtick.labelsize":      8,
    "ytick.labelsize":      8,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "grid.alpha":           0.2,
    "grid.linestyle":       "--",
    "lines.linewidth":      1.6,
    "lines.markersize":     5,
    "pdf.fonttype":         42,
    "ps.fonttype":          42,
})

METHOD_COLORS = {
    "lad_dva_full":                "#D62728",
    "baseline_zero_shot":         "#1F77B4",
    "baseline_cot_reasoning":     "#FF7F0E",
    "baseline_self_refine":       "#2CA02C",
    "baseline_rule_only":         "#9467BD",
    "baseline_standard_prompting":"#8C564B",
}
METHOD_MARKERS = {
    "lad_dva_full":                "o",
    "baseline_zero_shot":         "s",
    "baseline_cot_reasoning":     "^",
    "baseline_self_refine":       "D",
    "baseline_rule_only":         "v",
    "baseline_standard_prompting":"x",
}
METHOD_STYLE = {
    "lad_dva_full":                dict(linestyle="-",  linewidth=2.2),
    "baseline_zero_shot":         dict(linestyle="--", linewidth=1.5),
    "baseline_cot_reasoning":     dict(linestyle="--", linewidth=1.5),
    "baseline_self_refine":       dict(linestyle="--", linewidth=1.5),
    "baseline_rule_only":         dict(linestyle=":",  linewidth=1.8),
    "baseline_standard_prompting":dict(linestyle="-.", linewidth=1.5),
}
METHOD_LABELS = {
    "lad_dva_full":                "LAD-DVA Full (DEER + DFA)",
    "baseline_zero_shot":         "Zero-Shot",
    "baseline_cot_reasoning":     "CoT Reasoning",
    "baseline_self_refine":       "Self-Refine",
    "baseline_rule_only":         "Rule-Only (DFA)",
    "baseline_standard_prompting":"Standard Prompting",
}

DATASET_LABELS = {
    "msra":       "MSRA",
    "conll2003":  "CoNLL-2003",
    "wnut17":     "WNUT-17",
    "fewnerd":    "Few-NERD",
    "ontonotes5": "OntoNotes 5.0",
}

METHODS_ORDER = [
    "lad_dva_full",
    "baseline_zero_shot",
    "baseline_cot_reasoning",
    "baseline_self_refine",
    "baseline_rule_only",
    "baseline_standard_prompting",
]
DATASETS_ORDER = ["msra", "conll2003", "wnut17", "fewnerd", "ontonotes5"]


def load_csv(p: Path):
    rows = defaultdict(list)
    with p.open() as f:
        cleaned = (ln for ln in f if ln.strip() and not ln.lstrip().startswith("#"))
        reader = csv.DictReader(cleaned)
        for r in reader:
            if r.get("method") is None or r.get("f1_mean") is None:
                continue
            try:
                for k in ("ratio", "f1_mean", "f1_std", "ser_mean", "ser_std"):
                    r[k] = float(r[k])
                r["n_seeds"] = int(r["n_seeds"])
            except (TypeError, ValueError):
                continue
            rows[(r["method"], r["dataset"])].append(r)
    for k in rows:
        rows[k].sort(key=lambda x: x["ratio"])
    return rows


def plot_one(rows, dataset: str, out: Path):
    """Single figure: left=F1, right=SER, 6 method curves each."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    ax_f1, ax_ser = axes[0], axes[1]

    # --- F1 panel ---
    for method in METHODS_ORDER:
        key = (method, dataset)
        if key not in rows:
            continue
        series = rows[key]
        xs   = [r["ratio"]   for r in series]
        ys   = [r["f1_mean"] for r in series]
        errs = [r["f1_std"]  for r in series]
        color = METHOD_COLORS[method]
        marker = METHOD_MARKERS[method]
        style = METHOD_STYLE.get(method, {})
        ax_f1.plot(xs, ys, color=color, marker=marker, markersize=5, **style)
        ax_f1.fill_between(xs,
                            [y - e for y, e in zip(ys, errs)],
                            [y + e for y, e in zip(ys, errs)],
                            color=color, alpha=0.10, linewidth=0)

    ax_f1.set_title("F1 (Span-level)", fontweight="bold")
    ax_f1.set_xlabel("Noise rate")
    ax_f1.set_ylabel("F1")
    ax_f1.set_ylim(bottom=0.0, top=1.0)
    ax_f1.set_xticks([0.05, 0.15, 0.25, 0.35, 0.45])
    ax_f1.set_xticklabels(["5%", "15%", "25%", "35%", "45%"])
    ax_f1.grid(True, alpha=0.2, linestyle="--")

    # --- SER panel ---
    for method in METHODS_ORDER:
        key = (method, dataset)
        if key not in rows:
            continue
        series = rows[key]
        xs   = [r["ratio"]           for r in series]
        ys   = [r["ser_mean"] * 1000  for r in series]
        errs = [r["ser_std"]  * 1000  for r in series]
        color = METHOD_COLORS[method]
        marker = METHOD_MARKERS[method]
        style = METHOD_STYLE.get(method, {})
        ax_ser.plot(xs, ys, color=color, marker=marker, markersize=5, **style)
        ax_ser.fill_between(xs,
                             [max(0, y - e) for y, e in zip(ys, errs)],
                             [y + e         for y, e in zip(ys, errs)],
                             color=color, alpha=0.10, linewidth=0)

    ax_ser.set_title("SER (Syntax Error Rate ‰)", fontweight="bold")
    ax_ser.set_xlabel("Noise rate")
    ax_ser.set_ylabel("SER (‰)")
    ax_ser.set_ylim(bottom=0.0)
    ax_ser.set_xticks([0.05, 0.15, 0.25, 0.35, 0.45])
    ax_ser.set_xticklabels(["5%", "15%", "25%", "35%", "45%"])
    ax_ser.grid(True, alpha=0.2, linestyle="--")

    # --- Shared title + legend ---
    fig.suptitle(f"{DATASET_LABELS.get(dataset, dataset)} — BT Noise Sweep",
                 fontsize=12, fontweight="bold", y=1.02)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=METHOD_COLORS[m], marker=METHOD_MARKERS[m],
               linestyle=METHOD_STYLE[m]["linestyle"],
               linewidth=METHOD_STYLE[m].get("linewidth", 1.5),
               markersize=5, label=METHOD_LABELS[m])
        for m in METHODS_ORDER
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6,
               frameon=True, fontsize=7.5)

    fig.savefig(out, bbox_inches="tight")
    print(f"# wrote {out}", file=sys.stderr)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default="sweep_results.csv")
    ap.add_argument("--outdir", default=".", help="Output directory for PDFs")
    args = ap.parse_args(argv)

    inp = Path(args.input)
    if not inp.exists():
        print(f"input not found: {inp}; run sweep_results.py first", file=sys.stderr)
        sys.exit(1)
    rows = load_csv(inp)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS_ORDER:
        out = outdir / f"noise_sweep_{ds}.pdf"
        plot_one(rows, ds, out)


if __name__ == "__main__":
    main()
