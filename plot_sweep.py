"""
plot_sweep.py — produce a publication-ready 2-panel figure of the noise sweep.

Reads `sweep_results.csv` (from sweep_results.py) and writes a PDF figure.
Left panel:  F1 vs noise ratio, mean ± std bands.
Right panel: SER vs noise ratio, mean ± std bands.
Solid lines = TD-DVA Full, dashed lines = Semantic RAG Baseline.
Three colors = three datasets.

Usage:
    python plot_sweep.py                          # writes f1_ser_vs_noise.pdf
    python plot_sweep.py --input sweep_results.csv --output paper_fig2.pdf
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl


# --- styling ---------------------------------------------------------------
mpl.rcParams.update({
    "font.size":            10,
    "axes.titlesize":       11,
    "axes.labelsize":       10,
    "legend.fontsize":      8.5,
    "xtick.labelsize":      9,
    "ytick.labelsize":      9,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.grid":            True,
    "grid.alpha":           0.25,
    "grid.linestyle":       "--",
    "lines.linewidth":      1.8,
    "lines.markersize":     5,
    "pdf.fonttype":         42,         # editable in Illustrator
    "ps.fonttype":          42,
})

# Color-blind friendly palette (Wong 2011)
DATASET_COLORS = {
    "msra":      "#0072B2",  # blue
    "conll2003": "#E69F00",  # orange
    "wnut17":    "#009E73",  # green
}
DATASET_LABELS = {
    "msra":      "MSRA",
    "conll2003": "CoNLL-2003",
    "wnut17":    "WNUT-17",
}
METHOD_STYLE = {
    "td_dva_full":           dict(linestyle="-",  marker="o"),
    "semantic_rag_baseline": dict(linestyle="--", marker="s"),
}
METHOD_LABELS = {
    "td_dva_full":           "TD-DVA Full",
    "semantic_rag_baseline": "Semantic RAG",
}


def load_csv(p: Path):
    """Return dict: rows[(method, dataset)] -> sorted list of dicts.
    Skips blank lines and lines starting with '#'."""
    rows = defaultdict(list)
    with p.open() as f:
        # Filter out comment/blank lines before passing to DictReader
        cleaned = (ln for ln in f if ln.strip() and not ln.lstrip().startswith("#"))
        reader = csv.DictReader(cleaned)
        for r in reader:
            # Skip rows without expected fields (e.g. residual artifacts)
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


def plot(rows, noise_label: str, out: Path):
    fig, (ax_f1, ax_ser) = plt.subplots(1, 2, figsize=(9.0, 3.6),
                                          constrained_layout=True)

    # F1 panel
    for (method, ds), series in rows.items():
        xs   = [r["ratio"]   for r in series]
        ys   = [r["f1_mean"] for r in series]
        errs = [r["f1_std"]  for r in series]
        color = DATASET_COLORS.get(ds, "#666666")
        style = METHOD_STYLE.get(method, dict(linestyle="-", marker="x"))
        ax_f1.plot(xs, ys, color=color, **style)
        ax_f1.fill_between(xs,
                            [y - e for y, e in zip(ys, errs)],
                            [y + e for y, e in zip(ys, errs)],
                            color=color, alpha=0.12, linewidth=0)
    ax_f1.set_xlabel("Noise rate")
    ax_f1.set_ylabel("Span-level F1")
    ax_f1.set_title(f"(a) F1 vs noise rate — {noise_label}")
    ax_f1.set_ylim(bottom=0.0)

    # SER panel (in percent)
    for (method, ds), series in rows.items():
        xs   = [r["ratio"]              for r in series]
        ys   = [r["ser_mean"] * 100     for r in series]
        errs = [r["ser_std"]  * 100     for r in series]
        color = DATASET_COLORS.get(ds, "#666666")
        style = METHOD_STYLE.get(method, dict(linestyle="-", marker="x"))
        ax_ser.plot(xs, ys, color=color, **style)
        ax_ser.fill_between(xs,
                             [max(0, y - e) for y, e in zip(ys, errs)],
                             [y + e         for y, e in zip(ys, errs)],
                             color=color, alpha=0.12, linewidth=0)
    ax_ser.set_xlabel("Noise rate")
    ax_ser.set_ylabel("Syntax Error Rate (%)")
    ax_ser.set_title(f"(b) SER vs noise rate — {noise_label}")
    ax_ser.set_ylim(bottom=0.0)

    # Custom legend: separate handles for color (dataset) and linestyle (method)
    from matplotlib.lines import Line2D
    dataset_handles = [Line2D([0], [0], color=DATASET_COLORS[d], lw=2,
                              label=DATASET_LABELS[d])
                       for d in DATASET_COLORS
                       if any(k[1] == d for k in rows)]
    method_handles  = [Line2D([0], [0], color="black",
                              linestyle=METHOD_STYLE[m]["linestyle"],
                              marker=METHOD_STYLE[m]["marker"],
                              label=METHOD_LABELS[m])
                       for m in METHOD_STYLE
                       if any(k[0] == m for k in rows)]
    # Place dataset legend on the F1 panel, method legend on the SER panel
    ax_f1.legend(handles=dataset_handles,  loc="lower left", frameon=False,
                 title="Dataset")
    ax_ser.legend(handles=method_handles, loc="upper left", frameon=False,
                  title="Method")

    fig.savefig(out, bbox_inches="tight")
    print(f"# wrote {out}", file=sys.stderr)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default="sweep_results.csv")
    ap.add_argument("--output", default="f1_ser_vs_noise.pdf")
    ap.add_argument("--noise-label", default="Boundary Truncation (BT)",
                    help="Human-readable noise label for the figure title.")
    args = ap.parse_args(argv)

    inp = Path(args.input)
    if not inp.exists():
        print(f"input not found: {inp}; run sweep_results.py first",
              file=sys.stderr)
        sys.exit(1)
    rows = load_csv(inp)
    plot(rows, args.noise_label, Path(args.output))


if __name__ == "__main__":
    main()