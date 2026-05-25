"""
aggregate_seeds.py — read predictions_multiseed/, compute mean±std and
paired bootstrap, emit publication-ready LaTeX Table 1.

Usage:
    python aggregate_seeds.py                          # prints LaTeX to stdout
    python aggregate_seeds.py > new_table1.tex         # capture
    python aggregate_seeds.py --reference td_dva_full  # change bootstrap pivot
    python aggregate_seeds.py --bootstrap-iter 5000    # speed up at the cost of CI tightness
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from metrics import (compute_prf1, compute_ser,
                     paired_bootstrap_f1, significance_marker)
from run_multiseed import (CONFIGURATIONS, DATASETS, NOISE_TYPES,
                           PRED_DIR, SEEDS, _pred_path)


METHOD_LABEL = {
    "td_dva_full":              r"\textbf{LAD-DVA Full (DEER + DFA)}",
    "baseline_zero_shot":       "Zero-Shot (2025 baseline)",
    "baseline_cot_reasoning":   "CoT Reasoning",
    "baseline_self_refine":     "Self-Refine",
    "baseline_rule_only":       "Rule-Only (DFA fix)",
    "baseline_standard_prompting": "Standard Prompting (Nagar et al., 2024)",
}


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _load_pred_file(p: Path) -> Tuple[List[List[str]], List[List[str]]]:
    gold, pred = [], []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            gold.append(row["gold_tags"])
            pred.append(row["pred_tags"])
    return gold, pred


def _load_cell(method: str, dataset: str, noise: str
              ) -> List[Tuple[int, List[List[str]], List[List[str]]]]:
    """Return [(seed, gold, pred), ...] for all seeds available."""
    out = []
    for s in SEEDS:
        p = _pred_path(method, dataset, noise, s)
        if p.exists() and p.stat().st_size > 0:
            g, pr = _load_pred_file(p)
            out.append((s, g, pr))
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate_cell(method: str, dataset: str, noise: str) -> Optional[dict]:
    seeds_data = _load_cell(method, dataset, noise)
    if not seeds_data:
        return None
    f1s, ps, rs, sers = [], [], [], []
    per_seed = []
    for s, g, pr in seeds_data:
        m = compute_prf1(g, pr)
        ser = compute_ser(pr)
        f1s.append(m["f1"]); ps.append(m["precision"])
        rs.append(m["recall"]); sers.append(ser)
        per_seed.append({"seed": s, **m, "ser": ser})

    def m_sd(a):
        a = np.asarray(a)
        return float(a.mean()), (float(a.std(ddof=1)) if len(a) > 1 else 0.0)

    f1_m, f1_sd = m_sd(f1s)
    p_m,  p_sd  = m_sd(ps)
    r_m,  r_sd  = m_sd(rs)
    ser_m, ser_sd = m_sd(sers)
    return {
        "n_seeds":  len(seeds_data),
        "f1_mean":  f1_m,  "f1_std":  f1_sd,
        "p_mean":   p_m,   "p_std":   p_sd,
        "r_mean":   r_m,   "r_std":   r_sd,
        "ser_mean": ser_m, "ser_std": ser_sd,
        "per_seed": per_seed,
    }


def _bootstrap_vs(method: str, ref: str, dataset: str, noise: str,
                  n_iter: int) -> Optional[dict]:
    if method == ref:
        return None
    A = _load_cell(method, dataset, noise)
    B = _load_cell(ref,    dataset, noise)
    if not A or not B:
        return None

    # Match seeds; concat across the intersection
    A_by = {s: (g, p) for s, g, p in A}
    B_by = {s: (g, p) for s, g, p in B}
    common = sorted(set(A_by) & set(B_by))
    gold_all, a_all, b_all = [], [], []
    for s in common:
        gA, pA = A_by[s]
        gB, pB = B_by[s]
        if gA != gB:
            print(f"[warn] gold mismatch {method}/{ref} @ {dataset}/{noise}/seed{s};"
                  f" skipping", file=sys.stderr)
            continue
        gold_all.extend(gA); a_all.extend(pA); b_all.extend(pB)
    if not gold_all:
        return None
    return paired_bootstrap_f1(gold_all, a_all, b_all, n_iter=n_iter, seed=0)


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------

def _fmt_f1(cell: dict) -> str:
    f1m, f1s = cell["f1_mean"], cell["f1_std"]
    marker = significance_marker(cell.get("bootstrap_vs_ref", {}).get("p_value", 1.0))
    if cell["n_seeds"] > 1:
        return f"{f1m:.4f}{{\\scriptsize $\\pm${f1s:.4f}}}{marker}"
    return f"{f1m:.4f}{marker}"


def _fmt_ser(cell: dict) -> str:
    return f"{cell['ser_mean'] * 100:.2f}\\%"


def emit_latex(cells: Dict[Tuple[str, str, str], dict],
               methods: List[str], reference: str,
               n_iter: int, n_seeds_max: int) -> str:
    L: List[str] = []
    cols = "l" + "cc" * len(DATASETS)
    L.append(r"\begin{table*}[t]")
    L.append(r"\centering\small")
    L.append(r"\renewcommand{\arraystretch}{1.18}")
    L.append(rf"\begin{{tabular}}{{{cols}}}")
    L.append(r"\toprule")
    hdr = r"\multirow{2}{*}{\textbf{Method}}"
    for ds in DATASETS:
        hdr += rf" & \multicolumn{{2}}{{c}}{{\textbf{{{ds.upper()}}}}}"
    L.append(hdr + r" \\")
    L.append(" " + " ".join(r"& F1 & SER" for _ in DATASETS) + r" \\")
    L.append(r"\midrule")

    n_cols = 1 + 2 * len(DATASETS)
    for ni, nt in enumerate(NOISE_TYPES):
        L.append(rf"\multicolumn{{{n_cols}}}{{l}}"
                 rf"{{\textit{{{nt} noise (15\%)}}}} \\")
        for method in methods:
            label = METHOD_LABEL.get(method, method.replace("_", r"\_"))
            row = label
            for ds in DATASETS:
                cell = cells.get((method, ds, nt))
                row += (f" & {_fmt_f1(cell)} & {_fmt_ser(cell)}"
                        if cell else " & --- & ---")
            L.append(row + r" \\")
        if ni != len(NOISE_TYPES) - 1:
            L.append(r"\midrule")

    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(
        r"\caption{Multi-seed denoising performance. "
        rf"Mean $\pm$ std over up to {n_seeds_max} seeds; "
        r"$^{*}/^{**}/^{***}$: paired sentence-level bootstrap "
        rf"({n_iter} iterations) vs.\ "
        rf"\textit{{{METHOD_LABEL.get(reference, reference)}}}, "
        r"$p<0.05/0.01/0.001$. "
        r"SER averaged across seeds.}"
    )
    L.append(r"\label{tab:main_results}")
    L.append(r"\end{table*}")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="baseline_zero_shot",
                    help="Method to bootstrap against.")
    ap.add_argument("--bootstrap-iter", type=int, default=10_000)
    ap.add_argument("--methods", nargs="+", default=list(CONFIGURATIONS.keys()))
    args = ap.parse_args(argv)

    cells: Dict[Tuple[str, str, str], dict] = {}
    for m in args.methods:
        for ds in DATASETS:
            for nt in NOISE_TYPES:
                cell = _aggregate_cell(m, ds, nt)
                if cell is None:
                    continue
                cells[(m, ds, nt)] = cell

    if not cells:
        print(f"No predictions found in {PRED_DIR}/. "
              "Run gen_noisy.py and run_multiseed.py first.", file=sys.stderr)
        sys.exit(1)

    # Bootstrap pass
    for (m, ds, nt), cell in cells.items():
        if m == args.reference:
            continue
        bs = _bootstrap_vs(m, args.reference, ds, nt, args.bootstrap_iter)
        if bs is not None:
            cell["bootstrap_vs_ref"] = bs

    # Persist JSON for downstream use
    out_json = PRED_DIR / "aggregated.json"
    serial = {f"{m}|{d}|{n}": v for (m, d, n), v in cells.items()}
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(serial, f, indent=2)
    print(f"# wrote {out_json}", file=sys.stderr)

    # Console summary
    print(f"# {'Method':<28} {'Dataset':<10} {'Noise':<5} "
          f"{'F1 (mean±std, n)':<25} {'SER':<8} {'p':<10}",
          file=sys.stderr)
    for (m, d, n), c in sorted(cells.items()):
        p_str = ""
        if "bootstrap_vs_ref" in c:
            p_str = f"p={c['bootstrap_vs_ref']['p_value']:.4f}"
        print(f"# {m:<28} {d:<10} {n:<5} "
              f"{c['f1_mean']:.4f}±{c['f1_std']:.4f} (n={c['n_seeds']:>2})  "
              f"{c['ser_mean']*100:.2f}%   {p_str}",
              file=sys.stderr)

    n_seeds_max = max(c["n_seeds"] for c in cells.values())
    print(emit_latex(cells, args.methods, args.reference,
                     args.bootstrap_iter, n_seeds_max))


if __name__ == "__main__":
    main()
