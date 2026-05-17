"""
sweep_results.py — aggregate F1 and SER across noise rates for plotting.

Reads predictions_multiseed/pred_seed{S}__{config}__{dataset}__{NT}[__r{rate}].jsonl
files produced by `run_multiseed.py --ratios ...`, and writes a tidy CSV
suitable for plotting:

    method,dataset,noise,ratio,f1_mean,f1_std,ser_mean,ser_std,n_seeds

Usage:
    python sweep_results.py --noise BT \
        --ratios 0.05 0.15 0.25 0.35 0.45 \
        > sweep_results.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List

import numpy as np

from metrics import compute_prf1, compute_ser
from run_multiseed import (CONFIGURATIONS, DATASETS, SEEDS,
                            _pred_path, PRED_DIR)


def _load_cell(method: str, dataset: str, noise: str, ratio: float):
    """Return [(seed, gold, pred), ...] for all seeds available at this ratio."""
    out = []
    for s in SEEDS:
        p = _pred_path(method, dataset, noise, s, ratio)
        if p.exists() and p.stat().st_size > 0:
            gold, pred = [], []
            with p.open(encoding="utf-8") as f:
                for line in f:
                    import json
                    row = json.loads(line)
                    gold.append(row["gold_tags"])
                    pred.append(row["pred_tags"])
            out.append((s, gold, pred))
    return out


def _aggregate(method, dataset, noise, ratio):
    cells = _load_cell(method, dataset, noise, ratio)
    if not cells:
        return None
    f1s, sers = [], []
    for _, gold, pred in cells:
        f1s.append(compute_prf1(gold, pred)["f1"])
        sers.append(compute_ser(pred))
    f1, ser = np.asarray(f1s), np.asarray(sers)
    return {
        "n_seeds":  len(cells),
        "f1_mean":  float(f1.mean()),
        "f1_std":   float(f1.std(ddof=1)) if len(f1) > 1 else 0.0,
        "ser_mean": float(ser.mean()),
        "ser_std":  float(ser.std(ddof=1)) if len(ser) > 1 else 0.0,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods",  nargs="+", default=list(CONFIGURATIONS))
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--noise",    default="BT")
    ap.add_argument("--ratios",   nargs="+", type=float,
                    default=[0.05, 0.15, 0.25, 0.35, 0.45])
    args = ap.parse_args(argv)

    w = csv.writer(sys.stdout)
    w.writerow(["method", "dataset", "noise", "ratio",
                "f1_mean", "f1_std", "ser_mean", "ser_std", "n_seeds"])

    n_rows = 0
    for m in args.methods:
        for d in args.datasets:
            for r in args.ratios:
                agg = _aggregate(m, d, args.noise, r)
                if agg is None:
                    print(f"  [warn] missing cell: {m}/{d}/{args.noise}/r{r}",
                          file=sys.stderr)
                    continue
                w.writerow([m, d, args.noise, f"{r:.2f}",
                            f"{agg['f1_mean']:.6f}", f"{agg['f1_std']:.6f}",
                            f"{agg['ser_mean']:.6f}", f"{agg['ser_std']:.6f}",
                            agg["n_seeds"]])
                n_rows += 1
    print(f"# wrote {n_rows} rows", file=sys.stderr)


if __name__ == "__main__":
    main()