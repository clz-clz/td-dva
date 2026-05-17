"""
run_multiseed.py — multi-seed multi-config experiment driver.

Wraps your existing `run_agent_pipeline` from multi_agent_v2.py. Does NOT
modify that pipeline; treats it as a black box.

For each (config, dataset, noise_type, seed):
  1. Reads the per-seed noisy file produced by gen_noisy.py
  2. Runs run_agent_pipeline concurrently (asyncio + semaphore)
  3. Writes predictions to:
        predictions_multiseed/pred_seed{S}__{config}__{dataset}__{noise}.jsonl
  4. Each line: {"tokens": [...], "gold_tags": [...], "pred_tags": [...]}

Resumable: cells whose prediction file already exists are skipped.

Usage:
    python run_multiseed.py                 # full matrix, real pipeline
    python run_multiseed.py --dummy         # offline; mocks run_agent_pipeline
    python run_multiseed.py --configs td_dva_full semantic_rag_baseline
    python run_multiseed.py --seeds 13 42 2024
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

# Silence telemetry noise from chroma / langchain
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("POSTHOG_DISABLED", "1")
os.environ.setdefault("HUGGINGFACE_HUB_DISABLE_TELEMETRY", "1")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATASETS    = ["msra", "conll2003", "wnut17"]
NOISE_TYPES = ["BT", "IF", "ATF"]
SEEDS       = [13, 42, 2024]
SAMPLE_SIZE = 200
MAX_CONCURRENCY = 20      # match ablation_runner3.py

# Configurations mirror ablation_runner3.CONFIGURATIONS, keyed by safe names.
CONFIGURATIONS: Dict[str, dict] = {
    "td_dva_full":           {"lambda_bias": 1.0,  "use_dfa": True,  "use_topology_rag": True},
    "semantic_rag_baseline": {"lambda_bias": 1.0,  "use_dfa": False, "use_topology_rag": False},
    # Uncomment to add ablations:
    # "wo_topology_rag":     {"lambda_bias": 1.0,  "use_dfa": True,  "use_topology_rag": False},
    # "wo_dfa_wash":         {"lambda_bias": 1.0,  "use_dfa": False, "use_topology_rag": True},
    # "late_fusion_bias":    {"lambda_bias": 1.35, "use_dfa": True,  "use_topology_rag": True},
}

NOISY_DIR = Path("results_multiseed")
PRED_DIR  = Path("predictions_multiseed")
PRED_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Pipeline import (real or mock)
# ---------------------------------------------------------------------------

def _import_pipeline(dummy: bool):
    """Return an async function run_agent_pipeline(tokens, dirty_tags, config)."""
    if dummy:
        async def mock_pipeline(tokens, dirty_tags, config):
            # Trivial differentiator: td_dva_full "fixes" 60% of corrupted slots
            # (by copying from gold), semantic_rag_baseline only 30%. Gold is
            # smuggled via `config['__gold__']` only in dummy mode.
            import random as _rnd
            gold = config.get("__gold__", dirty_tags)
            rate = 0.6 if config.get("use_dfa") and config.get("use_topology_rag") else 0.3
            rng = _rnd.Random(config.get("__seed__", 0))
            out = list(dirty_tags)
            for j, t in enumerate(out):
                if j < len(gold) and t != gold[j] and rng.random() < rate:
                    out[j] = gold[j]
            await asyncio.sleep(0)
            return out
        return mock_pipeline

    try:
        from multi_agent_v2 import run_agent_pipeline
        return run_agent_pipeline
    except ImportError as e:                              # pragma: no cover
        raise ImportError(
            "Could not import run_agent_pipeline from multi_agent_v2.py. "
            "Either run from the directory containing multi_agent_v2.py, "
            "or add it to PYTHONPATH."
        ) from e


# ---------------------------------------------------------------------------
# I/O paths
# ---------------------------------------------------------------------------

def _noisy_path(dataset: str, noise: str, seed: int, size: int,
                ratio: float = 0.15) -> Path:
    # Backward compat: 0.15 keeps the legacy filename so existing files
    # generated before the rate-sweep extension still resolve.
    if abs(ratio - 0.15) < 1e-9:
        return NOISY_DIR / f"noisy_seed{seed}__{noise}__{dataset}__N{size}.jsonl"
    rate_pct = int(round(ratio * 100))
    return NOISY_DIR / f"noisy_seed{seed}__{noise}__{dataset}__N{size}__r{rate_pct}.jsonl"


def _pred_path(config_name: str, dataset: str, noise: str, seed: int,
               ratio: float = 0.15) -> Path:
    # Same backward-compat scheme as _noisy_path.
    if abs(ratio - 0.15) < 1e-9:
        return PRED_DIR / f"pred_seed{seed}__{config_name}__{dataset}__{noise}.jsonl"
    rate_pct = int(round(ratio * 100))
    return PRED_DIR / f"pred_seed{seed}__{config_name}__{dataset}__{noise}__r{rate_pct}.jsonl"


def _load_noisy(p: Path) -> List[dict]:
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Per-cell async runner
# ---------------------------------------------------------------------------

async def _run_one_cell(config_name: str, config: dict,
                        dataset: str, noise: str, seed: int,
                        size: int, pipeline_fn,
                        *, max_concurrency: int, dummy: bool,
                        ratio: float = 0.15):
    pred_p = _pred_path(config_name, dataset, noise, seed, ratio)
    if pred_p.exists() and pred_p.stat().st_size > 0:
        logging.info(f"[skip] {pred_p.name}")
        return

    noisy_p = _noisy_path(dataset, noise, seed, size, ratio)
    if not noisy_p.exists():
        logging.warning(f"[miss] noisy file not found: {noisy_p}; "
                        f"run gen_noisy.py first")
        return

    rows = _load_noisy(noisy_p)
    logging.info(f"[run]  {pred_p.name}  ({len(rows)} sentences)")
    t0 = time.time()

    sem = asyncio.Semaphore(max_concurrency)
    # Buffer results IN INPUT ORDER so prediction files line up across methods.
    buffer: List[Optional[dict]] = [None] * len(rows)

    async def _process(i, row):
        async with sem:
            tokens = row["tokens"]
            gold   = row["ner_tags"]
            dirty  = row["dirty_tags"]
            cfg = dict(config)
            cfg["__dataset__"] = dataset   # so the pipeline knows the ontology
            if dummy:
                cfg["__gold__"] = gold
                cfg["__seed__"] = seed
            try:
                # Try the new signature (dataset_name kwarg); fall back to old.
                try:
                    pred = await pipeline_fn(tokens, dirty, cfg,
                                             dataset_name=dataset)
                except TypeError:
                    pred = await pipeline_fn(tokens, dirty, cfg)
            except Exception as e:                       # noqa: BLE001
                logging.error(f"[!] sentence {i} failed ({e!r}); using dirty as fallback")
                pred = list(dirty)
            # Length alignment (defensive)
            if not isinstance(pred, list):
                pred = ["O"] * len(gold)
            if len(pred) < len(gold):
                pred = list(pred) + ["O"] * (len(gold) - len(pred))
            elif len(pred) > len(gold):
                pred = pred[:len(gold)]
            buffer[i] = {"tokens": tokens, "gold_tags": gold, "pred_tags": pred}

    rate_tag = f"r{int(round(ratio*100))}"
    try:
        from tqdm.asyncio import tqdm as async_tqdm
        await async_tqdm.gather(
            *(_process(i, r) for i, r in enumerate(rows)),
            desc=f"{dataset}/{noise}/seed{seed}/{config_name}/{rate_tag}",
            leave=False,
        )
    except ImportError:
        await asyncio.gather(*(_process(i, r) for i, r in enumerate(rows)))

    # Write to disk in input order, atomically (via temp file + rename)
    tmp_p = pred_p.with_suffix(pred_p.suffix + ".tmp")
    with tmp_p.open("w", encoding="utf-8") as f_out:
        for rec in buffer:
            if rec is None:
                continue
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp_p.replace(pred_p)

    logging.info(f"[done] {pred_p.name} in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs",  nargs="+", default=list(CONFIGURATIONS.keys()))
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--noise",    nargs="+", default=NOISE_TYPES,
                    choices=NOISE_TYPES)
    ap.add_argument("--seeds",    nargs="+", type=int, default=SEEDS)
    ap.add_argument("--size",     type=int, default=SAMPLE_SIZE)
    ap.add_argument("--ratios",   nargs="+", type=float, default=[0.15],
                    help="Noise ratios to evaluate. Default: just the legacy 0.15.")
    ap.add_argument("--max-concurrency", type=int, default=MAX_CONCURRENCY)
    ap.add_argument("--dummy",    action="store_true",
                    help="Use a mock pipeline; no API calls.")
    args = ap.parse_args(argv)

    pipeline_fn = _import_pipeline(args.dummy)

    total = (len(args.configs) * len(args.datasets)
             * len(args.noise) * len(args.seeds) * len(args.ratios))
    logging.info(f"running {total} cells "
                 f"({'DUMMY' if args.dummy else 'REAL'} pipeline)")

    n_done = n_err = 0
    for cfg_name in args.configs:
        if cfg_name not in CONFIGURATIONS:
            logging.error(f"Unknown config: {cfg_name}; "
                          f"known: {list(CONFIGURATIONS)}")
            continue
        cfg = CONFIGURATIONS[cfg_name]
        for ds in args.datasets:
            for nt in args.noise:
                for s in args.seeds:
                    for r in args.ratios:
                        try:
                            asyncio.run(_run_one_cell(
                                cfg_name, cfg, ds, nt, s, args.size,
                                pipeline_fn,
                                max_concurrency=args.max_concurrency,
                                dummy=args.dummy,
                                ratio=r,
                            ))
                            n_done += 1
                        except Exception:                    # noqa: BLE001
                            logging.error("[!] cell raised")
                            traceback.print_exc(file=sys.stderr)
                            n_err += 1

    logging.info(f"summary: {n_done} ok, {n_err} failed")


if __name__ == "__main__":
    main()