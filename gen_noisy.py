"""
gen_noisy.py — generate per-seed noisy JSONL files.

For each (dataset, noise_type, seed) combination, writes:
    results_multiseed/noisy_seed{S}__{noise_type}__{dataset}__N{size}.jsonl

Each line of the output file has the field names your existing
ablation_runner3.py / multi_agent_v2.py already expect:
    {
      "tokens":    [...],     # list of token strings
      "ner_tags":  [...],     # CLEAN gold tag strings
      "dirty_tags":[...]      # tag strings with noise applied (input to pipeline)
    }

Usage:
    python gen_noisy.py                                  # default: 3 seeds, 200 samples
    python gen_noisy.py --seeds 13 42 2024 --size 200    # explicit
    python gen_noisy.py --dummy                          # tiny smoke test (offline)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from noise_injector import inject_noise


# ----------------------------------------------------------------------------
# Config — edit `OUT_DIR` if you want results elsewhere
# ----------------------------------------------------------------------------

OUT_DIR = Path("results_multiseed")
OUT_DIR.mkdir(exist_ok=True)

DATASETS    = ["msra", "conll2003", "wnut17"]
NOISE_TYPES = ["BT", "IF", "ATF"]
SEEDS       = [13, 42, 2024]
SAMPLE_SIZE = 200
NOISE_RATIO = 0.15


# ----------------------------------------------------------------------------
# HuggingFace loaders (one per dataset)
# ----------------------------------------------------------------------------

_HF_CONFIG = {
    "msra": {
        "path": "msra_ner", "split": "test",
        "tokens_field": "tokens", "tags_field": "ner_tags",
        # MSRA-NER mapping (Hub canonical):
        "id2tag": ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"],
    },
    "conll2003": {
        "path": "conll2003", "split": "test",
        "tokens_field": "tokens", "tags_field": "ner_tags",
        "id2tag": ["O", "B-PER", "I-PER", "B-ORG", "I-ORG",
                   "B-LOC", "I-LOC", "B-MISC", "I-MISC"],
    },
    "wnut17": {
        "path": "wnut_17", "split": "test",
        "tokens_field": "tokens", "tags_field": "ner_tags",
        # wnut_17 canonical mapping (per HF datasets script):
        # 0=O, 1=B-corporation, 2=I-corporation, 3=B-creative-work, 4=I-creative-work,
        # 5=B-group, 6=I-group, 7=B-location, 8=I-location, 9=B-person, 10=I-person,
        # 11=B-product, 12=I-product
        "id2tag": ["O",
                   "B-corporation",   "I-corporation",
                   "B-creative-work", "I-creative-work",
                   "B-group",         "I-group",
                   "B-location",      "I-location",
                   "B-person",        "I-person",
                   "B-product",       "I-product"],
        # Coarsening map: WNUT-17 fine types -> standard 4-class ontology.
        # We map to PER/LOC/ORG/MISC because (a) LLM backbones strongly
        # prefer these familiar tag names and collapse to O when prompted
        # with unfamiliar vocabularies, and (b) the resulting metric is
        # comparable with CoNLL-2003. This is a stated experimental choice;
        # paper §4.1 discloses the mapping.
        "type_remap": {
            "person":        "PER",
            "location":      "LOC",
            "corporation":   "ORG",
            "group":         "ORG",
            "product":       "MISC",
            "creative-work": "MISC",
        },
    },
}


def _apply_type_remap(tag: str, remap: Dict[str, str]) -> str:
    """Map a single IOB2 tag's type field through `remap`."""
    if tag == "O" or "-" not in tag:
        return tag
    prefix, t = tag.split("-", 1)
    new_t = remap.get(t, t)
    return f"{prefix}-{new_t}"


def _load_hf(name: str, size: int) -> Tuple[List[List[str]], List[List[str]]]:
    """Load the test split via HuggingFace; return (tokens, tags_str)."""
    try:
        from datasets import load_dataset
    except ImportError as e:                       # pragma: no cover
        raise ImportError("Install datasets:  pip install datasets") from e

    cfg = _HF_CONFIG[name]
    print(f"  [hf] loading {cfg['path']} / {cfg['split']}", file=sys.stderr)
    ds = load_dataset(cfg["path"], split=cfg["split"], trust_remote_code=True)

    id2tag = cfg["id2tag"]
    remap = cfg.get("type_remap")
    tokens_field, tags_field = cfg["tokens_field"], cfg["tags_field"]

    tokens, tags = [], []
    for ex in ds:
        toks = list(ex[tokens_field])
        raw = ex[tags_field]
        if not toks or len(toks) != len(raw):
            continue
        if isinstance(raw[0], int):
            tag_strs = [id2tag[i] for i in raw]
        else:
            tag_strs = [str(t) for t in raw]
        if remap:
            tag_strs = [_apply_type_remap(t, remap) for t in tag_strs]
        tokens.append(toks)
        tags.append(tag_strs)
        if size and len(tokens) >= size:
            break
    print(f"  [hf] kept {len(tokens)} sentences (cap={size})", file=sys.stderr)
    return tokens, tags


# Dummy mode loader — no network, used by tests
def _load_dummy(name: str, size: int) -> Tuple[List[List[str]], List[List[str]]]:
    base = [
        (["Barack", "Obama", "visited", "Berlin", "."],
         ["B-PER", "I-PER", "O", "B-LOC", "O"]),
        (["The", "United", "Nations", "released", "a", "report", "."],
         ["O", "B-ORG", "I-ORG", "O", "O", "O", "O"]),
        (["Tim", "Cook", "announced", "the", "Apple", "Vision", "Pro", "."],
         ["B-PER", "I-PER", "O", "O", "B-ORG", "I-ORG", "I-ORG", "O"]),
        (["She", "studied", "at", "Stanford", "University", "in", "California", "."],
         ["O", "O", "O", "B-ORG", "I-ORG", "O", "B-LOC", "O"]),
    ]
    out_t, out_g = [], []
    for i in range(size or len(base)):
        toks, tags = base[i % len(base)]
        out_t.append(list(toks))
        out_g.append(list(tags))
    return out_t, out_g


# ----------------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------------

def _path(dataset: str, noise_type: str, seed: int, size: int,
          ratio: float = 0.15) -> Path:
    # Backward compat: 0.15 keeps the legacy filename so existing files
    # (and run_multiseed.py callers that don't pass a ratio) still resolve.
    if abs(ratio - 0.15) < 1e-9:
        return OUT_DIR / f"noisy_seed{seed}__{noise_type}__{dataset}__N{size}.jsonl"
    rate_pct = int(round(ratio * 100))
    return OUT_DIR / f"noisy_seed{seed}__{noise_type}__{dataset}__N{size}__r{rate_pct}.jsonl"


# Cache the loaded dataset across seeds/noise types — load once per dataset.
_DATASET_CACHE: Dict[Tuple[str, int, bool], Tuple[List[List[str]], List[List[str]]]] = {}


def _get_dataset(dataset: str, size: int, dummy: bool):
    key = (dataset, size, dummy)
    if key not in _DATASET_CACHE:
        loader = _load_dummy if dummy else _load_hf
        _DATASET_CACHE[key] = loader(dataset, size)
    return _DATASET_CACHE[key]


def generate_one(dataset: str, noise_type: str, seed: int,
                 size: int, ratio: float,
                 *, dummy: bool, force: bool) -> Path:
    """Generate a single per-seed JSONL file. Skips if it exists (unless --force)."""
    out = _path(dataset, noise_type, seed, size, ratio)
    if out.exists() and not force:
        print(f"  [skip] {out.name}", file=sys.stderr)
        return out

    tokens, gold_strs = _get_dataset(dataset, size, dummy)

    rng = random.Random(seed * 7919 + hash(f"{dataset}|{noise_type}|r{ratio}") % 100000)
    with out.open("w", encoding="utf-8") as f:
        for toks, gold in zip(tokens, gold_strs):
            dirty = inject_noise(gold, noise_type, ratio, dataset, rng)
            row = {"tokens": toks, "ner_tags": gold, "dirty_tags": dirty}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"  [write] {out.name}  ({len(tokens)} sentences)", file=sys.stderr)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--noise",    nargs="+", default=NOISE_TYPES)
    ap.add_argument("--seeds",    nargs="+", type=int, default=SEEDS)
    ap.add_argument("--size",     type=int, default=SAMPLE_SIZE)
    ap.add_argument("--ratios",   nargs="+", type=float, default=[NOISE_RATIO],
                    help="One or more noise ratios. Default: just the legacy 0.15.")
    ap.add_argument("--dummy",    action="store_true",
                    help="No HF download; tiny in-memory data.")
    ap.add_argument("--force",    action="store_true",
                    help="Overwrite existing files.")
    args = ap.parse_args(argv)

    total = len(args.datasets) * len(args.noise) * len(args.seeds) * len(args.ratios)
    print(f"Generating {total} files into {OUT_DIR}/ "
          f"({'DUMMY' if args.dummy else 'REAL'} mode)",
          file=sys.stderr)
    n_ok = 0
    for ds in args.datasets:
        for nt in args.noise:
            for s in args.seeds:
                for r in args.ratios:
                    try:
                        generate_one(ds, nt, s, args.size, r,
                                     dummy=args.dummy, force=args.force)
                        n_ok += 1
                    except Exception as e:                       # noqa: BLE001
                        print(f"  [error] {ds}/{nt}/seed{s}/r{r}: {e}",
                              file=sys.stderr)
    print(f"\nDone: {n_ok}/{total} files OK", file=sys.stderr)


if __name__ == "__main__":
    main()