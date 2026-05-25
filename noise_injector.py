"""
noise_injector.py — entity-level noise injection, matching ablation_runner4.py.

Pure functions; no I/O. Seedable per-call via a `random.Random` instance.

BT  : Boundary Truncation       — drop head or tail of a span
IF  : Internal Fragmentation    — drop an interior I- token (len>=3) else degrade
ATF : Adversarial Type Flipping — relabel the whole span to a different type

`ratio` is the fraction of ENTITIES to corrupt (matches the existing pipeline).
"""
from __future__ import annotations

import random
from typing import List, Dict


_VALID_TYPES_BY_DATASET: Dict[str, List[str]] = {
    "msra":      ["PER", "LOC", "ORG"],
    "conll2003": ["PER", "LOC", "ORG", "MISC"],
    # WNUT-17 coarsened to 4-class — see gen_noisy.py:_HF_CONFIG['wnut17']['type_remap']
    "wnut17":    ["PER", "LOC", "ORG", "MISC"],
    # Few-NERD coarsened to 4-class
    "fewnerd":   ["PER", "LOC", "ORG", "MISC"],
    # OntoNotes5 coarsened to 4-class
    "ontonotes5": ["PER", "LOC", "ORG", "MISC"],
}


def _extract_spans(tags: List[str]) -> List[Dict]:
    spans: List[Dict] = []
    cur = None
    for i, tag in enumerate(tags):
        if tag.startswith("B-"):
            if cur:
                spans.append(cur)
            cur = {"type": tag[2:], "start": i, "end": i}
        elif tag.startswith("I-") and cur and tag[2:] == cur["type"]:
            cur["end"] = i
        else:
            if cur:
                spans.append(cur)
                cur = None
    if cur:
        spans.append(cur)
    return spans


def inject_noise(tags: List[str],
                 noise_type: str,
                 ratio: float,
                 dataset: str,
                 rng: random.Random) -> List[str]:
    """Entity-level noise injection. `rng` carries the seed state."""
    if ratio == 0.0 or not tags:
        return list(tags)

    out = list(tags)
    spans = _extract_spans(tags)
    if not spans:
        return out

    valid_types = list(_VALID_TYPES_BY_DATASET.get(dataset.lower(),
                                                   ["PER", "LOC", "ORG", "MISC"]))

    n_corrupt = max(1, int(len(spans) * ratio))
    targets = rng.sample(spans, min(n_corrupt, len(spans)))

    for ent in targets:
        s, e = ent["start"], ent["end"]
        length = e - s + 1

        if noise_type == "BT":
            if length == 1:
                out[s] = "O"
            else:
                if rng.random() < 0.5:
                    out[s] = "O"          # drop head
                else:
                    out[e] = "O"          # drop tail

        elif noise_type == "IF":
            if length >= 3:
                idx = rng.randint(s + 1, e - 1)
                out[idx] = "O"
            else:
                out[s] = "O"              # degrade to BT for short spans

        elif noise_type == "ATF":
            alts = [t for t in valid_types if t != ent["type"]]
            if not alts:
                continue
            fake = rng.choice(alts)
            out[s] = f"B-{fake}"
            for i in range(s + 1, e + 1):
                out[i] = f"I-{fake}"
        else:
            raise ValueError(f"Unknown noise_type: {noise_type!r}")

    return out


if __name__ == "__main__":
    sample = ["O", "B-PER", "I-PER", "I-PER", "O", "B-LOC", "I-LOC", "O"]
    for nt in ("BT", "IF", "ATF"):
        rng = random.Random(42)
        print(f"{nt}: {inject_noise(sample, nt, 0.5, 'conll2003', rng)}")