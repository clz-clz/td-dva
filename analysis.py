"""F1 gap analysis: why lad_dva_full < baselines, and where."""
import json, sys
from pathlib import Path
from collections import defaultdict

from metrics import compute_prf1

PRED_DIR = Path("predictions_multiseed")
SEED = 13
DATASETS = ["msra", "conll2003", "wnut17", "fewnerd", "ontonotes5"]
NOISES  = ["BT", "IF", "ATF"]
METHODS = ["lad_dva_full", "baseline_standard_prompting", "baseline_zero_shot",
           "baseline_rule_only", "baseline_cot_reasoning", "baseline_self_refine"]

def load(method, dataset, noise):
    p = PRED_DIR / f"pred_seed{SEED}__{method}__{dataset}__{noise}.jsonl"
    if not p.exists(): return None
    return [json.loads(l) for l in open(p, encoding="utf-8")]

def get_entities(tags):
    ents, i = set(), 0
    while i < len(tags):
        if tags[i].startswith(("B-", "I-")):
            t = tags[i][2:]
            j = i + 1
            while j < len(tags) and tags[j] == f"I-{t}": j += 1
            ents.add((t, i, j)); i = j
        else: i += 1
    return ents


def analyze_one(dataset, noise):
    dva = load("lad_dva_full", dataset, noise)
    dirty = load("baseline_standard_prompting", dataset, noise)
    zero  = load("baseline_zero_shot", dataset, noise)
    rule  = load("baseline_rule_only", dataset, noise)
    if not dva or not dirty: return None

    all_g = [r["gold_tags"] for r in dva]
    all_dva = [r["pred_tags"] for r in dva]
    all_dirty = [r["pred_tags"] for r in dirty]

    m_dva   = compute_prf1(all_g, all_dva)
    m_dirty = compute_prf1(all_g, all_dirty)
    m_zero  = compute_prf1(all_g, [r["pred_tags"] for r in zero]) if zero else None
    m_rule  = compute_prf1(all_g, [r["pred_tags"] for r in rule]) if rule else None

    # Entity-level counters
    fixed = broken = both_correct = both_wrong = 0
    entity_to_o = o_to_entity = type_change = 0
    better = worse = same = 0
    gold_ent_total = 0

    for g, dva_t, dirty_t in zip(all_g, all_dva, all_dirty):
        ge = get_entities(g)
        de = get_entities(dirty_t)
        dve = get_entities(dva_t)
        gold_ent_total += len(ge)

        for e in ge:
            if e not in de and e in dve: fixed += 1
            elif e in de and e not in dve: broken += 1
            elif e in de and e in dve: both_correct += 1
            elif e not in de and e not in dve: both_wrong += 1

        for t, p, d in zip(g, dva_t, dirty_t):
            if d != "O" and p == "O": entity_to_o += 1
            if d == "O" and p != "O": o_to_entity += 1
            if d != "O" and p != "O" and d != p: type_change += 1

        f_dva = compute_prf1([g], [dva_t])["f1"]
        f_dirty = compute_prf1([g], [dirty_t])["f1"]
        if f_dva > f_dirty + 0.001: better += 1
        elif f_dva < f_dirty - 0.001: worse += 1
        else: same += 1

    n = len(all_g)
    return {
        "dataset": dataset, "noise": noise,
        "f1_dva": m_dva["f1"], "f1_dirty": m_dirty["f1"],
        "f1_zero": m_zero["f1"] if m_zero else None,
        "f1_rule": m_rule["f1"] if m_rule else None,
        "p_dva": m_dva["precision"], "r_dva": m_dva["recall"],
        "p_dirty": m_dirty["precision"], "r_dirty": m_dirty["recall"],
        "gold_ents": gold_ent_total,
        "fixed": fixed, "broken": broken,
        "both_correct": both_correct, "both_wrong": both_wrong,
        "net_entity": fixed - broken,
        "entity_to_o": entity_to_o, "o_to_entity": o_to_entity,
        "type_change": type_change,
        "better": better, "worse": worse, "same": same,
        "total_sents": n,
    }


def main():
    results = []
    for ds in DATASETS:
        for nt in NOISES:
            r = analyze_one(ds, nt)
            if r: results.append(r)

    # ---- Summary table ----
    print(f"{'Dataset':<12} {'Noise':<5} {'DVA_F1':<8} {'Dirty_F1':<8} "
          f"{'Zero_F1':<8} {'Rule_F1':<8} {'Delta_Dirty':<11} "
          f"{'NetEnt':<8} {'E->O':<6} {'O->E':<6} {'TypeCh':<7} "
          f"{'Better':<8} {'Worse':<8}")
    print("-" * 120)

    for r in sorted(results, key=lambda x: x["f1_dirty"] - x["f1_dva"], reverse=True):
        print(f"{r['dataset']:<12} {r['noise']:<5} "
              f"{r['f1_dva']:<8.4f} {r['f1_dirty']:<8.4f} "
              f"{r['f1_zero'] or 0:<8.4f} {r['f1_rule'] or 0:<8.4f} "
              f"{r['f1_dva']-r['f1_dirty']:<+11.4f} "
              f"{r['net_entity']:<+8d} {r['entity_to_o']:<6d} {r['o_to_entity']:<6d} "
              f"{r['type_change']:<7d} "
              f"{r['better']:<8d} {r['worse']:<8d}")

    # ---- Top failure patterns ----
    print("\n=== Failure Patterns ===")
    for r in sorted(results, key=lambda x: x["f1_dirty"] - x["f1_dva"], reverse=True):
        if r["f1_dva"] >= r["f1_dirty"]:
            continue
        n = r["total_sents"]
        print(f"\n{r['dataset']} | {r['noise']}: DVA={r['f1_dva']:.4f} < Dirty={r['f1_dirty']:.4f}")
        print(f"  P: {r['p_dva']:.4f} (DVA) vs {r['p_dirty']:.4f} (Dirty)")
        print(f"  R: {r['r_dva']:.4f} (DVA) vs {r['r_dirty']:.4f} (Dirty)")
        print(f"  Entity changes: fixed={r['fixed']} broken={r['broken']} (net={r['net_entity']:+d})")
        print(f"  Tag changes: entity->O={r['entity_to_o']} O->entity={r['o_to_entity']} type_change={r['type_change']}")
        print(f"  Sentences: better={r['better']} worse={r['worse']} same={r['same']} "
              f"({r['worse']/n*100:.1f}% made worse by DVA)")

    # ---- Best strategy per cell ----
    print("\n=== Optimal Strategy Per Cell ===")
    for r in sorted(results, key=lambda x: (x["dataset"], x["noise"])):
        best = max(
            ("lad_dva_full", r["f1_dva"]),
            ("dirty (no-op)", r["f1_dirty"]),
            ("zero_shot", r["f1_zero"] or 0),
            ("rule_only", r["f1_rule"] or 0),
            key=lambda x: x[1]
        )
        gap = best[1] - r["f1_dva"]
        marker = f"<<< GAP={gap:.4f}" if gap > 0.005 else "(already optimal)"
        print(f"  {r['dataset']:<12} {r['noise']:<5}  best={best[0]:<18} "
              f"{best[1]:.4f}  (DVA={r['f1_dva']:.4f})  {marker}")


if __name__ == "__main__":
    main()
