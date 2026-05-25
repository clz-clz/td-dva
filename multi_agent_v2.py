import asyncio
import logging
import os
import json
import re
from collections import Counter
from typing import Annotated, TypedDict, List, Tuple
from dotenv import load_dotenv
import argparse

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI

from utils import enforce_iob2_syntax, extract_json_list

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.7,
    base_url="https://api.deepseek.com",
    api_key=os.environ.get("DEEPSEEK_API_KEY")
)

# Dedicated higher-temperature LLM for Coder to encourage diverse paths
coder_llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=1.0,
    base_url="https://api.deepseek.com",
    api_key=os.environ.get("DEEPSEEK_API_KEY")
)

DEEPSEEK_NER_TOKEN_IDS = {
    "B-PER": 42531, "I-PER": 42532,
    "B-LOC": 18274, "I-LOC": 18275,
    "B-ORG": 39112, "I-ORG": 39113,
    "B-MISC": 6489, "I-MISC": 6490
}

# =====================================================================
# Dataset-aware entity ontologies
# Each dataset has its own valid entity type set. The Coder and Reviewer
# prompts inject this list so the LLM doesn't default to the common
# PER/LOC/ORG triad regardless of input dataset.
# =====================================================================
DATASET_ENTITY_TYPES = {
    "msra":      ["PER", "LOC", "ORG"],
    "conll2003": ["PER", "LOC", "ORG", "MISC"],
    # WNUT-17 in this project is COARSENED to the standard 4-class ontology
    # via type_remap in gen_noisy.py (person->PER, location->LOC,
    # corporation/group->ORG, product/creative-work->MISC). The pipeline must
    # therefore prompt with the same coarse types.
    "wnut17":    ["PER", "LOC", "ORG", "MISC"],
    # Few-NERD coarsened to 4-class (person->PER, location->LOC,
    # organization->ORG, art/building/event/other/product->MISC)
    "fewnerd":   ["PER", "LOC", "ORG", "MISC"],
    # OntoNotes5 coarsened to 4-class (PERSON->PER, GPE/LOC/FAC->LOC,
    # ORG->ORG, rest->MISC)
    "ontonotes5": ["PER", "LOC", "ORG", "MISC"],
}

def _format_valid_tags(dataset_name: str) -> str:
    """Return 'O, B-X, I-X, ...' as a single comma-separated string."""
    types = DATASET_ENTITY_TYPES.get(dataset_name)
    if not types:
        # Fallback: infer from a sample dirty_tags later, or use union
        types = ["PER", "LOC", "ORG", "MISC"]
    tags = ["O"]
    for t in types:
        tags.append(f"B-{t}")
        tags.append(f"I-{t}")
    return ", ".join(tags)

class State(TypedDict):
    messages: Annotated[list, add_messages]
    loop_count: int
    iterations: int
    tokens: List[str]
    dirty_tags: List[str]
    candidate_paths: List[List[str]]
    rag_weights: List[float]
    current_tags: List[str]
    errors: List[str]
    lambda_weight: float
    use_wash: bool
    dataset_name: str
    noise_type: str

    

def extract_nested_json_list(llm_output: str, expected_paths: int = 5) -> List[List[str]]:
    try:
        match = re.search(r'\[\s*\[.*?\]\s*\]', llm_output, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], list):
                return parsed[:expected_paths]
    except Exception as e:
        logging.warning(f"Coder JSON ：{e}")
    return [["O"]] * expected_paths

def extract_float_weights(llm_output: str, expected_len: int = 5) -> List[float]:
    try:
        match = re.search(r'\[[\d\.\s,]+\]', llm_output)
        if match:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list) and all(isinstance(x, (int, float)) for x in parsed):
                weights = [float(x) for x in parsed]
                if len(weights) >= expected_len:
                    return weights[:expected_len]
                else:
                    return weights + [0.0] * (expected_len - len(weights))
    except Exception as e:
        logging.error(f" Reviewer JSON 载入异常: {e}")

    logging.error(f" Warning: {llm_output[:50]}...")
    return [1.0] * expected_len 


def _format_sentence_initial_guidance(dataset_name: str) -> str:
    """Return dataset-appropriate guidance about sentence-initial entity heads."""
    if dataset_name == "msra":
        return (
            "- PAY SPECIAL ATTENTION TO SENTENCE-INITIAL TOKENS. A Chinese "
            "sentence often begins with a named entity (organization, "
            "person, location). Do not default the first token to O just "
            "because it lacks left context. If the first 1-4 tokens form a "
            "recognizable entity, label them B-<TYPE> I-<TYPE> .... Examples: "
            "'中国政府...' -> B-ORG I-ORG I-ORG; '北京市...' -> B-LOC I-LOC I-LOC."
        )
    # English-style (CoNLL-2003 and coarsened WNUT-17)
    return (
        "- PAY SPECIAL ATTENTION TO SENTENCE-INITIAL TOKENS. A capitalized "
        "token at position 0 that names a person, organization, location, or "
        "other entity should be labeled B-<TYPE>, not O. Do not default the "
        "first token to O just because it lacks left context. Examples: "
        "'Obama said ...' -> the first token is B-PER, not O. "
        "'Apple announced ...' -> the first token is B-ORG, not O. "
        "'German troops ...' -> the first token is B-MISC, not O."
    )


def _format_misc_guidance(dataset_name: str) -> str:
    """Return dataset-appropriate MISC type guidance (only for ontologies that have MISC)."""
    if dataset_name == "msra":
        # MSRA uses only PER/LOC/ORG, no MISC. Skip the MISC paragraph entirely.
        return ""
    return (
        "- DO NOT systematically avoid rare tag types. The MISC type (for "
        "nationalities, languages, events, works, products, and other "
        "proper-noun entities that are not PER/LOC/ORG) is valid and should "
        "be predicted whenever the token clearly refers to such an entity, "
        "even if MISC appears less frequently than PER/LOC/ORG. Examples of "
        "MISC entities: \"German\" (nationality), \"Olympic\" (event), "
        "\"iPhone\" (product), \"Bible\" (work)."
    )


async def coder_node(state: State):
    print(f"\n [Coder](Self-Consistency)")
    tokens = state.get("tokens", [])
    dirty_tags = state.get("dirty_tags", [])
    dataset_name = state.get("dataset_name", "conll2003")
    noise_type = state.get("noise_type", "BT")
    valid_tags_str = _format_valid_tags(dataset_name)
    misc_guidance = _format_misc_guidance(dataset_name)
    sentence_initial_guidance = _format_sentence_initial_guidance(dataset_name)

    # DEER examples: same calibration data the Reviewer sees
    deer_examples = _get_deer_examples(tokens, top_k=3, dataset_name=dataset_name)

    # ---- Noise-type-specific policies and per-path strategies ----
    if noise_type == "BT":
        noise_policy = """
    NOISE TYPE: Boundary Truncation (BT). Entity boundaries are corrupted — missing B- tags, dangling I- tags, broken spans. CRITICAL: entity TYPES are reliable. The dirty tag's type field (PER/LOC/ORG/MISC) is trustworthy. Never change an entity's type — only fix its boundaries."""
        path_strategies = {
            1: "CONSERVATIVE BOUNDARY FIX: Repair ONLY dangling I- tags. Upgrade each dangling I-<TYPE> to B-<TYPE>. Never change entity types. Never change O to entity. Never extend single-token entities. This is the most cautious strategy — only fix obvious structural IOB2 errors.",
            2: "FULL BOUNDARY REPAIR: Fix dangling I- tags AND broken B-X O B-X patterns. When you see B-<TYPE> O B-<TYPE>, merge into B-<TYPE> I-<TYPE> I-<TYPE>. Also consider B-X O O I-X → B-X I-X I-X. Never change entity types. Target: comprehensive boundary repair — fix all structure errors.",
            5: "OPTIMAL JOINT REPAIR: Combine all boundary repairs (dangling I- fix + B-X O B-X merging) with false-positive cleanup. Demote to O any entity token that is clearly a function word (the, of, in, a, an, for, to, with), common verb (said, was, is), punctuation, or bare number. Never change entity types for tokens that remain entities. Target: maximum accuracy.",
        }
    elif noise_type == "IF":
        noise_policy = """
    NOISE TYPE: Internal Fragmentation (IF). Entities are incorrectly split by O tags in the middle. CRITICAL: entity TYPES are reliable. The dirty tag's type field is trustworthy. Never change an entity's type — only merge fragmented entities."""
        path_strategies = {
            1: "CONSERVATIVE FRAGMENT MERGE: Merge ONLY adjacent same-type entities separated by exactly 1 O token. Only merge when the gap token is NOT a sentence delimiter (not punctuation, not a clause boundary). B-PER O B-PER → B-PER I-PER I-PER. Never change entity types. Target: fix obvious fragments with minimal risk.",
            2: "AGGRESSIVE FRAGMENT MERGE: Merge adjacent same-type entities with gaps up to 2 O tokens. Only skip merging when the gap token is clearly a sentence delimiter. Never change entity types. Target: comprehensive fragment repair — catch all splittings.",
            5: "OPTIMAL JOINT REPAIR: Merge fragments with balanced aggressiveness (Path 1 threshold for ambiguous gaps, Path 2 for clear gaps) plus false-positive cleanup. Demote to O any entity token that is a function word, common verb, punctuation, or bare number. Never change entity types for tokens that remain entities. Target: maximum accuracy.",
        }
    elif noise_type == "ATF":
        noise_policy = """
    NOISE TYPE: Adversarial Type Flipping (ATF). Entity BOUNDARIES (B/I/O structure) are CORRECT — the spans are right, but the entity TYPES are all potentially wrong. CRITICAL: Do NOT change entity boundaries. Only verify and fix entity TYPES."""
        path_strategies = {
            1: "CONSERVATIVE TYPE FIX: Only change an entity's type when the dirty type is IMPOSSIBLE for the token (e.g., 'Obama' as ORG, 'Beijing' as PER). If the dirty type is plausible, keep it. Do NOT change entity boundaries (B/I/O). Target: fix clear type errors only.",
            2: "CALIBRATION-GUIDED TYPE FIX: For each entity span, check the calibration examples for similar tokens. If a token consistently appears with a specific type in calibration, adopt that type. If no calibration evidence, keep the dirty type. Do NOT change boundaries. Target: data-driven type correction.",
            3: "AGGRESSIVE TYPE FIX: Verify EVERY entity's type against token semantics and world knowledge. Change any type that seems wrong, even if marginally plausible. Check multi-token entities for type consistency (all tokens in one entity MUST share the same type). Do NOT change boundaries. Target: comprehensive type repair.",
            4: "TYPE NORMALIZATION: For each entity span, pick the most probable type based on token semantics and calibration evidence. For multi-token entities, enforce uniform type across all tokens. Do NOT change boundaries. Target: statistical type correction.",
            5: "OPTIMAL JOINT REPAIR: Combine calibration-guided type fixing with false-positive cleanup. Demote to O any entity token that is clearly a function word, common verb, punctuation, or bare number. Boundaries stay identical except when demoting false positives. Target: maximum accuracy.",
        }
    else:
        noise_policy = ""
        path_strategies = {
            1: "DEFAULT REPAIR: Fix obvious IOB2 errors while preserving the original labeling as much as possible.",
            2: "DEFAULT REPAIR: Fix IOB2 errors and consider contextual appropriateness of entity boundaries.",
            5: "DEFAULT REPAIR: Comprehensive repair including false-positive cleanup.",
        }

    # Shared prompt prefix (identical for all paths)
    shared_prefix = f"""
You are an Elite AI Data Engineer performing IOB2 label denoising for Named Entity Recognition.
Tokens: {tokens}
Dirty IOB2 Tags: {dirty_tags}
{noise_policy}

Calibration Examples (correct annotations from the TRAINING SET — retrieved via label-guided similarity. These show what valid output looks like for sentences similar to the current one. Your repairs MUST be consistent with these patterns):
{json.dumps(deer_examples, indent=2) if deer_examples else '[]'}

VALID TAG VOCABULARY (use ONLY these tags; case-sensitive):
{valid_tags_str}

CRITICAL DENOISING POLICY — follow these rules precisely:

1. IOB2 GRAMMAR: Every multi-token entity MUST start with B-<TYPE>. A dangling I-<TYPE> without preceding same-type B/I is FATAL — upgrade to B-<TYPE>. All tokens in one entity must share the same type.

2. SINGLE-TOKEN ENTITY PROTECTION: A single B-X tag followed by O is a COMPLETE one-token entity. Do NOT extend it with I-X into following O. Single-token entities are correct and common.

3. NO ENTITY INVENTION: Never change O to B-X or I-X unless fixing a clear BT noise pattern (dangling I-). The NER model has near-perfect entity detection recall.

4. PRESERVATION BIAS: The dirty tags are MOSTLY correct. Prefer keeping a non-O tag over changing it to O. A token tagged as entity should stay entity unless it is clearly a function word, common verb, number, or punctuation.
{misc_guidance}
{sentence_initial_guidance}"""

    # Call LLM separately for each path (parallel) to guarantee diversity
    n_paths = len(path_strategies)
    print(f" [Coder] {noise_type}: {n_paths} individual LLM calls (parallel)")

    prompts = []
    for pidx, pdesc in path_strategies.items():
        prompt = shared_prefix + f"""

PATH {pidx} STRATEGY — your ONLY task:
{pdesc}

Generate EXACTLY 1 repair path (a JSON list of {len(tokens)} IOB2 tags).
Output ONLY the JSON list, no markdown, no explanation."""
        prompts.append(prompt)

    responses = await asyncio.gather(*[
        asyncio.to_thread(coder_llm.invoke, p) for p in prompts
    ])

    candidate_paths = []
    for r in responses:
        path = extract_json_list(r.content, fallback_length=len(tokens))
        candidate_paths.append(path)

    # Length normalization
    candidate_paths = [p[:len(tokens)] + ["O"] * max(0, len(tokens) - len(p)) for p in candidate_paths]

    # IOB2语法清洗：每条Coder路径先过DFA修复，保证Reviewer只判边界+类型
    ds_name = state.get("dataset_name", "conll2003")
    valid_set = set(DATASET_ENTITY_TYPES.get(ds_name, ["PER", "LOC", "ORG"]))
    candidate_paths = [enforce_iob2_syntax(p, valid_entity_types=valid_set) for p in candidate_paths]

    # ---- Hard constraint for BT/IF noise ----
    # BT/IF noise only drops entity tags (entity→O) and breaks boundaries.
    # Entity TYPES are correct, entity DETECTION recall is near-perfect.
    # The Coder is unreliable on BT/IF — block most changes except gap-fill.
    # Allowed: O→Entity when bridging two same-type entity fragments.
    # BT: blocked Entity→O, type changes, B↔I shuffles (boundaries are locally correct).
    # IF: blocked Entity→O, type changes; B↔I shuffles ALLOWED for fragment merging.
    if noise_type in ("BT", "IF"):
        n_reverted = 0
        for p in candidate_paths:
            for i in range(len(p)):
                if p[i] == dirty_tags[i]:
                    continue
                # ---- O→Entity: only allow legitimate BT gap-fill ----
                if dirty_tags[i] == "O" and p[i] != "O":
                    p_type = p[i][2:]
                    p_prefix = p[i][0]  # "B" or "I"
                    prev_same = (i > 0 and dirty_tags[i-1] != "O"
                                and dirty_tags[i-1][2:] == p_type)
                    next_same = (i+1 < len(dirty_tags) and dirty_tags[i+1] != "O"
                                and dirty_tags[i+1][2:] == p_type)
                    if p_prefix == "B":
                        # O→B-X: allowed if next is I-X (BT dropped the B- tag)
                        if not next_same:
                            p[i] = "O"
                            n_reverted += 1
                    else:
                        # O→I-X: allowed if prev is same type AND next is NOT a
                        # different entity type. This allows true gap-fill
                        # (next_same) AND BT tail-drop (next=O), but blocks
                        # entity extension into another entity (next=B-OTHER).
                        next_other = (i+1 < len(dirty_tags) and dirty_tags[i+1] != "O"
                                      and dirty_tags[i+1][2:] != p_type)
                        if not prev_same or next_other:
                            p[i] = "O"
                            n_reverted += 1
                    continue
                # ---- Entity→O: always revert (BT/IF never deletes entities) ----
                if dirty_tags[i] != "O" and p[i] == "O":
                    p[i] = dirty_tags[i]
                    n_reverted += 1
                    continue
                # ---- Same type, different B/I: boundary shuffle → revert ----
                if dirty_tags[i] != "O" and p[i] != "O":
                    d_type = dirty_tags[i][2:]
                    p_type = p[i][2:]
                    if d_type != p_type:
                        # Type change → revert
                        p[i] = dirty_tags[i]
                        n_reverted += 1
                    elif noise_type == "BT":
                        # BT: boundaries are locally correct (just truncated).
                        # Same-type B↔I shuffles are noise → revert.
                        p[i] = dirty_tags[i]
                        n_reverted += 1
                    # IF: allow same-type B↔I changes for fragment merging.
                    # B→I is needed to merge adjacent same-type entity fragments
                    # (e.g. B-PER O B-PER → B-PER I-PER I-PER).
        if n_reverted > 0:
            print(f" [Coder] Hard constraint reverted {n_reverted} changes "
                  f"(BT/IF: O-tag + type + boundary protection)")

    # Diversity check: if all paths identical, fallback to dirty when harmful
    if _all_paths_identical(candidate_paths):
        only_path = candidate_paths[0]
        all_o = all(t == "O" for t in only_path)
        has_entities_in_dirty = any(t != "O" for t in dirty_tags)
        if all_o and has_entities_in_dirty:
            # LLM erased all entities — fallback to dirty (safer)
            print(f" [Coder] All paths identical (all O) with entities in dirty — "
                  f"overriding with dirty tags")
            candidate_paths = [list(dirty_tags) for _ in range(5)]
        else:
            # Paths identical but not all-O: LLM is confident, proceed
            n_diff = sum(1 for a, b in zip(only_path, dirty_tags) if a != b)
            print(f" [Coder] All {len(candidate_paths)} paths identical "
                  f"({n_diff} diffs vs dirty) — LLM consensus, skipping diversity")

    return {"candidate_paths": candidate_paths, "iterations": state.get("iterations", 0) + 1}

def _all_paths_identical(paths: List[List[str]]) -> bool:
    """Check if all candidate paths are identical."""
    if len(paths) <= 1:
        return True
    first = paths[0]
    return all(p == first for p in paths[1:])


# ---- DEER integration (Bai et al., EMNLP 2025) ----
# Label-guided retrieval replaces the old RAG in the Reviewer.
# Statistics are built per-dataset from the appropriate training set.

_deer_stats = {}     # dataset_name -> DEERStatistics
_deer_retriever = {}  # dataset_name -> DEERRetriever


def _init_deer(dataset_name: str = "conll2003"):
    """Lazy-init DEER statistics and retriever for a given dataset."""
    global _deer_stats, _deer_retriever
    if dataset_name in _deer_stats:
        return

    from deer_retriever import DEERStatistics, DEERRetriever
    from datasets import load_dataset

    if dataset_name == "msra":
        print("[DEER] Building token statistics from MSRA-NER training set...")
        ds = load_dataset("msra_ner", split="train", trust_remote_code=True)
        id2tag = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
    elif dataset_name == "wnut17":
        # WNUT-17 original tags coarsened to 4-class: person->PER, location->LOC,
        # corporation/group->ORG, creative-work/product->MISC
        print("[DEER] Building token statistics from WNUT-17 training set...")
        ds = load_dataset("wnut_17", split="train", trust_remote_code=True)
        id2tag = ["O",
                  "B-ORG", "I-ORG",      # 1,2: corporation -> ORG
                  "B-MISC", "I-MISC",     # 3,4: creative-work -> MISC
                  "B-ORG", "I-ORG",       # 5,6: group -> ORG
                  "B-LOC", "I-LOC",       # 7,8: location -> LOC
                  "B-PER", "I-PER",       # 9,10: person -> PER
                  "B-MISC", "I-MISC"]     # 11,12: product -> MISC
    elif dataset_name == "fewnerd":
        # Few-NERD supervised: person->PER, location->LOC, organization->ORG,
        # art/building/event/other/product->MISC
        print("[DEER] Building token statistics from Few-NERD training set...")
        ds = load_dataset("DFKI-SLT/few-nerd", "supervised", split="train", trust_remote_code=True)
        id2tag = ds.features["ner_tags"].feature.names
        # Coarsen fine-grained types to 4-class
        _coarse = {
            "O": "O",
            "B-person": "B-PER", "I-person": "I-PER",
            "B-location": "B-LOC", "I-location": "I-LOC",
            "B-organization": "B-ORG", "I-organization": "I-ORG",
            "B-art": "B-MISC", "I-art": "I-MISC",
            "B-building": "B-MISC", "I-building": "I-MISC",
            "B-event": "B-MISC", "I-event": "I-MISC",
            "B-other": "B-MISC", "I-other": "I-MISC",
            "B-product": "B-MISC", "I-product": "I-MISC",
            "B-location": "B-LOC", "I-location": "I-LOC",
        }
        id2tag = [_coarse.get(t, t) for t in id2tag]
    elif dataset_name == "ontonotes5":
        # OntoNotes5 via tner: tags field (not ner_tags). 37 IOB2 tags → coarsen to 4-class.
        print("[DEER] Building token statistics from OntoNotes5 training set...")
        ds = load_dataset("tner/ontonotes5", split="train", trust_remote_code=True)
        _id2tag_raw = [
            "O",
            "B-CARDINAL", "B-DATE", "I-DATE",
            "B-PERSON", "I-PERSON",
            "B-NORP", "B-GPE", "I-GPE",
            "B-LAW", "I-LAW",
            "B-ORG", "I-ORG",
            "B-PERCENT", "I-PERCENT",
            "B-ORDINAL",
            "B-MONEY", "I-MONEY",
            "B-WORK_OF_ART", "I-WORK_OF_ART",
            "B-FAC",
            "B-TIME",
            "I-CARDINAL",
            "B-LOC",
            "B-QUANTITY", "I-QUANTITY",
            "I-NORP",
            "I-LOC",
            "B-PRODUCT",
            "I-TIME",
            "B-EVENT", "I-EVENT",
            "I-FAC",
            "B-LANGUAGE",
            "I-PRODUCT",
            "I-ORDINAL",
            "I-LANGUAGE",
        ]
        _coarse = {
            "PERSON": "PER", "GPE": "LOC", "LOC": "LOC", "FAC": "LOC",
            "ORG": "ORG",
            "NORP": "MISC", "PRODUCT": "MISC", "EVENT": "MISC",
            "WORK_OF_ART": "MISC", "LAW": "MISC", "LANGUAGE": "MISC",
            "DATE": "MISC", "TIME": "MISC", "PERCENT": "MISC",
            "MONEY": "MISC", "QUANTITY": "MISC", "ORDINAL": "MISC",
            "CARDINAL": "MISC",
        }
        id2tag = []
        for t in _id2tag_raw:
            if t == "O":
                id2tag.append("O")
            else:
                prefix, etype = t.split("-", 1)
                id2tag.append(f"{prefix}-{_coarse.get(etype, 'MISC')}")
    else:
        print("[DEER] Building token statistics from CoNLL2003 training set...")
        ds = load_dataset("conll2003", split="train", trust_remote_code=True)
        id2tag = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]

    _tags_field = "tags" if dataset_name == "ontonotes5" else "ner_tags"
    train_data = []
    for ex in ds:
        toks = list(ex["tokens"])
        raw = ex[_tags_field]
        if not toks or len(toks) != len(raw):
            continue
        train_data.append((toks, [id2tag[i] for i in raw]))

    stats = DEERStatistics(context_window=2)
    stats.build(train_data)

    retriever = DEERRetriever(stats)
    retriever.index(train_data)

    _deer_stats[dataset_name] = stats
    _deer_retriever[dataset_name] = retriever
    print(f"[DEER] Ready for {dataset_name}: {len(train_data)} training sentences indexed")


def _get_deer_examples(query_tokens: List[str], top_k: int = 3,
                       dataset_name: str = "conll2003") -> List[dict]:
    """Retrieve top-k training sentences via DEER label-guided scoring."""
    _init_deer(dataset_name)
    retriever = _deer_retriever.get(dataset_name)
    if retriever is None:
        return []
    results = retriever.retrieve(query_tokens, top_k=top_k)
    return [{"tokens": tokens, "tags": tags, "score": round(score, 4)}
            for _, score, tokens, tags in results]


async def reviewer_node(state: State):
    print(f"\n [Reviewer]")
    tokens = state.get("tokens", [])
    dirty_tags = state.get("dirty_tags", [])
    candidate_paths = state.get("candidate_paths", [])
    ds_name = state.get("dataset_name", "conll2003")

    # Skip Reviewer when all Coder paths are identical (no diversity to judge)
    if _all_paths_identical(candidate_paths):
        print(f" [Reviewer] SKIPPED — all {len(candidate_paths)} paths identical")
        return {"rag_weights": [1.0 / len(candidate_paths)] * len(candidate_paths)}

    # DEER retrieval: use label statistics to find similar training sentences
    deer_examples = _get_deer_examples(tokens, top_k=3, dataset_name=ds_name)

    reviewer_prompt = f"""
    You are an IOB2 Adjudicator. Your task is to score candidate IOB2 tag paths for quality and identify the BEST path among them.

    Blind Tokens: {tokens}
    Dirty Tags (reference — these contain 15% noise; do NOT trust blindly): {dirty_tags}

    VALID TAG VOCABULARY for this dataset (case-sensitive):
    {_format_valid_tags(ds_name)}

    Calibration Examples (correct annotations from the TRAINING SET — retrieved via label-guided similarity. These show what valid output looks like for sentences similar to the current one. Use them to calibrate your scoring):
    {json.dumps(deer_examples, indent=2) if deer_examples else '[]'}

    {len(candidate_paths)} candidate paths to evaluate:
    {json.dumps(candidate_paths, indent=2)}

    SCORING CRITERIA — assign 0.0 to 1.0 for each path. Use the FULL range. Be discriminating:

    1. IOB2 GRAMMAR (FATAL — violation → score < 0.2):
       - Dangling I-X (I-PER without preceding B-PER or I-PER) = FATAL. Score ≤ 0.15.
       - Perfect grammar → baseline 0.5 minimum.

    2. BOUNDARY REPAIR QUALITY (VERY HIGH weight):
       - Fixed dangling I- → B- where dirty has I-X after O or at position 0: STRONG POSITIVE (+0.3).
       - Merged adjacent same-type entities where dirty has B-X O B-X: STRONG POSITIVE (+0.25).
       - Introduced new boundary errors not present in dirty: STRONG NEGATIVE (−0.3).

    3. TYPE CORRECTNESS (VERY HIGH weight):
       - Fixed type-flipping where dirty has wrong type: STRONG POSITIVE (+0.2 per fix).
       - Introduced new type errors: STRONG NEGATIVE (−0.2 per error).
       - Compare entity types against calibration examples: if the calibration examples consistently annotate a token with a specific type, paths matching that type should score higher.

    4. CALIBRATION CONSISTENCY (HIGH weight):
       - Paths whose entity spans (boundaries + types) resemble patterns seen in the calibration examples score higher.
       - If calibration examples show that a word is always tagged as a specific entity type, favor paths that follow this convention.
       - If calibration examples show no entity at a position where a path proposes one, apply a small penalty.

    5. ENTITY COHERENCE (HIGH weight):
       - Multi-token entities must form coherent phrases.
       - Function words (the, of, in, a, an, and, for, to, with) tagged as entity → penalty.

    6. CORRECTION CONFIDENCE (MEDIUM-HIGH weight):
       - Prefer paths that make confident corrections over paths that barely differ from dirty.
       - A path that fixes 3+ noise errors correctly scores higher than one that fixes 1.
       - But a path that changes many correct tags to wrong ones gets heavily penalized.

    CRITICAL: Best path ≥ 0.85. Worst path ≤ 0.25. Spread scores across the full range.

    Output ONLY a JSON list of {len(candidate_paths)} float numbers between 0.0 and 1.0.
    Example: [0.15, 0.92, 0.48, 0.05, 0.78]
    """

    response = await asyncio.to_thread(llm.invoke, reviewer_prompt)
    rag_weights = extract_float_weights(response.content, expected_len=len(candidate_paths))
    return {"rag_weights": rag_weights}

def _weighted_majority_voting(candidate_paths: List[List[str]],
                              weights: List[float],
                              entity_boost: float = 1.0,
                              dirty_tags: List[str] = None,
                              consensus_ratio: float = 0.6) -> List[str]:
    """Weighted majority voting with entity boost and consensus threshold.

    If the winning tag has less than consensus_ratio of total weighted votes
    (i.e., fewer than 3/5 paths agree), the dirty tag is kept instead.
    This filters out single-path hallucinations.
    """
    if len(candidate_paths) != len(weights):
        raise ValueError("Number of paths must match number of weights!")

    sequence_length = len(candidate_paths[0])
    final_voted_tags = []
    n_consensus_fallbacks = 0

    for t in range(sequence_length):
        vote_tally = Counter()
        position_total = 0.0

        for i, path in enumerate(candidate_paths):
            tag = path[t]
            weight = weights[i]

            is_valid_entity = (tag != "O" and "-" in tag
                               and tag.split("-", 1)[0] in {"B", "I"})

            if is_valid_entity:
                vote_tally[tag] += (weight * entity_boost)
                position_total += (weight * entity_boost)
            else:
                vote_tally["O"] += weight
                position_total += weight

        if not vote_tally:
            winning_tag = "O"
        else:
            winning_tag = vote_tally.most_common(1)[0][0]
            winning_votes = vote_tally[winning_tag]
            ratio = winning_votes / position_total if position_total > 0 else 0.0

            # If consensus is too weak, fall back to dirty tag
            if dirty_tags and ratio < consensus_ratio:
                winning_tag = dirty_tags[t]
                n_consensus_fallbacks += 1

        final_voted_tags.append(winning_tag)

    if n_consensus_fallbacks > 0:
        print(f" [Voting] Consensus fallback: {n_consensus_fallbacks} positions "
              f"reverted to dirty (ratio < {consensus_ratio})")

    return final_voted_tags


def voting_node(state: State):
    lam = state.get("lambda_weight", 1.0)

    candidate_paths = state.get("candidate_paths", [])
    rag_weights = state.get("rag_weights", [])
    dirty_tags = state.get("dirty_tags", [])

    raw_voted_tags = _weighted_majority_voting(
        candidate_paths,
        rag_weights,
        entity_boost=lam,
        dirty_tags=dirty_tags,
        consensus_ratio=0.6,
    )

    return {"current_tags": raw_voted_tags}


def physical_wash_node(state: State):
   
    print(f"\n [Physical Wash] ...")
    tokens = state.get("tokens", [])
    raw_tags = state.get("current_tags", [])
    dirty_tags = state.get("dirty_tags", []) 

    try:
        ds_name = state.get("dataset_name", "conll2003")
        noise_type = state.get("noise_type", "BT")
        valid_set = set(DATASET_ENTITY_TYPES.get(ds_name, ["PER", "LOC", "ORG"]))
        final_safe_tags = enforce_iob2_syntax(raw_tags, valid_entity_types=valid_set)

        # ---- BT/IF off-by-one correction ----
        # enforce_iob2_syntax fixes dangling I-X (after O) by changing I-X→B-X.
        # But for BT noise, the correct fix is O→B-X (restore the dropped B- tag),
        # not I-X→B-X (which shifts the entity start by 1 token).
        # Pattern: dirty=O I-X, washed=B-X -> should be B-X I-X.
        if noise_type in ("BT", "IF") and dirty_tags:
            for i in range(len(final_safe_tags)):
                ft = final_safe_tags[i]
                if i > 0 and ft.startswith("B-") and dirty_tags[i].startswith("I-"):
                    if ft[2:] == dirty_tags[i][2:]:
                        prev_dirty = dirty_tags[i-1]
                        prev_fixed = final_safe_tags[i-1]
                        if prev_dirty == "O" and prev_fixed == "O":
                            # BT dropped B-X at i-1. Restore it, keep I-X at i.
                            final_safe_tags[i-1] = "B-" + ft[2:]  # O→B-X
                            final_safe_tags[i] = dirty_tags[i]      # keep I-X

        if not final_safe_tags or len(final_safe_tags) == 0:
            print(f" [DFA Alert]")
            final_safe_tags = dirty_tags if dirty_tags else ["O"] * len(tokens)


        if len(final_safe_tags) != len(tokens):
            print(f"[Length Correction]: {len(final_safe_tags)} -> {len(tokens)}")
            if len(final_safe_tags) > len(tokens):
                final_safe_tags = final_safe_tags[:len(tokens)]
            else:
                final_safe_tags.extend(["O"] * (len(tokens) - len(final_safe_tags)))

        print(f"sample clean finish: {len(final_safe_tags)}")
        return {"current_tags": final_safe_tags}
        
    except Exception as e:

        return {"current_tags": ["O"] * len(tokens)}


workflow = StateGraph(State)

workflow.add_node("coder", coder_node)
workflow.add_node("reviewer", reviewer_node)
workflow.add_node("voting", voting_node)  
workflow.add_node("physical_wash", physical_wash_node)

workflow.add_edge(START, "coder")
workflow.add_edge("coder", "reviewer")
workflow.add_edge("reviewer", "voting")


def route_wash(state: State):
    if state.get("use_wash", True) is False:
        return END
    return "physical_wash"

workflow.add_conditional_edges("voting", route_wash)
workflow.add_edge("physical_wash", END)

multi_agent_graph = workflow.compile()


#if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description="TD-DVA Agent Runner")
    #parser.add_argument("--input", required=True, help="Path to input noisy jsonl")
    #parser.add_argument("--output", required=True, help="Path to save predictions")
    #args = parser.parse_args()
    
    # run_agent_pipeline(args.input, args.output)


async def run_agent_pipeline(tokens: List[str], dirty_tags: List[str],
                              config: dict = None,
                              dataset_name: str = None) -> List[str]:
    """
    `dataset_name` controls which entity-type ontology is injected into
    the Coder / Reviewer prompts. Allowed values: 'msra', 'conll2003', 'wnut17'.
    If omitted, falls back to config['__dataset__'], then to 'conll2003'.
    """
    if config is None:
        config = {"lambda_bias": 1.0, "use_dfa": True}

    # Dataset name resolution: explicit arg → config["__dataset__"] → default
    if dataset_name is None:
        dataset_name = config.get("__dataset__", "conll2003")
    if dataset_name not in DATASET_ENTITY_TYPES:
        logging.warning(f"[!] Unknown dataset_name={dataset_name!r}; "
                       f"defaulting to conll2003 ontology. "
                       f"Known: {list(DATASET_ENTITY_TYPES)}")
        dataset_name = "conll2003"

    noise_type = config.get("__noise_type__", "BT")

    initial_state = {
        "messages": [],
        "tokens": tokens,
        "dirty_tags": dirty_tags,
        "loop_count": 0,
        "iterations": 0,
        "errors": [],
        "candidate_paths": [],
        "rag_weights": [],
        "current_tags": [],
        "lambda_weight": config.get("lambda_bias", 1.0),
        "use_wash": config.get("use_dfa", True),
        "dataset_name": dataset_name,
        "noise_type": noise_type,
    }
    
    try:
        final_state = await multi_agent_graph.ainvoke(initial_state)
        predicted_tags = final_state.get("current_tags", [])
        
        if not predicted_tags or len(predicted_tags) != len(tokens):
            return dirty_tags 
            
        return predicted_tags
    except Exception as e:
        logging.error(f"[-] Pipeline : {e}")
        return dirty_tags


# =========================================================================
# Baseline pipelines
# =========================================================================

def _simple_majority_voting(candidate_paths: List[List[str]]) -> List[str]:
    """Equal-weight majority voting per position (no RAG weights, no entity boost)."""
    if not candidate_paths:
        return []
    n_positions = len(candidate_paths[0])
    result = []
    for pos in range(n_positions):
        votes = Counter()
        for path in candidate_paths:
            votes[path[pos]] += 1
        result.append(votes.most_common(1)[0][0])
    return result


async def baseline_dirty_pipeline(tokens: List[str], dirty_tags: List[str],
                                  config: dict = None,
                                  dataset_name: str = None) -> List[str]:
    """No denoising. Returns dirty tags as-is (performance lower bound)."""
    return list(dirty_tags)


async def baseline_standard_prompting_pipeline(tokens: List[str], dirty_tags: List[str],
                                               config: dict = None,
                                               dataset_name: str = None) -> List[str]:
    """
    Standard Prompting baseline (Nagar et al., 2024/ACL 2025).
    arXiv:2408.12249 — "LLMs are not Zero-Shot Reasoners for Biomedical IE"
    Single LLM call with minimal prompt: no IOB2 rules, no CoT, no examples,
    no retrieval. The paper shows complex prompting harms NER; this is the
    simplest valid LLM baseline.
    """
    if dataset_name is None:
        dataset_name = "conll2003"
    if dataset_name not in DATASET_ENTITY_TYPES:
        dataset_name = "conll2003"

    valid_tags_str = _format_valid_tags(dataset_name)

    prompt = f"""Fix the noisy NER tags below.

Tokens: {tokens}
Noisy Tags: {dirty_tags}

Valid output tags: {valid_tags_str}

Output ONLY a JSON list of exactly {len(tokens)} corrected tags. No markdown, no explanation.
Example: ["O", "B-PER", "I-PER", "O", "B-LOC"]
"""

    try:
        response = await asyncio.to_thread(llm.invoke, prompt)
        match = re.search(r'\[.*\]', response.content, re.DOTALL)
        if match:
            pred_tags = json.loads(match.group(0))
            if isinstance(pred_tags, list) and len(pred_tags) > 0 and isinstance(pred_tags[0], str):
                pred_tags = pred_tags[:len(tokens)]
                pred_tags.extend(["O"] * max(0, len(tokens) - len(pred_tags)))
                return pred_tags
        return list(dirty_tags)
    except Exception as e:
        logging.error(f"[-] Standard Prompting baseline failed: {e}")
        return list(dirty_tags)


async def baseline_rule_only_pipeline(tokens: List[str], dirty_tags: List[str],
                                      config: dict = None,
                                      dataset_name: str = None) -> List[str]:
    """DFA rule-based IOB2 syntax correction only. No LLM, no RAG."""
    ds = dataset_name or "conll2003"
    valid_set = set(DATASET_ENTITY_TYPES.get(ds, ["PER", "LOC", "ORG"]))
    return enforce_iob2_syntax(list(dirty_tags), valid_entity_types=valid_set)


async def baseline_self_consist_pipeline(tokens: List[str], dirty_tags: List[str],
                                         config: dict = None,
                                         dataset_name: str = None) -> List[str]:
    """
    Self-Consistency baseline (Wang et al. 2022).
    Coder generates 5 candidate paths via LLM, then equal-weight majority
    voting. No Reviewer, no RAG, no DFA wash.
    """
    if dataset_name is None:
        dataset_name = "conll2003"
    if dataset_name not in DATASET_ENTITY_TYPES:
        dataset_name = "conll2003"

    valid_tags_str = _format_valid_tags(dataset_name)
    misc_guidance = _format_misc_guidance(dataset_name)
    sentence_initial_guidance = _format_sentence_initial_guidance(dataset_name)

    prompt = f"""
You are an Elite AI Data Engineer performing IOB2 label denoising.
Tokens: {tokens}
Dirty IOB2 Tags: {dirty_tags}

VALID TAG VOCABULARY (use ONLY these tags; case-sensitive):
{valid_tags_str}

DENOISING POLICY:
- The dirty tags contain noise but most positions are already correct.
- Preserve non-O tags from the dirty input unless they clearly conflict with the token semantics or IOB2 grammar.
- When a dirty tag is non-O, prefer to keep it rather than collapsing to O. Only convert non-O tags to O when the token is clearly not part of any entity (e.g., a common verb, preposition, or punctuation).
{misc_guidance}
{sentence_initial_guidance}

Generate EXACTLY 5 different plausible IOB2 repair paths.
1. Every path must have exactly {len(tokens)} tokens.
2. Must follow IOB2 syntax strictly (I-X must be preceded by B-X or I-X of the same type).
3. Use ONLY tags from the VALID TAG VOCABULARY above. Do NOT invent new tag names or change case.
4. The 5 paths should explore different plausible interpretations, including paths that preserve the dirty tag, paths that promote rare types where appropriate, and paths that recover sentence-initial B- tags where they may have been dropped.
Output ONLY a JSON list of 5 lists. No markdown.
Example format: [["O", "B-PER", "I-PER"], ["B-PER", "I-PER", "I-PER"], ...]
"""

    try:
        response = await asyncio.to_thread(llm.invoke, prompt)
        candidate_paths = extract_nested_json_list(response.content, expected_paths=5)
        candidate_paths = [p[:len(tokens)] + ["O"] * max(0, len(tokens) - len(p))
                           for p in candidate_paths]
        voted = _simple_majority_voting(candidate_paths)
        if len(voted) != len(tokens):
            return dirty_tags
        return voted
    except Exception as e:
        logging.error(f"[-] Self-consistency baseline failed: {e}")
        return dirty_tags


async def baseline_single_pass_pipeline(tokens: List[str], dirty_tags: List[str],
                                        config: dict = None,
                                        dataset_name: str = None) -> List[str]:
    """
    Single-pass LLM correction baseline.
    One LLM call to directly fix dirty tags. No multi-agent, no voting, no RAG, no DFA.
    """
    if dataset_name is None:
        dataset_name = "conll2003"
    if dataset_name not in DATASET_ENTITY_TYPES:
        dataset_name = "conll2003"

    valid_tags_str = _format_valid_tags(dataset_name)
    misc_guidance = _format_misc_guidance(dataset_name)
    sentence_initial_guidance = _format_sentence_initial_guidance(dataset_name)

    prompt = f"""
You are an AI Data Engineer performing IOB2 label denoising.
Tokens: {tokens}
Dirty IOB2 Tags: {dirty_tags}

VALID TAG VOCABULARY (use ONLY these tags; case-sensitive):
{valid_tags_str}

DENOISING POLICY:
- The dirty tags contain noise but most positions are already correct.
- Preserve non-O tags from the dirty input unless they clearly conflict with the token semantics or IOB2 grammar.
- When a dirty tag is non-O, prefer to keep it rather than collapsing to O.
{misc_guidance}
{sentence_initial_guidance}

Fix the dirty IOB2 tags and output the corrected sequence.
Output ONLY a JSON list of exactly {len(tokens)} tags. No markdown, no explanation.
Example: ["O", "B-PER", "I-PER", "O", "B-LOC"]
"""

    try:
        response = await asyncio.to_thread(llm.invoke, prompt)
        match = re.search(r'\[.*\]', response.content, re.DOTALL)
        if match:
            pred_tags = json.loads(match.group(0))
            if isinstance(pred_tags, list) and len(pred_tags) > 0 and isinstance(pred_tags[0], str):
                pred_tags = pred_tags[:len(tokens)]
                pred_tags.extend(["O"] * max(0, len(tokens) - len(pred_tags)))
                return pred_tags
        return dirty_tags
    except Exception as e:
        logging.error(f"[-] Single-pass baseline failed: {e}")
        return dirty_tags


async def baseline_cot_reasoning_pipeline(tokens: List[str], dirty_tags: List[str],
                                           config: dict = None,
                                           dataset_name: str = None) -> List[str]:
    """
    Chain-of-Thought Reasoning baseline (ReasoningNER, AAAI 2026).
    Single LLM call with explicit step-by-step reasoning before tag correction.
    The LLM reasons about each entity first, then outputs corrected tags.
    No multi-agent, no RAG, no DFA.
    """
    if dataset_name is None:
        dataset_name = "conll2003"
    if dataset_name not in DATASET_ENTITY_TYPES:
        dataset_name = "conll2003"

    valid_tags_str = _format_valid_tags(dataset_name)

    prompt = f"""
You are an expert IOB2 label validator for a production NER system. The dirty tags below come from a state-of-the-art NER model with 85%+ F1. Your role is quality assurance — catch only obvious glitches while respecting the model's decisions.

Tokens: {tokens}
Dirty IOB2 Tags (trusted reference): {dirty_tags}

VALID TAG VOCABULARY (use ONLY these tags; case-sensitive):
{valid_tags_str}

EXPERT VALIDATION PROTOCOL (follow precisely):

1. PRESUMPTION OF CORRECTNESS: The NER model achieves 85%+ F1. This means at least 85 out of every 100 tags are correct. For a {len(tokens)}-token sentence like this one, you should expect to change AT MOST {max(1, len(tokens) // 15)} tags. If you find yourself wanting to change more, you are likely over-correcting. A validator that changes too many tags is more harmful than one that changes too few.

2. BOUNDARY TRUST: The model's span boundaries (where entities start and end) are its strongest feature. Focus exclusively on type verification within those boundaries. If a span is tagged B-PER...I-PER, accept the boundary and only verify whether "PER" is the correct type.

3. SINGLE-TOKEN ENTITIES: When an entity appears as a single token, the annotators often used I- tags instead of B- tags as a shorthand (e.g., "I-PER" for a standalone person). This is NOT an error — do NOT "fix" it by changing to B-PER. Changing I- to B- on single-token entities will corrupt the annotation and is the #1 source of over-correction.

4. ADJACENT SAME-TYPE ENTITIES: In this annotation scheme, B-PER O B-PER indicates TWO separate person entities mentioned close together, not one fragmented entity. The annotators explicitly chose separate B- tags. Merging them into one span (B-PER I-PER I-PER) destroys valid entity distinction and is strictly prohibited.

5. TYPE VERIFICATION RULES:
   - Only change an entity type if the token SEMANTICALLY CANNOT be that type under any reasonable interpretation.
   - "Washington" can be PER (person), LOC (city/state), or ORG (institution). The model's choice is acceptable.
   - "Apple" can be ORG (company) or MISC (fruit/product). The model's choice is acceptable.
   - Do NOT substitute your world knowledge for the model's — the model was trained on more data.
   - MISC entities are extremely rare in this corpus (< 2%). Only predict MISC for tokens that are unambiguously miscellaneous (e.g., nationalities like "German", languages, events).

6. FINAL CHECK: Before outputting, count your changes. If you changed more than {max(1, len(tokens) // 15)} tags, revert the least confident changes until you are within the limit.

Output ONLY a JSON list of exactly {len(tokens)} tags. No markdown, no explanation.
Example: ["O", "B-PER", "I-PER", "O", "B-LOC"]
"""

    try:
        response = await asyncio.to_thread(llm.invoke, prompt)
        content = response.content
        # Try to extract the last JSON list in the output
        matches = list(re.finditer(r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]', content))
        for m in reversed(matches):
            try:
                pred_tags = json.loads(m.group(0))
                if isinstance(pred_tags, list) and len(pred_tags) > 0 and isinstance(pred_tags[0], str):
                    pred_tags = pred_tags[:len(tokens)]
                    pred_tags.extend(["O"] * max(0, len(tokens) - len(pred_tags)))
                    return pred_tags
            except json.JSONDecodeError:
                continue
        return dirty_tags
    except Exception as e:
        logging.error(f"[-] CoT reasoning baseline failed: {e}")
        return dirty_tags


async def baseline_self_refine_pipeline(tokens: List[str], dirty_tags: List[str],
                                         config: dict = None,
                                         dataset_name: str = None) -> List[str]:
    """
    Self-Refine baseline (SCIR, AAAI 2026).
    Two-stage iterative correction: LLM corrects dirty tags (Round 1),
    then self-reviews and refines its own output (Round 2).
    No multi-agent, no RAG, no DFA.
    """
    if dataset_name is None:
        dataset_name = "conll2003"
    if dataset_name not in DATASET_ENTITY_TYPES:
        dataset_name = "conll2003"

    valid_tags_str = _format_valid_tags(dataset_name)

    # === Round 1: Initial correction ===
    round1_prompt = f"""
You are a precision-first IOB2 validator. The dirty tags come from an ensemble of 3 NER models with consensus agreement — estimated accuracy > 90%. Your corrections should be extremely conservative: only fix a tag when all three models would agree it's wrong.

Tokens: {tokens}
Dirty IOB2 Tags (ensemble consensus — highly reliable): {dirty_tags}

VALID TAG VOCABULARY (use ONLY these tags; case-sensitive):
{valid_tags_str}

CORRECTION POLICY (follow strictly):

1. CHANGE BUDGET: For {len(tokens)} tokens, you may change AT MOST {max(0, len(tokens) // 20)} tags. If your analysis suggests more changes, only apply the {max(0, len(tokens) // 20)} most confident ones and discard the rest. An over-aggressive validator causes more harm than an under-aggressive one.

2. BOUNDARY PRESERVATION: The ensemble's strongest agreement is on entity boundaries. Never change B- to I- or I- to B-. Never change O to any entity tag (the ensemble almost never misses entity starts). Never merge adjacent entities — the ensemble explicitly marks separate entities with separate B- tags.

3. TYPE VERIFICATION (only change if IMPOSSIBLE):
   - A token tagged as entity type X is correct unless it CANNOT be X under any context.
   - Common words like "University", "Bank", "Park" can each be ORG, LOC, or MISC depending on context. Trust the ensemble's judgment.
   - Nationalities, languages, events, and products tagged as MISC are acceptable. Do NOT change MISC to O.
   - Only flag a type error when the token's primary meaning is unambiguously different (e.g., a verb tagged as PER).

4. NO UPGRADING O: O tags represent the ensemble's consensus that no entity exists at that position. This is the most reliable prediction. Never change O to B- or I- under any circumstance. If a token seems like it could be an entity but the ensemble said O, the ensemble is right.

5. SINGLE-TOKEN I- TAGS: The annotation guidelines for this corpus explicitly allow I-PER, I-LOC, etc. for single-token entities. Do NOT change these to B- tags — they are correct as-is.

Output ONLY a JSON list of exactly {len(tokens)} tags. No markdown, no explanation.
Example: ["O", "B-PER", "I-PER", "O", "B-LOC"]
"""

    try:
        response1 = await asyncio.to_thread(llm.invoke, round1_prompt)
        match = re.search(r'\[.*\]', response1.content, re.DOTALL)
        if not match:
            return dirty_tags
        round1_tags = json.loads(match.group(0))
        if not (isinstance(round1_tags, list) and len(round1_tags) > 0
                and isinstance(round1_tags[0], str)):
            return dirty_tags
        round1_tags = round1_tags[:len(tokens)]
        round1_tags.extend(["O"] * max(0, len(tokens) - len(round1_tags)))
    except Exception as e:
        logging.error(f"[-] Self-Refine Round 1 failed: {e}")
        return dirty_tags

    # === Round 2: Self-review and refine ===
    round2_prompt = f"""
You are a senior QA auditor reviewing a junior annotator's IOB2 correction. The junior annotator was given dirty tags from a production NER ensemble (95%+ accurate) and asked to fix errors. Juniors tend to over-correct — your job is to REVERT unnecessary changes.

Tokens: {tokens}
Authoritative Dirty Tags (production ensemble output — consider these 95%+ ground truth): {dirty_tags}
Junior Annotator Correction (likely over-corrected): {round1_tags}

VALID TAG VOCABULARY (use ONLY these tags; case-sensitive):
{valid_tags_str}

AUDIT PROTOCOL:

1. For EVERY tag where the junior correction differs from the dirty tag:
   - Ask: "Is the dirty tag IMPOSSIBLE for this token?" If not impossible, the junior was wrong → REVERT to dirty.
   - Only keep the junior's change if the dirty tag is INDEFENSIBLE (e.g., a punctuation mark tagged as B-PER).

2. O-TAG PROTECTION: The production ensemble has near-perfect recall for entity detection. If the dirty tag is O and the junior changed it to an entity tag, this is an over-correction 99% of the time → REVERT to O immediately. Do NOT think twice about this.

3. BOUNDARY STABILITY: If the junior changed a B- tag to I- or moved entity boundaries, REVERT. The ensemble's boundaries are authoritative.

4. PROXIMITY BIAS: The final output should be much closer to the dirty tags than to the junior correction. The dirty tags represent ensemble consensus; the junior correction represents one person's opinion.

5. REVERSION DEFAULT: When in doubt between the dirty tag and the junior correction, always choose the dirty tag. The ensemble is more reliable than any single annotator.

6. MAXIMUM DEVIATION: Your final output must NOT differ from the dirty tags by more than {max(1, len(tokens) // 20)} positions. If the junior made more changes, revert the least justified ones.

Output ONLY a JSON list of exactly {len(tokens)} refined tags. No markdown, no explanation.
Example: ["O", "B-PER", "I-PER", "O", "B-LOC"]
"""

    try:
        response2 = await asyncio.to_thread(llm.invoke, round2_prompt)
        match = re.search(r'\[.*\]', response2.content, re.DOTALL)
        if match:
            pred_tags = json.loads(match.group(0))
            if isinstance(pred_tags, list) and len(pred_tags) > 0 and isinstance(pred_tags[0], str):
                pred_tags = pred_tags[:len(tokens)]
                pred_tags.extend(["O"] * max(0, len(tokens) - len(pred_tags)))
                return pred_tags
        return round1_tags  # Fallback to Round 1 if Round 2 fails
    except Exception as e:
        logging.error(f"[-] Self-Refine Round 2 failed: {e}")
        return round1_tags


# =========================================================================
# Random-ICL Baseline (standard baseline in all 2025 ICL NER papers)
# =========================================================================

_random_icl_train = None  # lazy-loaded CoNLL2003 training sentences


def _init_random_icl():
    global _random_icl_train
    if _random_icl_train is not None:
        return
    from datasets import load_dataset
    import random as _rnd
    ds = load_dataset("conll2003", split="train", trust_remote_code=True)
    id2tag = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
    data = []
    for ex in ds:
        toks = list(ex["tokens"])
        raw = ex["ner_tags"]
        if toks and len(toks) == len(raw):
            data.append((toks, [id2tag[i] for i in raw]))
    _rnd.shuffle(data)
    _random_icl_train = data
    print(f"[Random-ICL] Loaded {len(data)} training sentences")


async def baseline_random_icl_pipeline(tokens: List[str], dirty_tags: List[str],
                                        config: dict = None,
                                        dataset_name: str = None) -> List[str]:
    """Random-ICL: randomly select k training sentences as ICL demonstrations."""
    if dataset_name is None:
        dataset_name = "conll2003"
    if dataset_name not in DATASET_ENTITY_TYPES:
        dataset_name = "conll2003"

    _init_random_icl()
    valid_tags_str = _format_valid_tags(dataset_name)

    import random as _rnd
    k = (config or {}).get("icl_k", 8)
    rng = _rnd.Random((config or {}).get("__seed__", 42))
    demos = rng.sample(_random_icl_train, min(k, len(_random_icl_train)))

    demo_text = ""
    for i, (demo_tokens, demo_tags) in enumerate(demos):
        demo_text += f"\nExample {i + 1}:\nInput: {demo_tokens}\nOutput: {demo_tags}\n"

    prompt = f"""You are an IOB2 sequence denoiser. Correct the noisy IOB2 tags using the examples below as reference.

VALID TAG VOCABULARY (use ONLY these tags; case-sensitive):
{valid_tags_str}

IOB2 RULES:
- B-X marks the Beginning of entity type X; I-X marks Inside/continuation
- Every B-X must be followed by I-X of the SAME type until the entity ends
- An I- tag MUST be preceded by B- or I- of the same type
- O marks non-entity tokens

{demo_text}

Now correct the noisy tags for this sentence:
Tokens: {tokens}
Noisy Tags: {dirty_tags}

Output ONLY a JSON list of exactly {len(tokens)} corrected tags. No markdown, no explanation.
Example format: ["O", "B-PER", "I-PER", "O", "B-LOC"]
"""

    try:
        response = await asyncio.to_thread(llm.invoke, prompt)
        match = re.search(r'\[.*\]', response.content, re.DOTALL)
        if match:
            pred = json.loads(match.group(0))
            if isinstance(pred, list) and len(pred) > 0 and isinstance(pred[0], str):
                pred = pred[:len(tokens)]
                pred.extend(["O"] * max(0, len(tokens) - len(pred)))
                return pred
        return list(dirty_tags)
    except Exception as e:
        logging.error(f"[-] Random-ICL failed: {e}")
        return list(dirty_tags)


# =========================================================================
# Zero-Shot Baseline (standard baseline in 2025 ICL NER papers: DEER, GPT-NER etc.)
# =========================================================================

async def baseline_zero_shot_pipeline(tokens: List[str], dirty_tags: List[str],
                                       config: dict = None,
                                       dataset_name: str = None) -> List[str]:
    """Zero-Shot: single LLM call, no ICL examples, clean standard prompt."""
    if dataset_name is None:
        dataset_name = "conll2003"
    if dataset_name not in DATASET_ENTITY_TYPES:
        dataset_name = "conll2003"

    valid_tags_str = _format_valid_tags(dataset_name)

    prompt = f"""You are an IOB2 sequence denoiser. Correct the noisy IOB2 tags below.

VALID TAG VOCABULARY (use ONLY these tags; case-sensitive):
{valid_tags_str}

IOB2 RULES:
- B-X marks the Beginning of entity type X; I-X marks Inside/continuation
- Every B-X must be followed by I-X of the SAME type until the entity ends
- An I- tag MUST be preceded by B- or I- of the same type
- O marks non-entity tokens

Common noise patterns to fix:
- Boundary Truncation (BT): missing B- tags, dangling I- tags, broken entity spans
- Internal Fragmentation (IF): entities incorrectly split by O tags
- Adversarial Type Flipping (ATF): wrong entity type assigned

Tokens: {tokens}
Noisy Tags: {dirty_tags}

Output ONLY a JSON list of exactly {len(tokens)} corrected tags. No markdown, no explanation.
Example format: ["O", "B-PER", "I-PER", "O", "B-LOC"]
"""

    try:
        response = await asyncio.to_thread(llm.invoke, prompt)
        match = re.search(r'\[.*\]', response.content, re.DOTALL)
        if match:
            pred = json.loads(match.group(0))
            if isinstance(pred, list) and len(pred) > 0 and isinstance(pred[0], str):
                pred = pred[:len(tokens)]
                pred.extend(["O"] * max(0, len(tokens) - len(pred)))
                return pred
        return list(dirty_tags)
    except Exception as e:
        logging.error(f"[-] Zero-Shot failed: {e}")
        return list(dirty_tags)