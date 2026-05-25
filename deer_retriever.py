"""
DEER (Label-Guided In-Context Learning for NER) adapted for IOB2 denoising.
Bai et al., EMNLP 2025 — https://arxiv.org/abs/2505.23722

Core idea: use training label statistics to weight tokens for retrieval,
so entity-indicative tokens and context tokens drive example selection.

Adaptation for denoising:
  - Step 0: Build P(token|entity/context/other) from CoNLL2003 train set
  - Step 1: Retrieve top-k training sentences using DEER weighted scoring
  - Step 2: Error reflection using token statistics
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np


class DEERStatistics:
    """Build and query token-level statistics from training labels."""

    def __init__(self, context_window: int = 2):
        self.context_window = context_window
        # Per-token counts: entity, context, other
        self._entity_count: Dict[str, int] = defaultdict(int)
        self._context_count: Dict[str, int] = defaultdict(int)
        self._other_count: Dict[str, int] = defaultdict(int)
        self._total_occurrences: Dict[str, int] = defaultdict(int)
        self._total_entity_tokens = 0
        self._total_context_tokens = 0
        self._total_other_tokens = 0
        self._vocab: set = set()
        self._built = False

    def build(self, sentences: List[Tuple[List[str], List[str]]]):
        """Build statistics from (tokens, IOB2_tags) pairs."""
        cw = self.context_window

        for tokens, tags in sentences:
            n = len(tokens)
            # Find entity spans
            in_entity = [False] * n
            entity_spans = []
            i = 0
            while i < n:
                if tags[i].startswith("B-"):
                    j = i + 1
                    while j < n and tags[j].startswith("I-"):
                        j += 1
                    entity_spans.append((i, j))
                    for k in range(i, j):
                        in_entity[k] = True
                    i = j
                else:
                    i += 1

            # Mark context: up to cw tokens on each side of entity spans
            in_context = [False] * n
            for start, end in entity_spans:
                ctx_start = max(0, start - cw)
                ctx_end = min(n, end + cw)
                for k in range(ctx_start, ctx_end):
                    if not in_entity[k]:
                        in_context[k] = True

            # Count tokens
            for k, tok in enumerate(tokens):
                tok_lower = tok.lower()
                self._vocab.add(tok_lower)
                self._total_occurrences[tok_lower] += 1
                if in_entity[k]:
                    self._entity_count[tok_lower] += 1
                    self._total_entity_tokens += 1
                elif in_context[k]:
                    self._context_count[tok_lower] += 1
                    self._total_context_tokens += 1
                else:
                    self._other_count[tok_lower] += 1
                    self._total_other_tokens += 1

        self._built = True
        print(f"DEER stats built: {len(self._vocab)} unique tokens, "
              f"{len(sentences)} sentences")

    def entity_prob(self, token: str) -> float:
        """P(token | entity) — fraction of total entity tokens that are this word."""
        if self._total_entity_tokens == 0:
            return 0.0
        return self._entity_count.get(token.lower(), 0) / self._total_entity_tokens

    def context_prob(self, token: str) -> float:
        """P(token | context) — fraction of total context tokens that are this word."""
        if self._total_context_tokens == 0:
            return 0.0
        return self._context_count.get(token.lower(), 0) / self._total_context_tokens

    def other_prob(self, token: str) -> float:
        """P(token | other) — fraction of total other tokens that are this word."""
        if self._total_other_tokens == 0:
            return 1.0
        return self._other_count.get(token.lower(), 0) / self._total_other_tokens

    def is_seen(self, token: str) -> bool:
        return token.lower() in self._vocab

    def token_weight(self, token: str,
                     w_e: float = 1.0, w_c: float = 1.0, w_o: float = 0.01) -> float:
        """W(t) = w_e * P(t|entity) + w_c * P(t|context) + w_o * P(t|other). Default 1.0 for unseen."""
        if not self._built:
            return 1.0
        return (w_e * self.entity_prob(token) +
                w_c * self.context_prob(token) +
                w_o * self.other_prob(token))


class DEERRetriever:
    """Retrieve training sentences using DEER weighted scoring."""

    def __init__(self, stats: DEERStatistics,
                 w_e: float = 1.0, w_c: float = 1.0, w_o: float = 0.01,
                 lambda_token: float = 1.0, lambda_embed: float = 0.0):
        self.stats = stats
        self.w_e = w_e
        self.w_c = w_c
        self.w_o = w_o
        self.lambda_token = lambda_token
        self.lambda_embed = lambda_embed
        self._train_data: List[Tuple[List[str], List[str]]] = []
        # Precomputed token sets for fast matching
        self._train_token_sets: List[set] = []
        # Weighted BoW embeddings
        self._train_embeddings: List[np.ndarray] = []

    def index(self, sentences: List[Tuple[List[str], List[str]]]):
        """Index training sentences for retrieval."""
        self._train_data = sentences
        for tokens, _tags in sentences:
            tok_set = set(t.lower() for t in tokens)
            self._train_token_sets.append(tok_set)
            # Build weighted BoW embedding
            if self.lambda_embed > 0:
                emb = np.zeros(1)  # placeholder; real impl would use GloVe
                self._train_embeddings.append(emb)
        print(f"DEER index: {len(sentences)} training sentences")

    def _token_score(self, query_tokens: List[str], train_idx: int) -> float:
        """S_token(s_i, s_test) = sum over query tokens of 1(token in train) * W(token)."""
        train_set = self._train_token_sets[train_idx]
        score = 0.0
        for t in query_tokens:
            t_lower = t.lower()
            if t_lower in train_set:
                score += self.stats.token_weight(t, self.w_e, self.w_c, self.w_o)
        return score

    def retrieve(self, query_tokens: List[str], top_k: int = 8,
                 exclude_indices: set = None) -> List[Tuple[int, float, List[str], List[str]]]:
        """
        Retrieve top-k training sentences.
        Returns: list of (index, score, tokens, tags).
        """
        if not self._train_data:
            return []

        scores = np.zeros(len(self._train_data), dtype=np.float32)
        for i in range(len(self._train_data)):
            if exclude_indices and i in exclude_indices:
                continue
            scores[i] = self._token_score(query_tokens, i)

        # Normalize by query length to avoid bias toward long queries
        if len(query_tokens) > 0:
            scores = scores / len(query_tokens)

        top_indices = np.argsort(scores)[-top_k:][::-1]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((int(idx), float(scores[idx]),
                                list(self._train_data[idx][0]),
                                list(self._train_data[idx][1])))
        return results

    # ---- Error Reflection (Step 2) ----

    def find_unseen_tokens(self, tokens: List[str], pred_tags: List[str]) -> List[int]:
        """Find unseen tokens that may be missed entities."""
        candidates = []
        for i, (tok, tag) in enumerate(zip(tokens, pred_tags)):
            if not self.stats.is_seen(tok) and tag == "O":
                # Check if surrounded by entity/context-frequent tokens
                ctx_entity_score = 0.0
                for offset in [-2, -1, 1, 2]:
                    j = i + offset
                    if 0 <= j < len(tokens):
                        ctx_entity_score += self.stats.entity_prob(tokens[j])
                        ctx_entity_score += self.stats.context_prob(tokens[j])
                if ctx_entity_score > 0.5:
                    candidates.append(i)
        return candidates

    def find_false_negatives(self, tokens: List[str], pred_tags: List[str],
                             threshold: float = 0.3) -> List[int]:
        """Find tokens with high entity probability but predicted as O."""
        candidates = []
        for i, (tok, tag) in enumerate(zip(tokens, pred_tags)):
            if tag == "O" and self.stats.entity_prob(tok) > threshold:
                candidates.append(i)
        return candidates

    def find_boundary_issues(self, tokens: List[str], pred_tags: List[str],
                             k: int = 2) -> List[Tuple[int, str]]:
        """Find boundary tokens where stats suggest misclassification.
        Returns: list of (index, issue_description).
        """
        issues = []
        n = len(tokens)
        in_entity = [t != "O" for t in pred_tags]

        for i, (tok, tag) in enumerate(zip(tokens, pred_tags)):
            if tag == "O":
                continue
            e_prob = self.stats.entity_prob(tok)
            c_prob = self.stats.context_prob(tok)
            # Token appears more as context than entity => possible boundary error
            if c_prob > e_prob and c_prob > 0.1:
                issues.append((i, f"boundary_suspect: context_prob={c_prob:.2f} > entity_prob={e_prob:.2f}"))
            # Token at entity edge
            is_edge = (i == 0 or pred_tags[i - 1] == "O" or
                       (pred_tags[i - 1].startswith("B-") and tag.startswith("I-") and
                        pred_tags[i - 1][2:] != tag[2:]))
            if is_edge and e_prob < 0.05 and self.stats.is_seen(tok):
                issues.append((i, f"edge_low_entity_prob: {e_prob:.2f}"))
        return issues

    def get_span_examples(self, token: str, category: str = "entity",
                          top_m: int = 3) -> List[Tuple[List[str], List[str]]]:
        """Get span-level examples containing a specific token from a category."""
        t_lower = token.lower()
        results = []
        for tokens, tags in self._train_data:
            if t_lower not in set(x.lower() for x in tokens):
                continue
            # Find spans containing this token
            i = 0
            while i < len(tokens):
                if tags[i].startswith("B-"):
                    j = i + 1
                    while j < len(tokens) and tags[j].startswith("I-"):
                        j += 1
                    span_tokens = [x.lower() for x in tokens[i:j]]
                    if t_lower in span_tokens:
                        ctx_start = max(0, i - 3)
                        ctx_end = min(len(tokens), j + 3)
                        span_ex = (list(tokens[ctx_start:ctx_end]),
                                   list(tags[ctx_start:ctx_end]))
                        results.append(span_ex)
                        break
                    i = j
                else:
                    i += 1
            if len(results) >= top_m:
                break
        return results[:top_m]
