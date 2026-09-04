# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Derived semantics for indexed documents.

Ported from ``next-integration/data-indexer/indexing/pipeline.py``. Two
independent derivations, both purely statistical or structural — no model, no
heuristics that drift:

* **chunk semantics** — the terms that distinguish one chunk from its siblings,
  by term frequency against inverse document frequency within the document.
* **structural semantics** — identity tokens expanded from a path or fully
  qualified name, so ``billing.service`` is findable as ``billing`` and
  ``service`` too.

They feed two different searches. Chunk semantics improve content retrieval;
structural semantics back symbol and identifier lookup, which is why the
prototype writes them as a *separate* vector per document rather than mixing
them into the content vectors.
"""

from __future__ import annotations

import re
from collections import Counter

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Deterministic tokenisation: regex only, no stopwords, no stemming.

    Single source of truth for both derivations, so a term that survives one
    survives the other.
    """
    return [
        match.group(0).lower()
        for match in _TOKEN_RE.finditer(text or "")
        if len(match.group(0)) >= 2 or match.group(0).isdigit()
    ]


def derive_chunk_semantics(
    chunks: list[str],
    *,
    top_k: int = 12,
    min_df: int = 1,
    max_df_ratio: float = 0.80,
) -> list[list[str]]:
    """Terms that distinguish each chunk from the rest of its document.

    Scored as term frequency divided by document frequency: a word appearing in
    almost every chunk describes the document, not the chunk, so
    ``max_df_ratio`` drops it. What remains is what makes a chunk findable on
    its own.
    """
    chunk_tokens = [tokenize(chunk) for chunk in chunks]
    if not chunk_tokens:
        return []

    document_frequency: Counter[str] = Counter()
    for tokens in chunk_tokens:
        document_frequency.update(set(tokens))

    max_df = int(len(chunk_tokens) * max_df_ratio)
    eligible = {
        token
        for token, frequency in document_frequency.items()
        if min_df <= frequency <= max_df
    }

    semantics: list[list[str]] = []
    for tokens in chunk_tokens:
        frequencies = Counter(tokens)
        scored = sorted(
            (
                (count / document_frequency[token], token)
                for token, count in frequencies.items()
                if token in eligible
            ),
            reverse=True,
        )
        semantics.append([token for _, token in scored[:top_k]])
    return semantics
