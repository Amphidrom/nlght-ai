# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Derive deterministic textual terms for indexed document chunks."""

from __future__ import annotations

import re
from collections import Counter

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Tokenize text consistently for semantic term derivation.

    Args:
        text (str): Source text.

    Returns:
        list[str]: Lowercase alphanumeric tokens of at least two characters, plus digits.
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
    """Rank terms that distinguish chunks within one document.

    Args:
        chunks (list[str]): Chunk texts from one document.
        top_k (int): Maximum terms returned for each chunk.
        min_df (int): Minimum document frequency for an eligible term.
        max_df_ratio (float): Maximum share of chunks in which a term may appear.

    Returns:
        list[list[str]]: Distinctive terms for each chunk, in the same order as the input.
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
