# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Derived semantics: what distinguishes a chunk, and what identifies a document."""

from __future__ import annotations

from nlght.core.ingestion import (
    derive_chunk_semantics,
    tokenize,
)


def test_tokenization_is_deterministic_and_keeps_digits() -> None:
    assert tokenize("Spring Boot 3.2 uses Java17!") == [
        "spring", "boot", "3", "2", "uses", "java17",
    ]


def test_a_term_common_to_every_chunk_describes_the_document_not_the_chunk() -> None:
    chunks = ["widget alpha", "widget beta", "widget gamma"]

    semantics = derive_chunk_semantics(chunks, max_df_ratio=0.5)

    # "widget" is in all three, so it cannot distinguish any of them.
    assert all("widget" not in terms for terms in semantics)
    assert semantics[0] == ["alpha"]


def test_distinguishing_terms_are_kept() -> None:
    chunks = ["authentication token expiry", "database connection pooling"]

    semantics = derive_chunk_semantics(chunks)

    assert "authentication" in semantics[0]
    assert "database" in semantics[1]


def test_top_k_bounds_the_term_list() -> None:
    chunks = [" ".join(f"term{i}" for i in range(50)), "other"]

    semantics = derive_chunk_semantics(chunks, top_k=5)

    assert len(semantics[0]) == 5


def test_no_chunks_yields_no_semantics() -> None:
    assert derive_chunk_semantics([]) == []
