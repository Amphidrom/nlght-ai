# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Layer C — controlled scenarios with a ranking worked out beforehand.

The invariants in the contract file say what must never happen. These say what
*does* happen, for a corpus small enough to compute by hand:

    score(finding) = Σ weight(source) / (damping + rank in that source)

The expected order below is derived from that formula and written out, not
observed from a run and pasted back. A golden test that records whatever the
code did is a change detector; one that records what the contract requires is a
specification, and only the second is worth keeping when the fusion algorithm is
replaced.

Each scenario states the arithmetic in its docstring, so a failure is read as
"the contract says 0.0328 and we got 0.0164" rather than as a list that moved.
"""

from __future__ import annotations

import pytest

from nlght.core.retrieval import (
    ASSERTION,
    CHUNK,
    DOCUMENT,
    KNOWLEDGE,
    LEXICAL,
    VECTOR,
    FusionWeights,
    Provenance,
    RetrievalHit,
    fuse,
    rank_within_sources,
)

K = 60.0


def rrf(*ranks: int, weight: float = 1.0) -> float:
    """The contract, spelled out, so the expectations are computed and not copied."""
    return sum(weight / (K + rank) for rank in ranks)


def lexical(document: str, text: str) -> RetrievalHit:
    return RetrievalHit(
        source=LEXICAL, carrier=DOCUMENT, carrier_id=document, content=text,
        raw_score=9.0, provenance=Provenance(document_id=document, path=f"{document}.adoc"),
    )


def chunk(chunk_id: str, document: str, text: str) -> RetrievalHit:
    return RetrievalHit(
        source=VECTOR, carrier=CHUNK, carrier_id=chunk_id, content=text,
        raw_score=0.7, provenance=Provenance(document_id=document, chunk_id=chunk_id),
    )


def claim(assertion: str, document: str, text: str) -> RetrievalHit:
    return RetrievalHit(
        source=KNOWLEDGE, carrier=ASSERTION, carrier_id=assertion, content=text,
        raw_score=0.9, provenance=Provenance(assertion_id=assertion, document_id=document),
    )


def ranked(*hits: RetrievalHit):  # noqa: ANN201
    return fuse(rank_within_sources(list(hits)))


# --- 1 -----------------------------------------------------------------------

def test_required_java_version() -> None:
    """Three systems, one topic, and the order the contract requires.

        lexical    1. d1  mentions Java in passing
                   2. d2  "Spring Boot requires Java 17"
        vector     1. c2  the relevant passage of d2
        knowledge  1. a1  the reviewed claim

    Nothing is found twice, so every finding scores its single reciprocal rank:

        a1  1/61 = 0.016393      d1  1/61 = 0.016393
        c2  1/61 = 0.016393      d2  1/62 = 0.016129

    Three findings tie at the top, which is the honest answer — each system's
    first choice, and no evidence yet to separate them. The tie is broken by
    carrier and id so the list is stable, and `d2` comes last because its own
    store ranked it second.
    """
    fused = ranked(
        lexical("d1", "Java is mentioned in passing."),
        lexical("d2", "Spring Boot requires Java 17."),
        chunk("c2", "d2", "Spring Boot requires Java 17."),
        claim("a1", "d2", "Spring Boot requires Java 17."),
    )

    assert [item.key for item in fused] == [
        (ASSERTION, "a1"), (CHUNK, "c2"), (DOCUMENT, "d1"), (DOCUMENT, "d2"),
    ]
    assert [round(item.score, 6) for item in fused] == [
        round(rrf(1), 6), round(rrf(1), 6), round(rrf(1), 6), round(rrf(2), 6),
    ]


# --- 2 -----------------------------------------------------------------------

def test_two_systems_finding_the_same_chunk() -> None:
    """Agreement is the only thing that separates them, and it is decisive.

        vector   1. c7   2. c9
        lexical  1. c7            (a lexical index over chunks)

        c7  1/61 + 1/61 = 0.032787
        c9  1/62         = 0.016129
    """
    both_c7 = [
        chunk("c7", "d1", "keystore password"),
        chunk("c9", "d2", "something else"),
        RetrievalHit(source=LEXICAL, carrier=CHUNK, carrier_id="c7",
                     content="keystore password", raw_score=8.0,
                     provenance=Provenance(document_id="d1", chunk_id="c7")),
    ]

    fused = ranked(*both_c7)

    assert [item.key for item in fused] == [(CHUNK, "c7"), (CHUNK, "c9")]
    assert round(fused[0].score, 6) == round(rrf(1, 1), 6)
    assert round(fused[1].score, 6) == round(rrf(2), 6)
    assert fused[0].sources == (LEXICAL, VECTOR)


# --- 3 -----------------------------------------------------------------------

def test_a_weighted_knowledge_store_outranks_a_lexical_first_place() -> None:
    """What a weight is *for*, computed rather than asserted.

        weights   knowledge 2.0, lexical 1.0
        a1  2.0/61 = 0.032787
        d1  1.0/61 = 0.016393

    An operator who trusts reviewed claims over prose says so here, and the
    arithmetic says exactly how much that trust is worth.
    """
    fused = fuse(
        rank_within_sources([lexical("d1", "prose"), claim("a1", "d1", "the reviewed claim")]),
        weights=FusionWeights(sources={KNOWLEDGE: 2.0, LEXICAL: 1.0, VECTOR: 1.0}),
    )

    assert [item.key for item in fused] == [(ASSERTION, "a1"), (DOCUMENT, "d1")]
    assert round(fused[0].score, 6) == round(rrf(1, weight=2.0), 6)
    assert round(fused[1].score, 6) == round(rrf(1), 6)


# --- 4 -----------------------------------------------------------------------

def test_a_deep_agreement_against_a_shallow_certainty() -> None:
    """The case that shows what the damping actually decides.

        lexical  1. top      ...  11. deep
        vector                    10. deep

        deep  1/71 + 1/70 = 0.028370
        top   1/61        = 0.016393

    At the default damping the two systems agreeing outweigh one system's first
    place — which is a property of the damping, not of the corpus, and is held
    in both directions in the contract tests.
    """
    hits = [
        lexical("top", "the best single answer"),
        *(lexical(f"pad{index}", "filler") for index in range(9)),
        RetrievalHit(source=LEXICAL, carrier=CHUNK, carrier_id="deep", content="deep",
                     raw_score=1.0, provenance=Provenance(chunk_id="deep")),
        *(chunk(f"c{index}", "dx", "filler") for index in range(9)),
        chunk("deep", "d9", "deep"),
    ]

    fused = ranked(*hits)

    assert fused[0].key == (CHUNK, "deep")
    assert round(fused[0].score, 6) == round(rrf(11, 10), 6)
    assert round(
        next(item.score for item in fused if item.key == (DOCUMENT, "top")), 6
    ) == round(rrf(1), 6)


# --- 5 -----------------------------------------------------------------------

@pytest.mark.parametrize("repeats", [1, 2, 3])
def test_the_same_corpus_always_ranks_the_same_way(repeats: int) -> None:
    """A golden ranking is worth nothing if the order can drift between runs."""
    hits = [
        lexical("d1", "one"), lexical("d2", "two"),
        chunk("c1", "d1", "three"), claim("a1", "d1", "four"),
    ]

    orders = [[item.key for item in ranked(*hits)] for _ in range(repeats + 1)]

    assert all(order == orders[0] for order in orders)
