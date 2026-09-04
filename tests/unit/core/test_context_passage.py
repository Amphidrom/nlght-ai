# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""The passage contract, as invariants.

This layer is defined by what it refuses to do, so most of these pin an absence:
no expansion, no truncation, no re-ranking, no judgement about overlap. A test
suite that only checked the happy path would be green for an implementation that
quietly did all four.
"""

from __future__ import annotations

import random

import pytest

from nlght.core.context import (
    ContextBudget,
    ContextPassage,
    build_context,
    passages_from,
    select,
)
from nlght.core.retrieval import (
    ASSERTION,
    CHUNK,
    DOCUMENT,
    KNOWLEDGE,
    LEXICAL,
    VECTOR,
    Provenance,
    RetrievalHit,
    fuse,
    rank_within_sources,
)


def _hit(
    source: str,
    carrier: str,
    carrier_id: str,
    content: str,
    *,
    document_id: str = "",
    processing_revision_id: str = "",
    path: str = "",
    assertion_id: str = "",
) -> RetrievalHit:
    return RetrievalHit(
        source=source,
        carrier=carrier,
        carrier_id=carrier_id,
        content=content,
        provenance=Provenance(
            document_id=document_id,
            chunk_id=carrier_id if carrier == CHUNK else "",
            assertion_id=assertion_id,
            processing_revision_id=processing_revision_id,
            path=path,
        ),
    )


def _fused(*hits: RetrievalHit):  # noqa: ANN202
    return fuse(rank_within_sources(list(hits)))


# 1. The text is the finding's text
# ---------------------------------------------------------------------------

def test_a_passage_is_exactly_what_the_store_returned() -> None:
    """No expansion and no reconstruction.

    Neighbour expansion is not possible from what the pipeline indexes — the
    chunk payload carries no position and no offsets — and expanding a chunk to
    its whole document would inflate a finding retrieval deliberately narrowed.
    Neither is approximated here.
    """
    hit = _hit(VECTOR, CHUNK, "c1", "the chunk text", document_id="d1")

    passages = passages_from(_fused(hit))

    assert [passage.text for passage in passages] == ["the chunk text"]


# 2. Order comes from retrieval, and only from retrieval
# ---------------------------------------------------------------------------

def test_the_order_survives_a_permutation_of_the_input() -> None:
    hits = [
        _hit(LEXICAL, DOCUMENT, "d1", "first"),
        _hit(LEXICAL, DOCUMENT, "d2", "second"),
        _hit(VECTOR, CHUNK, "c1", "third", document_id="d1"),
        _hit(VECTOR, CHUNK, "c2", "fourth", document_id="d2"),
    ]
    expected = [passage.carrier_id for passage in passages_from(_fused(*hits))]

    shuffled = list(hits)
    random.Random(7).shuffle(shuffled)
    # Each store's own order is its ranking, so a permutation *within* a store
    # is a different ranking and would rightly change the answer. What must not
    # change the answer is the order the stores happened to be asked in.
    by_source = {LEXICAL: [], VECTOR: []}
    for hit in shuffled:
        by_source[hit.source].append(hit)
    regrouped = [
        *sorted(by_source[LEXICAL], key=lambda hit: hit.carrier_id),
        *sorted(by_source[VECTOR], key=lambda hit: hit.carrier_id),
    ]

    assert [p.carrier_id for p in passages_from(_fused(*regrouped))] == expected


def test_ranks_are_positions_and_run_from_one() -> None:
    passages = passages_from(_fused(
        _hit(LEXICAL, DOCUMENT, "d1", "a"),
        _hit(LEXICAL, DOCUMENT, "d2", "b"),
    ))

    assert [passage.rank for passage in passages] == [1, 2]


# 3-5. What is one passage, and what is two
# ---------------------------------------------------------------------------

def test_two_stores_finding_one_thing_make_one_passage() -> None:
    passages = passages_from(_fused(
        _hit(LEXICAL, DOCUMENT, "d1", "the document", processing_revision_id="r1"),
        _hit(VECTOR, DOCUMENT, "d1", "the document", processing_revision_id="r1"),
    ))

    assert len(passages) == 1
    assert passages[0].found_by == (LEXICAL, VECTOR), "both finders survive"


def test_a_document_and_a_chunk_of_it_stay_two_passages() -> None:
    """Overlapping text is not evidence that either finding is redundant.

    Dropping the document because a chunk of it is present would be this layer
    deciding which finding is the better one — an interpretation, in the one
    place in the pipeline that is supposed to make none.
    """
    passages = passages_from(_fused(
        _hit(LEXICAL, DOCUMENT, "d1", "the whole document, including the part"),
        _hit(VECTOR, CHUNK, "c1", "the part", document_id="d1"),
    ))

    assert len(passages) == 2
    assert {passage.carrier for passage in passages} == {DOCUMENT, CHUNK}


def test_an_assertion_and_a_document_of_one_source_stay_two_passages() -> None:
    passages = passages_from(_fused(
        _hit(LEXICAL, DOCUMENT, "d1", "the document text", path="a.adoc"),
        _hit(KNOWLEDGE, ASSERTION, "fp1", "the claim", assertion_id="fp1"),
    ))

    assert len(passages) == 2


def test_sharing_a_document_id_is_not_being_a_duplicate() -> None:
    # The guard against the tempting shortcut: two findings about one file are
    # two findings.
    passages = passages_from(_fused(
        _hit(VECTOR, CHUNK, "c1", "first part", document_id="d1"),
        _hit(VECTOR, CHUNK, "c2", "second part", document_id="d1"),
    ))

    assert len(passages) == 2
    assert {passage.provenance.document_id for passage in passages} == {"d1"}


def test_one_carrier_in_two_revisions_is_two_passages() -> None:
    """Two states of one thing, and choosing between them is not this layer's job."""
    passages = passages_from(_fused(
        _hit(LEXICAL, DOCUMENT, "d1", "the old wording", processing_revision_id="r1"),
        _hit(VECTOR, DOCUMENT, "d1", "the new wording", processing_revision_id="r2"),
    ))

    assert len(passages) == 2
    assert {passage.provenance.processing_revision_id for passage in passages} == {"r1", "r2"}


# 6-8. The budget
# ---------------------------------------------------------------------------

def _passage(rank: int, text: str) -> ContextPassage:
    return ContextPassage(
        text=text, carrier=DOCUMENT, carrier_id=f"d{rank}",
        found_by=(LEXICAL,), rank=rank,
    )


def test_the_budget_is_spent_strictly_in_rank_order() -> None:
    selection = select(
        [_passage(1, "a" * 10), _passage(2, "b" * 10), _passage(3, "c" * 10)],
        ContextBudget(max_chars=20),
    )

    assert [passage.rank for passage in selection.passages] == [1, 2]
    assert selection.used_chars == 20
    assert [passage.rank for passage in selection.omitted] == [3]


def test_a_passage_that_does_not_fit_is_dropped_and_never_cut() -> None:
    selection = select([_passage(1, "x" * 500)], ContextBudget(max_chars=300))

    assert selection.passages == ()
    assert selection.used_chars == 0
    assert selection.omitted[0].text == "x" * 500, "the finding is untouched"


def test_a_smaller_budget_only_removes_from_the_bottom() -> None:
    """The reason a non-fitting passage stops the selection rather than being skipped.

    Skipping ahead to a smaller passage would let a *smaller* budget drop
    something from the middle of the list, and two budgets would no longer be
    comparable. Taking a prefix is what makes "less budget, less context, from
    the bottom" true — at the cost of sometimes leaving budget unspent, which
    `omitted` shows rather than hides.
    """
    passages = [_passage(1, "a" * 10), _passage(2, "b" * 50), _passage(3, "c" * 5)]

    generous = select(passages, ContextBudget(max_chars=100)).passages
    tight = select(passages, ContextBudget(max_chars=20)).passages

    assert [p.rank for p in generous] == [1, 2, 3]
    assert [p.rank for p in tight] == [1]
    assert list(tight) == list(generous[: len(tight)]), "a prefix, never a gap"


def test_a_budget_of_nothing_takes_nothing() -> None:
    selection = select([_passage(1, "a")], ContextBudget(max_chars=0))

    assert selection.passages == ()
    assert len(selection.omitted) == 1


def test_a_negative_budget_is_refused() -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        ContextBudget(max_chars=-1)


def test_cost_is_characters_and_says_so() -> None:
    # Not estimated tokens. The platform's TokenBudget owns the model's context
    # window; translating its share into characters happens where the model is
    # known, and this layer claims no precision it does not have.
    assert _passage(1, "hello").cost == 5


# 9-10. What must survive, and what must not appear
# ---------------------------------------------------------------------------

def test_provenance_survives_whole() -> None:
    selection = build_context(
        _fused(_hit(
            LEXICAL, DOCUMENT, "d1", "text",
            document_id="d1", processing_revision_id="r7", path="docs/a.adoc",
        )),
        ContextBudget(max_chars=100),
    )

    provenance = selection.passages[0].provenance
    assert (provenance.document_id, provenance.processing_revision_id, provenance.path) == (
        "d1", "r7", "docs/a.adoc",
    )


def test_a_knowledge_passage_claims_no_location_it_does_not_have() -> None:
    """The knowledge route records an assertion id and nothing else.

    So a passage from one has no path and no document, and must not pretend
    otherwise: a citation to a line nobody can look up is worse than no line.
    """
    passages = passages_from(_fused(
        _hit(KNOWLEDGE, ASSERTION, "fp1", "the claim", assertion_id="fp1")
    ))

    provenance = passages[0].provenance
    assert provenance.assertion_id == "fp1"
    assert (provenance.path, provenance.document_id) == ("", "")


def test_no_score_reaches_a_passage() -> None:
    # A passage carries a position and no relevance of its own. A score here
    # would be a second ranking, formed from less information than the first.
    passage = passages_from(_fused(_hit(LEXICAL, DOCUMENT, "d1", "text")))[0]

    assert not hasattr(passage, "score")
    assert not hasattr(passage, "raw_score")
    assert passage.rank == 1
