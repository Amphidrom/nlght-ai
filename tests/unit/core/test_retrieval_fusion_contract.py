# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Layer A — the fusion contract, as invariants rather than examples.

    a store decides local relevance
    retrieval fuses rankings
    retrieval never reads store scores as a shared unit

Hand-written examples prove that one arrangement came out a particular way.
These prove properties that must hold for every arrangement, which is what makes
the fusion algorithm replaceable later without reopening the question of what
"clean ranking" means.

Most of them are metamorphic: change the input in a way that must not matter,
and require the output not to move. A rank-based fusion is invariant under any
monotone transformation of a store's scores, so scaling BM25 by a thousand is a
test that can only fail one way — by magnitude having leaked back in.
"""

from __future__ import annotations

from dataclasses import replace

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

_CARRIER = {LEXICAL: DOCUMENT, VECTOR: CHUNK, KNOWLEDGE: ASSERTION}


def hit(source: str, carrier_id: str, *, score: float = 1.0, rank: int = 0) -> RetrievalHit:
    return RetrievalHit(
        source=source,
        carrier=_CARRIER[source],
        carrier_id=carrier_id,
        content=f"{source}:{carrier_id}",
        raw_score=score,
        source_rank=rank,
        provenance=Provenance(document_id=carrier_id),
    )


def order(fused) -> list[tuple[str, str]]:  # noqa: ANN001
    return [item.key for item in fused]


# --- 1. a store's own ranking survives -------------------------------------

def test_one_store_alone_keeps_its_own_order() -> None:
    """Fusion may not reorder a store that nothing else contradicts."""
    ranked = rank_within_sources([hit(LEXICAL, name) for name in ("a", "b", "c")])

    assert order(fuse(ranked)) == [(DOCUMENT, "a"), (DOCUMENT, "b"), (DOCUMENT, "c")]


@pytest.mark.parametrize("source", [LEXICAL, VECTOR, KNOWLEDGE])
def test_that_holds_for_every_store(source: str) -> None:
    ranked = rank_within_sources([hit(source, name) for name in ("a", "b", "c")])

    assert [key[1] for key in order(fuse(ranked))] == ["a", "b", "c"]


# --- 2. magnitude means nothing across stores ------------------------------

def test_scaling_one_stores_scores_changes_nothing() -> None:
    """The regression test for the defect a first attempt shipped.

    BM25 near 100 outranked an assertion at 0.82 on magnitude alone. If this
    goes red, raw scores have leaked back into the comparison.
    """
    lexical = [hit(LEXICAL, "d1", score=100.0), hit(LEXICAL, "d2", score=99.0)]
    knowledge = [hit(KNOWLEDGE, "a1", score=0.82)]
    baseline = fuse(rank_within_sources([*lexical, *knowledge]))

    scaled = fuse(rank_within_sources([
        *(replace(item, raw_score=item.raw_score * 1000) for item in lexical),
        *knowledge,
    ]))

    assert order(scaled) == order(baseline)


def test_shifting_one_stores_scores_changes_nothing() -> None:
    lexical = [hit(LEXICAL, "d1", score=7.1), hit(LEXICAL, "d2", score=7.0)]
    knowledge = [hit(KNOWLEDGE, "a1", score=0.5)]
    baseline = fuse(rank_within_sources([*lexical, *knowledge]))

    shifted = fuse(rank_within_sources([
        *(replace(item, raw_score=item.raw_score + 1000) for item in lexical),
        *knowledge,
    ]))

    assert order(shifted) == order(baseline)


def test_a_flat_score_distribution_is_not_stretched_into_a_ranking() -> None:
    """Why min-max normalisation was rejected.

    Store A returns 7.1, 7.0, 6.9 — three near-identical results. Normalising
    per store gives its top a 1.0 and its bottom a 0.0, inventing a spread that
    the store did not express. Under rank fusion the three stay adjacent.
    """
    flat = rank_within_sources([
        hit(LEXICAL, "d1", score=7.1), hit(LEXICAL, "d2", score=7.0),
        hit(LEXICAL, "d3", score=6.9),
    ])
    spread = rank_within_sources([
        hit(LEXICAL, "d1", score=0.81), hit(LEXICAL, "d2", score=0.42),
        hit(LEXICAL, "d3", score=0.10),
    ])

    assert [item.score for item in fuse(flat)] == [item.score for item in fuse(spread)]


# --- 3. a better position never hurts --------------------------------------

def test_moving_a_hit_up_its_own_list_cannot_lower_its_fused_score() -> None:
    deeper = fuse(rank_within_sources([hit(LEXICAL, f"d{i}") for i in range(5)]))
    higher = fuse(rank_within_sources(
        [hit(LEXICAL, "d4"), *(hit(LEXICAL, f"d{i}") for i in range(4))]
    ))

    before = next(item for item in deeper if item.key == (DOCUMENT, "d4"))
    after = next(item for item in higher if item.key == (DOCUMENT, "d4"))

    assert after.score >= before.score


# --- 4. a second store agreeing strengthens --------------------------------

def _chunk_from(source: str, carrier_id: str) -> RetrievalHit:
    """The same chunk, found by a different system.

    Two stores agree when they return the *same carrier*, not when two ids
    happen to read alike: an assertion called `c1` and a chunk called `c1` are
    different things, and the helper above ties each source to its usual carrier
    precisely so that is not glossed over.
    """
    return RetrievalHit(
        source=source, carrier=CHUNK, carrier_id=carrier_id,
        content=f"{source}:{carrier_id}", provenance=Provenance(chunk_id=carrier_id),
    )


def test_a_finding_two_stores_made_outranks_the_same_finding_from_one() -> None:
    """The central reason to fuse at all.

    Both stores returned `c1`; only one of them returned `c9`. The agreement is
    the difference, and it is the only difference.
    """
    ranked = rank_within_sources([
        _chunk_from(VECTOR, "c9"), _chunk_from(VECTOR, "c1"),
        _chunk_from(LEXICAL, "c1"), _chunk_from(LEXICAL, "c8"),
    ])

    fused = fuse(ranked)
    agreed = next(item for item in fused if item.key == (CHUNK, "c1"))

    assert agreed.sources == (LEXICAL, VECTOR)
    assert agreed is fused[0]
    assert agreed.score > next(item for item in fused if item.key == (CHUNK, "c9")).score


def test_how_far_agreement_outweighs_position_is_the_damping(  # noqa: D103
) -> None:
    """The knob that decides what fusion is actually *for*, pinned in both directions.

    Reciprocal rank fusion adds `damping` to every rank, so the damping decides
    how much of the score a position can still move:

        damping 60   one store's first place   0.0164
                     two stores at rank 10/11  0.0284   → agreement wins
        damping 5    one store's first place   0.1667
                     two stores at rank 10/11  0.1292   → position wins

    At the published 60 and lists of tens, the fusion is close to counting
    sources; low damping makes it close to trusting whichever store was most
    confident. Neither is wrong, and the choice is not the algorithm's to make —
    so it is configuration, and both ends of it are held here rather than
    discovered later by someone tuning it blind.
    """
    lexical = [hit(LEXICAL, "top"), *(hit(LEXICAL, f"pad{i}") for i in range(9)),
               _chunk_from(LEXICAL, "deep")]
    vector = [*(_chunk_from(VECTOR, f"c{i}") for i in range(9)), _chunk_from(VECTOR, "deep")]
    ranked = rank_within_sources([*lexical, *vector])

    def rank_of(fused, key):  # noqa: ANN001, ANN202
        return next(item.score for item in fused if item.key == key)

    damped = fuse(ranked, weights=FusionWeights(damping=60.0))
    sharp = fuse(ranked, weights=FusionWeights(damping=5.0))

    # Compared against each other rather than against the whole list: the point
    # is which of these two the fusion prefers, not who else happened to be
    # first in their own store.
    assert rank_of(damped, (CHUNK, "deep")) > rank_of(damped, (DOCUMENT, "top"))
    assert rank_of(sharp, (CHUNK, "deep")) < rank_of(sharp, (DOCUMENT, "top"))


# --- 5. weights act monotonically ------------------------------------------

def test_raising_a_stores_weight_cannot_weaken_its_contribution() -> None:
    ranked = rank_within_sources([hit(LEXICAL, "d1"), hit(KNOWLEDGE, "a1")])

    plain = fuse(ranked, weights=FusionWeights(sources={KNOWLEDGE: 1.0, LEXICAL: 1.0}))
    louder = fuse(ranked, weights=FusionWeights(sources={KNOWLEDGE: 2.0, LEXICAL: 1.0}))

    before = next(item for item in plain if item.key == (ASSERTION, "a1")).score
    after = next(item for item in louder if item.key == (ASSERTION, "a1")).score

    assert after > before


def test_a_zero_weight_means_the_store_does_not_shape_the_ranking() -> None:
    """And nothing else. Its hits still arrive, with their provenance.

    Not searching a store and giving it no vote are different things, and an
    operator may legitimately want the second.
    """
    ranked = rank_within_sources([hit(LEXICAL, "d1"), hit(KNOWLEDGE, "a1")])

    fused = fuse(ranked, weights=FusionWeights(sources={KNOWLEDGE: 0.0, LEXICAL: 1.0}))
    silent = next(item for item in fused if item.key == (ASSERTION, "a1"))

    assert silent.score == 0.0
    assert silent.hits, "a store with no vote still returned something"


# --- 6. an absent store changes nothing about the present ones -------------

def test_a_store_that_was_never_searched_leaves_no_trace() -> None:
    """Available-and-not-planned and absent must be the same result.

    Otherwise a platform without a vector index gets a subtly different ranking
    from one that has it and did not use it, and neither is explainable.
    """
    without = fuse(rank_within_sources([hit(LEXICAL, "d1"), hit(KNOWLEDGE, "a1")]))
    also_without = fuse(rank_within_sources([hit(KNOWLEDGE, "a1"), hit(LEXICAL, "d1")]))

    assert order(without) == order(also_without)
    assert all(VECTOR not in item.sources for item in without)


# --- 7. a long tail cannot outvote a short strong list ---------------------

def test_adding_irrelevant_tail_hits_does_not_disturb_the_top() -> None:
    """A store returning a hundred results has not thereby said more.

    The strongest guard against any per-batch normalisation creeping back:
    lengthening one store's list changes what that store said about its tail and
    nothing about its head.
    """
    core = [hit(LEXICAL, "d1"), hit(LEXICAL, "d2"), hit(KNOWLEDGE, "a1")]
    baseline = fuse(rank_within_sources(core))

    padded = fuse(rank_within_sources(
        [*core, *(hit(LEXICAL, f"tail{i}") for i in range(97))]
    ))

    assert order(padded)[:3] == order(baseline)[:3]
    assert [item.score for item in padded[:3]] == [item.score for item in baseline[:3]]


# --- 8. order of arrival is not information --------------------------------

def test_permuting_the_input_changes_nothing_when_ranks_are_given() -> None:
    hits = [
        hit(LEXICAL, "d1", rank=1), hit(LEXICAL, "d2", rank=2),
        hit(KNOWLEDGE, "a1", rank=1), hit(VECTOR, "c1", rank=1),
    ]

    assert order(fuse(hits)) == order(fuse(list(reversed(hits))))


# --- 9. ties are deterministic ---------------------------------------------

def test_two_findings_with_one_score_always_come_out_in_the_same_order() -> None:
    """A golden ranking test is worth nothing if the order can drift.

    Both are their own store's first result, so the score is identical and the
    tie is broken by the carrier and its id.
    """
    hits = [hit(LEXICAL, "d1", rank=1), hit(KNOWLEDGE, "a1", rank=1)]

    first = order(fuse(hits))
    again = order(fuse(list(reversed(hits))))

    assert first == again
    assert first[0] < first[1], "the tie-break is the carrier and id, ascending"


# --- 10. granularity is not identity ---------------------------------------

def test_a_document_a_chunk_of_it_and_a_claim_from_it_stay_three_findings() -> None:
    """Three systems each found something, and that is the result.

    Merging them on a shared `document_id` would throw away exactly that, and it
    would answer a question — which text does the model get — that belongs to
    context building, with no view of a token budget.
    """
    document = RetrievalHit(source=LEXICAL, carrier=DOCUMENT, carrier_id="d1", content="doc",
                            provenance=Provenance(document_id="d1"))
    chunk = RetrievalHit(source=VECTOR, carrier=CHUNK, carrier_id="c7", content="chunk",
                         provenance=Provenance(document_id="d1", chunk_id="c7"))
    claim = RetrievalHit(source=KNOWLEDGE, carrier=ASSERTION, carrier_id="a1", content="claim",
                         provenance=Provenance(document_id="d1", assertion_id="a1"))

    fused = fuse(rank_within_sources([document, chunk, claim]))

    assert len(fused) == 3
    assert {item.carrier for item in fused} == {DOCUMENT, CHUNK, ASSERTION}
    # And the relationship is visible without being acted on.
    assert {item.best.provenance.document_id for item in fused} == {"d1"}


def test_the_same_carrier_from_two_stores_is_one_finding() -> None:
    # Relatedness is not identity; sameness is. Two stores returning chunk `c7`
    # found one thing twice, and that is the agreement fusion counts.
    ranked = rank_within_sources([
        RetrievalHit(source=VECTOR, carrier=CHUNK, carrier_id="c7", content="a"),
        RetrievalHit(source=LEXICAL, carrier=CHUNK, carrier_id="c7", content="b"),
    ])

    [fused] = fuse(ranked)

    assert fused.sources == (LEXICAL, VECTOR)
    assert len(fused.hits) == 2, "both provenances survive"


# --- 11. a store cannot vote twice -----------------------------------------

def test_one_store_returning_a_carrier_twice_votes_once() -> None:
    """Otherwise a store could outweigh the others by repeating itself."""
    ranked = rank_within_sources([
        hit(LEXICAL, "d1"), hit(LEXICAL, "d1"), hit(LEXICAL, "d2"),
    ])

    assert [item.source_rank for item in ranked] == [1, 2]
    assert [item.carrier_id for item in ranked] == ["d1", "d2"]


def test_a_hit_with_no_rank_is_refused_rather_than_ranked_last() -> None:
    # Silently treating it as the worst result would make a wiring bug look like
    # a poor search.
    with pytest.raises(ValueError, match="no source rank"):
        fuse([hit(LEXICAL, "d1")])


# ---------------------------------------------------------------------------
# three revisions, three names (ADR-0062)
# ---------------------------------------------------------------------------


def test_the_three_revision_identities_do_not_share_a_field() -> None:
    """Each name answers one question, and none of them answers another's.

    Before the split there was one `revision_id` carrying the source fassung,
    the processed fassung or a claim's state depending on the carrier — so a
    consumer reading it got a different kind of answer per hit and nothing said
    so.
    """
    document = Provenance(
        document_id="d1",
        source_revision_id="src-7",
        processing_revision_id="proc-7",
    )
    assertion = Provenance(assertion_id="a1", knowledge_revision_id="know-7")

    # A document has both of its own fassungen and no claim state.
    assert document.source_revision_id == "src-7"
    assert document.processing_revision_id == "proc-7"
    assert document.knowledge_revision_id == ""
    # A claim has a state and neither document fassung.
    assert assertion.knowledge_revision_id == "know-7"
    assert assertion.source_revision_id == ""
    assert assertion.processing_revision_id == ""


def test_the_state_revision_reads_whichever_the_carrier_actually_has() -> None:
    """One derived question — which state is this finding on — for deduplication.

    It is not a fourth stored value: the fields above keep one meaning each, and
    this reads the one the carrier has.
    """
    document = Provenance(document_id="d1", processing_revision_id="proc-7")
    assertion = Provenance(assertion_id="a1", knowledge_revision_id="know-7")

    assert document.state_revision == "proc-7"
    assert assertion.state_revision == "know-7"
    # A processing revision and a knowledge revision are never compared, because
    # a document and an assertion are never the same carrier to begin with.
    assert Provenance().state_revision == ""


def test_the_source_fassung_does_not_move_when_only_the_pipeline_changed() -> None:
    """Which is why a citation names it and the currency check does not.

    Re-chunking a document produces a new processed fassung and the same source
    fassung. A reader comparing the wrong one would either re-cite everything or
    drop every hit after a pipeline change.
    """
    before = Provenance(document_id="d1", source_revision_id="src-7",
                        processing_revision_id="proc-a")
    after = Provenance(document_id="d1", source_revision_id="src-7",
                       processing_revision_id="proc-b")

    assert before.source_revision_id == after.source_revision_id
    assert before.processing_revision_id != after.processing_revision_id
    assert before.state_revision != after.state_revision
