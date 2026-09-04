# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Knowledge graph authority: identity, reinforcement, quarantine, review audit."""

from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nlght.adapters.outbound.persistence.knowledge_models import KnowledgeObservation
from nlght.adapters.outbound.persistence.knowledge_repository import (
    SqlAlchemyKnowledgeRepository,
)
from nlght.core.knowledge import (
    DecisionPayload,
    FactPayload,
    KnowledgeWrite,
    PatternPayload,
    ReviewConflictError,
    ReviewDecision,
    RulePayload,
    reinforce_confidence,
)


def _fact(
    identity: str = "fact-1",
    *,
    confidence: float = 0.6,
    run_id: str = "run-1",
    source_id: str = "source-1",
    review_required: bool = False,
    review_reason: str | None = None,
    metadata: dict[str, object] | None = None,
) -> KnowledgeWrite:
    return KnowledgeWrite(
        identity=identity,
        payload=FactPayload(subject="Spring", predicate="requires", object="Java 17"),
        type="dependency",
        confidence=confidence,
        source_id=source_id,
        run_id=run_id,
        product="spring-boot",
        version="3.2",
        review_required=review_required,
        review_reason=review_reason,
        evidence={"quote": "Spring Boot 3.2 requires Java 17"},
        metadata=metadata or {},
    )


@pytest.fixture
async def sqlite_engine():
    """A throwaway knowledge database — separate from the platform schema."""
    from sqlalchemy.ext.asyncio import create_async_engine

    from nlght.adapters.outbound.persistence.knowledge_models import KnowledgeBase

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(KnowledgeBase.metadata.create_all)
    yield engine
    await engine.dispose()


async def test_a_new_assertion_is_stored_with_its_payload_and_metadata(sqlite_engine) -> None:
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)

    stored = await repository.upsert(_fact(metadata={"origin": "owasp"}))

    assert stored.identity == "fact-1"
    assert stored.kind == "fact"
    assert stored.payload == {"subject": "Spring", "predicate": "requires", "object": "Java 17"}
    assert stored.metadata == {"origin": "owasp"}
    assert stored.review_required is False


async def test_an_independent_sighting_reinforces_confidence(sqlite_engine) -> None:
    """Independent means another source, not another run.

    Two documents saying the same thing is corroboration. The same document read
    twice is not, however many runs it takes — so what makes a sighting
    independent is the source it came from.
    """
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    first = await repository.upsert(_fact(confidence=0.6, source_id="source-1"))

    second = await repository.upsert(
        _fact(confidence=0.7, run_id="run-2", source_id="source-2")
    )

    assert second.confidence == pytest.approx(reinforce_confidence(0.6, 0.7))
    assert second.confidence > first.confidence


async def test_a_retried_step_does_not_count_the_same_evidence_twice(sqlite_engine) -> None:
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    await repository.upsert(_fact(run_id="run-1", source_id="source-1"))
    await repository.upsert(_fact(run_id="run-1", source_id="source-1"))

    async with AsyncSession(sqlite_engine) as session:
        observations = (
            await session.execute(
                select(KnowledgeObservation).where(KnowledgeObservation.identity == "fact-1")
            )
        ).scalars().all()

    assert len(observations) == 1


async def test_a_retried_step_does_not_reinforce_confidence_either(sqlite_engine) -> None:
    """The crash case: assertions written, the extraction record not yet.

    A run that dies between persisting its assertions and recording that it
    extracted the document is retried, and the retry writes the same assertions
    again under the same run. The observation is already guarded — but
    confidence was not, so the retry reinforced the node against itself and a
    claim grew more certain for having been interrupted.

    Reinforcement is for *independent* sightings. Seeing the same evidence twice
    because a worker crashed is one sighting.
    """
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    first = await repository.upsert(_fact(run_id="run-1", source_id="source-1"))

    retried = await repository.upsert(_fact(run_id="run-1", source_id="source-1"))

    assert retried.confidence == pytest.approx(first.confidence)


async def test_the_same_claim_from_another_source_still_reinforces(sqlite_engine) -> None:
    # The guard must not turn into "never reinforce": a second, genuinely
    # independent sighting is the reason the mechanism exists.
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    first = await repository.upsert(_fact(run_id="run-1", source_id="source-1"))

    second = await repository.upsert(_fact(run_id="run-1", source_id="source-2"))

    assert second.confidence > first.confidence


async def test_a_later_run_over_the_same_source_does_not_reinforce(sqlite_engine) -> None:
    """The correction. This test asserted the opposite and the premise was wrong.

    A later run over the same document is that document saying the same thing a
    second time. Calling it corroboration made a claim grow more certain for
    having been re-read — the extraction skip usually prevents ever getting
    here, but "usually" is not what belief should rest on.
    """
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    first = await repository.upsert(_fact(run_id="run-1", source_id="source-1"))

    second = await repository.upsert(_fact(run_id="run-2", source_id="source-1"))

    assert second.confidence == pytest.approx(first.confidence)


async def test_reusing_an_identity_for_another_kind_is_refused(sqlite_engine) -> None:
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    await repository.upsert(_fact("shared-identity"))

    conflicting = KnowledgeWrite(
        identity="shared-identity",
        payload=RulePayload(rule_text="Never log secrets"),
        type="security",
        confidence=0.9,
        source_id="source-2",
        run_id="run-2",
    )
    # Its own type, and a permanent one. Retrying cannot make the same input
    # resolve differently, and where a model produced that input a retry can
    # succeed by accident and hide the matcher fault that caused it.
    from nlght.core.errors.errors import PermanentError  # noqa: PLC0415
    from nlght.core.knowledge import KnowledgeKindConflictError  # noqa: PLC0415

    assert issubclass(KnowledgeKindConflictError, PermanentError)
    with pytest.raises(KnowledgeKindConflictError, match="already exists as kind"):
        await repository.upsert(conflicting)


async def test_quarantined_knowledge_is_invisible_to_retrieval(sqlite_engine) -> None:
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    await repository.upsert(_fact("visible"))
    await repository.upsert(
        _fact("quarantined", review_required=True, review_reason="low confidence")
    )

    resolved = await repository.resolve(["visible", "quarantined"])
    assert [item.identity for item in resolved] == ["visible"]

    with_quarantine = await repository.resolve(
        ["visible", "quarantined"], include_quarantined=True
    )
    assert [item.identity for item in with_quarantine] == ["visible", "quarantined"]


async def test_resolve_preserves_the_backend_ranking_order(sqlite_engine) -> None:
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    for identity in ("a", "b", "c"):
        await repository.upsert(_fact(identity))

    resolved = await repository.resolve(["c", "a", "b"])

    assert [item.identity for item in resolved] == ["c", "a", "b"]


async def test_a_later_run_may_not_raise_the_quarantine_on_a_released_assertion(
    sqlite_engine,
) -> None:
    """Ingestion says what a source asserts, not whether a person has ruled on it.

    This test used to assert the opposite — that a later run raises the
    quarantine — and the premise was wrong. The review policy flags *every*
    candidate on *every* run, so letting a run raise the flag meant an approved
    assertion was quarantined again on the next pass over an unchanged page. The
    approval was undone by the ingestion that found nothing new.
    """
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    await repository.upsert(_fact("fact-1"))

    flagged = await repository.upsert(
        _fact("fact-1", run_id="run-2", review_required=True, review_reason="policy: all")
    )

    assert flagged.review_required is False


async def test_pending_review_lists_the_least_confident_first(sqlite_engine) -> None:
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    await repository.upsert(_fact("clean"))
    await repository.upsert(
        _fact("sure", confidence=0.8, review_required=True, review_reason="r")
    )
    await repository.upsert(
        _fact("unsure", confidence=0.2, review_required=True, review_reason="r")
    )

    queue = await repository.pending_review()

    assert [item.identity for item in queue] == ["unsure", "sure"]
    assert await repository.pending_review_count() == 2


async def test_pending_review_pages_through_the_queue(sqlite_engine) -> None:
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    for index in range(3):
        await repository.upsert(
            _fact(
                f"q-{index}",
                confidence=0.1 * (index + 1),
                review_required=True,
                review_reason="r",
            )
        )

    first = await repository.pending_review(limit=2)
    second = await repository.pending_review(limit=2, offset=2)

    assert [item.identity for item in first] == ["q-0", "q-1"]
    assert [item.identity for item in second] == ["q-2"]


async def test_queue_neighbours_walk_the_queue_by_confidence(sqlite_engine) -> None:
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    for index in range(3):
        await repository.upsert(
            _fact(
                f"q-{index}",
                confidence=0.1 * (index + 1),
                review_required=True,
                review_reason="r",
            )
        )

    assert await repository.queue_neighbours("q-1") == ("q-0", "q-2")
    # The ends of the queue have nothing beyond them.
    assert await repository.queue_neighbours("q-0") == (None, "q-1")
    assert await repository.queue_neighbours("q-2") == ("q-1", None)


async def test_approval_lifts_the_quarantine_and_is_audited(sqlite_engine) -> None:
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    await repository.upsert(_fact("dirty", review_required=True, review_reason="low confidence"))

    review = await repository.record_review(
        identity="dirty",
        reviewer_id="reviewer-1",
        decision=ReviewDecision.APPROVED,
        comment="checked against the spec",
    )

    assert review.decision is ReviewDecision.APPROVED
    assert [item.identity for item in await repository.resolve(["dirty"])] == ["dirty"]
    assert await repository.pending_review() == []


async def test_rejection_keeps_knowledge_out_of_every_retrieval_path(sqlite_engine) -> None:
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    await repository.upsert(_fact("dirty", review_required=True, review_reason="wrong"))

    await repository.record_review(
        identity="dirty", reviewer_id="reviewer-1", decision=ReviewDecision.REJECTED
    )

    assert await repository.resolve(["dirty"]) == []
    # Rejection is a decision, so the item leaves the queue — the quarantine
    # flag alone cannot distinguish "unseen" from "seen and refused", and a
    # queue built on it could never be emptied.
    assert await repository.pending_review() == []


async def test_a_decided_assertion_returns_to_the_queue_on_fresh_evidence(
    sqlite_engine,
) -> None:
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    await repository.upsert(_fact("dirty", review_required=True, review_reason="wrong"))
    await repository.record_review(
        identity="dirty", reviewer_id="reviewer-1", decision=ReviewDecision.REJECTED
    )
    assert await repository.pending_review() == []

    # Another document asserts the same thing: that is new information about a
    # claim a reviewer has already refused, and deserves a second look.
    await repository.upsert(
        _fact(
            "dirty",
            run_id="run-2",
            source_id="source-2",
            review_required=True,
            review_reason="seen again elsewhere",
        )
    )

    assert [item.identity for item in await repository.pending_review()] == ["dirty"]


async def test_evidence_and_decision_history_are_readable(sqlite_engine) -> None:
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    await repository.upsert(_fact("dirty", review_required=True, review_reason="low"))
    await repository.upsert(_fact("dirty", run_id="run-2", source_id="source-2"))
    await repository.record_review(
        identity="dirty",
        reviewer_id="reviewer-1",
        decision=ReviewDecision.APPROVED,
        comment="checked the quote",
    )

    observations = await repository.observations("dirty")
    reviews = await repository.reviews("dirty")

    assert {observation.run_id for observation in observations} == {"run-1", "run-2"}
    assert observations[0].evidence == {"quote": "Spring Boot 3.2 requires Java 17"}
    assert [review.decision for review in reviews] == [ReviewDecision.APPROVED]
    assert reviews[0].comment == "checked the quote"


async def test_reviewing_unknown_knowledge_is_an_error(sqlite_engine) -> None:
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)

    with pytest.raises(ValueError, match="unknown knowledge identity"):
        await repository.record_review(
            identity="missing", reviewer_id="r", decision=ReviewDecision.APPROVED
        )


async def test_every_builtin_kind_round_trips(sqlite_engine) -> None:
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    writes = [
        _fact("k-fact"),
        KnowledgeWrite(
            identity="k-rule",
            payload=RulePayload(rule_text="Never log secrets"),
            type="security",
            confidence=0.8,
            source_id="s",
            run_id="r",
        ),
        KnowledgeWrite(
            identity="k-pattern",
            payload=PatternPayload(pattern_name="Outbox", description="Atomic write plus publish"),
            type="architecture",
            confidence=0.7,
            source_id="s",
            run_id="r",
        ),
        KnowledgeWrite(
            identity="k-decision",
            payload=DecisionPayload(decision="Use PostgreSQL", effect="No new dependency"),
            type="governance",
            confidence=0.9,
            source_id="s",
            run_id="r",
        ),
    ]
    for write in writes:
        await repository.upsert(write)

    resolved = await repository.resolve([w.identity for w in writes])

    assert [item.kind for item in resolved] == ["fact", "rule", "pattern", "decision"]
    # A rule written without the short pair round-trips with it absent — which
    # is what a corpus stored before those fields existed looks like.
    assert resolved[1].payload == {
        "rule_text": "Never log secrets", "subject": None, "rule_property": None,
    }
    assert resolved[3].payload == {
        "decision": "Use PostgreSQL", "effect": "No new dependency",
        "subject": None, "decision_type": None,
    }


async def test_a_rule_round_trips_its_subject_and_property(sqlite_engine) -> None:
    # And one written with them keeps them, canonically — the model's casing
    # does not survive into the database.
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    write = KnowledgeWrite(
        identity="rule-subjectful", type="security",
        payload=RulePayload(
            rule_text="Access tokens must not be logged.",
            subject="Access Token", rule_property="Logging Prohibition",
        ),
        confidence=0.9, source_id="source-1", run_id="run-1",
    )
    await repository.upsert(write)

    resolved = await repository.resolve(["rule-subjectful"])

    assert resolved[0].payload["subject"] == "access_token"
    assert resolved[0].payload["rule_property"] == "logging_prohibition"


# -- merge -------------------------------------------------------------------


async def test_merging_moves_the_evidence_and_retires_the_source(sqlite_engine) -> None:
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    await repository.upsert(_fact("survivor", confidence=0.6))
    await repository.upsert(
        _fact("duplicate", confidence=0.6, run_id="run-2", source_id="source-2")
    )

    review = await repository.record_review(
        identity="duplicate",
        reviewer_id="reviewer-1",
        decision=ReviewDecision.MERGED,
        merge_into="survivor",
        comment="same claim, other wording",
    )

    assert review.changes["merged_into"]["target"] == "survivor"
    assert review.changes["merged_into"]["observations_moved"] == 1

    # The evidence that justified the merge now belongs to the survivor.
    assert {item.run_id for item in await repository.observations("survivor")} == {
        "run-1", "run-2",
    }
    assert await repository.observations("duplicate") == []

    # Two independent sightings of one claim reinforce, exactly as a re-extraction
    # under the same identity would have done.
    survivor = (await repository.resolve(["survivor"]))[0]
    assert survivor.confidence > 0.6

    # The source keeps its history but is no longer canonical.
    assert await repository.resolve(["duplicate"]) == []
    retired = (await repository.resolve(["duplicate"], include_quarantined=True))[0]
    assert retired.merged_into == "survivor"
    assert [item.decision for item in await repository.reviews("duplicate")] == [
        ReviewDecision.MERGED
    ]


async def test_a_merged_assertion_leaves_the_queue_and_stays_out(sqlite_engine) -> None:
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    await repository.upsert(_fact("survivor"))
    await repository.upsert(
        _fact("duplicate", review_required=True, review_reason="r", run_id="run-2")
    )
    assert [item.identity for item in await repository.pending_review()] == ["duplicate"]

    await repository.record_review(
        identity="duplicate",
        reviewer_id="reviewer-1",
        decision=ReviewDecision.MERGED,
        merge_into="survivor",
    )

    assert await repository.pending_review() == []
    # Even fresh evidence must not reopen it: the claim now lives on the target.
    await repository.upsert(
        _fact("duplicate", run_id="run-3", source_id="source-3",
              review_required=True, review_reason="seen again")
    )
    assert await repository.pending_review() == []


async def test_a_merged_assertion_is_not_offered_as_a_merge_target(sqlite_engine) -> None:
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    for identity in ("survivor", "duplicate", "third"):
        await repository.upsert(_fact(identity))
    await repository.record_review(
        identity="duplicate",
        reviewer_id="reviewer-1",
        decision=ReviewDecision.MERGED,
        merge_into="survivor",
    )

    assert [item.identity for item in await repository.similar("third")] == ["survivor"]


async def test_merge_chains_are_refused(sqlite_engine) -> None:
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    for identity in ("survivor", "duplicate", "third"):
        await repository.upsert(_fact(identity))
    await repository.record_review(
        identity="duplicate", reviewer_id="r", decision=ReviewDecision.MERGED,
        merge_into="survivor",
    )

    # A chain would eventually point at something that is not canonical either.
    with pytest.raises(ValueError, match="merge into that instead"):
        await repository.record_review(
            identity="third", reviewer_id="r", decision=ReviewDecision.MERGED,
            merge_into="duplicate",
        )
    # `duplicate` has already been ruled on, so reaching the merge rule at all
    # means deliberately replacing that decision — the guard comes first, and
    # this is checking the rule behind it.
    with pytest.raises(ValueError, match="already merged"):
        await repository.record_review(
            identity="duplicate", reviewer_id="r", decision=ReviewDecision.MERGED,
            merge_into="third", supersede=True,
        )


async def test_a_merge_needs_a_real_and_different_target(sqlite_engine) -> None:
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    await repository.upsert(_fact("only"))

    with pytest.raises(ValueError, match="cannot be merged into itself"):
        await repository.record_review(
            identity="only", reviewer_id="r", decision=ReviewDecision.MERGED,
            merge_into="only",
        )
    with pytest.raises(ValueError, match="unknown merge target"):
        await repository.record_review(
            identity="only", reviewer_id="r", decision=ReviewDecision.MERGED,
            merge_into="never-extracted",
        )
    with pytest.raises(ValueError, match="requires a merge target"):
        await repository.record_review(
            identity="only", reviewer_id="r", decision=ReviewDecision.MERGED,
        )
    with pytest.raises(ValueError, match="only it may have one"):
        await repository.record_review(
            identity="only", reviewer_id="r", decision=ReviewDecision.APPROVED,
            merge_into="only",
        )


# ---------------------------------------------------------------------------
# Two reviewers, one assertion
# ---------------------------------------------------------------------------


async def test_a_second_decision_on_the_same_version_is_refused(sqlite_engine) -> None:
    # The write below `record_review` is unconditional, so without this guard
    # the later decision silently replaces the earlier one and the outcome is
    # decided by which reviewer pressed last. Two browser tabs are enough.
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    await repository.upsert(_fact("contested", review_required=True, review_reason="unsure"))
    await repository.record_review(
        identity="contested", reviewer_id="alice@example.com", decision=ReviewDecision.APPROVED
    )

    with pytest.raises(ReviewConflictError) as conflict:
        await repository.record_review(
            identity="contested", reviewer_id="bob@example.com", decision=ReviewDecision.REJECTED
        )

    # The message has to name whose decision is in the way, or a reviewer
    # cannot judge whether replacing it is reasonable.
    assert "alice@example.com" in str(conflict.value)
    assert "approved" in str(conflict.value)
    # And the first decision stands.
    assert len(await repository.resolve(["contested"])) == 1


async def test_a_decision_can_be_replaced_deliberately(sqlite_engine) -> None:
    # Without this a misclick would be uncorrectable until new evidence arrived.
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    await repository.upsert(_fact("misclicked", review_required=True, review_reason="unsure"))
    await repository.record_review(
        identity="misclicked", reviewer_id="alice@example.com", decision=ReviewDecision.APPROVED
    )

    await repository.record_review(
        identity="misclicked",
        reviewer_id="alice@example.com",
        decision=ReviewDecision.REJECTED,
        supersede=True,
    )

    assert await repository.resolve(["misclicked"]) == []
    # Both decisions stay in the history — replacing is not erasing.
    assert len(await repository.reviews("misclicked")) == 2


async def test_fresh_evidence_reopens_an_assertion_without_superseding(sqlite_engine) -> None:
    # The guard must not close the legitimate second look: new evidence bumps
    # the node past its last review, so this is a different version.
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    await repository.upsert(_fact("revisited", review_required=True, review_reason="unsure"))
    await repository.record_review(
        identity="revisited", reviewer_id="alice@example.com", decision=ReviewDecision.REJECTED
    )
    await repository.upsert(
        _fact(
            "revisited",
            run_id="run-2",
            source_id="source-2",
            review_required=True,
            review_reason="seen again elsewhere",
        )
    )

    # No supersede needed, and no conflict raised.
    await repository.record_review(
        identity="revisited", reviewer_id="bob@example.com", decision=ReviewDecision.APPROVED
    )

    assert len(await repository.resolve(["revisited"])) == 1


async def test_a_reviewer_id_may_be_any_opaque_string(sqlite_engine) -> None:
    # It will be whatever the identity provider hands over — an email, an
    # opaque id, something with symbols in it. Nothing here parses it.
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    await repository.upsert(_fact("opaque", review_required=True, review_reason="unsure"))

    review = await repository.record_review(
        identity="opaque",
        reviewer_id="k.müller+review@example.co.uk",
        decision=ReviewDecision.APPROVED,
    )

    assert review.reviewer_id == "k.müller+review@example.co.uk"


async def test_a_document_never_extracted_has_no_state(sqlite_engine) -> None:
    # The absent case is the safe one: no row licenses no skip.
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)

    assert await repository.extraction_state("doc-never-seen") is None


async def test_recording_an_extraction_makes_the_document_skippable(sqlite_engine) -> None:
    from nlght.core.knowledge import ExtractionStateWrite, extraction_version

    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    version = extraction_version(
        workflow="ingest-knowledge", model="ollama/llama3",
        prompt="Extract from: {text}", settings={"min_confidence": 0.3},
    )
    await repository.record_extraction(
        ExtractionStateWrite(
            document_id="doc-1", content_hash="c-1", extraction_version=version,
            run_id="run-1", assertion_count=4,
        )
    )

    state = await repository.extraction_state("doc-1")

    assert state is not None
    assert state.is_current(content_hash="c-1", extraction_version=version)
    assert state.extracted_at is not None


async def test_a_changed_model_makes_a_recorded_document_current_no_longer(sqlite_engine) -> None:
    # The document did not change; the extraction did. A content-only key would
    # skip it here, which is exactly when the corpus needs redoing.
    from nlght.core.knowledge import ExtractionStateWrite, extraction_version

    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    parts = dict(
        workflow="ingest-knowledge", prompt="Extract from: {text}",
        settings={"min_confidence": 0.3},
    )
    await repository.record_extraction(
        ExtractionStateWrite(
            document_id="doc-1", content_hash="c-1",
            extraction_version=extraction_version(model="ollama/llama3", **parts),
            run_id="run-1",
        )
    )

    state = await repository.extraction_state("doc-1")

    assert state is not None
    assert not state.is_current(
        content_hash="c-1",
        extraction_version=extraction_version(model="ollama/llama3.1", **parts),
    )


async def test_re_extracting_a_document_replaces_its_state(sqlite_engine) -> None:
    # One document, one answer to "may this be skipped", and it is the latest.
    from nlght.core.knowledge import ExtractionStateWrite, extraction_version

    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    version = extraction_version(
        workflow="w", model="m", prompt="p", settings={},
    )
    await repository.record_extraction(
        ExtractionStateWrite(
            document_id="doc-1", content_hash="c-1", extraction_version=version,
            run_id="run-1", assertion_count=4,
        )
    )
    await repository.record_extraction(
        ExtractionStateWrite(
            document_id="doc-1", content_hash="c-2", extraction_version=version,
            run_id="run-2", assertion_count=6,
        )
    )

    state = await repository.extraction_state("doc-1")

    assert state is not None
    assert not state.is_current(content_hash="c-1", extraction_version=version)
    assert state.is_current(content_hash="c-2", extraction_version=version)


# -- ingestion does not overwrite what a person decided -----------------------


async def test_a_later_run_does_not_undo_an_approval(sqlite_engine) -> None:
    """The defect: a reviewer's decision survived until the next ingestion.

    Somebody looked at the assertion and approved it. The next run over the
    same unchanged document re-extracts it, the review policy flags every
    candidate as it always does, and `upsert` raised `review_required` again —
    so the approval was gone and the assertion left retrieval, for a page nobody
    had touched.

    Ingestion says what a source asserts. It does not get to say whether a
    person has ruled on it.
    """
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    await repository.upsert(_fact(review_required=True, review_reason="policy: all"))
    await repository.record_review(
        identity="fact-1", decision=ReviewDecision.APPROVED, reviewer_id="ada",
    )

    await repository.upsert(
        _fact(run_id="run-2", review_required=True, review_reason="policy: all")
    )

    resolved = await repository.resolve(["fact-1"])
    assert len(resolved) == 1
    assert resolved[0].review_required is False


async def test_a_later_run_does_not_return_a_rejected_assertion_to_the_queue(
    sqlite_engine,
) -> None:
    """The same defect through the other door, and the quieter one.

    A rejection keeps `review_required` set — that is what holds the assertion
    out of retrieval — so the queue tells an unseen assertion from a judged one
    by comparing the review against the node's last change. `upsert` stamped
    `updated_at` on every sighting, so a re-ingestion made the node look changed
    and the rejected assertion came back to be judged again. On every run.
    """
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    await repository.upsert(_fact(review_required=True, review_reason="policy: all"))
    await repository.record_review(
        identity="fact-1", decision=ReviewDecision.REJECTED, reviewer_id="ada",
    )
    assert await repository.pending_review_count() == 0

    await repository.upsert(
        _fact(run_id="run-2", review_required=True, review_reason="policy: all")
    )

    assert await repository.pending_review_count() == 0


async def test_an_unjudged_assertion_still_waits_in_the_queue(sqlite_engine) -> None:
    # The guard on both fixes: nothing here may hide an assertion nobody has
    # looked at yet.
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    await repository.upsert(_fact(review_required=True, review_reason="policy: all"))

    await repository.upsert(
        _fact(run_id="run-2", review_required=True, review_reason="policy: all")
    )

    assert await repository.pending_review_count() == 1
