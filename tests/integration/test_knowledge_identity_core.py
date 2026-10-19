# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Verify stable knowledge identity across document and extraction changes.

Identity follows the persisted assertion, stable slot, and proposition match.
These tests vary model-generated descriptions while asserting exact database
transitions. They also guard against merging materially different claims.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nlght.adapters.outbound.persistence.knowledge_models import (
    Knowledge,
    KnowledgeAssertion,
    KnowledgeEvidence,
    KnowledgeRevision,
    KnowledgeVariant,
)
from nlght.adapters.outbound.persistence.knowledge_repository import (
    SqlAlchemyKnowledgeRepository,
)
from nlght.core.knowledge import (
    APPROVED,
    DERIVED_ANCHOR,
    NO_ANCHOR,
    FactPayload,
    KnowledgeWrite,
    LineageWrite,
    ObservedSection,
    Proposition,
    entity_key,
)

pytestmark = pytest.mark.integration


SLOT_X = "unit:actuator#customizing-sanitization"
SLOT_Y = "unit:actuator#sanitization"
DOC_A = "doc-actuator"
DOC_B = "doc-reference"

#: One proposition, described twice. The two differ in exactly the way the live
#: run differed: same claim, different `subject` and `rule_property`. If identity
#: followed the description, these would be two assertions.
SANITIZER_TEXT = "Each SanitizingFunction is called in order until a function changes the value."
SANITIZER_AS_EXECUTION = entity_key(
    "rule", subject="sanitizing_function_execution", rule_property="ordered_until_changed"
)
SANITIZER_AS_ORDER = entity_key(
    "rule", subject="sanitizing_functions", rule_property="processing_order"
)

#: A different claim, not a different wording. No matcher may merge these.
JAVA_17 = entity_key("fact", predicate="requires", subject="spring", object="java 17")
JAVA_21 = entity_key("fact", predicate="requires", subject="spring", object="java 21")
MAVEN = entity_key("fact", predicate="requires", subject="spring", object="maven")
GRADLE = entity_key("fact", predicate="requires", subject="spring", object="gradle")

#: One rule, edited in the smallest visible way there is: a full stop.
LIMIT = entity_key("rule", subject="value", rule_property="max_length")


def _write(
    entity,
    *,
    text: str,
    kind: str = "rule",
    slot_id: str = SLOT_X,
    document_id: str = DOC_A,
    document_revision: str = "rev-1",
    run_id: str = "run-1",
    fingerprint: str | None = None,
    extraction_version: str = "ver-1",
) -> LineageWrite:
    return LineageWrite(
        entity=entity,
        kind=kind,
        scope=(),
        text=text,
        # The fingerprint is derived from the wording, so two descriptions of one
        # proposition have different fingerprints, as observed in the live run.
        fingerprint=fingerprint or f"fp:{text}",
        extraction_version=extraction_version,
        document_id=document_id,
        document_revision=document_revision,
        slot_id=slot_id,
        run_id=run_id,
        # Every sighting carries a structured claim. These tests are about which
        # assertion it continues, so the structure is the minimum its kind needs.
        proposition=Proposition(
            {"rule_text": text} if kind == "rule"
            else {"subject": "spring", "predicate": "requires", "object": text}
        ),
    )


@pytest_asyncio.fixture
async def repository():
    from sqlalchemy import event
    from sqlalchemy.ext.asyncio import create_async_engine

    from nlght.adapters.outbound.persistence.knowledge_models import KnowledgeBase

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    @event.listens_for(engine.sync_engine, "connect")
    def _enforce_foreign_keys(dbapi_connection, _record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with engine.begin() as conn:
        await conn.run_sync(KnowledgeBase.metadata.create_all)
    repository = SqlAlchemyKnowledgeRepository(engine)
    repository._test_engine = engine
    yield repository
    await engine.dispose()


async def _rows(repository, model):
    async with AsyncSession(repository._test_engine) as session:
        return list((await session.execute(select(model))).scalars().all())


async def _revisions(repository, assertion_id: str):
    variants = {
        row.variant_id
        for row in await _rows(repository, KnowledgeVariant)
        if row.assertion_id == assertion_id
    }
    rows = [r for r in await _rows(repository, KnowledgeRevision) if r.variant_id in variants]
    return sorted(rows, key=lambda row: row.revision)


async def _evidence(repository, assertion_id: str):
    revisions = {row.revision_id for row in await _revisions(repository, assertion_id)}
    return [r for r in await _rows(repository, KnowledgeEvidence) if r.revision_id in revisions]


async def test_same_proposition_same_slot_keeps_the_assertion_id(repository) -> None:
    """The central test. It reproduces the live defect exactly.

    The source text did not change. The model described the same rule with
    different fields on the second pass, and because identity is looked up by
    those fields, the claim became a different claim.

    This must be decided by the slot and the proposition, never by the
    description. Note the two entity keys below are deliberately different: a
    version of this test where they matched would be green today and green after
    the bug, which is to say it would test nothing.
    """
    first = await repository.record_lineage(
        _write(SANITIZER_AS_EXECUTION, text=SANITIZER_TEXT)
    )

    second = await repository.record_lineage(
        _write(SANITIZER_AS_ORDER, text=SANITIZER_TEXT, run_id="run-2")
    )

    assert second.assertion_id == first.assertion_id
    assert len(await _rows(repository, KnowledgeAssertion)) == 1
    # Same proposition, so nothing about the claim is new and no reviewer is owed
    # a second look.
    assert second.revision == 1
    assert second.review_state == "approved"


async def test_a_redescribed_assertion_is_not_withdrawn(repository) -> None:
    # The other half of the same failure: the old assertion must not be retired
    # for having been described differently. Retraction belongs to a source that
    # stopped carrying the claim.
    #
    # Unmarked, and it took the ratchet to notice: this holds today, though only
    # because nothing withdraws anything yet. It has to still hold once
    # retraction exists, which is precisely when it becomes easy to break.
    first = await repository.record_lineage(_write(SANITIZER_AS_EXECUTION, text=SANITIZER_TEXT))

    await repository.record_lineage(
        _write(SANITIZER_AS_ORDER, text=SANITIZER_TEXT, run_id="run-2")
    )

    stored = next(
        row for row in await _rows(repository, KnowledgeAssertion)
        if row.assertion_id == first.assertion_id
    )
    assert stored.retired_at is None
    assert await repository.supersessions_of(first.assertion_id) == ()


@pytest.mark.parametrize(
    ("kind", "fields"),
    [
        ("rule", {"subject": "line-length", "rule_property": "maximum"}),
        ("decision", {"subject": "postgres-as-queue", "decision_type": "architecture"}),
        ("pattern", {"pattern_name": "retry-with-backoff"}),
    ],
)
async def test_a_rephrased_body_keeps_the_assertion_whatever_the_kind(
    repository, kind: str, fields: dict[str, str]
) -> None:
    """Keep an assertion when the model rephrases its description.

    The stable slot and observed wording establish continuity across claim kinds;
    the model-generated description may change.
    """
    key = entity_key(kind, **fields)
    body = "Lines are limited to 100 characters."

    first = await repository.record_lineage(_write(key, kind=kind, text=body))
    second = await repository.record_lineage(
        _write(key, kind=kind, text=body, run_id="run-2")
    )

    assert second.assertion_id == first.assertion_id
    assert second.revision == 1


async def test_reflowing_a_line_is_not_a_new_state(repository) -> None:
    # The one tolerance the observed wording has. A parser may rewrap a line and
    # that changes no word, so it is not a state anybody entered.
    key = entity_key("rule", subject="line-length", rule_property="maximum")

    first = await repository.record_lineage(
        _write(key, text="Lines are limited to 100 characters")
    )
    second = await repository.record_lineage(
        _write(key, text="Lines  are\n  limited to 100 characters", run_id="run-2")
    )

    assert second.assertion_id == first.assertion_id
    assert second.revision == 1, "a reflow is not a state the corpus should record"


async def test_a_visible_change_to_the_wording_is_a_new_state(repository) -> None:
    """Record a visible wording edit as a new revision without a new approval.

    Whitespace reflow preserves wording, while punctuation changes the observed
    text. The underlying claim remains the same.
    """
    key = entity_key("rule", subject="line-length", rule_property="maximum")

    first = await repository.record_lineage(
        _write(key, text="Lines are limited to 100 characters")
    )
    second = await repository.record_lineage(
        _write(key, text="Lines are limited to 100 characters.", run_id="run-2")
    )

    assert second.assertion_id == first.assertion_id, "still one claim"
    assert second.revision == 2
    assert second.review_state == "approved"


async def test_material_change_same_slot_is_a_different_assertion(repository) -> None:
    """Create a new assertion when a claim changes materially in the same slot.

    Treating Java 17 and Java 21 as one approved assertion would conceal a
    changed claim under the previous approval.
    """
    first = await repository.record_lineage(
        _write(JAVA_17, kind="fact", text="Spring requires Java 17.")
    )

    second = await repository.record_lineage(
        _write(JAVA_21, kind="fact", text="Spring requires Java 21.", run_id="run-2")
    )

    assert second.assertion_id != first.assertion_id
    assert len(await _rows(repository, KnowledgeAssertion)) == 2


async def test_parallel_assertions_in_one_slot_do_not_become_revisions(repository) -> None:
    # A slot holds several claims. A second one appearing beside the first is an
    # addition, not the first one changing: two assertions, and the first keeps
    # exactly one revision.
    first = await repository.record_lineage(
        _write(JAVA_17, kind="fact", text="Spring requires Java 17.")
    )

    await repository.record_lineage(
        _write(JAVA_17, kind="fact", text="Spring requires Java 17.", run_id="run-2")
    )
    second = await repository.record_lineage(
        _write(MAVEN, kind="fact", text="Spring requires Maven.", run_id="run-2")
    )

    assert second.assertion_id != first.assertion_id
    assert len(await _rows(repository, KnowledgeAssertion)) == 2
    assert [row.revision for row in await _revisions(repository, first.assertion_id)] == [1]


async def test_an_unchanged_assertion_survives_its_neighbour_being_edited(repository) -> None:
    # Editing one claim in a slot must not disturb the other. Without this, a
    # slot-scoped matcher could rewrite everything it touches.
    stable = await repository.record_lineage(
        _write(JAVA_17, kind="fact", text="Spring requires Java 17.")
    )
    await repository.record_lineage(
        _write(MAVEN, kind="fact", text="Spring requires Maven.")
    )

    await repository.record_lineage(
        _write(JAVA_17, kind="fact", text="Spring requires Java 17.", run_id="run-2")
    )
    await repository.record_lineage(
        _write(MAVEN, kind="fact", text="Spring requires Maven 3.9.", run_id="run-2")
    )

    unchanged = await _revisions(repository, stable.assertion_id)
    assert [row.revision for row in unchanged] == [1]
    assert len(await _rows(repository, KnowledgeAssertion)) == 2


async def test_an_ambiguous_match_does_not_merge(repository) -> None:
    """Create a new ambiguous assertion when two prior claims match equally.

    Choosing the first plausible match would silently attach the new claim to
    the wrong history.
    """
    left = await repository.record_lineage(
        _write(entity_key("rule", subject="timeout", rule_property="connect"),
               text="The connect timeout is 30 seconds.")
    )
    right = await repository.record_lineage(
        _write(entity_key("rule", subject="timeout", rule_property="read"),
               text="The read timeout is 30 seconds.")
    )

    merged = await repository.record_lineage(
        _write(entity_key("rule", subject="timeout", rule_property="default"),
               text="The timeout is 30 seconds.", run_id="run-2")
    )

    assert merged.assertion_id not in {left.assertion_id, right.assertion_id}
    assert merged.resolution == "ambiguous"


async def test_a_changed_value_under_one_key_is_a_new_revision_needing_review(
    repository,
) -> None:
    """Keep the assertion identity but require review for a changed value.

    The rule key identifies the approval threshold. Changing CHF 500 to CHF 550
    changes the rule state and invalidates its previous approval.
    """
    threshold = entity_key("rule", subject="expense", rule_property="approval_threshold")

    first = await repository.record_lineage(
        _write(threshold, text="Expenses above CHF 500 require approval.")
    )
    second = await repository.record_lineage(
        _write(threshold, text="Expenses above CHF 550 require approval.", run_id="run-2")
    )

    assert second.assertion_id == first.assertion_id
    assert second.variant_id == first.variant_id
    assert second.revision == 2
    assert second.supersedes == 1
    assert second.review_state == "needs_review"


async def test_a_reworded_rule_under_one_key_keeps_its_approval(repository) -> None:
    """The counterpart, and the reason the distinction cannot be a threshold.

    Lexical similarity inverts these two: the reworded rule scores 0.46 against
    its predecessor and the changed threshold 0.97. What separates them is the
    quantity the body asserts, compared apart from the words carrying it.
    """
    threshold = entity_key("rule", subject="expense", rule_property="approval_threshold")

    first = await repository.record_lineage(
        _write(threshold, text="Expenses above CHF 500 require approval.")
    )
    second = await repository.record_lineage(
        _write(threshold, text="Expenses over CHF 500 must be signed off.", run_id="run-2")
    )

    assert second.assertion_id == first.assertion_id
    assert second.revision == 2
    assert second.review_state == "approved"


async def test_two_documents_disagreeing_on_a_value_do_not_collapse_silently(
    repository,
) -> None:
    # One key, two documents, two different thresholds. They continue one
    # assertion because it is one business quantity. They must not land as the
    # same state with the approval intact, because nobody has ruled on the
    # second one.
    threshold = entity_key("rule", subject="expense", rule_property="approval_threshold")

    first = await repository.record_lineage(
        _write(threshold, text="Expenses above CHF 500 require approval.", document_id=DOC_A)
    )
    second = await repository.record_lineage(
        _write(threshold, text="Expenses above CHF 5000 require board approval.",
               document_id=DOC_B, slot_id="unit:reference#expenses", run_id="run-2")
    )

    assert second.assertion_id == first.assertion_id
    assert second.revision != first.revision
    assert second.review_state == "needs_review"


async def test_an_assertion_that_moved_within_a_document_keeps_its_id(repository) -> None:
    # A section renamed or reordered moves the slot, not the claim. The slot
    # lineage records the move so the anchor can be followed. The assertion id
    # does not move with it.
    first = await repository.record_lineage(_write(SANITIZER_AS_EXECUTION, text=SANITIZER_TEXT))

    second = await repository.record_lineage(
        _write(SANITIZER_AS_ORDER, text=SANITIZER_TEXT, slot_id=SLOT_Y, run_id="run-2")
    )

    assert second.assertion_id == first.assertion_id
    slots = {row.slot_id for row in await _evidence(repository, first.assertion_id)}
    assert slots == {SLOT_X, SLOT_Y}


async def test_the_same_claim_in_two_documents_keeps_its_provenance_apart(repository) -> None:
    """Keep evidence separate when two documents support one assertion.

    Retraction depends on whether each document still carries the claim.
    """
    first = await repository.record_lineage(
        _write(JAVA_17, kind="fact", text="Spring requires Java 17.", document_id=DOC_A)
    )

    second = await repository.record_lineage(
        _write(JAVA_17, kind="fact", text="Spring requires Java 17.",
               document_id=DOC_B, slot_id="unit:reference#stack", run_id="run-2")
    )

    assert second.assertion_id == first.assertion_id
    evidence = await _evidence(repository, first.assertion_id)
    assert {row.document_id for row in evidence} == {DOC_A, DOC_B}
    assert {row.slot_id for row in evidence} == {SLOT_X, "unit:reference#stack"}


async def _slot(repository, *writes, slot_id: str = SLOT_X, document_id: str = DOC_A,
                run_id: str = "run-1"):
    """One document with a single section, in one call.

    Args:
        repository (Any): Repository used by the test.
        *writes (Any): Claim writes included in the section.
        slot_id (str): Identifier of the claim slot.
        document_id (str): Document identifier.
        run_id (str): Ingestion run identifier.
    """
    return await _document(
        repository, {slot_id: writes}, document_id=document_id, run_id=run_id
    )


async def _document(repository, sections, *, document_id: str = DOC_A,
                    run_id: str = "run-1"):
    """Record every section observed in a complete document read.

    A missing claim can be retired only after the entire document has been read.
    Section paths identify sections; ordinals do not determine identity.

    Args:
        repository (Any): Repository used by the test.
        sections (Any): Document sections to store.
        document_id (str): Document identifier.
        run_id (str): Ingestion run identifier.
    """
    return await repository.record_document(
        document_id=document_id,
        sections=[
            ObservedSection.as_read(
                # An anchor of "" is a section with nothing to be found by. It is
                # unscoped part of a document, which PDF and prose are entirely.
                anchor="" if anchor == UNSCOPED else anchor,
                anchor_strength=NO_ANCHOR if anchor == UNSCOPED else DERIVED_ANCHOR,
                # From what the section says, not from what it is called. A
                # fingerprint taken off the anchor is identical however much the
                # text under it changed, so a section that lost a sentence looked
                # untouched, and retraction now reads an untouched section as
                # evidence that the source did *not* drop anything.
                content=anchor + "|" + "|".join(sorted(write.text for write in writes)),
                unit_ordinal=anchor,
                assertions=tuple((_graph(write), write) for write in writes),
            )
            for anchor, writes in sections.items()
        ],
        run_id=run_id,
    )


#: A section with no anchor. Its claims are diffed against the document rather
#: than against a slot, because there is nothing to scope them to.
UNSCOPED = "<unscoped>"


def _graph(write) -> KnowledgeWrite:
    """The graph half of the same sighting, which travels in one transaction.

    Args:
        write (Any): Knowledge write being converted to graph form.

    Returns:
        KnowledgeWrite: Graph write corresponding to the selected sighting.
    """
    return KnowledgeWrite(
        identity=write.fingerprint,
        payload=FactPayload(subject="spring", predicate="requires", object=write.text),
        type="dependency",
        confidence=0.9,
        source_id="corpus",
        run_id=write.run_id,
        evidence={"document_id": write.document_id},
    )


async def test_an_assertion_the_source_dropped_is_retired_not_deleted(repository) -> None:
    """Retire a dropped claim while preserving its evidence.

    Historical citations must still show what the document asserted before the
    claim disappeared.
    """
    kept = (await _slot(
        repository,
        _write(JAVA_17, kind="fact", text="Spring requires Java 17."),
        _write(MAVEN, kind="fact", text="Spring requires Maven."),
    )).recorded
    dropped_id = kept[1].assertion_id

    outcome = await _slot(
        repository,
        _write(JAVA_17, kind="fact", text="Spring requires Java 17.", run_id="run-2"),
        run_id="run-2",
    )

    assert [row.assertion_id for row in outcome.retired] == [dropped_id]
    rows = {row.assertion_id: row for row in await _rows(repository, KnowledgeAssertion)}
    assert rows[dropped_id].retired_at is not None
    assert rows[kept[0].assertion_id].retired_at is None
    # The sighting stands. Only the claim stopped being carried.
    assert len(await _evidence(repository, dropped_id)) == 1


async def test_a_section_that_says_the_same_thing_cannot_lose_a_claim(repository) -> None:
    """Keep claims from an unchanged section when extraction omits one.

    The stable-slot guard prevents model nondeterminism from retiring knowledge
    that the source text still supports.
    """
    both = (await _slot(
        repository,
        _write(JAVA_17, kind="fact", text="Spring requires Java 17."),
        _write(MAVEN, kind="fact", text="Spring requires Maven."),
    )).recorded
    forgotten = both[1].assertion_id

    # The same section, unchanged, and the model mentions only one claim.
    outcome = await repository.record_document(
        document_id=DOC_A,
        sections=[
            ObservedSection.as_read(
                anchor=SLOT_X,
                anchor_strength=DERIVED_ANCHOR,
                # What the first run was given, exactly.
                content=SLOT_X + "|" + "|".join(sorted(
                    ["Spring requires Java 17.", "Spring requires Maven."]
                )),
                unit_ordinal=SLOT_X,
                assertions=((
                    _graph(_write(JAVA_17, kind="fact",
                                  text="Spring requires Java 17.", run_id="run-2")),
                    _write(JAVA_17, kind="fact",
                           text="Spring requires Java 17.", run_id="run-2"),
                ),),
            )
        ],
        run_id="run-2",
    )

    assert outcome.retired == ()
    rows = {row.assertion_id: row for row in await _rows(repository, KnowledgeAssertion)}
    assert rows[forgotten].retired_at is None, "the model forgot it; the source did not drop it"


def _unchanged_section(*writes, run_id: str):
    """The section exactly as the first run was given it, whatever is claimed now.

    Args:
        *writes (Any): Writes retained from the first run.
        run_id (str): Ingestion run identifier.
    """
    return ObservedSection.as_read(
        anchor=SLOT_X,
        anchor_strength=DERIVED_ANCHOR,
        content=SLOT_X + "|" + "|".join(sorted(
            ["Spring requires Java 17.", "Spring requires Maven."]
        )),
        unit_ordinal=SLOT_X,
        assertions=tuple((_graph(write), write) for write in writes),
    )


async def test_a_section_that_says_the_same_thing_cannot_gain_one_either(repository) -> None:
    """Reject a new claim from unchanged text and extraction input.

    A second model response to the same input must not add a claim absent from
    the earlier complete observation.
    """
    before = (await _slot(
        repository,
        _write(JAVA_17, kind="fact", text="Spring requires Java 17."),
        _write(MAVEN, kind="fact", text="Spring requires Maven."),
    )).recorded
    assert len(before) == 2

    invented = _write(
        GRADLE, kind="fact", text="Spring requires Gradle.", run_id="run-2"
    )
    await repository.record_document(
        document_id=DOC_A,
        sections=[_unchanged_section(
            _write(JAVA_17, kind="fact", text="Spring requires Java 17.", run_id="run-2"),
            _write(MAVEN, kind="fact", text="Spring requires Maven.", run_id="run-2"),
            invented,
            run_id="run-2",
        )],
        run_id="run-2",
    )

    live = [row for row in await _rows(repository, KnowledgeAssertion) if row.retired_at is None]
    assert len(live) == 2, "the section said nothing new, so the corpus gained nothing"


async def test_a_sighting_of_what_the_section_already_says_still_lands(repository) -> None:
    # Reinforcement is the point of a second run. Only wording the slot has never
    # carried is refused. Seeing again what is already there is evidence.
    first = (await _slot(
        repository,
        _write(JAVA_17, kind="fact", text="Spring requires Java 17."),
        _write(MAVEN, kind="fact", text="Spring requires Maven."),
    )).recorded

    outcome = await repository.record_document(
        document_id=DOC_A,
        sections=[_unchanged_section(
            _write(JAVA_17, kind="fact", text="Spring requires Java 17.", run_id="run-2"),
            _write(MAVEN, kind="fact", text="Spring requires Maven.", run_id="run-2"),
            run_id="run-2",
        )],
        run_id="run-2",
    )

    assert {row.assertion_id for row in outcome.recorded} == {
        row.assertion_id for row in first
    }
    assert len(await _evidence(repository, first[0].assertion_id)) == 2


async def test_a_full_stop_is_a_different_question_to_the_model(repository) -> None:
    """Treat punctuation changes as changed extraction input.

    The slot fingerprint normalizes punctuation for location matching. The input
    fingerprint preserves exact text so a corrected sentence can be extracted.
    """
    before = (await _slot(
        repository,
        _write(LIMIT, text="The value is limited to 100 characters"),
    )).recorded[0]

    outcome = await _slot(
        repository,
        _write(LIMIT, text="The value is limited to 100 characters.", run_id="run-2"),
        run_id="run-2",
    )

    assert len(outcome.recorded) == 1, "the section changed, so the claim was recorded"
    after = outcome.recorded[0]
    # One claim, two states of it. The assertion id is what carries the approval
    # across the edit, and it is the same id.
    assert after.assertion_id == before.assertion_id
    assert (after.revision, after.supersedes) == (2, 1)
    revisions = await _revisions(repository, before.assertion_id)
    assert [row.text for row in revisions] == [
        "The value is limited to 100 characters",
        "The value is limited to 100 characters.",
    ]
    # Nothing a proposition depends on moved, so the approval given to the first
    # state covers the second. A punctuation edit must not reach a reviewer.
    assert revisions[1].review_state == APPROVED


async def test_a_byte_identical_section_still_freezes_its_claim_set(repository) -> None:
    """The guard's own case, on the exact fingerprint rather than the folded one.

    Same input, same extraction version: the model may not add a claim there and
    may not lose one. This is the half that must survive making the question
    finer.
    """
    before = (await _slot(
        repository,
        _write(JAVA_17, kind="fact", text="Spring requires Java 17."),
        _write(MAVEN, kind="fact", text="Spring requires Maven."),
    )).recorded

    # The same section text matters. `_slot` builds it from the claims, so the third
    # claim would change it. It is passed as a section that says what it said.
    await repository.record_document(
        document_id=DOC_A,
        sections=[_unchanged_section(
            _write(JAVA_17, kind="fact", text="Spring requires Java 17.", run_id="run-2"),
            _write(GRADLE, kind="fact", text="Spring requires Gradle.", run_id="run-2"),
            run_id="run-2",
        )],
        run_id="run-2",
    )

    live = {
        row.assertion_id
        for row in await _rows(repository, KnowledgeAssertion)
        if row.retired_at is None
    }
    assert live == {row.assertion_id for row in before}, (
        "nothing arrived and nothing left: the model was asked the same question"
    )


async def test_the_guard_freezes_the_slot_and_not_the_lifecycle(repository) -> None:
    """Keep the claim set stable without freezing reviews or evidence.

    The guard controls contradictory model extraction from unchanged text. It
    does not block other documents or reviewer decisions.
    """
    from nlght.core.knowledge.knowledge import ReviewDecision

    first = (await _slot(
        repository,
        _write(JAVA_17, kind="fact", text="Spring requires Java 17."),
        _write(MAVEN, kind="fact", text="Spring requires Maven."),
    )).recorded

    # A second sighting of the unchanged section, then a human decision on top.
    await repository.record_document(
        document_id=DOC_A,
        sections=[_unchanged_section(
            _write(JAVA_17, kind="fact", text="Spring requires Java 17.", run_id="run-2"),
            _write(MAVEN, kind="fact", text="Spring requires Maven.", run_id="run-2"),
            run_id="run-2",
        )],
        run_id="run-2",
    )
    review = await repository.record_review(
        identity="fp:Spring requires Java 17.",
        reviewer_id="a-person",
        decision=ReviewDecision.APPROVED,
    )

    assert review.decision == ReviewDecision.APPROVED
    # And a second document may still support it, unchanged section or not.
    elsewhere = await _slot(
        repository,
        _write(JAVA_17, kind="fact", text="Spring requires Java 17.",
               document_id=DOC_B, slot_id=SLOT_Y, run_id="run-3"),
        document_id=DOC_B, slot_id=SLOT_Y, run_id="run-3",
    )

    assert elsewhere.recorded[0].assertion_id == first[0].assertion_id


async def test_a_removed_section_still_retires_what_it_held(repository) -> None:
    """Retire claims when a section disappears from a complete document read.

    An unchanged last fingerprint cannot prove that a section still exists.
    """
    held = (await _slot(
        repository,
        _write(JAVA_17, kind="fact", text="Spring requires Java 17."),
    )).recorded[0]

    # The document is observed in full and no longer contains that section.
    outcome = await repository.record_document(
        document_id=DOC_A,
        sections=[ObservedSection.as_read(
            anchor=SLOT_Y,
            anchor_strength=DERIVED_ANCHOR,
            content="a different section entirely",
            unit_ordinal=SLOT_Y,
            assertions=((
                _graph(_write(MAVEN, kind="fact", text="Spring requires Maven.",
                              slot_id=SLOT_Y, run_id="run-2")),
                _write(MAVEN, kind="fact", text="Spring requires Maven.",
                       slot_id=SLOT_Y, run_id="run-2"),
            ),),
        )],
        run_id="run-2",
    )

    assert [row.assertion_id for row in outcome.retired] == [held.assertion_id]
    rows = {row.assertion_id: row for row in await _rows(repository, KnowledgeAssertion)}
    assert rows[held.assertion_id].retired_at is not None
    # Its evidence stands: it stopped being asserted, it did not stop having been.
    assert len(await _evidence(repository, held.assertion_id)) == 1


async def test_a_new_extraction_version_may_say_something_new(repository) -> None:
    """Allow new claims when the extraction contract changes.

    A new prompt version asks a different question, even when source text is
    unchanged.
    """
    await _slot(
        repository,
        _write(JAVA_17, kind="fact", text="Spring requires Java 17."),
        _write(MAVEN, kind="fact", text="Spring requires Maven."),
    )

    await repository.record_document(
        document_id=DOC_A,
        sections=[_unchanged_section(
            _write(JAVA_17, kind="fact", text="Spring requires Java 17.",
                   run_id="run-2", extraction_version="ver-2"),
            _write(MAVEN, kind="fact", text="Spring requires Maven.",
                   run_id="run-2", extraction_version="ver-2"),
            _write(GRADLE, kind="fact", text="Spring requires Gradle.",
                   run_id="run-2", extraction_version="ver-2"),
            run_id="run-2",
        )],
        run_id="run-2",
    )

    live = [row for row in await _rows(repository, KnowledgeAssertion) if row.retired_at is None]
    assert len(live) == 3


async def test_inserting_a_paragraph_does_not_retire_what_moved_below_it(
    repository,
) -> None:
    """Preserve claims moved by an inserted paragraph.

    Position-based slot IDs change when sections shift. Document-level matching
    must retain claims that the document still asserts.
    """
    first = (await _document(
        repository,
        {"/stack": [_write(JAVA_17, kind="fact", text="Spring requires Java 17.",
                           slot_id="s:doc:1")]},
    )).recorded[0]

    # A paragraph is inserted above. Every ordinal below it shifts. The headings
    # do not, which is the difference the registry is built to notice.
    outcome = await _document(
        repository,
        {
            "/intro": [_write(MAVEN, kind="fact", text="Spring requires Maven.",
                              slot_id="s:doc:1", run_id="run-2")],
            "/stack": [_write(JAVA_17, kind="fact", text="Spring requires Java 17.",
                              slot_id="s:doc:2", run_id="run-2")],
        },
        run_id="run-2",
    )

    assert outcome.retired == ()
    rows = {row.assertion_id: row for row in await _rows(repository, KnowledgeAssertion)}
    assert rows[first.assertion_id].retired_at is None
    assert len(rows) == 2, "the moved claim must not have been re-created"
    # And it did not "move" at all any more. The ordinal shifted. The section did
    # not, so the evidence names one slot before and after, which is the whole
    # difference between an identity and a position.
    slots = {row.slot_id for row in await _evidence(repository, first.assertion_id)}
    assert len(slots) == 1
    assert "s:doc:1" not in slots and "s:doc:2" not in slots


async def test_a_claim_dropped_from_one_section_is_still_retired(repository) -> None:
    # The other direction, and the reason widening the container is a fix rather
    # than a retreat: what the document genuinely stopped saying still goes.
    kept, dropped = (await _document(
        repository,
        {
            "s:doc:1": [_write(JAVA_17, kind="fact", text="Spring requires Java 17.",
                               slot_id="s:doc:1")],
            "s:doc:2": [_write(MAVEN, kind="fact", text="Spring requires Maven.",
                               slot_id="s:doc:2")],
        },
    )).recorded

    outcome = await _document(
        repository,
        {"s:doc:1": [_write(JAVA_17, kind="fact", text="Spring requires Java 17.",
                            slot_id="s:doc:1", run_id="run-2")]},
        run_id="run-2",
    )

    assert [row.assertion_id for row in outcome.retired] == [dropped.assertion_id]
    rows = {row.assertion_id: row for row in await _rows(repository, KnowledgeAssertion)}
    assert rows[kept.assertion_id].retired_at is None


async def test_a_claim_another_document_still_carries_is_not_retired(repository) -> None:
    # The second guard, and the one that decides whether retraction is safe to
    # run at all. Keeping it may cost a stale row that a reviewer can retire.
    # erring the other way deletes knowledge a second source was carrying.
    first = (await _slot(
        repository, _write(JAVA_17, kind="fact", text="Spring requires Java 17.")
    )).recorded[0]
    await _slot(
        repository,
        _write(JAVA_17, kind="fact", text="Spring requires Java 17.",
               document_id=DOC_B, slot_id="unit:reference#stack", run_id="run-2"),
        document_id=DOC_B, slot_id="unit:reference#stack", run_id="run-2",
    )

    outcome = await _slot(repository, run_id="run-3")

    assert outcome.retired == ()
    rows = {row.assertion_id: row for row in await _rows(repository, KnowledgeAssertion)}
    assert rows[first.assertion_id].retired_at is None


async def test_a_retired_claim_coming_back_is_not_silently_reactivated(repository) -> None:
    # A claim removed and later restored is a new decision, not an undo. Reviving
    # the retired row would carry an approval across a gap nobody looked at.
    original = (await _slot(
        repository, _write(MAVEN, kind="fact", text="Spring requires Maven.")
    )).recorded[0]
    await _slot(
        repository,
        _write(JAVA_17, kind="fact", text="Spring requires Java 17.", run_id="run-2"),
        run_id="run-2",
    )

    restored = (await _slot(
        repository,
        _write(MAVEN, kind="fact", text="Spring requires Maven.", run_id="run-3"),
        run_id="run-3",
    )).recorded[0]

    assert restored.assertion_id != original.assertion_id
    rows = {row.assertion_id: row for row in await _rows(repository, KnowledgeAssertion)}
    assert rows[original.assertion_id].retired_at is not None


async def test_a_retry_changes_nothing(repository) -> None:
    """A crashed run replayed must leave exactly the state it would have left.

    Unmarked: this holds today and every transition above is only meaningful if
    it keeps holding. A matcher that is not idempotent would make each retry a
    new reading of the slot, which is the drift arriving through the mechanism
    built to remove it.
    """
    write = _write(SANITIZER_AS_EXECUTION, text=SANITIZER_TEXT)
    first = await repository.record_lineage(write)
    before = (await repository.lineage_report()).state_digest

    replay = await repository.record_lineage(write)

    assert replay.assertion_id == first.assertion_id
    assert replay.revision == first.revision
    assert (await repository.lineage_report()).state_digest == before
    assert len(await _rows(repository, KnowledgeAssertion)) == 1
    assert len(await _revisions(repository, first.assertion_id)) == 1
    assert len(await _evidence(repository, first.assertion_id)) == 1


# A claim can move between scopes without changing what the source says. A
# heading gains an id, or loses one, or a section is written above the first
# heading. The scopes exist to make a diff precise, and a claim crossing between
# them must not read as one claim ending and another beginning.


async def test_a_claim_gaining_a_slot_is_not_retired_and_recreated(repository) -> None:
    """Unscoped becomes anchored: somebody added a heading above a paragraph.

    The claim did not move, the text did not change, and a diff that decided
    scope by scope would have found it missing from the unscoped part and gone
    before it was found in the new slot.
    """
    first = (await _document(
        repository,
        {UNSCOPED: [_write(JAVA_17, kind="fact", text="Spring requires Java 17.")]},
    )).recorded[0]

    outcome = await _document(
        repository,
        {"/stack": [_write(JAVA_17, kind="fact", text="Spring requires Java 17.",
                           run_id="run-2")]},
        run_id="run-2",
    )

    assert outcome.retired == ()
    assert outcome.recorded[0].assertion_id == first.assertion_id
    rows = await _rows(repository, KnowledgeAssertion)
    assert len(rows) == 1
    assert rows[0].retired_at is None


async def test_a_claim_losing_its_slot_is_not_retired_and_recreated(repository) -> None:
    # The other direction: a heading's id removed, or the heading itself, and the
    # section falls back into the unscoped part of its document.
    first = (await _document(
        repository,
        {"/stack": [_write(JAVA_17, kind="fact", text="Spring requires Java 17.")]},
    )).recorded[0]

    outcome = await _document(
        repository,
        {UNSCOPED: [_write(JAVA_17, kind="fact", text="Spring requires Java 17.",
                           run_id="run-2")]},
        run_id="run-2",
    )

    assert outcome.retired == ()
    assert outcome.recorded[0].assertion_id == first.assertion_id


async def test_a_claim_that_moved_between_two_slots_keeps_its_lineage(repository) -> None:
    # Content cut from one section and pasted into another. Both slots are stable
    # and neither is where it was, and the claim is still the same claim.
    first = (await _document(
        repository,
        {
            "/stack": [_write(JAVA_17, kind="fact", text="Spring requires Java 17.")],
            "/build": [_write(MAVEN, kind="fact", text="Spring requires Maven.")],
        },
    )).recorded[0]

    outcome = await _document(
        repository,
        {
            "/stack": [_write(MAVEN, kind="fact", text="Spring requires Maven.",
                              run_id="run-2")],
            "/build": [_write(JAVA_17, kind="fact", text="Spring requires Java 17.",
                              run_id="run-2")],
        },
        run_id="run-2",
    )

    assert outcome.retired == ()
    assert {row.assertion_id for row in outcome.recorded} >= {first.assertion_id}
    assert len(await _rows(repository, KnowledgeAssertion)) == 2


async def test_a_deleted_section_still_loses_its_claims(repository) -> None:
    """The guard on the other side of all of that.

    Nothing above may be bought by making retraction timid. A section the source
    removed takes its claims with it, and the fact that scopes moved underneath
    does not excuse keeping them.
    """
    kept, dropped = (await _document(
        repository,
        {
            "/stack": [_write(JAVA_17, kind="fact", text="Spring requires Java 17.")],
            "/build": [_write(MAVEN, kind="fact", text="Spring requires Maven.")],
        },
    )).recorded

    outcome = await _document(
        repository,
        {"/stack": [_write(JAVA_17, kind="fact", text="Spring requires Java 17.",
                           run_id="run-2")]},
        run_id="run-2",
    )

    assert [row.assertion_id for row in outcome.retired] == [dropped.assertion_id]
    rows = {row.assertion_id: row for row in await _rows(repository, KnowledgeAssertion)}
    assert rows[kept.assertion_id].retired_at is None


async def test_two_kinds_read_from_one_sentence_stay_two_assertions(repository) -> None:
    """Keep different claim kinds separate when they share observed wording.

    A fact and rule extracted from one sentence may quote the same text without
    sharing an assertion identity.
    """
    sentence = "The grace period must not exceed 30 seconds."

    fact = await repository.record_lineage(
        _write(
            entity_key("fact", predicate="limits", subject="grace_period", object="30 seconds"),
            text=sentence,
            kind="fact",
        )
    )
    rule = await repository.record_lineage(
        _write(
            entity_key("rule", subject="grace_period", rule_property="maximum"),
            text=sentence,
            kind="rule",
        )
    )

    assert rule.assertion_id != fact.assertion_id
    kinds = {row.kind for row in await _rows(repository, KnowledgeAssertion)}
    assert kinds == {"fact", "rule"}


async def test_the_same_kind_in_one_slot_still_continues(repository) -> None:
    # The guard must not cost the rung its purpose: within a kind, the same slot
    # saying the same thing is still one claim.
    first = await repository.record_lineage(
        _write(SANITIZER_AS_EXECUTION, text=SANITIZER_TEXT, kind="rule")
    )
    second = await repository.record_lineage(
        _write(SANITIZER_AS_ORDER, text=SANITIZER_TEXT, kind="rule")
    )

    assert second.assertion_id == first.assertion_id


async def test_a_candidate_with_no_kind_is_not_continued_into(repository) -> None:
    """An unknown kind is refused, not treated as a wildcard.

    The difference is the whole point of the rule above. A wildcard would let the
    fused cross-kind match back in through any candidate that happened to carry
    no kind, which is a hole in exactly the wall the filter is.

    The direction is chosen by the harm. Failing to continue an old assertion
    costs a duplicate somebody can merge; merging two identity namespaces costs a
    claim, and nothing afterwards shows which one.
    """
    from nlght.core.knowledge import Candidate, match_proposition

    nameless = Candidate(
        assertion_id="a_legacy",
        entity_key="k",
        entity_key_version="1",
        text=SANITIZER_TEXT,
        document_id=DOC_A,
        slot_ids=frozenset({SLOT_X}),
        kind="",
    )

    decision = match_proposition(
        entity_key="k",
        entity_key_version="1",
        text=SANITIZER_TEXT,
        slot_id=SLOT_X,
        document_id=DOC_A,
        candidates=[nameless],
        kind="rule",
    )

    assert decision.assertion_id is None


#
# The graph and the lineage used to be joined by an equality between two hashes,
# `KnowledgeRevision.fingerprint == Knowledge.identity`, which is not a
# relationship: a fingerprint is a content address and two revisions may carry
# one. A revision is new when the observed *wording* moves, even if the
# fingerprinted structure stays the same. Every read that needed "which state is this
# node" therefore quantified over the set and hoped it answered.
#
# `Knowledge.revision_id` is the edge. These five pin what it means, and in
# particular the two things it must never be allowed to blur: which node's
# pointer may move, and the two very different reasons a pointer can be absent.

async def test_a_structural_change_leaves_the_old_node_on_the_old_state(repository) -> None:
    """Keep the old graph node attached to its original revision.

    A structural change creates a new canonical state instead of repointing the
    old node to content it does not represent.
    """
    first = (await _slot(
        repository,
        _write(JAVA_17, kind="fact", text="Spring requires Java 17."),
    )).recorded[0]

    second = (await _slot(
        repository,
        _write(JAVA_17, kind="fact", text="Spring requires Java 17 exactly.",
               run_id="run-2"),
        run_id="run-2",
    )).recorded[0]

    assert second.assertion_id == first.assertion_id, "one claim, two states"
    assert first.fingerprint != second.fingerprint, "the structure moved, so the address did"
    nodes = {row.identity: row for row in await _rows(repository, Knowledge)}
    assert str(nodes[first.fingerprint].revision_id) == first.revision_id
    assert str(nodes[second.fingerprint].revision_id) == second.revision_id

    resolved = await repository.resolve([first.fingerprint, second.fingerprint])

    assert [row.identity for row in resolved] == [second.fingerprint]
    assert resolved[0].observed_text == "Spring requires Java 17 exactly."


async def test_one_state_reworded_keeps_its_node_and_moves_its_pointer(repository) -> None:
    """Move the graph node's revision pointer after a wording change.

    The structured fingerprint remains stable, while the lineage records a newer
    revision for the same node.
    """
    shared = "fp:one-structured-state"
    first = (await _slot(
        repository,
        _write(JAVA_17, kind="fact", text="Spring requires Java 17",
               fingerprint=shared),
    )).recorded[0]

    second = (await _slot(
        repository,
        _write(JAVA_17, kind="fact", text="Spring requires Java 17.",
               fingerprint=shared, run_id="run-2"),
        run_id="run-2",
    )).recorded[0]

    assert (first.revision, second.revision) == (1, 2)
    assert first.revision_id != second.revision_id
    nodes = await _rows(repository, Knowledge)
    assert [row.identity for row in nodes] == [shared], "one structured state, one node"
    assert str(nodes[0].revision_id) == second.revision_id

    resolved = await repository.resolve([shared])

    assert resolved[0].observed_text == "Spring requires Java 17."


async def test_a_node_with_no_lineage_at_all_stays_canonical(repository) -> None:
    """Absence of a record is not evidence of anything.

    Written before lineage existed, or by a path that keeps none. It names no
    revision and never will until something gives it one, and withholding it
    would be reading a gap in the record as a withdrawal.
    """
    written = await repository.upsert(
        KnowledgeWrite(
            identity="legacy-node",
            payload=FactPayload(subject="spring", predicate="requires", object="java 17"),
            type="dependency",
            confidence=0.9,
            source_id="corpus",
            run_id="run-1",
        )
    )
    node = (await _rows(repository, Knowledge))[0]
    assert (node.revision_id, node.revision_unresolved) == (None, False)

    resolved = await repository.resolve([written.identity])

    assert [row.identity for row in resolved] == ["legacy-node"]
    assert resolved[0].observed_text == "", "no revision, so no recorded wording"


async def test_a_node_whose_state_the_migration_could_not_prove_is_withheld(
    repository,
) -> None:
    """Withhold a node whose revision identity migration could not establish.

    Several matching revisions cannot prove which state a node represents.
    Retrieval must not present a possibly superseded state as current.
    """
    recorded = (await _slot(
        repository,
        _write(JAVA_17, kind="fact", text="Spring requires Java 17."),
    )).recorded[0]

    # Exactly the row state migration 0018 leaves behind for an unprovable node.
    async with AsyncSession(repository._test_engine) as session, session.begin():
        node = await session.get(Knowledge, recorded.fingerprint)
        node.revision_id = None
        node.revision_unresolved = True

    assert await repository.resolve([recorded.fingerprint]) == []
    # A review surface still sees it because unresolved provenance is not invisibility.
    assert len(await repository.resolve(
        [recorded.fingerprint], include_quarantined=True
    )) == 1


async def test_a_later_sighting_settles_what_the_migration_could_not(repository) -> None:
    """The only honest way out: the state stops being inferred and gets recorded."""
    recorded = (await _slot(
        repository,
        _write(JAVA_17, kind="fact", text="Spring requires Java 17."),
    )).recorded[0]
    async with AsyncSession(repository._test_engine) as session, session.begin():
        node = await session.get(Knowledge, recorded.fingerprint)
        node.revision_id = None
        node.revision_unresolved = True

    again = (await _slot(
        repository,
        _write(JAVA_17, kind="fact", text="Spring requires Java 17.", run_id="run-2"),
        run_id="run-2",
    )).recorded[0]

    node = (await _rows(repository, Knowledge))[0]
    assert node.revision_unresolved is False
    assert str(node.revision_id) == again.revision_id
    assert [row.identity for row in await repository.resolve([recorded.fingerprint])] == [
        recorded.fingerprint
    ]
