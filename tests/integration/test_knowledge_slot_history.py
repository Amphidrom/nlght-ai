# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Slots minted, resolved, and their anchor history advanced.

The write path. What is asserted here is the rule that keeps the history worth
having:

    the anchor history is advanced only on a real transition

An exact match writes nothing, which is every run over an unchanged document and
therefore almost every run. A rename resolves through an alias or the content,
and *then* the current anchor is closed and a new one opened. Writing a row per
run would turn an audit trail into a log of runs, and a table nobody can read is
a table nobody consults.

Resolution and the history are written in the same transaction as the claims. A
slot continued while its history still described the previous state is a corpus
disagreeing with itself, and nothing afterwards could say which half was right.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nlght.adapters.outbound.persistence.knowledge_models import (
    KnowledgeEvidence,
    KnowledgeSlot,
    KnowledgeSlotAnchor,
)
from nlght.adapters.outbound.persistence.knowledge_repository import (
    SqlAlchemyKnowledgeRepository,
)
from nlght.core.knowledge import (
    AUTHORED_ANCHOR,
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

DOC = "doc-a"
CLAIM = "Spring requires Java 17."


@pytest_asyncio.fixture
async def repository():
    from sqlalchemy import event
    from sqlalchemy.ext.asyncio import create_async_engine

    from nlght.adapters.outbound.persistence.knowledge_models import KnowledgeBase

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    @event.listens_for(engine.sync_engine, "connect")
    def _enforce_foreign_keys(dbapi_connection, _record):  # noqa: ANN001, ANN202
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with engine.begin() as conn:
        await conn.run_sync(KnowledgeBase.metadata.create_all)
    repository = SqlAlchemyKnowledgeRepository(engine)
    repository._test_engine = engine  # noqa: SLF001
    yield repository
    await engine.dispose()


def _section(  # noqa: ANN201
    anchor: str,
    strength: str = DERIVED_ANCHOR,
    unit: str = "s:doc-a:0",
    content: str = CLAIM,
):
    return ObservedSection.as_read(
        anchor=anchor,
        anchor_strength=strength,
        content=content,
        unit_ordinal=unit,
    )


def _pair(run_id: str, unit: str = "s:doc-a:0", fact: str = "java 17"):  # noqa: ANN201
    lineage = LineageWrite(
        entity=entity_key("fact", predicate="requires", subject="spring", object=fact),
        kind="fact", scope=(), text=CLAIM, fingerprint=f"fp-{fact}",
        proposition=Proposition({"subject": "spring", "predicate": "requires",
                                 "object": fact}),
        extraction_version="v1", document_id=DOC, document_revision="rev-1",
        slot_id=unit, run_id=run_id,
    )
    graph = KnowledgeWrite(
        identity=f"fp-{fact}",
        payload=FactPayload(subject="spring", predicate="requires", object=fact),
        type="dependency", confidence=0.9, source_id="s", run_id=run_id,
    )
    return graph, lineage


async def _run(repository, *sections, run_id: str = "run-1"):  # noqa: ANN001, ANN002, ANN201
    return await repository.record_document(
        document_id=DOC,
        sections=[
            replace(
                section,
                assertions=(_pair(run_id, section.unit_ordinal, section.unit_ordinal),),
            )
            for section in sections
        ],
        run_id=run_id,
    )


async def _anchors(repository):  # noqa: ANN001, ANN201
    async with AsyncSession(repository._test_engine) as session:  # noqa: SLF001
        rows = (
            await session.execute(
                select(KnowledgeSlotAnchor).order_by(KnowledgeSlotAnchor.valid_from)
            )
        ).scalars().all()
        return list(rows)


async def _slots(repository):  # noqa: ANN001, ANN201
    async with AsyncSession(repository._test_engine) as session:  # noqa: SLF001
        return list((await session.execute(select(KnowledgeSlot))).scalars().all())


# ---------------------------------------------------------------------------
# The history moves only when something moved
# ---------------------------------------------------------------------------

async def test_a_first_sighting_mints_a_slot_and_its_first_anchor(repository) -> None:
    await _run(repository, _section("/expenses"))

    slots, anchors = await _slots(repository), await _anchors(repository)
    assert len(slots) == 1
    assert [(a.anchor, a.strength, a.is_current) for a in anchors] == [
        ("/expenses", DERIVED_ANCHOR, True)
    ]


async def test_a_repeated_run_writes_no_second_row(repository) -> None:
    """The guard that decides whether this table stays readable.

    An exact match is almost every run over almost every document. A row per run
    would be a log of runs wearing the name of an audit trail.
    """
    await _run(repository, _section("/expenses"))
    await _run(repository, _section("/expenses"), run_id="run-2")
    await _run(repository, _section("/expenses"), run_id="run-3")

    assert len(await _anchors(repository)) == 1


async def test_a_rename_closes_the_old_anchor_and_opens_a_new_one(repository) -> None:
    # The section is recognised by its content, so the slot continues — and the
    # path it is now found by is recorded rather than the old one being edited.
    await _run(repository, _section("/expenses"))

    await _run(repository, _section("/spending"), run_id="run-2")

    anchors = await _anchors(repository)
    assert [(a.anchor, a.is_current) for a in anchors] == [
        ("/expenses", False),
        ("/spending", True),
    ]
    assert len(await _slots(repository)) == 1, "a rename is not a new section"


async def test_a_path_that_comes_back_is_a_third_row(repository) -> None:
    """A → B → A, and not the first row reopened.

    Reopening it would say the anchor was valid the whole time, which is the one
    thing the history exists to be able to deny. Three rows answer "which anchor
    was valid in March"; one row with a reopened range answers it wrongly.
    """
    await _run(repository, _section("/expenses"))
    await _run(repository, _section("/spending"), run_id="run-2")

    await _run(repository, _section("/expenses"), run_id="run-3")

    anchors = await _anchors(repository)
    assert [(a.anchor, a.is_current) for a in anchors] == [
        ("/expenses", False),
        ("/spending", False),
        ("/expenses", True),
    ]


async def test_an_upgrade_to_an_authored_anchor_is_recorded(repository) -> None:
    # Somebody added an id to a heading. The section is the same one, and the
    # slot is now anchored on something stronger than a path.
    await _run(repository, _section("/expenses"))

    await _run(repository, _section("expenses-policy", AUTHORED_ANCHOR), run_id="run-2")

    anchors = await _anchors(repository)
    assert [(a.anchor, a.strength, a.is_current) for a in anchors] == [
        ("/expenses", DERIVED_ANCHOR, False),
        ("expenses-policy", AUTHORED_ANCHOR, True),
    ]


async def test_a_degradation_is_recorded_rather_than_hidden(repository) -> None:
    """`authored → derived`: the id was removed and only the path is left.

    The slot continues, because it is the same section. What must not happen is
    continuing while the history still claims an authored anchor — a lineage
    carried across a change nobody could see.
    """
    await _run(repository, _section("expenses-policy", AUTHORED_ANCHOR))

    await _run(repository, _section("/expenses"), run_id="run-2")

    anchors = await _anchors(repository)
    assert [(a.strength, a.is_current) for a in anchors] == [
        (AUTHORED_ANCHOR, False),
        (DERIVED_ANCHOR, True),
    ]
    assert len(await _slots(repository)) == 1


# ---------------------------------------------------------------------------
# What is written against the claim
# ---------------------------------------------------------------------------

async def test_the_evidence_records_the_slot_and_not_the_ordinal(repository) -> None:
    """The point of the whole registry.

    `s:doc-a:0` is a position. It renumbers when a paragraph is inserted above,
    which is why retraction had to be widened to the document to stop that
    destroying approvals.
    """
    await _run(repository, _section("/expenses"))

    slot = (await _slots(repository))[0]
    async with AsyncSession(repository._test_engine) as session:  # noqa: SLF001
        evidence = (await session.execute(select(KnowledgeEvidence))).scalars().all()

    assert [row.slot_id for row in evidence] == [slot.slot_id]
    assert slot.slot_id != "s:doc-a:0"


async def test_a_section_with_nothing_to_be_found_by_gets_no_slot(repository) -> None:
    """PDF and prose, permanently.

    An id nothing can resolve back to would be new on every run, retiring and
    recreating everything it holds each time — worse than no slot, because it
    claims a stability it does not have. Those documents keep the document as
    their smallest dependable diff boundary.
    """
    await _run(
        repository,
        ObservedSection.as_read(
            anchor="",
            anchor_strength=NO_ANCHOR,
            content=CLAIM,
            unit_ordinal="s:doc-a:0",
        ),
    )

    assert await _slots(repository) == []
    assert await _anchors(repository) == []
    async with AsyncSession(repository._test_engine) as session:  # noqa: SLF001
        evidence = (await session.execute(select(KnowledgeEvidence))).scalars().all()
    # Still recorded, still diffable at the document level — just not slotted.
    assert [row.slot_id for row in evidence] == ["s:doc-a:0"]


async def test_two_sections_of_one_document_get_two_slots(repository) -> None:
    await _run(
        repository,
        _section("/expenses", unit="s:doc-a:0", content="A."),
        _section("/travel", unit="s:doc-a:1", content="B."),
    )

    assert len({slot.slot_id for slot in await _slots(repository)}) == 2


async def test_two_identical_sections_stay_two_slots(repository) -> None:
    """The fingerprint alone must never decide.

    Two sections of one document that say the same thing under different
    headings are two sections. Recovery exists to find a slot from a *previous*
    run; offering it a sibling would collapse them into one, and which of them
    won would depend on the order they happened to be written in.
    """
    await _run(
        repository,
        _section("/expenses", unit="s:doc-a:0"),
        _section("/travel", unit="s:doc-a:1"),
    )

    assert len({slot.slot_id for slot in await _slots(repository)}) == 2


async def test_a_section_keeps_its_slot_when_a_paragraph_is_inserted_above(
    repository,
) -> None:
    """The regression the whole registry exists for, at the persisted level.

    The ordinal renumbers; the anchor does not. The slot the claim was written
    against must be the same row before and after, or retraction narrowed back to
    the slot would retire everything below an insertion — which is exactly what
    it did before the container was widened to the document.
    """
    await _run(repository, _section("/expenses", unit="s:doc-a:0"))
    before = (await _slots(repository))[0].slot_id

    # A paragraph is inserted above, so every ordinal below it shifts.
    await _run(
        repository,
        _section("/intro", unit="s:doc-a:0", content="Intro."),
        _section("/expenses", unit="s:doc-a:1"),
        run_id="run-2",
    )

    slots = {slot.slot_id for slot in await _slots(repository)}
    assert before in slots, "the section that moved kept its slot"
    assert len(slots) == 2, "the inserted paragraph is the only new one"


async def test_recovery_that_finds_two_equally_good_slots_refuses(repository) -> None:
    """Ambiguity is the answer, not a tie broken by order.

    Two sections that once said the same thing, both now unfindable by their
    anchors. Picking either would be a coin toss recorded as an identity.
    """
    await _run(
        repository,
        _section("/expenses", unit="s:doc-a:0"),
        _section("/travel", unit="s:doc-a:1"),
    )
    minted = {slot.slot_id for slot in await _slots(repository)}

    await _run(repository, _section("/renamed", unit="s:doc-a:0"), run_id="run-2")

    slots = {slot.slot_id for slot in await _slots(repository)}
    assert len(slots) == 3, "neither of the two was claimed on a coin toss"
    assert minted < slots


# ---------------------------------------------------------------------------
# The slot's life, which is not the claim's life
# ---------------------------------------------------------------------------
#
# Keeping the two apart is the point. A section can disappear while everything it
# said is still asserted somewhere else, and a claim can be retired while the
# section it stood in is untouched. Deciding one from the other is the ordering
# failure this design already paid for once.


async def test_a_section_the_document_lost_retires_its_slot(repository) -> None:
    await _run(
        repository,
        _section("/expenses", unit="s:doc-a:0", content="A."),
        _section("/travel", unit="s:doc-a:1", content="B."),
    )
    slots = {slot.slot_id for slot in await _slots(repository)}

    outcome = await _run(
        repository, _section("/expenses", unit="s:doc-a:0", content="A."), run_id="run-2"
    )

    assert len(outcome.retired_slots) == 1
    rows = {slot.slot_id: slot for slot in await _slots(repository)}
    assert rows[outcome.retired_slots[0]].status == "retired"
    assert rows[outcome.retired_slots[0]].retired_at is not None
    # Retired, not deleted: a section that comes back must be distinguishable
    # from one that was never there.
    assert set(rows) == slots


async def test_a_retired_slot_keeps_its_anchors_closed(repository) -> None:
    # The path it had when it left is the one a returning section arrives under.
    await _run(
        repository,
        _section("/expenses", unit="s:doc-a:0", content="A."),
        _section("/travel", unit="s:doc-a:1", content="B."),
    )

    outcome = await _run(
        repository, _section("/expenses", unit="s:doc-a:0", content="A."), run_id="run-2"
    )

    gone = outcome.retired_slots[0]
    anchors = [row for row in await _anchors(repository) if row.slot_id == gone]
    assert anchors, "its history outlives it"
    assert not any(row.is_current for row in anchors)


async def test_a_renamed_section_keeps_its_slot_live(repository) -> None:
    # Resolution succeeded, so nothing was lost — the slot is the same one under
    # a different path.
    await _run(repository, _section("/expenses"))

    outcome = await _run(repository, _section("/spending"), run_id="run-2")

    assert outcome.retired_slots == ()
    assert [slot.status for slot in await _slots(repository)] == ["live"]


async def test_a_section_that_lost_its_heading_retires_the_slot_not_the_claim(
    repository,
) -> None:
    """`slot → unscoped`, and the two outcomes are deliberately different.

    There is no section to be found any more, so the slot is over. The claim is
    still asserted by the document, so it is not.
    """
    await _run(repository, _section("/expenses"))

    outcome = await _run(
        repository,
        ObservedSection.as_read(
            anchor="",
            anchor_strength=NO_ANCHOR,
            content=CLAIM,
            unit_ordinal="s:doc-a:0",
        ),
        run_id="run-2",
    )

    assert len(outcome.retired_slots) == 1
    assert outcome.retired == (), "the claim is still asserted, only not from a slot"


async def test_a_retired_slot_is_not_reactivated_by_the_section_returning(
    repository,
) -> None:
    """Deliberate or not at all.

    Reviving it would continue a lineage across a gap nobody looked at, which is
    what the surrogate ids exist to prevent.
    """
    await _run(
        repository,
        _section("/expenses", unit="s:doc-a:0", content="A."),
        _section("/travel", unit="s:doc-a:1", content="B."),
    )
    outcome = await _run(
        repository, _section("/expenses", unit="s:doc-a:0", content="A."), run_id="run-2"
    )
    gone = outcome.retired_slots[0]

    await _run(
        repository,
        _section("/expenses", unit="s:doc-a:0", content="A."),
        _section("/travel", unit="s:doc-a:1", content="B."),
        run_id="run-3",
    )

    rows = {slot.slot_id: slot for slot in await _slots(repository)}
    assert rows[gone].status == "retired"
    assert len(rows) == 3, "the returning section is a new slot, deliberately"


# ---------------------------------------------------------------------------
# The support view, which decides nothing
# ---------------------------------------------------------------------------

async def test_support_says_where_a_claim_stopped_being_said(repository) -> None:
    """The question a reviewer has when the number moves.

    A slot losing a claim is not the claim going — it may have moved, or lost its
    heading. What retirement asks is whether the document still carries it
    anywhere; what this answers is *where* it stopped.
    """
    await _run(
        repository,
        _section("/expenses", unit="s:doc-a:0", content="A."),
        _section("/travel", unit="s:doc-a:1", content="B."),
    )
    slots = {slot.slot_id for slot in await _slots(repository)}

    outcome = await _run(
        repository, _section("/expenses", unit="s:doc-a:0", content="A."), run_id="run-2"
    )

    withdrawn = [support for support in outcome.supports if not support.active]
    active = [support for support in outcome.supports if support.active]
    assert len(withdrawn) == 1
    assert withdrawn[0].slot_id in slots
    assert len(active) == 1
    assert all(support.document_id == DOC for support in outcome.supports)


async def test_a_claim_supported_from_two_slots_keeps_the_one_that_remains(
    repository,
) -> None:
    # One slot stops carrying it, another still does. The support view says so,
    # and the claim is untouched.
    same = "Spring requires Java 17."
    await repository.record_document(
        document_id=DOC,
        sections=[
            replace(_section("/a", unit="s:doc-a:0", content=same),
                    assertions=(_pair("run-1", "s:doc-a:0"),)),
            replace(_section("/b", unit="s:doc-a:1", content="other"),
                    assertions=(_pair("run-1", "s:doc-a:1"),)),
        ],
        run_id="run-1",
    )

    outcome = await repository.record_document(
        document_id=DOC,
        sections=[
            replace(_section("/a", unit="s:doc-a:0", content=same),
                    assertions=(_pair("run-2", "s:doc-a:0"),)),
            replace(_section("/b", unit="s:doc-a:1", content="other"), assertions=()),
        ],
        run_id="run-2",
    )

    assert outcome.retired == (), "one slot dropped it; the document still says it"
    assert any(not support.active for support in outcome.supports)
    assert any(support.active for support in outcome.supports)
