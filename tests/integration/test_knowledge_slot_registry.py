# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""The slot registry: a surrogate, and every path it has been found by.

Schema only. Nothing resolves a slot here — that is the next step, and these
properties have to hold whatever the resolver ends up doing with them.

The strength lives on the *anchor* rather than on the slot, and the reason is one
transition:

    authored → derived    a degradation

When `[[expenses]]` disappears and only the heading path is left, a resolver that
still believed it held an authored anchor would carry a lineage across a change
nobody could see. One strength per slot cannot express that; a strength per
anchor can, and the history is where the transition becomes visible.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nlght.adapters.outbound.persistence.knowledge_models import (
    KnowledgeSlot,
    KnowledgeSlotAnchor,
)
from nlght.core.knowledge import AUTHORED_ANCHOR, DERIVED_ANCHOR

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def engine():
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
    yield engine
    await engine.dispose()


async def _slot_with(engine, *anchors: tuple[str, str]):  # noqa: ANN001, ANN201
    """A slot and its anchor history, oldest first."""
    async with AsyncSession(engine) as session, session.begin():
        slot = KnowledgeSlot(slot_id="slot_1", document_id="doc-a")
        session.add(slot)
        await session.flush()
        for index, (anchor, strength) in enumerate(anchors):
            session.add(
                KnowledgeSlotAnchor(
                    slot_id=slot.slot_id,
                    anchor=anchor,
                    strength=strength,
                    valid_from=datetime(2026, 1, index + 1, tzinfo=UTC),
                    valid_to=(
                        None
                        if index == len(anchors) - 1
                        else datetime(2026, 1, index + 2, tzinfo=UTC)
                    ),
                )
            )
    async with AsyncSession(engine) as session:
        rows = (
            await session.execute(
                select(KnowledgeSlotAnchor)
                .where(KnowledgeSlotAnchor.slot_id == "slot_1")
                .order_by(KnowledgeSlotAnchor.valid_from)
            )
        ).scalars().all()
        return list(rows)


async def test_a_slot_keeps_every_path_it_has_been_found_by(engine) -> None:
    history = await _slot_with(
        engine,
        ("Expenses", DERIVED_ANCHOR),
        ("expenses-policy", AUTHORED_ANCHOR),
    )

    assert [row.anchor for row in history] == ["Expenses", "expenses-policy"]
    assert [row.strength for row in history] == [DERIVED_ANCHOR, AUTHORED_ANCHOR]


async def test_exactly_one_anchor_is_current(engine) -> None:
    """`valid_to IS NULL` *is* current.

    A separate `is_current` flag would be a second fact about the same thing, and
    two facts about one thing can disagree — usually a year later, quietly.
    """
    history = await _slot_with(
        engine,
        ("Expenses", DERIVED_ANCHOR),
        ("expenses-policy", AUTHORED_ANCHOR),
    )

    assert [row.is_current for row in history] == [False, True]


async def test_a_degradation_is_visible_in_the_history(engine) -> None:
    """The transition one strength per slot could not express.

    The authored id was removed from the source, so the section arrives with a
    heading path again. The slot continues — it is the same section — but a
    resolver reading only "this slot is authored" would not see that it stopped
    being. Here the last two rows say it plainly.
    """
    history = await _slot_with(
        engine,
        ("expenses-policy", AUTHORED_ANCHOR),
        ("Expenses", DERIVED_ANCHOR),
    )

    current, previous = history[-1], history[-2]
    assert previous.strength == AUTHORED_ANCHOR
    assert current.strength == DERIVED_ANCHOR
    assert current.is_current


async def test_retiring_a_slot_keeps_its_anchors(engine) -> None:
    # A section that comes back must be distinguishable from one that was never
    # there, and that decision needs the paths it used to have.
    await _slot_with(engine, ("expenses-policy", AUTHORED_ANCHOR))

    async with AsyncSession(engine) as session, session.begin():
        slot = await session.get(KnowledgeSlot, "slot_1")
        slot.status = "retired"
        slot.retired_at = datetime(2026, 6, 1, tzinfo=UTC)

    async with AsyncSession(engine) as session:
        anchors = (
            await session.execute(
                select(KnowledgeSlotAnchor).where(KnowledgeSlotAnchor.slot_id == "slot_1")
            )
        ).scalars().all()
        slot = await session.get(KnowledgeSlot, "slot_1")

    assert slot.status == "retired"
    assert [row.anchor for row in anchors] == ["expenses-policy"]


async def test_a_slot_id_is_not_derived_from_anything(engine) -> None:
    """The property the whole registry exists for.

    Derived from the position it renumbered under an insertion; derived from the
    anchor it would move when somebody renamed a heading. Minted once, it is
    neither.
    """
    async with AsyncSession(engine) as session, session.begin():
        session.add(KnowledgeSlot(slot_id="slot_1", document_id="doc-a"))
        session.add(
            KnowledgeSlotAnchor(
                slot_id="slot_1", anchor="expenses-policy", strength=AUTHORED_ANCHOR
            )
        )

    async with AsyncSession(engine) as session:
        slot = await session.get(KnowledgeSlot, "slot_1")
        anchor = (
            await session.execute(select(KnowledgeSlotAnchor))
        ).scalars().one()

    assert slot.slot_id != anchor.anchor
    assert slot.slot_id != slot.document_id
