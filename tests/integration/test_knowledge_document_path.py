# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Preserve the observed document path with each knowledge sighting.

Changing a path does not itself create an assertion revision. Reports use the
path recorded for each run rather than reconstructing it from a document ID.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nlght.adapters.outbound.persistence.knowledge_models import (
    KnowledgeAssertion,
    KnowledgeEvidence,
    KnowledgeRevision,
)
from nlght.adapters.outbound.persistence.knowledge_repository import (
    SqlAlchemyKnowledgeRepository,
)
from nlght.core.knowledge import LineageWrite, Proposition, entity_key

pytestmark = pytest.mark.integration

DOCUMENT = "doc-1"
SENTENCE = "The grace period must not exceed 30 seconds."
ENTITY = entity_key("rule", subject="grace_period", rule_property="maximum")


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


def _sighting(*, path: str, revision: str, run: str) -> LineageWrite:
    """The same claim, in the same document, seen under a given name.

    Args:
        path (str): Document path seen by the run.
        revision (str): Document revision identifier.
        run (str): Run identifier for the sighting.

    Returns:
        LineageWrite: Lineage write for the document path and revision.
    """
    return LineageWrite(
        entity=ENTITY,
        kind="rule",
        text=SENTENCE,
        fingerprint="fp-1",
        extraction_version="ver-1",
        document_id=DOCUMENT,
        document_revision=revision,
        document_path=path,
        run_id=run,
        slot_id="slot-1",
        proposition=Proposition({"rule_text": SENTENCE, "subject": "grace_period",
                                 "rule_property": "maximum"}),
    )


async def _rows(repository, model):
    async with AsyncSession(repository._test_engine) as session:
        return list((await session.execute(select(model))).scalars().all())


async def test_a_renamed_document_changes_nothing_about_the_claim(repository) -> None:
    """Keep the assertion and revision stable across a document-path rename."""
    first = await repository.record_lineage(
        _sighting(path="docs/a.md", revision="rev-1", run="run-1")
    )
    second = await repository.record_lineage(
        _sighting(path="docs/renamed-a.md", revision="rev-1", run="run-2")
    )

    assert second.assertion_id == first.assertion_id
    assert len(await _rows(repository, KnowledgeAssertion)) == 1
    # No revision: the claim did not move, only the file's name did.
    assert len(await _rows(repository, KnowledgeRevision)) == 1


async def test_each_sighting_keeps_the_name_it_saw(repository) -> None:
    """Keep the document path observed by each historical sighting."""
    await repository.record_lineage(_sighting(path="docs/a.md", revision="rev-1", run="run-1"))
    await repository.record_lineage(
        _sighting(path="docs/renamed-a.md", revision="rev-1", run="run-2")
    )

    paths = sorted(row.document_path for row in await _rows(repository, KnowledgeEvidence))

    assert paths == ["docs/a.md", "docs/renamed-a.md"]


async def test_the_report_can_name_the_document_after_a_rename(repository) -> None:
    """And a report names it by what it is called now.

    The latest sighting wins, so an expectation written against the current
    corpus finds the file under its current name while the older evidence keeps
    what it saw. Both runs remain nameable, which is the property that failed
    live: `document_paths` was read from an empty table, so every case reported
    "in neither snapshot" and nothing could be checked at all.
    """
    await repository.record_lineage(_sighting(path="docs/a.md", revision="rev-1", run="run-1"))
    before = await repository.lineage_report()

    await repository.record_lineage(
        _sighting(path="docs/renamed-a.md", revision="rev-1", run="run-2")
    )
    after = await repository.lineage_report()

    assert before.document_paths == {DOCUMENT: "docs/a.md"}
    assert after.document_paths == {DOCUMENT: "docs/renamed-a.md"}


async def test_a_document_with_no_recorded_path_is_absent_rather_than_guessed(
    repository,
) -> None:
    # Evidence written before the name was carried has none, and cannot be given
    # one: what the file was called at that moment was not recorded. Absent is
    # the honest answer, and an expectation naming it fails rather than matching
    # something else.
    await repository.record_lineage(_sighting(path="", revision="rev-1", run="run-1"))

    report = await repository.lineage_report()

    assert report.document_paths == {}
