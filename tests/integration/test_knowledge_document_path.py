# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""The document's name travels with the sighting, and decides nothing.

Three rules, and the third is why the first two matter:

    a path may change            → no assertion and no revision because of it
    a path is stored as observed → never reconstructed from `document_id`
    a report names both runs     → each sighting keeps the name it saw

`document_id` is a hash, so a report naming only that can say a claim moved and
never which file. The obvious place to look a name up was `ingestion_documents`,
and a live corpus had it empty: a knowledge-only flow does not write the
ingestion record, so every document came back unnameable and every expectation
reported "in neither snapshot". Joining the two lifecycles to borrow a string is
a decision of its own; carrying the name with the evidence is not.
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


def _sighting(*, path: str, revision: str, run: str) -> LineageWrite:
    """The same claim, in the same document, seen under a given name."""
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


async def _rows(repository, model):  # noqa: ANN001, ANN201
    async with AsyncSession(repository._test_engine) as session:  # noqa: SLF001
        return list((await session.execute(select(model))).scalars().all())


async def test_a_renamed_document_changes_nothing_about_the_claim(repository) -> None:
    """The gate: same document_id, new path, same knowledge.

    A rename is not an edit. The file is called something else and says the same
    thing, so the assertion continues, no revision is written, and the only new
    row is the sighting — which knows the new name.
    """
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
    """Stored as observed, never reconstructed.

    The two runs saw two names, and both are true of the moments they describe.
    Recomputing a path from `document_id` — or overwriting the old one — would
    make the earlier evidence say something that was not the case when it was
    written.
    """
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
