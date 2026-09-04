# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from dataclasses import replace

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from nlght.adapters.outbound.persistence.ingestion_models import (
    IngestionDocument,
    IngestionDocumentRevision,
)
from nlght.adapters.outbound.persistence.ingestion_repository import SqlAlchemyIngestionRepository
from nlght.core.ingestion import (
    DocumentRevisionWrite,
    IndexStateWrite,
    SnapshotCommit,
)


@pytest.fixture
async def sqlite_engine():
    """A throwaway knowledge database — ingestion state lives there, not in the
    platform schema (ADR-0033). Overrides the shared platform fixture.

    Foreign keys are switched on per connection: SQLite ignores them by default,
    which would let a wrong cross-table insert order pass here while failing on
    PostgreSQL, the store the code actually runs against.
    """
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
    yield engine
    await engine.dispose()


def _revision(name: str, version: int = 1) -> DocumentRevisionWrite:
    return DocumentRevisionWrite(
        document_id=f"document-{name}",
        external_id=f"fs:repo/{name}.md",
        processing_revision_id=f"revision-{name}-{version}",
        source_revision_id=f"source-revision-{name}-{version}",
        path=f"repo/{name}.md",
        artifact_ref=f"artifact://run/{name}-{version}",
        content=f"# {name}\n\nbody of {name} v{version}\n",
        metadata={"root": "repo"},
        processing={"classifier_version": "1", "enricher_version": "1"},
    )


def _snapshot(key: str, *documents: DocumentRevisionWrite, complete: bool = True) -> SnapshotCommit:
    return SnapshotCommit(
        snapshot_key=key,
        source_id="repo-source",
        source_type="filesystem",
        config_revision="config-v1",
        complete=complete,
        documents=documents,
    )


async def test_snapshot_commit_is_idempotent_and_only_complete_snapshot_tombstones(sqlite_engine) -> None:
    repository = SqlAlchemyIngestionRepository(sqlite_engine)
    first = _snapshot("snapshot-1", _revision("a"), _revision("b"))

    created = await repository.commit_snapshot(first)
    replay = await repository.commit_snapshot(first)
    incomplete = await repository.commit_snapshot(_snapshot("snapshot-2", _revision("a"), complete=False))

    assert created.created_processing_revision_ids == ("revision-a-1", "revision-b-1")
    assert replay.run_id == created.run_id
    assert replay.idempotent_replay
    assert incomplete.tombstoned_document_ids == ()
    async with AsyncSession(sqlite_engine) as session:
        b = await session.get(IngestionDocument, "document-b")
        assert b is not None and b.deleted_at is None

    completed = await repository.commit_snapshot(_snapshot("snapshot-3", _revision("a")))

    assert completed.tombstoned_document_ids == ("document-b",)
    async with AsyncSession(sqlite_engine) as session:
        b = await session.get(IngestionDocument, "document-b")
        assert b is not None and b.deleted_at is not None


async def test_snapshot_commit_persists_observations_under_their_run(sqlite_engine) -> None:
    # Regression: the run must be inserted before its observations. With foreign
    # keys enforced, a multi-document snapshot would raise IntegrityError if the
    # child rows were flushed first.
    from sqlalchemy import select

    from nlght.adapters.outbound.persistence.ingestion_models import (
        IngestionSnapshotObservation,
    )

    repository = SqlAlchemyIngestionRepository(sqlite_engine)
    result = await repository.commit_snapshot(
        _snapshot("snapshot-1", _revision("a"), _revision("b"), _revision("c"))
    )

    async with AsyncSession(sqlite_engine) as session:
        observations = (
            await session.execute(
                select(IngestionSnapshotObservation).where(
                    IngestionSnapshotObservation.run_id == result.run_id
                )
            )
        ).scalars().all()

    assert {obs.document_id for obs in observations} == {
        "document-a",
        "document-b",
        "document-c",
    }
    assert all(obs.run_id == result.run_id for obs in observations)


async def test_snapshot_key_and_immutable_revision_reject_conflicting_reuse(sqlite_engine) -> None:
    repository = SqlAlchemyIngestionRepository(sqlite_engine)
    first = _snapshot("snapshot-1", _revision("a"))
    await repository.commit_snapshot(first)

    with pytest.raises(ValueError, match="different snapshot"):
        await repository.commit_snapshot(replace(first, complete=False))
    conflicting = replace(_revision("a"), artifact_ref="artifact://different")
    with pytest.raises(ValueError, match="different immutable content"):
        await repository.commit_snapshot(_snapshot("snapshot-2", conflicting))


# ---------------------------------------------------------------------------
# per-document index state (replaces copy-on-write generations)
# ---------------------------------------------------------------------------


def _index_write(
    name: str,
    *,
    target: str = "data:main",
    enricher: str = "enricher-1",
    generation: str = "generation-1",
) -> IndexStateWrite:
    return IndexStateWrite(
        document_id=f"document-{name}",
        target=target,
        content_hash=f"content-{name}",
        enricher_hash=enricher,
        processing_revision_id=f"revision-{name}-1",
        index_generation=generation,
    )


async def test_unknown_document_has_no_index_state(sqlite_engine) -> None:
    repository = SqlAlchemyIngestionRepository(sqlite_engine)
    assert await repository.get_index_state(document_id="document-a", target="data:main") is None


async def test_recording_an_index_write_is_idempotent(sqlite_engine) -> None:
    repository = SqlAlchemyIngestionRepository(sqlite_engine)
    await repository.commit_snapshot(_snapshot("snapshot-1", _revision("a")))

    first = await repository.record_indexed(_index_write("a"))
    second = await repository.record_indexed(_index_write("a"))

    assert first.document_id == second.document_id == "document-a"
    assert second.content_hash == "content-a"
    state = await repository.get_index_state(document_id="document-a", target="data:main")
    assert state is not None and state.processing_revision_id == "revision-a-1"


async def test_changed_content_or_enricher_is_visible_as_a_reindex_reason(sqlite_engine) -> None:
    repository = SqlAlchemyIngestionRepository(sqlite_engine)
    await repository.commit_snapshot(_snapshot("snapshot-1", _revision("a")))
    await repository.record_indexed(_index_write("a"))

    # Same content, newer enricher version — the stored state must move on.
    await repository.record_indexed(_index_write("a", enricher="enricher-2"))
    state = await repository.get_index_state(document_id="document-a", target="data:main")
    assert state is not None
    assert state.enricher_hash == "enricher-2"
    assert state.content_hash == "content-a"


async def test_index_state_is_tracked_per_target(sqlite_engine) -> None:
    repository = SqlAlchemyIngestionRepository(sqlite_engine)
    await repository.commit_snapshot(_snapshot("snapshot-1", _revision("a")))

    await repository.record_indexed(_index_write("a", target="data:main"))
    await repository.record_indexed(_index_write("a", target="knowledge:main", enricher="enricher-9"))

    data = await repository.get_index_state(document_id="document-a", target="data:main")
    knowledge = await repository.get_index_state(document_id="document-a", target="knowledge:main")
    assert data is not None and data.enricher_hash == "enricher-1"
    assert knowledge is not None and knowledge.enricher_hash == "enricher-9"


async def test_clearing_index_state_reports_whether_a_row_existed(sqlite_engine) -> None:
    repository = SqlAlchemyIngestionRepository(sqlite_engine)
    await repository.commit_snapshot(_snapshot("snapshot-1", _revision("a")))
    await repository.record_indexed(_index_write("a"))

    assert await repository.clear_index_state(document_id="document-a", target="data:main") is True
    assert await repository.clear_index_state(document_id="document-a", target="data:main") is False
    assert await repository.get_index_state(document_id="document-a", target="data:main") is None


async def test_index_state_identity_fields_must_not_be_empty() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        IndexStateWrite(
            document_id="document-a",
            target="   ",
            content_hash="content-a",
            enricher_hash="enricher-1",
            processing_revision_id="revision-a-1",
            index_generation="generation-1",
        )


# ---------------------------------------------------------------------------
# index generation — which incarnation of a target the state describes
# ---------------------------------------------------------------------------


async def test_a_target_gets_a_generation_on_first_ask(sqlite_engine) -> None:
    repository = SqlAlchemyIngestionRepository(sqlite_engine)

    first = await repository.current_generation(target="data:main")
    second = await repository.current_generation(target="data:main")

    assert first and first == second


async def test_targets_do_not_share_a_generation(sqlite_engine) -> None:
    repository = SqlAlchemyIngestionRepository(sqlite_engine)

    data = await repository.current_generation(target="data:main")
    knowledge = await repository.current_generation(target="knowledge:main")

    assert data != knowledge


async def test_rotating_invalidates_every_document_of_that_target(sqlite_engine) -> None:
    # The whole point: a recreated index must make the recorded state stop
    # counting, so a rerun writes the corpus again instead of skipping it.
    repository = SqlAlchemyIngestionRepository(sqlite_engine)
    await repository.commit_snapshot(_snapshot("snapshot-1", _revision("a")))
    generation = await repository.current_generation(target="data:main")
    await repository.record_indexed(_index_write("a", generation=generation))

    rotated = await repository.rotate_generation(target="data:main")

    assert rotated != generation
    state = await repository.get_index_state(document_id="document-a", target="data:main")
    assert state is not None
    assert state.index_generation != rotated  # therefore: not current, write it again


async def test_a_summary_separates_outstanding_work_from_never_indexed(sqlite_engine) -> None:
    repository = SqlAlchemyIngestionRepository(sqlite_engine)
    await repository.commit_snapshot(
        _snapshot("snapshot-1", _revision("a"), _revision("b"), _revision("c"))
    )
    generation = await repository.current_generation(target="data:main")
    await repository.record_indexed(_index_write("a", generation=generation))
    await repository.record_indexed(_index_write("b", generation="an-older-generation"))

    summary = await repository.index_state_summary(target="data:main")

    assert summary.live_documents == 3      # c was never indexed at all
    assert summary.indexed == 2
    assert summary.on_current_generation == 1
    assert summary.outstanding == 1         # b, left behind by an interrupted reindex


async def test_stale_documents_are_reported_by_name(sqlite_engine) -> None:
    repository = SqlAlchemyIngestionRepository(sqlite_engine)
    await repository.commit_snapshot(_snapshot("snapshot-1", _revision("a"), _revision("b")))
    generation = await repository.current_generation(target="data:main")
    await repository.record_indexed(_index_write("a", generation=generation))
    await repository.record_indexed(_index_write("b", generation="an-older-generation"))

    stale = await repository.stale_document_ids(target="data:main")

    assert stale == ("document-b",)


async def test_a_rename_tombstones_the_old_path_and_creates_the_new_one(sqlite_engine) -> None:
    """A rename is a deletion and an addition, because identity is the path.

    Nothing tells the repository that a rename happened — it sees a complete
    snapshot in which one external id is gone and another has appeared, which is
    indistinguishable from deleting one file and adding another. That is the
    whole behaviour, and it is worth stating because the alternative is
    tempting: matching on content would carry a moved file's review history and
    its vectors across, at the price of merging two identical files into one
    document.
    """
    repository = SqlAlchemyIngestionRepository(sqlite_engine)
    await repository.commit_snapshot(_snapshot("before-rename", _revision("readme")))

    after = await repository.commit_snapshot(_snapshot("after-rename", _revision("guide")))

    assert after.tombstoned_document_ids == ("document-readme",)
    assert after.created_processing_revision_ids == ("revision-guide-1",)
    async with AsyncSession(sqlite_engine) as session:
        old = await session.get(IngestionDocument, "document-readme")
        new = await session.get(IngestionDocument, "document-guide")
        assert old is not None and old.deleted_at is not None
        assert new is not None and new.deleted_at is None


async def test_a_revision_stores_the_text_it_is_a_revision_of(sqlite_engine) -> None:
    """The corpus holds the content, byte for byte, and holds it immutably.

    This is the property the whole read side rests on: a chunk's offsets index
    `ProcessedDocument.text`, so that exact string has to survive somewhere that
    is not a search projection.
    """
    repository = SqlAlchemyIngestionRepository(sqlite_engine)
    written = _revision("a")

    await repository.commit_snapshot(_snapshot("snapshot-1", written))

    async with AsyncSession(sqlite_engine) as session:
        stored = await session.get(IngestionDocumentRevision, written.processing_revision_id)
        assert stored is not None
        assert stored.content == written.content
        # Provenance is kept and is not the read path.
        assert stored.artifact_ref == written.artifact_ref


async def test_a_document_with_no_usable_text_stores_null_and_replays(sqlite_engine) -> None:
    """A binary file is observed, has no content, and must not read as deleted.

    NULL carries exactly one meaning here, so committing the same snapshot twice
    is a replay rather than a conflict against a row that "changed".
    """
    repository = SqlAlchemyIngestionRepository(sqlite_engine)
    binary = replace(_revision("bin"), content=None)

    await repository.commit_snapshot(_snapshot("snapshot-1", binary))
    replay = await repository.commit_snapshot(_snapshot("snapshot-1", binary))

    assert replay.idempotent_replay is True
    async with AsyncSession(sqlite_engine) as session:
        stored = await session.get(IngestionDocumentRevision, binary.processing_revision_id)
        assert stored is not None
        assert stored.content is None
        # Observed, so not tombstoned.
        document = await session.get(IngestionDocument, binary.document_id)
        assert document is not None and document.deleted_at is None


async def test_differing_content_under_one_revision_id_is_refused(sqlite_engine) -> None:
    """A revision id hashes the bytes it came from, so this cannot happen honestly."""
    repository = SqlAlchemyIngestionRepository(sqlite_engine)
    await repository.commit_snapshot(_snapshot("snapshot-1", _revision("a")))

    forged = replace(_revision("a"), content="something else entirely")
    with pytest.raises(ValueError, match="different immutable content"):
        await repository.commit_snapshot(_snapshot("snapshot-2", forged))


async def test_a_row_written_before_the_column_existed_is_filled_in(sqlite_engine) -> None:
    """Migration 0020 leaves old rows without content; the next run completes them.

    Without this, the first re-ingest after upgrading would fail as "different
    immutable content" — a migration turning itself into a landmine.
    """
    repository = SqlAlchemyIngestionRepository(sqlite_engine)
    written = _revision("a")
    await repository.commit_snapshot(_snapshot("snapshot-1", written))

    async with AsyncSession(sqlite_engine) as session:
        stored = await session.get(IngestionDocumentRevision, written.processing_revision_id)
        assert stored is not None
        stored.content = None
        await session.commit()

    await repository.commit_snapshot(_snapshot("snapshot-2", written))

    async with AsyncSession(sqlite_engine) as session:
        stored = await session.get(IngestionDocumentRevision, written.processing_revision_id)
        assert stored is not None
        assert stored.content == written.content
