# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, cast

from sqlalchemy import func, select
from sqlalchemy.engine import CursorResult, Result
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from nlght.adapters.outbound.persistence.ingestion_models import (
    IngestionDocument,
    IngestionDocumentRevision,
    IngestionIndexGeneration,
    IngestionIndexState,
    IngestionSnapshotObservation,
    IngestionSnapshotRun,
    IngestionSource,
)
from nlght.core.ingestion import (
    IndexState,
    IndexStateSummary,
    IndexStateWrite,
    SnapshotCommit,
    SnapshotCommitResult,
)
from nlght.ports.outbound.ingestion_repository import IngestionRepository

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _rowcount(result: Result[Any]) -> int:
    return cast(CursorResult[Any], result).rowcount


def _command_digest(command: SnapshotCommit) -> str:
    payload = {
        "source_id": command.source_id,
        "source_type": command.source_type,
        "config_revision": command.config_revision,
        "complete": command.complete,
        "documents": [
            {
                "document_id": item.document_id,
                "external_id": item.external_id,
                "processing_revision_id": item.processing_revision_id,
                "source_revision_id": item.source_revision_id,
                "path": item.path,
                "artifact_ref": item.artifact_ref,
                # `content` is deliberately absent. A revision id already hashes
                # the source bytes it was derived from, so two snapshots cannot
                # differ in content while agreeing on every revision id, and
                # hashing whole documents here would make the digest of a
                # replay check cost a pass over the entire corpus.
                "metadata": item.metadata,
                "processing": item.processing,
            }
            for item in command.documents
        ],
        "diagnostics": command.diagnostics,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class SqlAlchemyIngestionRepository(IngestionRepository):
    """Short PostgreSQL transactions. No external backend call occurs here."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def commit_snapshot(self, command: SnapshotCommit) -> SnapshotCommitResult:
        try:
            return await self._commit_snapshot(command)
        except IntegrityError as first_error:
            try:
                return await self._commit_snapshot(command)
            except IntegrityError:
                raise first_error from None

    async def _commit_snapshot(self, command: SnapshotCommit) -> SnapshotCommitResult:
        digest = _command_digest(command)
        now = _utcnow()
        async with AsyncSession(self._engine, expire_on_commit=False) as session:
            async with session.begin():
                existing = (
                    await session.execute(select(IngestionSnapshotRun).where(IngestionSnapshotRun.snapshot_key == command.snapshot_key))
                ).scalar_one_or_none()
                if existing is not None:
                    if existing.command_digest != digest:
                        raise ValueError("snapshot_key already belongs to a different snapshot")
                    return SnapshotCommitResult(
                        run_id=existing.run_id,
                        created_processing_revision_ids=tuple(existing.result.get("created_processing_revision_ids", ())),
                        tombstoned_document_ids=tuple(existing.result.get("tombstoned_document_ids", ())),
                        idempotent_replay=True,
                    )

                source = await session.get(IngestionSource, command.source_id)
                if source is None:
                    source = IngestionSource(
                        source_id=command.source_id,
                        source_type=command.source_type,
                        config_revision=command.config_revision,
                        created_at=now,
                        updated_at=now,
                    )
                    session.add(source)
                    await session.flush()
                else:
                    if source.source_type != command.source_type:
                        raise ValueError("source_id already belongs to a different source type")
                    source.config_revision = command.config_revision
                    source.updated_at = now

                run_id = uuid.uuid4()
                created_revisions: list[str] = []
                observed_ids: set[str] = set()
                observations: list[IngestionSnapshotObservation] = []
                for item in command.documents:
                    observed_ids.add(item.document_id)
                    document = await session.get(IngestionDocument, item.document_id)
                    if document is None:
                        document = IngestionDocument(
                            document_id=item.document_id,
                            source_id=command.source_id,
                            external_id=item.external_id,
                            created_at=now,
                            updated_at=now,
                        )
                        session.add(document)
                        await session.flush()
                    elif document.source_id != command.source_id or document.external_id != item.external_id:
                        raise ValueError("document_id already belongs to a different source identity")

                    revision = await session.get(IngestionDocumentRevision, item.processing_revision_id)
                    if revision is None:
                        revision = IngestionDocumentRevision(
                            processing_revision_id=item.processing_revision_id,
                            document_id=item.document_id,
                            source_revision_id=item.source_revision_id,
                            path=item.path,
                            artifact_ref=item.artifact_ref,
                            content=item.content,
                            revision_metadata=dict(item.metadata),
                            processing=dict(item.processing),
                            created_at=now,
                        )
                        session.add(revision)
                        created_revisions.append(item.processing_revision_id)
                    elif (
                        revision.document_id != item.document_id
                        or revision.source_revision_id != item.source_revision_id
                        or revision.path != item.path
                        or revision.artifact_ref != item.artifact_ref
                        or revision.revision_metadata != item.metadata
                        or revision.processing != item.processing
                        # Only where the row already has content. A revision id
                        # hashes the bytes it came from, so a stored text and an
                        # incoming one cannot honestly differ under the same id.
                        or (revision.content is not None and revision.content != item.content)
                    ):
                        raise ValueError("processing_revision_id already belongs to different immutable content")
                    elif revision.content is None and item.content is not None:
                        # A row from before the column existed (migration 0020),
                        # meeting its text for the first time. This does not give
                        # NULL a second meaning: going forward NULL means "observed
                        # and unusable", and such a document arrives with no content
                        # either, so it takes neither this branch nor the one above.
                        revision.content = item.content
                    document.current_processing_revision_id = item.processing_revision_id
                    document.deleted_at = None
                    document.updated_at = now
                    observations.append(
                        IngestionSnapshotObservation(
                            run_id=run_id,
                            document_id=item.document_id,
                            processing_revision_id=item.processing_revision_id,
                        )
                    )

                tombstoned: list[str] = []
                if command.complete:
                    query = select(IngestionDocument).where(
                        IngestionDocument.source_id == command.source_id,
                        IngestionDocument.deleted_at.is_(None),
                    )
                    if observed_ids:
                        query = query.where(IngestionDocument.document_id.not_in(observed_ids))
                    missing = (await session.execute(query)).scalars().all()
                    for document in missing:
                        document.deleted_at = now
                        document.updated_at = now
                        tombstoned.append(document.document_id)

                result = {
                    "created_processing_revision_ids": sorted(created_revisions),
                    "tombstoned_document_ids": sorted(tombstoned),
                }
                session.add(
                    IngestionSnapshotRun(
                        run_id=run_id,
                        snapshot_key=command.snapshot_key,
                        command_digest=digest,
                        source_id=command.source_id,
                        config_revision=command.config_revision,
                        complete=command.complete,
                        document_count=len(command.documents),
                        diagnostics=list(command.diagnostics),
                        result=result,
                        created_at=now,
                    )
                )
                # The run must reach the database before its observations: their
                # run_id is a raw FK column with no relationship() to order the
                # unit-of-work, so a single flush is free to emit the child
                # inserts first and trip the foreign key. Flush the parent first,
                # as the source and document inserts above already do.
                await session.flush()
                session.add_all(observations)
            return SnapshotCommitResult(
                run_id=run_id,
                created_processing_revision_ids=tuple(sorted(created_revisions)),
                tombstoned_document_ids=tuple(sorted(tombstoned)),
                idempotent_replay=False,
            )

    # -- per-document index state -------------------------------------------

    async def current_processing_revisions(
        self, document_ids: Sequence[str]
    ) -> dict[str, str]:
        """Read published revisions for candidate documents in one query.

        Args:
            document_ids (Sequence[str]): Candidate document identities.

        Returns:
            dict[str, str]: Published processing revision by live document ID.

        Side Effects:
            Reads PostgreSQL.

        Notes:
            Tombstoned documents are omitted, making their index hits stale.
        """
        wanted = [identity for identity in dict.fromkeys(document_ids) if identity]
        if not wanted:
            return {}
        async with AsyncSession(self._engine) as session:
            rows = (
                await session.execute(
                    select(
                        IngestionDocument.document_id,
                        IngestionDocument.current_processing_revision_id,
                    ).where(
                        IngestionDocument.document_id.in_(wanted),
                        IngestionDocument.deleted_at.is_(None),
                        IngestionDocument.current_processing_revision_id.is_not(None),
                    )
                )
            ).all()
        return {row[0]: row[1] for row in rows}

    async def read_revision_contents(
        self, keys: Sequence[tuple[str, str]]
    ) -> dict[tuple[str, str], str]:
        """Read authoritative text for exact document-revision pairs.

        Args:
            keys (Sequence[tuple[str, str]]): Document ID and processing revision ID pairs.

        Returns:
            dict[tuple[str, str], str]: Stored text keyed by each exact pair.

        Side Effects:
            Reads PostgreSQL.

        Notes:
            A mismatched pair returns no text from another document.
        """
        wanted = [pair for pair in dict.fromkeys(keys) if pair[0] and pair[1]]
        if not wanted:
            return {}
        revisions = {pair[1] for pair in wanted}
        async with AsyncSession(self._engine) as session:
            rows = (
                await session.execute(
                    select(
                        IngestionDocumentRevision.document_id,
                        IngestionDocumentRevision.processing_revision_id,
                        IngestionDocumentRevision.content,
                    ).where(
                        IngestionDocumentRevision.processing_revision_id.in_(revisions)
                    )
                )
            ).all()
        stored = {(row[0], row[1]): row[2] for row in rows if row[2] is not None}
        return {pair: stored[pair] for pair in wanted if pair in stored}

    @staticmethod
    def _to_index_state(row: IngestionIndexState) -> IndexState:
        return IndexState(
            document_id=row.document_id,
            target=row.target,
            content_hash=row.content_hash,
            enricher_hash=row.enricher_hash,
            processing_revision_id=row.processing_revision_id,
            index_generation=row.index_generation,
            indexed_at=row.indexed_at,
            projection_fingerprint=row.projection_fingerprint,
        )

    async def get_index_state(self, *, document_id: str, target: str) -> IndexState | None:
        async with AsyncSession(self._engine) as session:
            row = await session.get(IngestionIndexState, (document_id, target))
            return self._to_index_state(row) if row is not None else None

    async def record_indexed(self, write: IndexStateWrite) -> IndexState:
        """Record one document as indexed for one target.

        Args:
            write (IndexStateWrite): Completed index-state write.

        Returns:
            IndexState: Persisted index state.

        Side Effects:
            Writes PostgreSQL in a short transaction after the store write.
        """
        async with AsyncSession(self._engine) as session, session.begin():
            row = await session.get(
                IngestionIndexState, (write.document_id, write.target), with_for_update=True
            )
            now = _utcnow()
            if row is None:
                row = IngestionIndexState(
                    document_id=write.document_id,
                    target=write.target,
                    content_hash=write.content_hash,
                    enricher_hash=write.enricher_hash,
                    processing_revision_id=write.processing_revision_id,
                    index_generation=write.index_generation,
                    projection_fingerprint=write.projection_fingerprint,
                    indexed_at=now,
                )
                session.add(row)
            else:
                row.content_hash = write.content_hash
                row.enricher_hash = write.enricher_hash
                row.processing_revision_id = write.processing_revision_id
                row.index_generation = write.index_generation
                row.projection_fingerprint = write.projection_fingerprint
                row.indexed_at = now
            await session.flush()
            return self._to_index_state(row)

    async def clear_index_state(self, *, document_id: str, target: str) -> bool:
        async with AsyncSession(self._engine) as session, session.begin():
            row = await session.get(IngestionIndexState, (document_id, target))
            if row is None:
                return False
            await session.delete(row)
            return True

    # -- index generation ----------------------------------------------------

    async def current_generation(self, *, target: str) -> str:
        """Read or establish the current generation of an index target.

        Args:
            target (str): Index target identity.

        Returns:
            str: Generation shared by workers for this target.

        Side Effects:
            Reads PostgreSQL and may create an initial generation.
        """
        async with AsyncSession(self._engine) as session, session.begin():
            row = await session.get(IngestionIndexGeneration, target)
            if row is None:
                row = IngestionIndexGeneration(
                    target=target, generation=str(uuid.uuid4()), rotated_at=_utcnow()
                )
                session.add(row)
                await session.flush()
            return row.generation

    async def rotate_generation(self, *, target: str) -> str:
        """Declare that an index target has a new generation.

        Args:
            target (str): Index target whose collection or index was recreated.

        Returns:
            str: Generation in force after rotation, including any concurrent rotation.

        Side Effects:
            Updates target generation in PostgreSQL.
        """
        async with AsyncSession(self._engine) as session, session.begin():
            row = await session.get(IngestionIndexGeneration, target, with_for_update=True)
            generation = str(uuid.uuid4())
            if row is None:
                session.add(
                    IngestionIndexGeneration(
                        target=target, generation=generation, rotated_at=_utcnow()
                    )
                )
            else:
                row.generation = generation
                row.rotated_at = _utcnow()
            await session.flush()
        logger.info("ingestion.index.generation.rotated | target=%s", target)
        return generation

    async def index_state_summary(self, *, target: str) -> IndexStateSummary:
        """Summarize recorded index state for reconciliation.

        Args:
            target (str): Index target to inspect.

        Returns:
            IndexStateSummary: Live, indexed, and current-generation document counts.

        Side Effects:
            Reads PostgreSQL.

        Notes:
            The caller compares these counts with the actual index backends.
        """
        generation = await self.current_generation(target=target)
        async with AsyncSession(self._engine) as session:
            indexed = (
                await session.execute(
                    select(func.count())
                    .select_from(IngestionIndexState)
                    .where(IngestionIndexState.target == target)
                )
            ).scalar_one()
            on_generation = (
                await session.execute(
                    select(func.count())
                    .select_from(IngestionIndexState)
                    .where(
                        IngestionIndexState.target == target,
                        IngestionIndexState.index_generation == generation,
                    )
                )
            ).scalar_one()
            live = (
                await session.execute(
                    select(func.count())
                    .select_from(IngestionDocument)
                    .where(IngestionDocument.deleted_at.is_(None))
                )
            ).scalar_one()
        return IndexStateSummary(
            target=target,
            generation=generation,
            live_documents=int(live),
            indexed=int(indexed),
            on_current_generation=int(on_generation),
        )

    async def stale_document_ids(self, *, target: str, limit: int = 100) -> tuple[str, ...]:
        """Find work left by an interrupted generation change.

        Args:
            target (str): Index target to inspect.
            limit (int): Maximum number of document IDs to return.

        Returns:
            tuple[str, ...]: Document IDs recorded under an older generation.

        Side Effects:
            Reads PostgreSQL.
        """
        generation = await self.current_generation(target=target)
        async with AsyncSession(self._engine) as session:
            rows = (
                await session.execute(
                    select(IngestionIndexState.document_id)
                    .where(
                        IngestionIndexState.target == target,
                        IngestionIndexState.index_generation != generation,
                    )
                    .limit(limit)
                )
            ).scalars()
            return tuple(rows)
