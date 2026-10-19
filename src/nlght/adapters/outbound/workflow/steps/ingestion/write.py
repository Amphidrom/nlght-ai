# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import logging
from typing import Any

from nlght.adapters.outbound.stores.connections import StoreConnections
from nlght.adapters.outbound.stores.data_writer import DataIndexWriterTool
from nlght.adapters.outbound.tools.activator import ResourceActivator
from nlght.adapters.outbound.tools.loader import ToolLoader
from nlght.core.entry.context import RequestContext
from nlght.core.errors.errors import WorkflowConfigurationError, WorkflowExecutionError
from nlght.core.ingestion import (
    ChunkProjection,
    DocumentRevisionWrite,
    IndexStateWrite,
    ProcessedDocument,
    SnapshotCommit,
    SourceSnapshot,
    stable_digest,
)
from nlght.core.workflow.step import (
    StepBase,
    StepOption,
    StepResult,
    WorkflowStepContext,
)
from nlght.ports.outbound.ingestion_repository import IngestionRepository
from nlght.ports.outbound.resource_repository import ResourceRepository

logger = logging.getLogger(__name__)


def _revision_of(document: ProcessedDocument) -> DocumentRevisionWrite:
    """Build a durable revision record from a processed document.

    Args:
        document (ProcessedDocument): Document and processing metadata to record.

    Returns:
        DocumentRevisionWrite: Revision with unchanged document text, source
        provenance, and processing metadata. Text may be ``None`` for a
        document that remains observed but has no usable content.
    """
    return DocumentRevisionWrite(
        document_id=document.document_id,
        external_id=document.external_id,
        processing_revision_id=document.processing_revision_id,
        source_revision_id=document.source_revision_id,
        path=document.path,
        artifact_ref=f"{document.source}:{document.external_id}@{document.source_revision_id}",
        content=document.text,
        metadata=dict(document.metadata),
        processing={
            "classifier_version": document.classification.classifier_version,
            "kind": document.classification.kind,
            "language": document.classification.language,
            "enricher": document.enrichment.enricher,
            "enricher_version": document.enrichment.version,
            "accepted": document.accepted,
        },
    )

NOTHING_TO_DO = "unchanged"
"""Verdict when every document was already indexed at this revision."""

MORE_DOCUMENTS = "more"
"""Verdict when processed documents remain unwritten. Route it back to this
step: one invocation writes a bounded batch, not every document of the run."""


class IngestionWriteStep(StepBase):
    """Writes processed documents into a configured index target.

    Step config:
        writer: resource name of the `data_index_writer` activation to use
        document_batch_size: documents written by one invocation (default 1)

    Indexing is idempotent per document: a document whose content and
    processing revision already match the recorded index state is skipped, so a
    rerun over an unchanged source costs nothing and retrieval is never
    disturbed. There is no generation switch. Replacement happens one document
    at a time.

    Reads ``ingestion.processed`` and, when present, ``ingestion.embedded``.
    """

    TYPE = "ingestion.write"

    class Verdict(StepBase.Verdict):
        """Verdicts returned by this step."""

        NOTHING_TO_DO = NOTHING_TO_DO
        MORE_DOCUMENTS = MORE_DOCUMENTS

    def __init__(
        self,
        *,
        config: dict[str, Any],
        resource_repository: ResourceRepository | None = None,
        resource_activator: ResourceActivator | None = None,
        tool_loader: ToolLoader | None = None,
        store_connections: StoreConnections | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(config=config, **kwargs)
        self._resources = resource_repository
        self._activator = resource_activator
        self._tools = tool_loader
        # This step re-enters itself once per document batch and activates its
        # writer each time. Without the runtime's shared pools every batch would
        # leave an abandoned engine and client behind.
        self._connections = store_connections

    @classmethod
    def options(cls) -> list[StepOption]:
        return [
            StepOption("writer", "string",
                       "Resource name of the data_index_writer activation to write through",
                       required=True, placeholder="primary-index"),
            StepOption("document_batch_size", "integer",
                       "Documents written by one step invocation", default=1),
        ]

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        documents = ctx.metadata.get("ingestion.processed")
        if not isinstance(documents, tuple):
            raise WorkflowConfigurationError(
                f"Step '{self.TYPE}' requires processed documents; "
                f"place an 'ingestion.process' step before it."
            )
        if self._resources is None or self._tools is None:
            raise WorkflowConfigurationError(
                f"Step '{self.TYPE}' requires a resource repository and tool loader "
                f"to activate its writer."
            )

        # The writer resource carries the index state too: the record of what is
        # indexed belongs to the activation that owns the indexes.
        writer = await self._writer(ctx.trigger.context)
        repository = writer.repository
        chunks_by_document = self._chunks_by_document(ctx)
        # Whether this pipeline embeds at all. A lexical-only pipeline is a
        # legitimate shape. A pipeline that embeds and then produces no chunk
        # for a document with text is not.
        embedded = ctx.metadata.get("ingestion.embedded") is not None
        # Collections and mappings must exist before the first write. A missing
        # Qdrant collection fails and an unmapped OpenSearch index produces
        # documents nothing can find.
        embedding = self._embedding_descriptor(ctx)
        dimension = embedding[2]
        # How this run projects a document, as against what the document is. A
        # changed embedding model moves this and nothing else, which is exactly
        # the case that used to reindex nothing.
        fingerprint = writer.projection_fingerprint(
            provider=embedding[0], model=embedding[1], dimension=dimension
        )
        # Creating a collection or index means the indexes this writer points at
        # are not the ones the recorded state describes: emptied, dropped, or a
        # fresh environment on an old database. Rotating the generation is what
        # turns that into "everything must be written again" instead of a run
        # that skips every document and reports success into an empty index.
        if writer.ensure_ready(dimension=dimension):
            generation = await repository.rotate_generation(target=writer.target)
            logger.info(
                "[%s] ingestion.write.generation_rotated | target=%s: index was created, "
                "every document will be written again",
                ctx.correlation_id, writer.target,
            )
        else:
            generation = await repository.current_generation(target=writer.target)

        try:
            document_batch_size = int(self.config.get("document_batch_size", 1))
        except ValueError as exc:
            raise WorkflowConfigurationError(
                f"Step '{self.TYPE}' has invalid document_batch_size: {exc}"
            ) from exc
        if document_batch_size < 1:
            raise WorkflowConfigurationError(
                f"Step '{self.TYPE}' requires document_batch_size to be at least 1."
            )

        # The corpus record first, in one short transaction with no external
        # call in it. It is what makes deletion possible: a document absent from
        # a *complete* snapshot is tombstoned here, and only then removed from
        # the index below. It is run once per workflow pass. Document writes
        # continue in bounded batches after this point.
        if not bool(ctx.metadata.get("ingestion.write.snapshot_committed", False)):
            removed = await self._commit_snapshot(ctx, repository, writer, documents)
            ctx.metadata["ingestion.write.snapshot_committed"] = True
            ctx.metadata["ingestion.removed"] = removed
        else:
            removed = int(ctx.metadata.get("ingestion.removed", 0))

        next_document = int(ctx.metadata.get("ingestion.write.next_document", 0))
        if next_document >= len(documents):
            written_total = int(ctx.metadata.get("ingestion.written", 0))
            return StepResult(ctx=ctx, verdict=self.Verdict.DEFAULT if written_total or removed else self.Verdict.NOTHING_TO_DO)

        batch_end = min(next_document + document_batch_size, len(documents))
        written = 0
        skipped = 0
        for document in documents[next_document:batch_end]:
            # This observed document is unusable: a binary file, or one that would not
            # decode. It belongs in the snapshot, so absence never mistakes it
            # for a deletion, but there is nothing to index.
            if not document.accepted:
                skipped += 1
                continue
            if await self._is_current(
                repository, document, writer.target, generation, fingerprint
            ):
                skipped += 1
                continue
            chunks = chunks_by_document.get(document.document_id, [])
            # A pipeline that embeds and a document that has text must produce
            # chunks. None means the document would be written lexically but not
            # vectorially, and then recorded as fully indexed. That divergence
            # nothing would ever repair. A pipeline with no embed step at all,
            # and a document whose text is empty, are both legitimate.
            if embedded and document.text and not chunks:
                raise WorkflowExecutionError(
                    f"Step '{self.TYPE}' will not record '{document.document_id}' as indexed: "
                    f"this run embedded, and the document has text, but it produced no chunks. "
                    f"Writing it would leave it findable lexically and absent from the vector "
                    f"index, with nothing to correct it later."
                )
            await writer.write_document(
                document_id=document.document_id,
                path=document.path,
                text=document.text or "",
                source_revision_id=document.source_revision_id,
                processing_revision_id=document.processing_revision_id,
                metadata={
                    "source": document.source,
                    "external_id": document.external_id,
                    **document.metadata,
                },
                chunks=chunks,
            )
            # Recorded only after the external write returned, so a crash in
            # between re-runs the document instead of marking it done.
            await repository.record_indexed(
                IndexStateWrite(
                    document_id=document.document_id,
                    target=writer.target,
                    content_hash=document.source_revision_id,
                    enricher_hash=document.processing_revision_id,
                    processing_revision_id=document.processing_revision_id,
                    index_generation=generation,
                    projection_fingerprint=fingerprint,
                )
            )
            written += 1

        written_total = int(ctx.metadata.get("ingestion.written", 0)) + written
        ctx.metadata["ingestion.written"] = written_total
        ctx.metadata["ingestion.removed"] = removed
        ctx.metadata["ingestion.write.next_document"] = batch_end
        logger.info(
            "[%s] ingestion.write.batch | target=%s documents=%d/%d written=%d total=%d unchanged=%d removed=%d",
            ctx.correlation_id, writer.target, batch_end, len(documents), written, written_total, skipped, removed,
        )
        if batch_end < len(documents):
            return StepResult(ctx=ctx, verdict=self.Verdict.MORE_DOCUMENTS)
        return StepResult(ctx=ctx, verdict=self.Verdict.DEFAULT if written_total or removed else self.Verdict.NOTHING_TO_DO)

    async def _commit_snapshot(
        self,
        ctx: WorkflowStepContext,
        repository: IngestionRepository,
        writer: DataIndexWriterTool,
        documents: tuple[ProcessedDocument, ...],
    ) -> int:
        """Commit a source snapshot and remove tombstoned index entries.

        Args:
            ctx (WorkflowStepContext): Context holding the acquired source snapshot.
            repository (IngestionRepository): Authority for snapshot and index state.
            writer (DataIndexWriterTool): Index writer used to remove tombstones.
            documents (tuple[ProcessedDocument, ...]): Observed documents,
                including documents excluded from indexing.

        Returns:
            int: Number of tombstoned documents removed from the index.

        Raises:
            WorkflowConfigurationError: No acquired snapshot is available.

        Side Effects:
            Commits the observation before removing tombstones. Clears index
            state only after each backend deletion succeeds, so a retry repeats
            an interrupted removal.
        """
        snapshot = ctx.metadata.get("ingestion.snapshot")
        if not isinstance(snapshot, SourceSnapshot):
            raise WorkflowConfigurationError(
                f"Step '{self.TYPE}' requires the acquired snapshot to record what a "
                f"source contained; place an 'ingestion.source' step before it."
            )

        result = await repository.commit_snapshot(
            SnapshotCommit(
                # Stable per source and run, so a retried attempt replays the
                # same commit instead of recording a second snapshot.
                snapshot_key=stable_digest(snapshot.source_id, str(ctx.correlation_id)),
                source_id=snapshot.source_id,
                source_type=str(ctx.metadata.get("ingestion.source_type", "unknown")),
                config_revision=str(ctx.metadata.get("ingestion.config_revision", "unknown")),
                complete=snapshot.complete,
                documents=tuple(_revision_of(document) for document in documents),
                diagnostics=tuple(
                    {"code": item.code, "location": item.location, "detail": item.detail}
                    for item in snapshot.diagnostics
                ),
            )
        )

        removed = 0
        for document_id in result.tombstoned_document_ids:
            await writer.delete_document(document_id=document_id)
            await repository.clear_index_state(document_id=document_id, target=writer.target)
            removed += 1
        if removed:
            logger.info(
                "[%s] ingestion.write.removed | target=%s documents=%d",
                ctx.correlation_id, writer.target, removed,
            )
        return removed

    async def _is_current(
        self,
        repository: IngestionRepository,
        document: ProcessedDocument,
        target: str,
        generation: str,
        fingerprint: str,
    ) -> bool:
        """Check whether a document matches the current index projection.

        Args:
            repository (IngestionRepository): Repository holding index state.
            document (ProcessedDocument): Document to compare with stored state.
            target (str): Index target name.
            generation (str): Current target incarnation.
            fingerprint (str): Current index projection fingerprint.

        Returns:
            bool: Whether generation, source and processing revisions, and
            projection fingerprint all match the stored index state.
        """
        state = await repository.get_index_state(
            document_id=document.document_id, target=target
        )
        return (
            state is not None
            and state.index_generation == generation
            and state.content_hash == document.source_revision_id
            and state.enricher_hash == document.processing_revision_id
            and state.projection_fingerprint == fingerprint
        )

    @staticmethod
    def _chunks_by_document(ctx: WorkflowStepContext) -> dict[str, list[ChunkProjection]]:
        """Group embedded chunk projections by document ID.

        Args:
            ctx (WorkflowStepContext): Context containing embedded chunks.

        Returns:
            dict[str, list[ChunkProjection]]: Locator, vector, and semantics
            for each chunk, grouped by its document ID.
        """
        grouped: dict[str, list[ChunkProjection]] = {}
        for chunk in ctx.metadata.get("ingestion.embedded") or ():
            grouped.setdefault(chunk.document_id, []).append(
                ChunkProjection(
                    chunk_id=chunk.chunk_id,
                    position=chunk.position,
                    start_offset=chunk.start_offset,
                    end_offset=chunk.end_offset,
                    content=chunk.content,
                    vector=chunk.vector,
                    semantics=chunk.semantics,
                )
            )
        return grouped

    @staticmethod
    def _embedding_descriptor(ctx: WorkflowStepContext) -> tuple[str, str, int]:
        """Describe the embedding used by this run.

        Args:
            ctx (WorkflowStepContext): Context containing embedded chunks.

        Returns:
            tuple[str, str, int]: Provider, model, and dimension from the first
            chunk, or empty names and zero for a lexical-only run.
        """
        for chunk in ctx.metadata.get("ingestion.embedded") or ():
            return (chunk.provider, chunk.model, chunk.dimension)
        return ("", "", 0)

    async def _writer(self, caller: RequestContext | None) -> DataIndexWriterTool:
        assert self._resources is not None and self._tools is not None
        name = str(self.config.get("writer", "")).strip()
        if not name:
            raise WorkflowConfigurationError(
                f"Step '{self.TYPE}' requires a 'writer' config key naming an "
                f"enabled '{DataIndexWriterTool.KIND}' resource."
            )

        if self._activator is None:
            raise WorkflowConfigurationError(
                f"Step '{self.TYPE}' has no resource activator, so it cannot "
                f"reach '{name}'."
            )
        instance = await self._activator.activate(
            kind=DataIndexWriterTool.KIND,
            name=name,
            caller=caller,
        )
        if not isinstance(instance, DataIndexWriterTool):
            raise WorkflowConfigurationError(
                f"Resource '{name}' resolves to {type(instance).__name__}, "
                f"which is not a {DataIndexWriterTool.__name__}."
            )
        return instance
