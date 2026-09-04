# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import logging

from nlght.adapters.outbound.ingestion.classifier import PygmentsDocumentClassifier
from nlght.adapters.outbound.ingestion.enrichers import BuiltinDocumentEnricher
from nlght.core.errors.errors import WorkflowConfigurationError
from nlght.core.ingestion import ProcessedDocument, SourceSnapshot
from nlght.core.ingestion.processing import ChunkingSettings, IngestionDocumentProcessor
from nlght.core.workflow.step import (
    StepBase,
    StepOption,
    StepResult,
    WorkflowStepContext,
)

logger = logging.getLogger(__name__)

EMPTY = "empty"
"""Verdict when a snapshot yielded nothing to process — a normal outcome for an
unchanged source, and one a workflow usually routes straight to done."""

MORE_DOCUMENTS = "more"
"""Verdict when documents of this snapshot remain unprocessed. Route it back to
this step: one invocation processes a bounded batch, not the whole snapshot."""


class IngestionProcessStep(StepBase):
    """Classifies, enriches, and chunks every document in the snapshot.

    Side-effect free and deterministic: the same document with the same config
    always produces the same processing revision, which is what makes the
    reindex decision stable.

    Step config:
        chunk_size:    characters per chunk (default 4000)
        chunk_overlap: overlap between chunks (default 200)
        document_batch_size: documents processed by one invocation (default 1)
        enrich:        set false to skip enrichment (default true)

    Reads ``ctx.metadata['ingestion.snapshot']`` and emits
    ``ctx.metadata['ingestion.processed']``.
    """

    TYPE = "ingestion.process"

    @classmethod
    def options(cls) -> list[StepOption]:
        return [
            StepOption("chunk_size", "integer", "Characters per chunk", default=4000),
            StepOption("chunk_overlap", "integer",
                       "Overlap between chunks; must be smaller than the size", default=200),
            StepOption("document_batch_size", "integer",
                       "Documents processed by one step invocation", default=1),
        ]

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        snapshot = ctx.metadata.get("ingestion.snapshot")
        if not isinstance(snapshot, SourceSnapshot):
            raise WorkflowConfigurationError(
                f"Step '{self.TYPE}' requires an acquired snapshot; "
                f"place an 'ingestion.source' step before it."
            )

        try:
            chunking = ChunkingSettings(
                size=int(self.config.get("chunk_size", 4000)),
                overlap=int(self.config.get("chunk_overlap", 200)),
            )
        except ValueError as exc:
            raise WorkflowConfigurationError(f"Step '{self.TYPE}' has invalid chunking: {exc}") from exc

        processor = IngestionDocumentProcessor(
            classifier=PygmentsDocumentClassifier(),
            enricher=BuiltinDocumentEnricher(),
            chunking=chunking,
        )

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

        next_document = int(ctx.metadata.get("ingestion.process.next_document", 0))
        if next_document >= len(snapshot.documents):
            existing = ctx.metadata.get("ingestion.processed")
            processed_count = len(existing) if isinstance(existing, tuple) else 0
            # Publish the (possibly empty) result even when there was nothing to
            # do. A later step that finds the key missing is entitled to say the
            # pipeline is misconfigured — "place an ingestion.process step before
            # it" — and an empty snapshot must not be made to look like that.
            if not isinstance(existing, tuple):
                ctx.metadata["ingestion.processed"] = ()
            return StepResult(ctx=ctx, verdict="DEFAULT" if processed_count else EMPTY)

        existing = ctx.metadata.get("ingestion.processed")
        processed: list[ProcessedDocument] = list(existing) if isinstance(existing, tuple) else []
        batch_end = min(next_document + document_batch_size, len(snapshot.documents))
        batch = [processor.process(document) for document in snapshot.documents[next_document:batch_end]]
        processed.extend(batch)
        ctx.metadata["ingestion.processed"] = tuple(processed)
        ctx.metadata["ingestion.process.next_document"] = batch_end

        chunk_count = sum(len(document.chunks) for document in batch)
        logger.info(
            "[%s] ingestion.process.batch | documents=%d/%d chunks=%d total=%d",
            ctx.correlation_id, batch_end, len(snapshot.documents), chunk_count, len(processed),
        )
        if batch_end < len(snapshot.documents):
            return StepResult(ctx=ctx, verdict=MORE_DOCUMENTS)
        return StepResult(ctx=ctx, verdict="DEFAULT" if processed else EMPTY)
