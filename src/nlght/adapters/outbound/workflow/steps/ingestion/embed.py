# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from nlght.core.errors.errors import WorkflowConfigurationError
from nlght.core.ingestion import (
    ProcessedDocument,
    derive_chunk_semantics,
)
from nlght.core.workflow.step import (
    StepBase,
    StepOption,
    StepResult,
    WorkflowStepContext,
)
from nlght.ports.outbound.embedding_client import EmbeddingClient

logger = logging.getLogger(__name__)

SKIPPED = "skipped"
"""Verdict when nothing needed embedding — no chunks, or the step is disabled.

Distinct from a failure so a lexical-only deployment can route straight on
instead of treating a missing embedding provider as an error.
"""

MORE_DOCUMENTS = "more"
"""Verdict when processed documents remain unembedded. Route it back to this
step: one invocation embeds a bounded batch, not every document of the run."""


@dataclass(slots=True, frozen=True)
class EmbeddedChunk:
    """One chunk with its vector and the provenance to persist alongside it.

    ``semantics`` holds the terms that distinguish this chunk from its siblings;
    they are written to the index so a keyword search can find the chunk even
    when the semantic neighbourhood is crowded.
    """

    document_id: str
    source_revision_id: str
    processing_revision_id: str
    chunk_id: str
    position: int
    #: Where this chunk sits in the processed text. Carried rather than dropped
    #: because it is the only thing that lets a hit name a span of the document
    #: it came from — without it the payload's own copy is the only text there is.
    start_offset: int
    end_offset: int
    content: str
    vector: tuple[float, ...]
    provider: str
    model: str
    dimension: int
    semantics: tuple[str, ...] = ()


class IngestionEmbedStep(StepBase):
    """Turns processed chunks into vectors.

    Step config:
        required:  fail instead of skipping when no embedding client is
                   configured (default false)
        semantics_top_k: distinguishing terms kept per chunk (default 12)
        document_batch_size: documents embedded by one invocation (default 1)

    Reads ``ctx.metadata['ingestion.processed']`` and emits
    ``ctx.metadata['ingestion.embedded']``.

    The embedding client is a runtime dependency of the loader, not step config:
    a local model is loaded into memory once at startup and must not be rebuilt
    per step run.
    """

    TYPE = "ingestion.embed"

    def __init__(
        self,
        *,
        config: dict[str, Any],
        embedding_client: EmbeddingClient | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(config=config, **kwargs)
        self._embedding = embedding_client

    @classmethod
    def options(cls) -> list[StepOption]:
        return [
            StepOption("required", "boolean",
                       "Fail instead of skipping when no embedding provider is configured",
                       default=False),
            StepOption("semantics_top_k", "integer",
                       "Distinguishing terms kept per chunk", default=12),
            StepOption("document_batch_size", "integer",
                       "Documents embedded by one step invocation", default=1),
        ]

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        documents = ctx.metadata.get("ingestion.processed")
        if not isinstance(documents, tuple):
            raise WorkflowConfigurationError(
                f"Step '{self.TYPE}' requires processed documents; "
                f"place an 'ingestion.process' step before it."
            )

        required = bool(self.config.get("required", False))
        if self._embedding is None:
            if required:
                raise WorkflowConfigurationError(
                    f"Step '{self.TYPE}' is configured as required, but no embedding "
                    f"provider is configured. Set 'embedding.provider' in platform.yaml."
                )
            logger.info(
                "[%s] ingestion.embed.skipped | no embedding provider configured",
                ctx.correlation_id,
            )
            ctx.metadata["ingestion.embedded"] = ()
            return StepResult(ctx=ctx, verdict=SKIPPED)

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

        next_document = int(ctx.metadata.get("ingestion.embed.next_document", 0))
        if next_document >= len(documents):
            existing = ctx.metadata.get("ingestion.embedded")
            embedded_count = len(existing) if isinstance(existing, tuple) else 0
            return StepResult(ctx=ctx, verdict="DEFAULT" if embedded_count else SKIPPED)

        batch_end = min(next_document + document_batch_size, len(documents))
        batch_documents = documents[next_document:batch_end]
        pending: list[tuple[ProcessedDocument, Any]] = [
            (document, chunk) for document in batch_documents for chunk in document.chunks
        ]
        existing_embedded = ctx.metadata.get("ingestion.embedded")
        embedded_so_far: list[EmbeddedChunk] = (
            list(existing_embedded) if isinstance(existing_embedded, tuple) else []
        )

        if pending:
            result = await self._embedding.embed([chunk.content for _, chunk in pending])

            top_k = int(self.config.get("semantics_top_k", 12))
            semantics_by_document = {
                document.document_id: derive_chunk_semantics(
                    [chunk.content for chunk in document.chunks], top_k=top_k
                )
                for document in batch_documents
            }

            embedded_batch = tuple(
                EmbeddedChunk(
                    document_id=document.document_id,
                    source_revision_id=document.source_revision_id,
                    processing_revision_id=document.processing_revision_id,
                    chunk_id=chunk.chunk_id,
                    position=chunk.position,
                    start_offset=chunk.start_offset,
                    end_offset=chunk.end_offset,
                    content=chunk.content,
                    vector=vector,
                    provider=result.provider,
                    model=result.model,
                    dimension=result.dimension,
                    semantics=tuple(
                        semantics_by_document.get(document.document_id, [])[chunk.position]
                        if chunk.position < len(semantics_by_document.get(document.document_id, []))
                        else []
                    ),
                )
                for (document, chunk), vector in zip(pending, result.vectors, strict=True)
            )
            embedded_so_far.extend(embedded_batch)
            model = result.model
            dimension = result.dimension
        else:
            model = ""
            dimension = 0

        ctx.metadata["ingestion.embedded"] = tuple(embedded_so_far)
        ctx.metadata["ingestion.embed.next_document"] = batch_end

        logger.info(
            "[%s] ingestion.embed.batch | documents=%d/%d chunks=%d total=%d "
            "model=%s dimension=%d",
            ctx.correlation_id,
            batch_end,
            len(documents),
            len(pending),
            len(embedded_so_far),
            model,
            dimension,
        )
        if batch_end < len(documents):
            return StepResult(ctx=ctx, verdict=MORE_DOCUMENTS)
        return StepResult(ctx=ctx, verdict="DEFAULT" if embedded_so_far else SKIPPED)
