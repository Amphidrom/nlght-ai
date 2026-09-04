# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Deterministic document processing.

Side-effect free: classification, enrichment, and chunking of one document, plus
the stable processing revision derived from them. It lives in ``core`` rather
than ``application`` because it orchestrates nothing — it is the domain rule
that decides what a processed document *is*, and both the application layer and
workflow steps depend on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from nlght.core.ingestion.document import (
    DocumentChunk,
    Enrichment,
    ProcessedDocument,
    SourceDocument,
    stable_digest,
)

if TYPE_CHECKING:
    from nlght.ports.outbound.document_processing import DocumentClassifier, DocumentEnricher


@dataclass(slots=True, frozen=True)
class ChunkingSettings:
    size: int = 4000
    overlap: int = 200

    def __post_init__(self) -> None:
        if self.size < 1 or self.overlap < 0 or self.overlap >= self.size:
            raise ValueError("chunk size must be positive and overlap smaller than size")


class IngestionDocumentProcessor:
    def __init__(
        self,
        *,
        classifier: DocumentClassifier,
        enricher: DocumentEnricher,
        chunking: ChunkingSettings | None = None,
    ) -> None:
        self._classifier = classifier
        self._enricher = enricher
        self._chunking = chunking or ChunkingSettings()

    def process(self, document: SourceDocument) -> ProcessedDocument:
        path = Path(document.path)
        classification, text, diagnostics = self._classifier.classify(
            path=path,
            content=document.content,
        )
        enrichment = (
            self._enricher.enrich(path=path, text=text, classification=classification) if text is not None else Enrichment(enricher="none", version="1")
        )
        processing_revision_id = stable_digest(
            document.source_revision_id,
            document.path,
            classification.classifier_version,
            enrichment.enricher,
            enrichment.version,
            str(self._chunking.size),
            str(self._chunking.overlap),
        )
        chunks = self._chunks(processing_revision_id, text) if text is not None else ()
        return ProcessedDocument(
            document_id=document.document_id,
            source_revision_id=document.source_revision_id,
            processing_revision_id=processing_revision_id,
            source=document.source,
            external_id=document.external_id,
            path=document.path,
            text=text,
            classification=classification,
            enrichment=enrichment,
            chunks=chunks,
            metadata=dict(document.metadata),
            diagnostics=diagnostics,
        )

    def _chunks(self, processing_revision_id: str, text: str) -> tuple[DocumentChunk, ...]:
        if not text:
            return ()
        chunks: list[DocumentChunk] = []
        start = 0
        position = 0
        while start < len(text):
            hard_end = min(len(text), start + self._chunking.size)
            end = hard_end
            if hard_end < len(text):
                boundary = text.rfind("\n", start, hard_end)
                if boundary > start:
                    end = boundary + 1
            content = text[start:end]
            chunks.append(
                DocumentChunk(
                    chunk_id=stable_digest(processing_revision_id, str(position), content),
                    position=position,
                    content=content,
                    start_offset=start,
                    end_offset=end,
                )
            )
            if end >= len(text):
                break
            start = max(start + 1, end - self._chunking.overlap)
            position += 1
        return tuple(chunks)
