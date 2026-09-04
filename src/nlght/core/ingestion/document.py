# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


def stable_digest(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(slots=True, frozen=True)
class SourceDocument:
    source: str
    external_id: str
    path: str
    #: What the pipeline consumes: classification, chunking and the search index
    #: are all derived from this.
    content: bytes
    metadata: dict[str, Any] = field(default_factory=dict)
    source_revision: str | None = None
    #: The document as the source actually holds it, when acquisition had to
    #: transform it to produce `content`. Kept *beside* rather than instead,
    #: because the three consumers want different things and only one of them
    #: wants text:
    #:
    #:     source form   the markup — what a knowledge parser reads
    #:     search text   markup removed — what the index holds
    #:     model input   derived from the source form
    #:
    #: Conflating the first two cost a whole source type. Confluence stripped its
    #: HTML at acquisition and stored the result under a path still ending
    #: `.html`, so the knowledge parser was handed tagless text and looked for
    #: elements that were no longer there. Every page produced zero units, with
    #: no error and an empty funnel that read like a page with nothing to say.
    #:
    #: `None` where the two coincide, which is every source that hands over what
    #: it holds.
    source_form: bytes | None = None

    @property
    def markup(self) -> bytes:
        """The form a parser should read: the source's own, or the content."""
        return self.source_form if self.source_form is not None else self.content

    def __post_init__(self) -> None:
        if not self.source.strip() or not self.external_id.strip() or not self.path.strip():
            raise ValueError("source, external_id, and path must not be empty")

    @property
    def document_id(self) -> str:
        return stable_digest(self.source, self.external_id)

    @property
    def source_revision_id(self) -> str:
        return stable_digest(
            self.document_id,
            self.source_revision or "",
            hashlib.sha256(self.content).hexdigest(),
        )


@dataclass(slots=True, frozen=True)
class DocumentClassification:
    kind: str
    language: str
    classifier_version: str


@dataclass(slots=True, frozen=True)
class Enrichment:
    enricher: str
    version: str
    declaration: str | None = None
    tags: tuple[str, ...] = ()
    symbols: tuple[str, ...] = ()
    references: tuple[str, ...] = ()
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class DocumentChunk:
    chunk_id: str
    position: int
    content: str
    start_offset: int
    end_offset: int


@dataclass(slots=True, frozen=True)
class ChunkProjection:
    """One chunk as an index will hold it: its vector, its text, and where it is.

    The locator travels with the vector rather than beside it. It used to be a
    ``(chunk_id, vector, content)`` tuple with the semantics passed as a second
    list zipped back on by position — so a chunk's terms were correct only while
    two lists stayed the same length in the same order, and ``position``,
    ``start_offset`` and ``end_offset`` were simply dropped on the way.

    Those three are what makes a hit resolvable at all. Without them a chunk
    found in the index names no span of the document it came from, and the only
    text anyone could show was the payload's own copy — which is a projection,
    not the source of truth.
    """

    chunk_id: str
    position: int
    start_offset: int
    end_offset: int
    content: str
    vector: tuple[float, ...]
    semantics: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class ProcessedDocument:
    document_id: str
    source_revision_id: str
    processing_revision_id: str
    source: str
    external_id: str
    path: str
    text: str | None
    classification: DocumentClassification
    enrichment: Enrichment
    chunks: tuple[DocumentChunk, ...]
    metadata: dict[str, Any]
    diagnostics: tuple[str, ...] = ()

    @property
    def accepted(self) -> bool:
        return self.text is not None and self.classification.kind != "binary"
