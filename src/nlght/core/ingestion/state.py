# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True, frozen=True)
class DocumentRevisionWrite:
    """One document revision, with the text it is a revision *of*.

    ``content`` is the authority: exactly ``ProcessedDocument.text``, the string
    chunking ran over and the string chunk offsets index into. Nothing derives
    it again later — a refetch through ``artifact_ref`` returns whatever the
    source holds now, which for a source that transforms at acquisition is not
    even the same representation, and the search payload is a projection that
    may be rebuilt or dropped at any time. Neither can be addressed by an
    offset, so neither can be the authority.

    ``artifact_ref`` stays what it always was: where this revision came from.
    It is provenance, never a read fallback.

    ``None`` where the document was observed and has no text — a binary file, or
    one that would not decode. Such a document belongs in the snapshot, because
    the tombstone rule is "absent from a complete snapshot" and leaving it out
    would declare it deleted, but there is nothing to store and nothing to
    index. That is the single meaning of ``None`` here, and it must stay single.
    """

    document_id: str
    external_id: str
    processing_revision_id: str
    source_revision_id: str
    path: str
    artifact_ref: str
    content: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    processing: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        required = (
            self.document_id,
            self.external_id,
            self.processing_revision_id,
            self.source_revision_id,
            self.path,
            self.artifact_ref,
        )
        if any(not value.strip() for value in required):
            raise ValueError("document revision identity and artifact_ref must not be empty")


@dataclass(slots=True, frozen=True)
class SnapshotCommit:
    snapshot_key: str
    source_id: str
    source_type: str
    config_revision: str
    complete: bool
    documents: tuple[DocumentRevisionWrite, ...]
    diagnostics: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if any(not value.strip() for value in (self.snapshot_key, self.source_id, self.source_type, self.config_revision)):
            raise ValueError("snapshot and source identity must not be empty")
        ids = [document.document_id for document in self.documents]
        if len(ids) != len(set(ids)):
            raise ValueError("snapshot document ids must be unique")


@dataclass(slots=True, frozen=True)
class SnapshotCommitResult:
    run_id: uuid.UUID
    created_processing_revision_ids: tuple[str, ...]
    tombstoned_document_ids: tuple[str, ...]
    idempotent_replay: bool


@dataclass(slots=True, frozen=True)
class IndexStateWrite:
    """Records that one document revision is present in one index target.

    ``target`` names the index a document was written to (e.g. a store resource
    name plus its collection). ``enricher_hash`` versions the processing that
    produced the payload, so a changed enricher forces a reindex even when the
    source content did not change.

    ``index_generation`` names *which* index it was written to. Content and
    enricher describe the document; without the generation the record would
    claim a document is indexed even after the index it names was dropped and
    recreated — and a run over an unchanged source would then write nothing and
    report success into an empty index. It is also what makes a large reindex
    resumable: a document already carrying the current generation is done, and
    an interrupted run picks up exactly the rest.
    """

    document_id: str
    target: str
    content_hash: str
    enricher_hash: str
    processing_revision_id: str
    index_generation: str
    #: How the document was projected into the indexes — payload schema,
    #: embedding provider, model, dimension, distance metric.
    #:
    #: Separate from the three above because it moves for different reasons.
    #: They describe the document and which incarnation of an index holds it;
    #: none of them notices when the embedding model changes. Without this, a
    #: model swap reindexed nothing: every document read as current, the run
    #: reported success, and the collection kept vectors from a model the
    #: queries were no longer embedded with.
    projection_fingerprint: str = ""

    def __post_init__(self) -> None:
        required = (
            self.document_id,
            self.target,
            self.content_hash,
            self.enricher_hash,
            self.processing_revision_id,
            self.index_generation,
        )
        if any(not value.strip() for value in required):
            raise ValueError("index state identity fields must not be empty")


@dataclass(slots=True, frozen=True)
class IndexState:
    document_id: str
    target: str
    content_hash: str
    enricher_hash: str
    processing_revision_id: str
    index_generation: str
    indexed_at: datetime
    projection_fingerprint: str = ""


@dataclass(slots=True, frozen=True)
class IndexStateSummary:
    """What the database expects of one index target.

    ``indexed`` minus ``on_current_generation`` is the work an interrupted
    reindex still owes. ``live_documents`` minus ``indexed`` is what was never
    written. Neither says anything about what the indexes actually hold — only
    a caller that can ask Qdrant and OpenSearch can compare that, and the gap
    between this expectation and their answer is the divergence worth reporting.
    """

    target: str
    generation: str
    live_documents: int
    indexed: int
    on_current_generation: int

    @property
    def outstanding(self) -> int:
        """Documents recorded against an older generation of this target."""
        return self.indexed - self.on_current_generation
