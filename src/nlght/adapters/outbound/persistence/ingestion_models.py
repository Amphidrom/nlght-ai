# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Data ingestion authority — the record of a corpus.

On the knowledge database's declarative base, not the platform's. What a
source contained, which revision of each document was seen, and what has been
written into which retrieval target describes the corpus, not the runtime: it
outlives any particular nlght deployment, it is what the knowledge graph is
derived from, and both ingestion verticals read it. The runtime database stays
about workflows, resources, policies, and executions (ADR-0033, superseding the
placement in ADR-0032).

Consequence, and the reason it is stated here: a deployment with no
``integrations.persistence.knowledge.url`` has no ingestion state either, so it
can run workflows but cannot ingest.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    TIMESTAMP,
    Boolean,
    ForeignKey,
    Index,
    Integer,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from nlght.adapters.outbound.persistence.knowledge_models import KnowledgeBase

_JSON = JSON().with_variant(JSONB(), "postgresql")


class IngestionSource(KnowledgeBase):
    __tablename__ = "ingestion_sources"

    source_id: Mapped[str] = mapped_column(Text, primary_key=True)
    source_type: Mapped[str] = mapped_column(Text, nullable=False)
    config_revision: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)


class IngestionSnapshotRun(KnowledgeBase):
    __tablename__ = "ingestion_snapshot_runs"

    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    snapshot_key: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    command_digest: Mapped[str] = mapped_column(Text, nullable=False)
    source_id: Mapped[str] = mapped_column(Text, ForeignKey("ingestion_sources.source_id", ondelete="RESTRICT"), nullable=False)
    config_revision: Mapped[str] = mapped_column(Text, nullable=False)
    complete: Mapped[bool] = mapped_column(Boolean, nullable=False)
    document_count: Mapped[int] = mapped_column(Integer, nullable=False)
    diagnostics: Mapped[list[dict[str, Any]]] = mapped_column(_JSON, nullable=False, default=list)
    result: Mapped[dict[str, Any]] = mapped_column(_JSON, nullable=False, default=dict[str, Any])
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)


Index("ingestion_snapshot_runs_by_source", IngestionSnapshotRun.source_id, IngestionSnapshotRun.created_at)


class IngestionDocument(KnowledgeBase):
    __tablename__ = "ingestion_documents"
    __table_args__ = (UniqueConstraint("source_id", "external_id", name="uq_ingestion_document_source_external"),)

    document_id: Mapped[str] = mapped_column(Text, primary_key=True)
    source_id: Mapped[str] = mapped_column(Text, ForeignKey("ingestion_sources.source_id", ondelete="RESTRICT"), nullable=False)
    external_id: Mapped[str] = mapped_column(Text, nullable=False)
    current_processing_revision_id: Mapped[str | None] = mapped_column(Text)
    deleted_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)


Index("ingestion_documents_visible", IngestionDocument.source_id, IngestionDocument.deleted_at)


class IngestionDocumentRevision(KnowledgeBase):
    """One immutable revision of a document, holding the text it is made of.

    ``content`` is the authoritative processed text — exactly what chunking ran
    over, so a chunk's offsets address this string and no other. The indexes
    hold projections of it and are not addressable by offset; ``artifact_ref``
    names where the revision came from and is provenance, not a read path.

    ``NULL`` means observed and unusable — binary, or undecodable. It carries
    that one meaning and no other, which is why nothing is backfilled into it.
    """

    __tablename__ = "ingestion_document_revisions"

    processing_revision_id: Mapped[str] = mapped_column(Text, primary_key=True)
    document_id: Mapped[str] = mapped_column(Text, ForeignKey("ingestion_documents.document_id", ondelete="CASCADE"), nullable=False)
    source_revision_id: Mapped[str] = mapped_column(Text, nullable=False)
    path: Mapped[str] = mapped_column(Text, nullable=False)
    artifact_ref: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str | None] = mapped_column(Text)
    revision_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", _JSON, nullable=False, default=dict[str, Any])
    processing: Mapped[dict[str, Any]] = mapped_column(_JSON, nullable=False, default=dict[str, Any])
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)


Index("ingestion_document_revisions_by_document", IngestionDocumentRevision.document_id, IngestionDocumentRevision.created_at)


class IngestionSnapshotObservation(KnowledgeBase):
    __tablename__ = "ingestion_snapshot_observations"

    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("ingestion_snapshot_runs.run_id", ondelete="CASCADE"), primary_key=True)
    document_id: Mapped[str] = mapped_column(Text, ForeignKey("ingestion_documents.document_id", ondelete="CASCADE"), primary_key=True)
    processing_revision_id: Mapped[str] = mapped_column(
        Text,
        ForeignKey("ingestion_document_revisions.processing_revision_id", ondelete="RESTRICT"),
        nullable=False,
    )


class IngestionIndexGeneration(KnowledgeBase):
    """Which incarnation of a target's indexes the recorded state describes.

    An index lives in Qdrant and OpenSearch; the record that it holds a document
    lives here. Those can diverge — a dropped collection, a container without a
    persistent volume, a restored database, a writer pointed at a fresh
    environment — and nothing in the record itself would notice. The generation
    is what ties the two together: it rotates whenever the bootstrap has to
    *create* a collection or index, and every index-state row carries the
    generation it was written under.

    PostgreSQL arbitrates it rather than the indexes themselves. OpenSearch does
    expose an intrinsic `index.uuid`, but Qdrant has no equivalent, so reading
    the generation out of the stores would mean a marker point and a racy
    read-back for one half and not the other. One arbiter, read straight after
    rotating, is both simpler and free of that race.
    """

    __tablename__ = "ingestion_index_generation"

    target: Mapped[str] = mapped_column(Text, primary_key=True)
    generation: Mapped[str] = mapped_column(Text, nullable=False)
    rotated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)


class IngestionIndexState(KnowledgeBase):
    """Per-document, per-target index state — the reindex decision.

    Replaces copy-on-write generations: writes are idempotent per document, so
    retrieval stays available throughout indexing without an atomic cutover.
    A document is reindexed when its content or its enricher version changed —
    or when it does not carry the target's current index generation, which is
    how a recreated index and an interrupted reindex both come out as "still to
    do" without anyone having to remember.
    """

    __tablename__ = "ingestion_index_state"

    document_id: Mapped[str] = mapped_column(Text, ForeignKey("ingestion_documents.document_id", ondelete="CASCADE"), primary_key=True)
    target: Mapped[str] = mapped_column(Text, primary_key=True)
    content_hash: Mapped[str] = mapped_column(Text, nullable=False)
    enricher_hash: Mapped[str] = mapped_column(Text, nullable=False)
    processing_revision_id: Mapped[str] = mapped_column(
        Text,
        ForeignKey("ingestion_document_revisions.processing_revision_id", ondelete="RESTRICT"),
        nullable=False,
    )
    index_generation: Mapped[str] = mapped_column(Text, nullable=False)
    projection_fingerprint: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    indexed_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)


Index("ingestion_index_state_by_target", IngestionIndexState.target, IngestionIndexState.indexed_at)
# "What of this target is still not on the current generation" is the question a
# resumed reindex asks, and the question the reconcile answers.
Index("ingestion_index_state_by_generation", IngestionIndexState.target, IngestionIndexState.index_generation)
