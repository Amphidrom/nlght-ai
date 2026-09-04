# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Authoritative ingestion snapshots, revisions, tombstones, and per-document index state.

Second revision of the knowledge database. The ingestion record of a corpus
belongs with the knowledge derived from it, not with the runtime's own metadata
(ADR-0033). This migration was authored against the platform chain as 0004 and
never executed anywhere, so it moves rather than migrating.

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-22
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "ingestion_sources",
        sa.Column("source_id", sa.Text(), primary_key=True),
        sa.Column("source_type", sa.Text(), nullable=False),
        sa.Column("config_revision", sa.Text(), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_table(
        "ingestion_snapshot_runs",
        sa.Column("run_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("snapshot_key", sa.Text(), nullable=False, unique=True),
        sa.Column("command_digest", sa.Text(), nullable=False),
        sa.Column("source_id", sa.Text(), sa.ForeignKey("ingestion_sources.source_id", ondelete="RESTRICT"), nullable=False),
        sa.Column("config_revision", sa.Text(), nullable=False),
        sa.Column("complete", sa.Boolean(), nullable=False),
        sa.Column("document_count", sa.Integer(), nullable=False),
        sa.Column("diagnostics", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("result", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ingestion_snapshot_runs_by_source", "ingestion_snapshot_runs", ["source_id", "created_at"])
    op.create_table(
        "ingestion_documents",
        sa.Column("document_id", sa.Text(), primary_key=True),
        sa.Column("source_id", sa.Text(), sa.ForeignKey("ingestion_sources.source_id", ondelete="RESTRICT"), nullable=False),
        sa.Column("external_id", sa.Text(), nullable=False),
        sa.Column("current_revision_id", sa.Text()),
        sa.Column("deleted_at", sa.TIMESTAMP(timezone=True)),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint("source_id", "external_id", name="uq_ingestion_document_source_external"),
    )
    op.create_index("ingestion_documents_visible", "ingestion_documents", ["source_id", "deleted_at"])
    op.create_table(
        "ingestion_document_revisions",
        sa.Column("revision_id", sa.Text(), primary_key=True),
        sa.Column("document_id", sa.Text(), sa.ForeignKey("ingestion_documents.document_id", ondelete="CASCADE"), nullable=False),
        sa.Column("source_revision_id", sa.Text(), nullable=False),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column("artifact_ref", sa.Text(), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("processing", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index(
        "ingestion_document_revisions_by_document",
        "ingestion_document_revisions",
        ["document_id", "created_at"],
    )
    op.create_table(
        "ingestion_snapshot_observations",
        sa.Column("run_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("ingestion_snapshot_runs.run_id", ondelete="CASCADE"), primary_key=True),
        sa.Column("document_id", sa.Text(), sa.ForeignKey("ingestion_documents.document_id", ondelete="CASCADE"), primary_key=True),
        sa.Column("revision_id", sa.Text(), sa.ForeignKey("ingestion_document_revisions.revision_id", ondelete="RESTRICT"), nullable=False),
    )
    op.create_table(
        "ingestion_index_state",
        sa.Column("document_id", sa.Text(), sa.ForeignKey("ingestion_documents.document_id", ondelete="CASCADE"), primary_key=True),
        sa.Column("target", sa.Text(), primary_key=True),
        sa.Column("content_hash", sa.Text(), nullable=False),
        sa.Column("enricher_hash", sa.Text(), nullable=False),
        sa.Column("revision_id", sa.Text(), sa.ForeignKey("ingestion_document_revisions.revision_id", ondelete="RESTRICT"), nullable=False),
        sa.Column("indexed_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ingestion_index_state_by_target", "ingestion_index_state", ["target", "indexed_at"])


def downgrade() -> None:
    op.drop_table("ingestion_index_state")
    op.drop_table("ingestion_snapshot_observations")
    op.drop_table("ingestion_document_revisions")
    op.drop_table("ingestion_documents")
    op.drop_table("ingestion_snapshot_runs")
    op.drop_table("ingestion_sources")
