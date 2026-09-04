# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Knowledge graph: assertions, kind payloads, observations, reviews, metadata.

First revision of the knowledge database. This chain is separate from the
platform's: nothing here is ever applied to the runtime database.

Revision ID: 0001
Revises:
Create Date: 2026-08-21
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "knowledge",
        sa.Column("identity", sa.Text(), primary_key=True),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("type", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("review_required", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("review_reason", sa.Text()),
        sa.Column("product", sa.Text()),
        sa.Column("version", sa.Text()),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.CheckConstraint("confidence >= 0.0 AND confidence <= 1.0", name="ck_knowledge_confidence"),
    )
    op.create_index("knowledge_visible_by_kind", "knowledge", ["kind", "review_required"])
    op.create_index("knowledge_review_queue", "knowledge", ["review_required", "updated_at"])

    op.create_table(
        "knowledge_facts",
        sa.Column("identity", sa.Text(), sa.ForeignKey("knowledge.identity", ondelete="CASCADE"), primary_key=True),
        sa.Column("subject", sa.Text(), nullable=False),
        sa.Column("predicate", sa.Text(), nullable=False),
        sa.Column("object", sa.Text(), nullable=False),
    )
    op.create_index("knowledge_facts_subject", "knowledge_facts", ["subject"])
    op.create_index("knowledge_facts_predicate", "knowledge_facts", ["predicate"])
    op.create_index("knowledge_facts_object", "knowledge_facts", ["object"])

    op.create_table(
        "knowledge_rules",
        sa.Column("identity", sa.Text(), sa.ForeignKey("knowledge.identity", ondelete="CASCADE"), primary_key=True),
        sa.Column("rule_text", sa.Text(), nullable=False),
    )
    op.create_table(
        "knowledge_patterns",
        sa.Column("identity", sa.Text(), sa.ForeignKey("knowledge.identity", ondelete="CASCADE"), primary_key=True),
        sa.Column("pattern_name", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
    )
    op.create_table(
        "knowledge_decisions",
        sa.Column("identity", sa.Text(), sa.ForeignKey("knowledge.identity", ondelete="CASCADE"), primary_key=True),
        sa.Column("decision", sa.Text(), nullable=False),
        sa.Column("effect", sa.Text(), nullable=False),
    )

    op.create_table(
        "knowledge_observations",
        sa.Column("observation_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("identity", sa.Text(), sa.ForeignKey("knowledge.identity", ondelete="CASCADE"), nullable=False),
        sa.Column("source_id", sa.Text(), nullable=False),
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column("evidence", postgresql.JSONB()),
        sa.Column("observed_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint("identity", "run_id", "source_id", name="uq_knowledge_observation"),
    )

    op.create_table(
        "knowledge_reviews",
        sa.Column("review_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("identity", sa.Text(), sa.ForeignKey("knowledge.identity", ondelete="CASCADE"), nullable=False),
        sa.Column("reviewer_id", sa.Text(), nullable=False),
        sa.Column("decision", sa.Text(), nullable=False),
        sa.Column("comment", sa.Text()),
        sa.Column("changes", postgresql.JSONB()),
        sa.Column("reviewed_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.CheckConstraint("decision IN ('approved', 'rejected', 'edited')", name="ck_knowledge_reviews_decision"),
    )
    op.create_index("knowledge_reviews_by_identity", "knowledge_reviews", ["identity", "reviewed_at"])

    op.create_table(
        "knowledge_metadata",
        sa.Column("identity", sa.Text(), sa.ForeignKey("knowledge.identity", ondelete="CASCADE"), primary_key=True),
        sa.Column("attributes", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("schema_version", sa.Text(), nullable=False, server_default="v1"),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index(
        "knowledge_metadata_attributes", "knowledge_metadata", ["attributes"], postgresql_using="gin"
    )


def downgrade() -> None:
    op.drop_table("knowledge_metadata")
    op.drop_table("knowledge_reviews")
    op.drop_table("knowledge_observations")
    op.drop_table("knowledge_decisions")
    op.drop_table("knowledge_patterns")
    op.drop_table("knowledge_rules")
    op.drop_table("knowledge_facts")
    op.drop_table("knowledge")
