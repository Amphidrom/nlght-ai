# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""An assertion over its lifetime: assertions, variants, revisions, evidence.

Four tables for the three levels the design keeps apart. `knowledge_assertions`
holds the surrogate that never moves and the entity key that says which business
question it is. `knowledge_variants` holds the simultaneously valid cases of that
question — CHF 500 for employees and CHF 5000 for executives are both true, and
revisions are a sequence, so two values true at once cannot be revisions of one
thing. `knowledge_revisions` holds the states of one case, with the review state
beside them because whether a claim is still the same claim and whether an
approval of it still holds are two questions. `knowledge_evidence` says where a
revision was found, with document and document revision as columns rather than
JSON, because the diff has to ask which assertions came from document D at
revision R.

**Nothing is migrated into these.** They start empty and are filled from the
next run onwards, exactly as the extraction state did. The existing `knowledge`
graph stays the read path and stays authoritative; deriving these rows from the
assertions already stored would mean guessing which business question each
sentence was about, which is the guessing this whole design removes.

Revision ID: 0007
Revises: 0006
Create Date: 2026-08-28
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_JSON = sa.JSON().with_variant(JSONB(), "postgresql")
_UUID = sa.Uuid().with_variant(UUID(as_uuid=True), "postgresql")


def upgrade() -> None:
    op.create_table(
        "knowledge_assertions",
        sa.Column("assertion_id", sa.Text(), primary_key=True),
        sa.Column("entity_key_version", sa.Text(), nullable=False),
        sa.Column("entity_key", sa.Text(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column(
            "created_at", sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(), nullable=False,
        ),
        sa.Column("retired_at", sa.TIMESTAMP(timezone=True)),
        sa.UniqueConstraint(
            "entity_key_version", "entity_key", name="uq_knowledge_assertion_entity"
        ),
    )
    op.create_table(
        "knowledge_variants",
        sa.Column("variant_id", sa.Text(), primary_key=True),
        sa.Column(
            "assertion_id", sa.Text(),
            sa.ForeignKey("knowledge_assertions.assertion_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("scope", _JSON, nullable=False),
        sa.Column("resolution", sa.Text(), nullable=False),
        sa.Column(
            "created_at", sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(), nullable=False,
        ),
        sa.Column("retired_at", sa.TIMESTAMP(timezone=True)),
    )
    op.create_table(
        "knowledge_revisions",
        sa.Column("revision_id", _UUID, primary_key=True),
        sa.Column(
            "variant_id", sa.Text(),
            sa.ForeignKey("knowledge_variants.variant_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("fingerprint", sa.Text(), nullable=False),
        sa.Column("extraction_version", sa.Text(), nullable=False),
        sa.Column("review_state", sa.Text(), nullable=False),
        sa.Column("supersedes", sa.Integer()),
        sa.Column(
            "created_at", sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(), nullable=False,
        ),
        sa.UniqueConstraint("variant_id", "revision", name="uq_knowledge_revision"),
    )
    op.create_table(
        "knowledge_evidence",
        sa.Column("evidence_id", _UUID, primary_key=True),
        sa.Column(
            "revision_id", _UUID,
            sa.ForeignKey("knowledge_revisions.revision_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("document_id", sa.Text(), nullable=False),
        sa.Column("document_revision", sa.Text(), nullable=False),
        sa.Column("slot_id", sa.Text()),
        sa.Column("source_span", _JSON),
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column(
            "observed_at", sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(), nullable=False,
        ),
        sa.UniqueConstraint(
            "revision_id", "document_id", "run_id", name="uq_knowledge_evidence"
        ),
    )
    op.create_index(
        "knowledge_evidence_by_document",
        "knowledge_evidence",
        ["document_id", "document_revision"],
    )


def downgrade() -> None:
    op.drop_index("knowledge_evidence_by_document", table_name="knowledge_evidence")
    op.drop_table("knowledge_evidence")
    op.drop_table("knowledge_revisions")
    op.drop_table("knowledge_variants")
    op.drop_table("knowledge_assertions")
