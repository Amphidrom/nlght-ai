# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""One assertion took another's place: knowledge_assertion_lineage.

A rewording keeps an assertion and adds a revision to it. A material change does
not — `CEO(OpenAI, Alice)` and `CEO(OpenAI, Bob)` say different things, so they
are two assertions, and what connects them is that the second replaced the first.

An **edge table**, not a column on the assertion, because succession is not
one-to-one:

    A → B          a claim replaced
    A → B, C       one rule split into two
    A, B → C       two folded into one

A `superseded_by` field can express none of the latter two, and a corpus of
policies does both routinely.

This exists because an earlier attempt tried to avoid needing it. Facts were to
be keyed on a projection of their roles, so that OpenAI's chief executive would
be one thing whose value moved. Deciding which roles identify a relation and
which are its state is domain knowledge no platform has without an ontology, and
neither a shipped vocabulary nor asking a customer to declare one before
ingesting is an answer. So the platform stops deciding it: an assertion is its
whole proposition, and succession is recorded — with who decided it and why —
rather than inferred from the words.

Empty, with no backfill. Nothing existing supersedes anything, and working out
retrospectively what replaced what is exactly the guessing this design removes.

Revision ID: 0008
Revises: 0007
Create Date: 2026-08-28
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_UUID = sa.Uuid().with_variant(UUID(as_uuid=True), "postgresql")


def upgrade() -> None:
    op.create_table(
        "knowledge_assertion_lineage",
        sa.Column("lineage_id", _UUID, primary_key=True),
        sa.Column(
            "predecessor_id", sa.Text(),
            sa.ForeignKey("knowledge_assertions.assertion_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "successor_id", sa.Text(),
            sa.ForeignKey("knowledge_assertions.assertion_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("relation", sa.Text(), nullable=False, server_default="supersedes"),
        sa.Column("reason", sa.Text()),
        sa.Column("recorded_by", sa.Text()),
        sa.Column(
            "recorded_at", sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(), nullable=False,
        ),
        sa.UniqueConstraint(
            "predecessor_id", "successor_id", "relation", name="uq_knowledge_lineage_edge"
        ),
    )
    op.create_index(
        "knowledge_lineage_by_successor", "knowledge_assertion_lineage", ["successor_id"]
    )


def downgrade() -> None:
    op.drop_index("knowledge_lineage_by_successor", table_name="knowledge_assertion_lineage")
    op.drop_table("knowledge_assertion_lineage")
