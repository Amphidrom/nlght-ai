# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Merge as a review decision.

A reviewer who finds two assertions saying the same thing in different words
merges one into the other. The source row stays: deleting it would cascade away
its observations and its decision history, which is exactly the trail a later
disagreement needs. Instead it points at the target and stops being canonical.

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-22
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("knowledge", sa.Column("merged_into", sa.Text(), nullable=True))
    # RESTRICT: a merge target must not be deleted while assertions point at it,
    # or the pointer would dangle and the merged evidence lose its home.
    op.create_foreign_key(
        "fk_knowledge_merged_into",
        "knowledge",
        "knowledge",
        ["merged_into"],
        ["identity"],
        ondelete="RESTRICT",
    )
    op.create_index("knowledge_merged_into", "knowledge", ["merged_into"])

    # SQLite cannot ALTER a CHECK constraint, and the knowledge database is
    # PostgreSQL; the batch path would rebuild the table for no gain.
    if op.get_bind().dialect.name == "postgresql":
        op.drop_constraint("ck_knowledge_reviews_decision", "knowledge_reviews", type_="check")
        op.create_check_constraint(
            "ck_knowledge_reviews_decision",
            "knowledge_reviews",
            "decision IN ('approved', 'rejected', 'edited', 'merged')",
        )


def downgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        op.drop_constraint("ck_knowledge_reviews_decision", "knowledge_reviews", type_="check")
        op.create_check_constraint(
            "ck_knowledge_reviews_decision",
            "knowledge_reviews",
            "decision IN ('approved', 'rejected', 'edited')",
        )
    op.drop_index("knowledge_merged_into", table_name="knowledge")
    op.drop_constraint("fk_knowledge_merged_into", "knowledge", type_="foreignkey")
    op.drop_column("knowledge", "merged_into")
