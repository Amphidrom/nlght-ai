# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Do not extract again what has not changed: knowledge_extraction_state.

The knowledge counterpart to the data layer's index state. Without it every run
asks the model about every document again, and the model answers the same
question in slightly different words each time — which, once assertions are
updated incrementally rather than rebuilt, is a retraction and a re-review of
something nobody edited.

Two columns decide it together. `content_hash` is the text the extraction was
given; `extraction_version` folds the assertion schema, the prompt and its
workflow, and the model. Either alone would be wrong in one direction: content
alone skips a document after a model change, version alone skips an edited one.

No backfill. An empty table means nothing is skipped, so the first run after
this migration extracts everything once and records what it did — the safe
direction, and the only honest one, because nothing here can know what version
produced the assertions that already exist.

Revision ID: 0005
Revises: 0004
Create Date: 2026-08-28
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "knowledge_extraction_state",
        sa.Column("document_id", sa.Text(), primary_key=True),
        sa.Column("content_hash", sa.Text(), nullable=False),
        sa.Column("extraction_version", sa.Text(), nullable=False),
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column("assertion_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "extracted_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_table("knowledge_extraction_state")
