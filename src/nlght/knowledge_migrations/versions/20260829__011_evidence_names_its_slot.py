# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""A sighting is distinct per slot, not only per document and run.

Two sections of one document saying the same thing are two sightings. The old key
— revision, document, run — recorded only the first, so nothing could say which
section had stopped carrying a claim when one of them dropped it: the corpus had
never known both were carrying it.

A retry still records nothing twice. The run, the document and the slot are all
the same on a retry, which is the case the key was written for and still covers.

`slot_id` becomes empty rather than null where a section has nothing to be found
by. PostgreSQL treats nulls as distinct in a unique constraint, so a null here
would quietly let a retry record the same sighting twice — the one thing this key
exists to prevent.

Revision ID: 0011
Revises: 0010
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0011"
down_revision: str | None = "0010"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.execute(
        sa.text("UPDATE knowledge_evidence SET slot_id = '' WHERE slot_id IS NULL")
    )
    with op.batch_alter_table("knowledge_evidence") as batch:
        batch.alter_column(
            "slot_id", existing_type=sa.Text(), nullable=False, server_default=""
        )
        batch.drop_constraint("uq_knowledge_evidence", type_="unique")
        batch.create_unique_constraint(
            "uq_knowledge_evidence", ["revision_id", "document_id", "slot_id", "run_id"]
        )


def downgrade() -> None:
    with op.batch_alter_table("knowledge_evidence") as batch:
        batch.drop_constraint("uq_knowledge_evidence", type_="unique")
        batch.create_unique_constraint(
            "uq_knowledge_evidence", ["revision_id", "document_id", "run_id"]
        )
        batch.alter_column("slot_id", existing_type=sa.Text(), nullable=True)
