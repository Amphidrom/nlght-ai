# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Which incarnation of an index the recorded state describes.

Index state used to say "this document is in this target" without saying which
incarnation of that target. Drop the Qdrant collection or the OpenSearch index
and the record still claimed everything was indexed, so a rerun over an
unchanged source skipped every document and reported success into an empty
index. The generation closes that, and doubles as the marker that makes a large
reindex resumable: a document already on the current generation is done.

Existing rows are backfilled with a generation minted here, one per target, so
an established corpus is not forced through a full reindex. What that cannot
know is whether an index was already wiped before this migration ran — the
reconcile path is what finds that.

Revision ID: 0004
Revises: 0003
Create Date: 2026-08-27
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "ingestion_index_generation",
        sa.Column("target", sa.Text(), primary_key=True),
        sa.Column("generation", sa.Text(), nullable=False),
        sa.Column(
            "rotated_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    op.add_column(
        "ingestion_index_state",
        sa.Column("index_generation", sa.Text(), nullable=True),
    )

    connection = op.get_bind()
    targets = [
        row[0]
        for row in connection.execute(
            sa.text("SELECT DISTINCT target FROM ingestion_index_state")
        )
    ]
    for target in targets:
        generation = str(uuid.uuid4())
        connection.execute(
            sa.text(
                "INSERT INTO ingestion_index_generation (target, generation) "
                "VALUES (:target, :generation)"
            ),
            {"target": target, "generation": generation},
        )
        connection.execute(
            sa.text(
                "UPDATE ingestion_index_state SET index_generation = :generation "
                "WHERE target = :target"
            ),
            {"target": target, "generation": generation},
        )

    # Only now: every row has a value, so the column can carry the invariant.
    op.alter_column("ingestion_index_state", "index_generation", nullable=False)

    op.create_index(
        "ingestion_index_state_by_generation",
        "ingestion_index_state",
        ["target", "index_generation"],
    )


def downgrade() -> None:
    op.drop_index("ingestion_index_state_by_generation", table_name="ingestion_index_state")
    op.drop_column("ingestion_index_state", "index_generation")
    op.drop_table("ingestion_index_generation")
