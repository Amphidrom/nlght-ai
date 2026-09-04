# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""What each run read in each slot, kept rather than overwritten.

`knowledge_slots.content_fingerprint` is a current state: the resolver
overwrites it every run so its recovery rung can match a renamed heading against
what the section says *now*. That makes it unable to answer the question a report
has to ask — whether the section a claim stood in was actually edited between two
pictures — because after a run the value is already that run's and the previous
one is gone.

So a reading is recorded per run and per slot. The two are not redundant: one is
what the slot says now, the other is what it said each time it was looked at, and
only the second lets a movement be explained rather than merely permitted:

    expected_change=true
    because=slot_input_changed

instead of the document-wide "something in this file was edited", which waves
through a claim that moved in a section nobody touched.

A log of readings and not of changes. A run that found a slot unchanged still
records that it looked, which is what distinguishes "unchanged" from "never
observed" — and the second must not be reported as drift.

Nothing is backfilled. Slots observed before this migration have no readings, and
a comparison across that boundary falls back to the document gate rather than
calling the whole corpus drift, which would be reporting the migration.

Revision ID: 0014
Revises: 0013
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision: str = "0014"
down_revision: str | None = "0013"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.create_table(
        "knowledge_slot_observations",
        sa.Column("observation_id", UUID(as_uuid=True), primary_key=True),
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column("document_id", sa.Text(), nullable=False),
        sa.Column("slot_id", sa.Text(), nullable=False),
        sa.Column("parser_input_fingerprint", sa.Text(), nullable=False),
        sa.Column("document_revision", sa.Text(), nullable=False, server_default=""),
        sa.Column(
            "observed_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        # One reading per run per slot. A retried step re-reads the same section
        # and must not look like a second sighting of it.
        sa.UniqueConstraint("run_id", "slot_id", name="uq_knowledge_slot_observation"),
    )
    # The report asks for the latest reading of each slot, so that is the shape
    # the index has.
    op.create_index(
        "knowledge_slot_observations_by_slot",
        "knowledge_slot_observations",
        ["slot_id", "observed_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "knowledge_slot_observations_by_slot",
        table_name="knowledge_slot_observations",
    )
    op.drop_table("knowledge_slot_observations")
