# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Where a claim stands, identified once and findable again.

A slot has been a section's *position* — `source:document:0`, `:1`, `:2` — which
is why inserting a paragraph renumbered every section below it and retraction had
to be widened to the whole document to stop that destroying approvals. The
position is useful for ordering and useless as identity.

`knowledge_slots` holds the surrogate. `knowledge_slot_anchors` holds every path
the slot has been found by, with how much that path was worth, because the
strength changes over a slot's life and the change is the part that matters:

    authored → authored    ordinary
    derived  → authored    an upgrade; very likely the same section
    derived  → derived     the fallback rungs are doing the work
    authored → derived     a degradation — do not continue blindly

One strength per slot could not express the last row. When an authored id
disappears and only a heading path is left, a resolver that still believed it
held an authored anchor would carry a lineage across a change nobody could see.

Additive and empty. Nothing is backfilled: deriving a slot for an assertion
already stored would mean guessing which section it came from, and that guess is
what this whole design removes. Existing evidence keeps its ordinal `slot_id`,
and the document-scoped retraction container means both can exist at once.

Revision ID: 0010
Revises: 0009
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.create_table(
        "knowledge_slots",
        sa.Column("slot_id", sa.Text(), primary_key=True),
        sa.Column("document_id", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False, server_default="live"),
        # What the section last said, hashed. The resolver's recovery rung
        # compares against it; without it a renamed heading would mint a new slot
        # and retract everything the old one held.
        sa.Column("content_fingerprint", sa.Text(), nullable=False, server_default=""),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("retired_at", sa.TIMESTAMP(timezone=True)),
    )
    # Matching never crosses a document, so every lookup starts here.
    op.create_index(
        "knowledge_slots_by_document", "knowledge_slots", ["document_id", "status"]
    )

    op.create_table(
        "knowledge_slot_anchors",
        sa.Column("anchor_id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "slot_id",
            sa.Text(),
            sa.ForeignKey("knowledge_slots.slot_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("anchor", sa.Text(), nullable=False),
        sa.Column("strength", sa.Text(), nullable=False),
        sa.Column(
            "valid_from",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        # `valid_to IS NULL` *is* current. A separate `is_current` flag would be
        # a second fact about the same thing, and two facts can disagree.
        sa.Column("valid_to", sa.TIMESTAMP(timezone=True)),
    )
    op.create_index(
        "knowledge_slot_anchors_by_slot", "knowledge_slot_anchors", ["slot_id", "valid_to"]
    )
    op.create_index("knowledge_slot_anchors_by_anchor", "knowledge_slot_anchors", ["anchor"])


def downgrade() -> None:
    op.drop_index("knowledge_slot_anchors_by_anchor", table_name="knowledge_slot_anchors")
    op.drop_index("knowledge_slot_anchors_by_slot", table_name="knowledge_slot_anchors")
    op.drop_table("knowledge_slot_anchors")
    op.drop_index("knowledge_slots_by_document", table_name="knowledge_slots")
    op.drop_table("knowledge_slots")
