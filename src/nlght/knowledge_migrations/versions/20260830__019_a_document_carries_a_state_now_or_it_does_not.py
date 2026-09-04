# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Which documents carry a state now, recorded rather than inferred.

`knowledge_evidence` is append-only and answers "was observed here". That is the
right shape for an audit and the wrong one for a citation. A document that has
since removed a sentence stays in the evidence forever — correctly, it really did
assert it once — so an answer built from evidence sends a reader to a file that no
longer says it, while the claim is perfectly alive because a second document
still carries it.

The alternatives were both worse. Deleting evidence destroys the record the whole
diff is built on. Inferring currency from absence — "this document was read again
later and produced nothing" — is a derivation across two tables with an edge for
every unslotted section, and a citation resting on it would be a guess presented
as a source.

So currency gets its own small relation, replaced from what a run saw:

    complete observation of D produces revision R   → D supports R
    complete observation of D no longer produces R  → the row goes
                                                      the evidence stays

**The document decides, never the slot.** A section may be renamed, split, or
lose its heading without the document ceasing to assert anything (ADR-0044), so
`slot_id` here is provenance and takes part in no removal. Only a complete
observation of the document maintains this at all — a partial run cannot say what
a document currently carries, because it did not see all of it.

**The revision on a row is where the state was last extracted**, not where the
document stands now. A section the stable-slot guard protected is not re-read, so
its row keeps the older number — the truthful answer to "where was this last seen
to be said". The column is named `observed_document_revision` so that a citation
cannot quietly read it as "the revision that currently carries this".

**Nothing is backfilled.** Which of the existing sightings are current cannot be
known: that is precisely the fact this table exists to record and the corpus never
recorded it. Backfilling from evidence would assert a currency nothing proves,
which is the error this migration is here to end. Until a document is observed in
full again, its claims resolve with no current support — visible, and counted in
`knowledge-report` as `claims_without_current_support`.

`ON DELETE CASCADE` because this is a projection of a state, not a record of one:
if a revision were ever removed, a row saying somebody carries it means nothing.

Revision ID: 0019
Revises: 0018
"""

from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision: str = "0019"
down_revision: str | None = "0018"
branch_labels: str | None = None
depends_on: str | None = None

logger = logging.getLogger("alembic.runtime.migration")


def upgrade() -> None:
    op.create_table(
        "knowledge_document_support",
        sa.Column("support_id", UUID(as_uuid=True), primary_key=True),
        sa.Column("document_id", sa.Text(), nullable=False),
        sa.Column(
            "revision_id",
            UUID(as_uuid=True),
            sa.ForeignKey("knowledge_revisions.revision_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("observed_document_revision", sa.Text(), nullable=False, server_default=""),
        sa.Column("document_path", sa.Text(), nullable=False, server_default=""),
        sa.Column("slot_id", sa.Text(), nullable=False, server_default=""),
        sa.Column(
            "observed_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        # One row per document, state and section. A run that re-observes the
        # same section is the same support, not a second one.
        sa.UniqueConstraint(
            "document_id", "revision_id", "slot_id",
            name="uq_knowledge_document_support",
        ),
    )
    op.create_index(
        "knowledge_document_support_by_revision",
        "knowledge_document_support",
        ["revision_id", "document_id"],
    )

    waiting = op.get_bind().execute(
        sa.text(
            "SELECT count(*) FROM knowledge k "
            "WHERE k.revision_id IS NOT NULL AND NOT EXISTS ("
            "  SELECT 1 FROM knowledge_document_support s "
            "   WHERE s.revision_id = k.revision_id)"
        )
    ).scalar_one()
    if waiting:
        logger.warning(
            "knowledge.migration.0019 | claims with no current support=%s — they "
            "resolve without a citable source until each document is observed in "
            "full again; not backfilled, because which sightings are current was "
            "never recorded",
            waiting,
        )


def downgrade() -> None:
    op.drop_index(
        "knowledge_document_support_by_revision",
        table_name="knowledge_document_support",
    )
    op.drop_table("knowledge_document_support")
