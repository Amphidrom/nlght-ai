# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""A graph node names the revision it stands for.

The relationship existed only as an equality between two hashes:

    KnowledgeRevision.fingerprint == Knowledge.identity

which is not a relationship. A fingerprint is a content address, and two
revisions may carry the same one — a revision is new when the *observed wording*
moves, and the structured claim it hashes may not have moved at all. So nothing
said which state a node stood for, and every read that needed one quantified over
the whole set instead: the deepest revision carrying the hash, or "no revision
carrying it is current", or "every assertion carrying it is retired". Set logic
compensating for a missing edge.

`knowledge.revision_id` is that edge. The node names one state; the history stays
on the revisions, where it belongs.

**Nullable, and the backfill proves rather than guesses.** A full rebuild of the
graph from the lineage is not possible and must not be attempted: the row carries
review decisions, the confidence built from independent sightings, metadata and
legacy payloads, none of which any revision records. Rebuilding it would destroy
the approvals this whole design exists to protect.

So the pointer is set exactly where it is provable — a fingerprint carried by
*one* revision and no other, which is an injection and not a resemblance — and
left NULL everywhere else. In particular the deepest revision carrying a shared
fingerprint is **not** chosen: that is precisely the set logic this migration
removes, and writing its answer into a column would make it permanent.

**And a NULL is not one thing.** Two very different nodes end up with no pointer,
and collapsing them is how a superseded wording gets handed back to retrieval:

    no revision ever            written before lineage existed, or by a path
                                that keeps none. Absence of a record is not
                                evidence of anything, and this node stays
                                canonical exactly as it always was.

    lineage exists, the         several revisions carried its fingerprint. One
    state is unresolved         of them may be a state the corpus has long
                                moved past, and nothing here can say which.

The second is marked `revision_unresolved` and withheld from canonical reads
until a sighting settles it. Withholding a claim that may be current costs a
reader one stale gap in a corpus that is being re-ingested anyway; publishing one
that is superseded is the failure this relationship was introduced to end.

Both counts are reported, so what could not be proved is visible rather than
silent.

`ON DELETE RESTRICT` because a node whose state has been deleted has no state.

Revision ID: 0018
Revises: 0017
"""

from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision: str = "0018"
down_revision: str | None = "0017"
branch_labels: str | None = None
depends_on: str | None = None

logger = logging.getLogger("alembic.runtime.migration")


def upgrade() -> None:
    op.add_column(
        "knowledge",
        sa.Column("revision_id", UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "knowledge",
        sa.Column(
            "revision_unresolved",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )
    op.create_foreign_key(
        "fk_knowledge_revision",
        "knowledge",
        "knowledge_revisions",
        ["revision_id"],
        ["revision_id"],
        ondelete="RESTRICT",
    )
    op.create_index("knowledge_by_revision", "knowledge", ["revision_id"])

    # One revision carrying this fingerprint and no other: the node can only have
    # been written by that one. Two or more is not a harder version of the same
    # question — it is the question this column exists to stop being asked.
    op.execute(
        sa.text(
            """
            UPDATE knowledge
               SET revision_id = (
                       SELECT r.revision_id
                         FROM knowledge_revisions r
                        WHERE r.fingerprint = knowledge.identity
                   )
             WHERE (
                       SELECT count(*)
                         FROM knowledge_revisions r
                        WHERE r.fingerprint = knowledge.identity
                   ) = 1
            """
        )
    )

    # A node the backfill could not settle, but which lineage does describe. Not
    # the same thing as a node no revision has ever mentioned, and marked so that
    # nothing downstream has to guess which kind of NULL it is looking at.
    op.execute(
        sa.text(
            """
            UPDATE knowledge
               SET revision_unresolved = true
             WHERE revision_id IS NULL
               AND EXISTS (
                       SELECT 1
                         FROM knowledge_revisions r
                        WHERE r.fingerprint = knowledge.identity
                   )
            """
        )
    )

    bind = op.get_bind()
    unresolved = bind.execute(
        sa.text("SELECT count(*) FROM knowledge WHERE revision_unresolved")
    ).scalar_one()
    lineageless = bind.execute(
        sa.text(
            "SELECT count(*) FROM knowledge "
            "WHERE revision_id IS NULL AND NOT revision_unresolved"
        )
    ).scalar_one()
    if unresolved:
        logger.warning(
            "knowledge.migration.0018 | lineage-backed nodes whose state could not be "
            "proved=%s — withheld from canonical reads until a sighting sets the "
            "pointer",
            unresolved,
        )
    if lineageless:
        logger.info(
            "knowledge.migration.0018 | nodes with no lineage at all=%s — unchanged, "
            "still canonical",
            lineageless,
        )


def downgrade() -> None:
    op.drop_index("knowledge_by_revision", table_name="knowledge")
    op.drop_constraint("fk_knowledge_revision", "knowledge", type_="foreignkey")
    op.drop_column("knowledge", "revision_unresolved")
    op.drop_column("knowledge", "revision_id")
