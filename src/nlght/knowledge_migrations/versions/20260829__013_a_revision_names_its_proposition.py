# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""A revision names the complete structure it recorded.

`knowledge_propositions` could hold a claim whole and nothing read it, so a fact
with four roles was stored where retrieval does not look — which is not better
than the malformed rejection it replaced, only quieter. This column is the link
that closes it.

It sits on the **revision** and not on the assertion or the graph node, and that
is the whole decision. An assertion is continuity; a revision is the form the
claim took at one time:

    Assertion A
      Revision 1 → Proposition P1
      Revision 2 → Proposition P2

Two propositions with different fingerprints can continue one assertion when an
equivalence judgement says they are one claim. A single `proposition_id` on the
assertion — or on the node — would flatten that into "one proposition per
assertion" and overwrite the earlier form of every claim that was ever reworded,
which is exactly the history this design exists to keep.

Nullable, and it stays that way. Every revision written before propositions
existed has none and never will: reconstructing one from a payload that could not
hold it is guessing. `rule`, `pattern` and `decision` have none by design — they
are documents of a fixed shape rather than n-ary claims.

`ON DELETE RESTRICT` because a proposition a revision names is what that revision
said. Cascading would delete history to tidy up a claims table.

Revision ID: 0013
Revises: 0012
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision: str = "0013"
down_revision: str | None = "0012"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.add_column(
        "knowledge_revisions",
        sa.Column("proposition_id", UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_knowledge_revision_proposition",
        "knowledge_revisions",
        "knowledge_propositions",
        ["proposition_id"],
        ["proposition_id"],
        ondelete="RESTRICT",
    )


def downgrade() -> None:
    op.drop_constraint(
        "fk_knowledge_revision_proposition", "knowledge_revisions", type_="foreignkey"
    )
    op.drop_column("knowledge_revisions", "proposition_id")
