# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""An extracted claim kept whole, and the judgements made about pairs of them.

`knowledge_propositions` holds a claim exactly as it arrived. `proposition_id` is
a surrogate and **not** an assertion's identity: whether two propositions are one
claim is a judgement, recorded in the other table, and reading it off this one
would be the content-derived identity this design spent four ADRs removing.

The unique key is on the *content*, so one claim seen twice is one row. Same
fingerprint means byte-identical after normalisation; different fingerprints say
nothing about whether the claims are one, which is precisely the open question.
`fingerprint_version` is a column of its own for the same reason
`entity_key_version` is — normalisation will change, and an old value must not
look like it lives in the new namespace.

`knowledge_proposition_equivalence` is **append-only and has no unique key on the
pair**. The same judge asked twice can answer differently, and a constraint
permitting one row per pair and judge would delete exactly the evidence that a
model is unstable — the thing anyone auditing this would most want to see.

Reusing a decision rather than asking again is a *cache*: a different table, with
a different key, added when there is a reason to. Audit answers "what was
decided"; a cache answers "what do we use". One table cannot answer both, because
the constraint that makes the second efficient is what makes the first blind.

Both tables are empty and nothing writes to them yet. Nothing is backfilled: a
proposition for an assertion already stored would have to be reconstructed from a
payload that could not hold it, which is guessing.

Revision ID: 0012
Revises: 0011
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "0012"
down_revision: str | None = "0011"
branch_labels: str | None = None
depends_on: str | None = None

_JSON = sa.JSON().with_variant(JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "knowledge_propositions",
        sa.Column("proposition_id", UUID(as_uuid=True), primary_key=True),
        sa.Column("fingerprint", sa.Text(), nullable=False),
        sa.Column("fingerprint_version", sa.Text(), nullable=False),
        sa.Column("fields", _JSON, nullable=False),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "fingerprint_version", "fingerprint", name="uq_knowledge_proposition_content"
        ),
    )

    op.create_table(
        "knowledge_proposition_equivalence",
        sa.Column("assessment_id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "left_id",
            UUID(as_uuid=True),
            sa.ForeignKey("knowledge_propositions.proposition_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "right_id",
            UUID(as_uuid=True),
            sa.ForeignKey("knowledge_propositions.proposition_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("verdict", sa.Text(), nullable=False),
        sa.Column("classifier", sa.Text()),
        sa.Column("model", sa.Text()),
        sa.Column("classifier_version", sa.Text()),
        sa.Column("run_id", sa.Text()),
        sa.Column(
            "decided_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    # An index and not a constraint. Looking a pair up must be quick; forbidding
    # a second answer would hide a model that gave one.
    op.create_index(
        "knowledge_proposition_equivalence_by_pair",
        "knowledge_proposition_equivalence",
        ["left_id", "right_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "knowledge_proposition_equivalence_by_pair",
        table_name="knowledge_proposition_equivalence",
    )
    op.drop_table("knowledge_proposition_equivalence")
    op.drop_table("knowledge_propositions")
