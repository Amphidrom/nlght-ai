# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Equivalence compares observations, not fact propositions.

`knowledge_proposition_equivalence` (migration `0012`) keyed a judgement on two
rows of `knowledge_propositions`. That is a fact's representation, so three of
the four kinds could never be audited at all — a reworded `rule` was joined or
kept apart with nothing recording why.

The comparison unit is an **observation**: what kind it is, what the source said,
and which assertion inside that sentence is meant. `knowledge_propositions` is
untouched and goes on holding the structured fact; nothing here generalises it,
because making a fact's representation everyone's would invent an abstraction no
invariant asks for.

The old table is dropped rather than kept beside the new one. It never held a
row, and two assessment tables would be the kind-specific split this is removing.

**The order is the reason these exist.** An assessment written after a merge can
only describe what happened; a reader cannot tell a judgement that *caused* a
continuation from one reconstructed to explain it. So both observations are
durable first, the assessment is appended next, and only then may a `yes`
continue an assertion. Nothing downstream could serve: `knowledge_evidence`
hangs off a revision, and the revision exists only once the decision has already
been taken.

An observation is a sighting and not an identity, so it is **not** deduplicated:
two runs that read one sentence saw it twice, and collapsing them would lose
which run saw what.

Revision ID: 0016
Revises: 0015
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "0016"
down_revision: str | None = "0015"
branch_labels: str | None = None
depends_on: str | None = None

_JSON = sa.JSON().with_variant(JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "knowledge_assertion_observations",
        sa.Column("observation_id", UUID(as_uuid=True), primary_key=True),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("observed_text", sa.Text(), nullable=False),
        sa.Column("representation", _JSON, nullable=False),
        sa.Column("representation_version", sa.Text(), nullable=False, server_default="1"),
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column("document_id", sa.Text(), nullable=False, server_default=""),
        sa.Column("slot_id", sa.Text(), nullable=False, server_default=""),
        sa.Column(
            "observed_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        # No unique key of any shape. An observation is an event, and an event
        # that happened twice happened twice.
    )

    op.create_table(
        "knowledge_equivalence_assessments",
        sa.Column("assessment_id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "left_observation_id",
            UUID(as_uuid=True),
            sa.ForeignKey(
                "knowledge_assertion_observations.observation_id", ondelete="CASCADE"
            ),
            nullable=False,
        ),
        sa.Column(
            "right_observation_id",
            UUID(as_uuid=True),
            sa.ForeignKey(
                "knowledge_assertion_observations.observation_id", ondelete="CASCADE"
            ),
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
        "knowledge_equivalence_by_pair",
        "knowledge_equivalence_assessments",
        ["left_observation_id", "right_observation_id"],
    )

    op.drop_index(
        "knowledge_proposition_equivalence_by_pair",
        table_name="knowledge_proposition_equivalence",
    )
    op.drop_table("knowledge_proposition_equivalence")


def downgrade() -> None:
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
    op.create_index(
        "knowledge_proposition_equivalence_by_pair",
        "knowledge_proposition_equivalence",
        ["left_id", "right_id"],
    )
    op.drop_index(
        "knowledge_equivalence_by_pair",
        table_name="knowledge_equivalence_assessments",
    )
    op.drop_table("knowledge_equivalence_assessments")
    op.drop_table("knowledge_assertion_observations")
