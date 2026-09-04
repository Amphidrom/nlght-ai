# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""A revision names the sighting it came from.

Migration `0016` made the observation the unit an equivalence judgement compares.
Without this column there was no way back from a claim already in the corpus to
the sighting it came from, so the comparison **synthesised** one — a fresh row
per candidate per comparison, stamped with the run that did the comparing rather
than the run that did the seeing.

That turns a sighting into a sighting-per-comparison. The audit would still be
complete and would be historically false, which is worse than incomplete: it
reads as evidence. The three levels only hold if each stays what it is:

    observation   a sighting happened
    assessment    a judgement between two sightings
    revision      the state that resulted

So the revision names its observation, the way it already names its proposition,
and a candidate's side of an assessment points at the row that actually recorded
it.

Nullable, and nothing is backfilled. Every revision written before this has no
observation and cannot be given one: which sighting produced it was not
recorded, and inventing one would put a fabricated event in an audit.

`ON DELETE RESTRICT` because a sighting a revision came from is part of how that
revision is explained.

Revision ID: 0017
Revises: 0016
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision: str = "0017"
down_revision: str | None = "0016"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.add_column(
        "knowledge_revisions",
        sa.Column("observation_id", UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_knowledge_revision_observation",
        "knowledge_revisions",
        "knowledge_assertion_observations",
        ["observation_id"],
        ["observation_id"],
        ondelete="RESTRICT",
    )


def downgrade() -> None:
    op.drop_constraint(
        "fk_knowledge_revision_observation", "knowledge_revisions", type_="foreignkey"
    )
    op.drop_column("knowledge_revisions", "observation_id")
