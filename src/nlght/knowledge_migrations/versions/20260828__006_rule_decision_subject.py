# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""A short subject and property for rules and decisions.

Both kinds were identified by free text — a rule by its whole sentence, a
decision by two of them — so two runs over one unchanged page agreed only when
the model answered word for word. `pattern` has had the right shape all along:
a short name beside a mutable description. These two get the same.

**Nullable, and no backfill.** A corpus exists whose rules have no subject and
no property, and there is no way to derive one from the stored wording that is
not the guessing this design exists to remove. Absent is a state the schema has
to express.

**This is a schema migration, not an identity migration.** Identity still comes
from content. Re-deriving it from these fields would rewrite what every existing
assertion is, and it belongs with entity keys, variants and lineage — after the
matcher that carries the continuity exists, not before it.

Revision ID: 0006
Revises: 0005
Create Date: 2026-08-28
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("knowledge_rules", sa.Column("subject", sa.Text(), nullable=True))
    op.add_column("knowledge_rules", sa.Column("rule_property", sa.Text(), nullable=True))
    op.add_column("knowledge_decisions", sa.Column("subject", sa.Text(), nullable=True))
    op.add_column("knowledge_decisions", sa.Column("decision_type", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("knowledge_decisions", "decision_type")
    op.drop_column("knowledge_decisions", "subject")
    op.drop_column("knowledge_rules", "rule_property")
    op.drop_column("knowledge_rules", "subject")
