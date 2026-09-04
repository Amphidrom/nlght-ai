# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""The entity key describes an assertion; it does not identify one.

`uq_knowledge_assertion_entity` said the opposite. While it stood, the model's
choice of `subject` and `rule_property` *was* the identity, enforced by the
database — and a run over an unedited sentence proved what that costs: the model
described one rule two ways across two passes, so the corpus gained an assertion
nobody wrote, and the approval stayed with the one nobody would read again.

Identity is now decided by where a claim stands and what it says. The key remains
on the row as the description it always was, and two assertions may legitimately
carry the same one: an ambiguous match mints a new assertion rather than picking
between equally plausible candidates, and a claim a source dropped and later
asserted again is a new decision, not an undo. Both are correct outcomes the
constraint would have rejected.

Nothing is dropped and no row moves. Only the claim that a description is unique.

Revision ID: 0009
Revises: 0008
"""

from __future__ import annotations

from alembic import op

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | None = None
depends_on: str | None = None

_CONSTRAINT = "uq_knowledge_assertion_entity"
_TABLE = "knowledge_assertions"


def upgrade() -> None:
    with op.batch_alter_table(_TABLE) as batch:
        batch.drop_constraint(_CONSTRAINT, type_="unique")
    # Still worth an index: every match asks "is this key already known", which
    # is a lookup and no longer an assertion of uniqueness.
    op.create_index(
        "knowledge_assertions_by_entity_key",
        _TABLE,
        ["entity_key_version", "entity_key"],
    )


def downgrade() -> None:
    op.drop_index("knowledge_assertions_by_entity_key", table_name=_TABLE)
    with op.batch_alter_table(_TABLE) as batch:
        batch.create_unique_constraint(_CONSTRAINT, ["entity_key_version", "entity_key"])
