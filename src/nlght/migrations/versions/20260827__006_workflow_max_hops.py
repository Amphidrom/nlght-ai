# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""How many steps a workflow may take: workflows.max_hops.

The step machine's budget used to be a constant, and constants are wrong here in
both directions. Twenty is plenty for a workflow that answers a request and far
too few for one that works through a corpus a document at a time — that one
spends a step on each, and only the workflow knows how many there will be.

Nullable on purpose, and the three states mean different things: NULL leaves it
to the runtime's default, a number sets the budget, and 0 removes the guard
altogether for a deployment that would rather have a run that never stops than
one that stops early.

Existing workflows get NULL, so nothing changes for them until somebody decides
otherwise.

Revision ID: 0006
Revises: 0005
Create Date: 2026-08-27
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
    op.add_column("workflows", sa.Column("max_hops", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("workflows", "max_hops")
