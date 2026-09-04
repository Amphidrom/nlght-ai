# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Execution fan-out: workflow_executions.parent_execution_id.

A run that spreads its work across workers submits one child execution per item.
A child is an ordinary execution — claimed, leased, fenced, and retried like any
other — and this nullable self-reference is what keeps the work of one logical
run findable once it has been spread out.

ON DELETE SET NULL rather than CASCADE: a child's own record is durable evidence
of work that was performed, and losing a parent must not erase it.

Revision ID: 0005
Revises: 0004
Create Date: 2026-08-22
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "workflow_executions",
        sa.Column("parent_execution_id", sa.UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_workflow_executions_parent",
        "workflow_executions",
        "workflow_executions",
        ["parent_execution_id"],
        ["execution_id"],
        ondelete="SET NULL",
    )
    # "How far are this run's children?" is answered as one aggregate over this
    # index instead of a scan of the children's rows.
    op.create_index(
        "workflow_executions_by_parent",
        "workflow_executions",
        ["parent_execution_id", "status"],
    )


def downgrade() -> None:
    op.drop_index("workflow_executions_by_parent", table_name="workflow_executions")
    op.drop_constraint(
        "fk_workflow_executions_parent",
        "workflow_executions",
        type_="foreignkey",
    )
    op.drop_column("workflow_executions", "parent_execution_id")
