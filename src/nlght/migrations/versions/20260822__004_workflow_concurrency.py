# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Per-workflow concurrency: workflows.concurrency and workflow_executions.exclusive.

A workflow is 'blocking' (one execution at a time) or 'non-blocking' (any number
in parallel). The policy is snapshotted onto each execution as ``exclusive`` and
enforced at claim: a blocking execution is only claimable when no sibling
execution of the same workflow is LEASED/RUNNING.

Revision ID: 0004
Revises: 0003
Create Date: 2026-08-22
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "workflows",
        sa.Column(
            "concurrency",
            sa.Text(),
            nullable=False,
            server_default=sa.text("'non-blocking'"),
        ),
    )
    op.add_column(
        "workflow_executions",
        sa.Column(
            "exclusive",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    # The claim's concurrency check looks for a sibling of the same workflow in a
    # LEASED/RUNNING state; index that lookup.
    op.create_index(
        "workflow_executions_by_workflow_status",
        "workflow_executions",
        ["workflow_id", "status"],
    )


def downgrade() -> None:
    op.drop_index(
        "workflow_executions_by_workflow_status",
        table_name="workflow_executions",
    )
    op.drop_column("workflow_executions", "exclusive")
    op.drop_column("workflows", "concurrency")
