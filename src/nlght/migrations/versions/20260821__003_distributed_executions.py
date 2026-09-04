# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Distributed workflow executions, workers, attempts, requirements, and artifacts.

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-21
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "workflow_workers",
        sa.Column("worker_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("boot_token", postgresql.UUID(as_uuid=True), nullable=False, unique=True),
        sa.Column("instance_name", sa.Text(), nullable=False),
        sa.Column("capacity", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("status", sa.Text(), nullable=False, server_default=sa.text("'active'")),
        sa.Column("metadata", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("heartbeat_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("expires_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.CheckConstraint("capacity > 0", name="ck_workflow_workers_capacity"),
        sa.CheckConstraint(
            "status IN ('active', 'stopping', 'offline')",
            name="ck_workflow_workers_status",
        ),
    )
    op.create_index("workflow_workers_by_expiry", "workflow_workers", ["status", "expires_at"])

    op.create_table(
        "workflow_worker_capabilities",
        sa.Column(
            "worker_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("workflow_workers.worker_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("capability", sa.Text(), primary_key=True),
    )
    op.create_index(
        "workflow_worker_capabilities_by_capability",
        "workflow_worker_capabilities",
        ["capability"],
    )

    op.create_table(
        "workflow_executions",
        sa.Column("execution_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "workflow_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("workflows.workflow_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "workflow_version_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("workflow_versions.workflow_version_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("idempotency_key", sa.Text(), nullable=False, unique=True),
        sa.Column("trigger", postgresql.JSONB(), nullable=False),
        sa.Column("source_config_revision", sa.Text()),
        sa.Column("priority", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("artifact_refs", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("max_attempts", sa.Integer(), nullable=False, server_default="3"),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("fencing_token", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("status", sa.Text(), nullable=False, server_default=sa.text("'queued'")),
        sa.Column(
            "owner_worker_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("workflow_workers.worker_id", ondelete="SET NULL"),
        ),
        sa.Column("owner_boot_token", postgresql.UUID(as_uuid=True)),
        sa.Column("lease_expires_at", sa.TIMESTAMP(timezone=True)),
        sa.Column("next_attempt_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("cancel_requested", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("progress", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("result", postgresql.JSONB()),
        sa.Column("diagnostics", postgresql.JSONB()),
        sa.Column("started_at", sa.TIMESTAMP(timezone=True)),
        sa.Column("completed_at", sa.TIMESTAMP(timezone=True)),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.CheckConstraint(
            "status IN ('queued', 'leased', 'running', 'succeeded', 'failed', 'cancelled')",
            name="ck_workflow_executions_status",
        ),
        sa.CheckConstraint("max_attempts > 0", name="ck_workflow_executions_max_attempts"),
        sa.CheckConstraint("attempt_count >= 0", name="ck_workflow_executions_attempt_count"),
        sa.CheckConstraint("fencing_token >= 0", name="ck_workflow_executions_fencing_token"),
    )
    op.create_index(
        "workflow_executions_claim_idx",
        "workflow_executions",
        ["status", "next_attempt_at", "priority", "created_at"],
    )
    op.create_index(
        "workflow_executions_by_lease",
        "workflow_executions",
        ["status", "lease_expires_at"],
    )
    op.create_index(
        "workflow_executions_by_workflow",
        "workflow_executions",
        ["workflow_id", "created_at"],
    )

    op.create_table(
        "workflow_execution_requirements",
        sa.Column(
            "execution_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("workflow_executions.execution_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("capability", sa.Text(), primary_key=True),
    )
    op.create_index(
        "workflow_execution_requirements_by_capability",
        "workflow_execution_requirements",
        ["capability"],
    )

    op.create_table(
        "workflow_execution_attempts",
        sa.Column("attempt_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "execution_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("workflow_executions.execution_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("attempt_number", sa.Integer(), nullable=False),
        sa.Column(
            "worker_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("workflow_workers.worker_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("boot_token", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("fencing_token", sa.BigInteger(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False, server_default=sa.text("'leased'")),
        sa.Column("lease_expires_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("heartbeat_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("started_at", sa.TIMESTAMP(timezone=True)),
        sa.Column("completed_at", sa.TIMESTAMP(timezone=True)),
        sa.Column("result", postgresql.JSONB()),
        sa.Column("diagnostics", postgresql.JSONB()),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint("execution_id", "attempt_number", name="uq_execution_attempt_number"),
        sa.UniqueConstraint("execution_id", "fencing_token", name="uq_execution_attempt_fencing"),
        sa.CheckConstraint(
            "status IN ('leased', 'running', 'succeeded', 'failed', 'cancelled', 'expired')",
            name="ck_workflow_execution_attempts_status",
        ),
    )
    op.create_index(
        "workflow_execution_attempts_by_execution",
        "workflow_execution_attempts",
        ["execution_id"],
    )
    op.create_index(
        "workflow_execution_attempts_by_lease",
        "workflow_execution_attempts",
        ["status", "lease_expires_at"],
    )

    op.create_table(
        "workflow_execution_artifacts",
        sa.Column("artifact_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "execution_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("workflow_executions.execution_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("content_type", sa.Text(), nullable=False),
        sa.Column("checksum", sa.Text(), nullable=False),
        sa.Column("content", sa.LargeBinary(), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint("execution_id", "name", name="uq_execution_artifact_name"),
    )
    op.create_index(
        "workflow_execution_artifacts_by_execution",
        "workflow_execution_artifacts",
        ["execution_id"],
    )


def downgrade() -> None:
    op.drop_table("workflow_execution_artifacts")
    op.drop_table("workflow_execution_attempts")
    op.drop_table("workflow_execution_requirements")
    op.drop_table("workflow_executions")
    op.drop_table("workflow_worker_capabilities")
    op.drop_table("workflow_workers")
