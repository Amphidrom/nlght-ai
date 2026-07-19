"""Initial schema — workflows, workflow_versions, workflow_steps, resources.

Revision ID: 0001
Revises:
Create Date: 2026-04-11
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # workflows
    # ------------------------------------------------------------------
    op.create_table(
        "workflows",
        sa.Column("workflow_id",  postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name",         sa.Text(),    nullable=False, unique=True),
        sa.Column("description",  sa.Text()),
        sa.Column("enabled",      sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("capabilities", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("created_at",   sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at",   sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("workflows_enabled_idx", "workflows", ["enabled"])

    # ------------------------------------------------------------------
    # workflow_versions
    # ------------------------------------------------------------------
    op.create_table(
        "workflow_versions",
        sa.Column("workflow_version_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("workflow_id",         postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("workflows.workflow_id", ondelete="CASCADE"), nullable=False),
        sa.Column("version",    sa.Integer(), nullable=False),
        sa.Column("status",     sa.Text(),    nullable=False, server_default=sa.text("'draft'")),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint("workflow_id", "version", name="uq_workflow_versions_wf_version"),
    )
    # Partial unique index: only one active version per workflow
    op.create_index(
        "workflow_versions_one_active",
        "workflow_versions",
        ["workflow_id"],
        unique=True,
        postgresql_where=sa.text("status = 'active'"),
    )

    # ------------------------------------------------------------------
    # workflow_steps
    # ------------------------------------------------------------------
    op.create_table(
        "workflow_steps",
        sa.Column("workflow_step_id",     postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("workflow_version_id",  postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("workflow_versions.workflow_version_id", ondelete="CASCADE"), nullable=False),
        sa.Column("position",     sa.Integer(), nullable=False),
        sa.Column("name",         sa.Text(),    nullable=False),
        sa.Column("type",         sa.Text(),    nullable=False),
        sa.Column("enabled",      sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("config",       postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("transitions",  postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("is_terminal",  sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("is_start",     sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("is_resume",    sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at",   sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at",   sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint("workflow_version_id", "position", name="uq_workflow_steps_version_position"),
        sa.UniqueConstraint("workflow_version_id", "name",     name="uq_workflow_steps_version_name"),
    )
    op.create_index("workflow_steps_by_version", "workflow_steps", ["workflow_version_id", "position"])
    op.create_index("workflow_steps_by_type",    "workflow_steps", ["type"])
    op.create_index(
        "workflow_steps_by_start",
        "workflow_steps",
        ["workflow_version_id"],
        postgresql_where=sa.text("is_start = true"),
    )

    # ------------------------------------------------------------------
    # resources
    # ------------------------------------------------------------------
    op.create_table(
        "resources",
        sa.Column("resource_id",     postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name",            sa.Text(), nullable=False),
        sa.Column("kind",            sa.Text(), nullable=False),
        sa.Column("provider",        sa.Text(), nullable=False),
        sa.Column("allowed_models",  postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("config",          postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("enabled",         sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at",      sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at",      sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("resources_by_kind",    "resources", ["kind"])
    op.create_index("resources_enabled_idx", "resources", ["enabled"])


def downgrade() -> None:
    op.drop_table("resources")
    op.drop_table("workflow_steps")
    op.drop_table("workflow_versions")
    op.drop_table("workflows")
