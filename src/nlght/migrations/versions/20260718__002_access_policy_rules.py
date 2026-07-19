"""Access-policy rules — add access_policies table, drop resources.allowed_models.

Revision ID: 0002
Revises: 0001
Create Date: 2026-07-18
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "access_policies",
        sa.Column("rule_id",      postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("subject_type", sa.Text(),    nullable=False),
        sa.Column("subject",      sa.Text(),    nullable=False),
        sa.Column("effect",       sa.Text(),    nullable=False, server_default=sa.text("'allow'")),
        sa.Column("conditions",   postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("priority",     sa.Integer(), nullable=False, server_default="0"),
        sa.Column("enabled",      sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at",   sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at",   sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.CheckConstraint("subject_type IN ('tool', 'model', 'playbook')", name="ck_access_policies_subject_type"),
        sa.CheckConstraint("effect IN ('allow', 'deny')", name="ck_access_policies_effect"),
    )
    op.create_index("access_policies_by_subject_type", "access_policies", ["subject_type"])
    op.create_index("access_policies_enabled_idx", "access_policies", ["enabled"])

    # allowed_models is replaced by access_policies rules. Existing entries are
    # intentionally not auto-converted — the mechanism never shipped in a
    # production release (pre-1.0, Experimental).
    op.drop_column("resources", "allowed_models")


def downgrade() -> None:
    op.add_column(
        "resources",
        sa.Column("allowed_models", postgresql.JSONB(), nullable=False, server_default="[]"),
    )
    op.drop_index("access_policies_enabled_idx", table_name="access_policies")
    op.drop_index("access_policies_by_subject_type", table_name="access_policies")
    op.drop_table("access_policies")
