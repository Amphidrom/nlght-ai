# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import (
    JSON,
    TIMESTAMP,
    Boolean,
    ForeignKey,
    Index,
    Integer,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Renders as JSONB on PostgreSQL, plain JSON on other dialects (e.g. SQLite in tests).
_JSON = JSON().with_variant(JSONB(), "postgresql")


class Base(DeclarativeBase):
    pass


# ============================================================
# Workflows
# ============================================================

class Workflow(Base):
    __tablename__ = "workflows"

    workflow_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    capabilities: Mapped[list[str]] = mapped_column(_JSON, nullable=False, default=list)
    created_at: Mapped[object] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[object] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )

    versions: Mapped[list[WorkflowVersion]] = relationship(
        back_populates="workflow",
        cascade="all, delete-orphan",
    )


Index("workflows_enabled_idx", Workflow.enabled)


# ============================================================
# Workflow Versions
# ============================================================

class WorkflowVersion(Base):
    __tablename__ = "workflow_versions"
    __table_args__ = (
        UniqueConstraint("workflow_id", "version"),
        Index(
            "workflow_versions_one_active",
            "workflow_id",
            unique=True,
            postgresql_where=(text("status = 'active'")),
        ),
    )

    workflow_version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    workflow_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("workflows.workflow_id", ondelete="CASCADE"),
        nullable=False,
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="draft")
    created_at: Mapped[object] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[object] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )

    workflow: Mapped[Workflow] = relationship(back_populates="versions")
    steps: Mapped[list[WorkflowStep]] = relationship(
        back_populates="workflow_version",
        cascade="all, delete-orphan",
        order_by="WorkflowStep.position",
    )


# ============================================================
# Workflow Steps  —  State Machine Nodes
# ============================================================

class WorkflowStep(Base):
    __tablename__ = "workflow_steps"
    __table_args__ = (
        UniqueConstraint("workflow_version_id", "position"),
        UniqueConstraint("workflow_version_id", "name"),
    )

    workflow_step_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    workflow_version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("workflow_versions.workflow_version_id", ondelete="CASCADE"),
        nullable=False,
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    type: Mapped[str] = mapped_column(Text, nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    config: Mapped[dict[str, Any]] = mapped_column(_JSON, nullable=False, default=dict[str, Any])

    # State machine transitions (verdict → next step name)
    transitions: Mapped[dict[str, Any]] = mapped_column(_JSON, nullable=False, default=dict[str, Any])
    is_terminal: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_start: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_resume: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    created_at: Mapped[object] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[object] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )

    workflow_version: Mapped[WorkflowVersion] = relationship(back_populates="steps")


Index("workflow_steps_by_version", WorkflowStep.workflow_version_id, WorkflowStep.position)
Index("workflow_steps_by_type", WorkflowStep.type)
Index(
    "workflow_steps_by_start",
    WorkflowStep.workflow_version_id,
    postgresql_where=(WorkflowStep.is_start.is_(True)),
)


# ============================================================
# Resources
# ============================================================

class Resource(Base):
    __tablename__ = "resources"

    resource_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    provider: Mapped[str] = mapped_column(Text, nullable=False)
    config: Mapped[dict[str, Any]] = mapped_column(_JSON, nullable=False, default=dict[str, Any])
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[object] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[object] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )


Index("resources_by_kind", Resource.kind)
Index("resources_enabled_idx", Resource.enabled)


# ============================================================
# Access policies
# ============================================================

class AccessPolicyRule(Base):
    __tablename__ = "access_policies"

    rule_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    subject_type: Mapped[str] = mapped_column(Text, nullable=False)
    subject: Mapped[str] = mapped_column(Text, nullable=False)
    effect: Mapped[str] = mapped_column(Text, nullable=False, default="allow")
    conditions: Mapped[dict[str, Any]] = mapped_column(_JSON, nullable=False, default=dict[str, Any])
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[object] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[object] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )


Index("access_policies_by_subject_type", AccessPolicyRule.subject_type)
Index("access_policies_enabled_idx", AccessPolicyRule.enabled)
