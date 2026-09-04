# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    TIMESTAMP,
    BigInteger,
    Boolean,
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
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

    workflow_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    capabilities: Mapped[list[str]] = mapped_column(_JSON, nullable=False, default=list)
    # "blocking" | "non-blocking" — one run at a time vs any number in parallel.
    concurrency: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("'non-blocking'"))
    # How many steps one run may take. NULL leaves it to the runtime's default;
    # 0 removes the guard. Per workflow because only the workflow knows its own
    # shape — one that spends a step per document needs a number that fits its
    # corpus, and one that answers a request needs twenty.
    max_hops: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[object] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[object] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)

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

    workflow_version_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workflow_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("workflows.workflow_id", ondelete="CASCADE"),
        nullable=False,
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="draft")
    created_at: Mapped[object] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[object] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)

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

    workflow_step_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
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

    created_at: Mapped[object] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[object] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)

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
    """One configured resource.

    ``resource_id`` is the identity; ``(kind, name)`` is the address, and it is
    unique because everything resolves a resource by it and an access rule names
    it. ``provider`` selects the implementation and is deliberately not part of
    the address — see migration ``0007``.
    """

    __tablename__ = "resources"
    __table_args__ = (UniqueConstraint("kind", "name", name="uq_resource_address"),)

    resource_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    provider: Mapped[str] = mapped_column(Text, nullable=False)
    config: Mapped[dict[str, Any]] = mapped_column(_JSON, nullable=False, default=dict[str, Any])
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[object] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[object] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)


Index("resources_by_kind", Resource.kind)
Index("resources_enabled_idx", Resource.enabled)


# ============================================================
# Access policies
# ============================================================


class AccessPolicyRule(Base):
    __tablename__ = "access_policies"

    rule_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    subject_type: Mapped[str] = mapped_column(Text, nullable=False)
    subject: Mapped[str] = mapped_column(Text, nullable=False)
    effect: Mapped[str] = mapped_column(Text, nullable=False, default="allow")
    conditions: Mapped[dict[str, Any]] = mapped_column(_JSON, nullable=False, default=dict[str, Any])
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[object] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[object] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)


Index("access_policies_by_subject_type", AccessPolicyRule.subject_type)
Index("access_policies_enabled_idx", AccessPolicyRule.enabled)


# ============================================================
# Distributed workflow execution (ADR-0031)
# ============================================================


class WorkflowWorker(Base):
    __tablename__ = "workflow_workers"
    __table_args__ = (
        CheckConstraint("capacity > 0", name="ck_workflow_workers_capacity"),
        CheckConstraint(
            "status IN ('active', 'stopping', 'offline')",
            name="ck_workflow_workers_status",
        ),
    )

    worker_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    boot_token: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, unique=True)
    instance_name: Mapped[str] = mapped_column(Text, nullable=False)
    capacity: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="active")
    worker_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", _JSON, nullable=False, default=dict[str, Any])
    heartbeat_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)


Index("workflow_workers_by_expiry", WorkflowWorker.status, WorkflowWorker.expires_at)


class WorkflowWorkerCapability(Base):
    __tablename__ = "workflow_worker_capabilities"

    worker_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("workflow_workers.worker_id", ondelete="CASCADE"),
        primary_key=True,
    )
    capability: Mapped[str] = mapped_column(Text, primary_key=True)


Index("workflow_worker_capabilities_by_capability", WorkflowWorkerCapability.capability)


class WorkflowExecution(Base):
    __tablename__ = "workflow_executions"
    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'leased', 'running', 'succeeded', 'failed', 'cancelled')",
            name="ck_workflow_executions_status",
        ),
        CheckConstraint("max_attempts > 0", name="ck_workflow_executions_max_attempts"),
        CheckConstraint("attempt_count >= 0", name="ck_workflow_executions_attempt_count"),
        CheckConstraint("fencing_token >= 0", name="ck_workflow_executions_fencing_token"),
    )

    execution_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workflow_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("workflows.workflow_id", ondelete="RESTRICT"),
        nullable=False,
    )
    workflow_version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("workflow_versions.workflow_version_id", ondelete="RESTRICT"),
        nullable=False,
    )
    # A fanned-out child points at the execution that submitted it. Self
    # referencing because a child *is* an execution: it claims, leases, fences,
    # and retries exactly like any other, and only its provenance differs.
    parent_execution_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("workflow_executions.execution_id", ondelete="SET NULL"),
    )
    idempotency_key: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    trigger: Mapped[dict[str, Any]] = mapped_column(_JSON, nullable=False)
    source_config_revision: Mapped[str | None] = mapped_column(Text)
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    artifact_refs: Mapped[list[str]] = mapped_column(_JSON, nullable=False, default=list)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    # Snapshot of the workflow's concurrency policy: when true, at most one
    # execution of this workflow is claimable at a time.
    exclusive: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    fencing_token: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="queued")
    owner_worker_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("workflow_workers.worker_id", ondelete="SET NULL"))
    owner_boot_token: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    lease_expires_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    next_attempt_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    progress: Mapped[dict[str, Any]] = mapped_column(_JSON, nullable=False, default=dict[str, Any])
    result: Mapped[dict[str, Any] | None] = mapped_column(_JSON)
    diagnostics: Mapped[dict[str, Any] | None] = mapped_column(_JSON)
    started_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)


Index(
    "workflow_executions_claim_idx",
    WorkflowExecution.status,
    WorkflowExecution.next_attempt_at,
    WorkflowExecution.priority,
    WorkflowExecution.created_at,
)
Index("workflow_executions_by_lease", WorkflowExecution.status, WorkflowExecution.lease_expires_at)
Index("workflow_executions_by_workflow", WorkflowExecution.workflow_id, WorkflowExecution.created_at)
# Answers "how far are this run's children?" as one aggregate over the index,
# without touching the children's rows.
Index(
    "workflow_executions_by_parent",
    WorkflowExecution.parent_execution_id,
    WorkflowExecution.status,
)


class WorkflowExecutionRequirement(Base):
    __tablename__ = "workflow_execution_requirements"

    execution_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("workflow_executions.execution_id", ondelete="CASCADE"),
        primary_key=True,
    )
    capability: Mapped[str] = mapped_column(Text, primary_key=True)


Index("workflow_execution_requirements_by_capability", WorkflowExecutionRequirement.capability)


class WorkflowExecutionAttempt(Base):
    __tablename__ = "workflow_execution_attempts"
    __table_args__ = (
        UniqueConstraint("execution_id", "attempt_number", name="uq_execution_attempt_number"),
        UniqueConstraint("execution_id", "fencing_token", name="uq_execution_attempt_fencing"),
        CheckConstraint(
            "status IN ('leased', 'running', 'succeeded', 'failed', 'cancelled', 'expired')",
            name="ck_workflow_execution_attempts_status",
        ),
    )

    attempt_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    execution_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("workflow_executions.execution_id", ondelete="CASCADE"),
        nullable=False,
    )
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False)
    worker_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("workflow_workers.worker_id", ondelete="RESTRICT"),
        nullable=False,
    )
    boot_token: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    fencing_token: Mapped[int] = mapped_column(BigInteger, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="leased")
    lease_expires_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    heartbeat_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    result: Mapped[dict[str, Any] | None] = mapped_column(_JSON)
    diagnostics: Mapped[dict[str, Any] | None] = mapped_column(_JSON)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)


Index("workflow_execution_attempts_by_execution", WorkflowExecutionAttempt.execution_id)
Index("workflow_execution_attempts_by_lease", WorkflowExecutionAttempt.status, WorkflowExecutionAttempt.lease_expires_at)


class WorkflowExecutionArtifact(Base):
    __tablename__ = "workflow_execution_artifacts"
    __table_args__ = (UniqueConstraint("execution_id", "name", name="uq_execution_artifact_name"),)

    artifact_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    execution_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("workflow_executions.execution_id", ondelete="CASCADE"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    content_type: Mapped[str] = mapped_column(Text, nullable=False)
    checksum: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    artifact_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", _JSON, nullable=False, default=dict[str, Any])
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)


Index("workflow_execution_artifacts_by_execution", WorkflowExecutionArtifact.execution_id)
