# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nlght.core.trigger.trigger import Trigger

BLOCKING = "blocking"
NON_BLOCKING = "non-blocking"
CONCURRENCY_MODES = (BLOCKING, NON_BLOCKING)


@dataclass(slots=True, frozen=True)
class WorkflowStepDef:
    step_id: uuid.UUID
    position: int
    name: str
    type: str
    enabled: bool
    config: dict[str, Any]
    transitions: dict[str, Any]
    is_start: bool
    is_terminal: bool
    is_resume: bool


@dataclass(slots=True, frozen=True)
class WorkflowVersionDef:
    version_id: uuid.UUID
    workflow_id: uuid.UUID
    version: int
    status: str
    steps: list[WorkflowStepDef] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class WorkflowDef:
    workflow_id: uuid.UUID
    name: str
    enabled: bool
    capabilities: list[str]
    description: str | None = None
    # "blocking" runs at most one execution of this workflow at a time;
    # "non-blocking" allows any number in parallel. Enforced at claim.
    concurrency: str = NON_BLOCKING
    #: How many steps one run of this workflow may take. ``None`` means the
    #: runtime's default; ``0`` means no guard at all. A workflow that works
    #: through a corpus one document per step needs a number that fits its
    #: corpus, and is the only thing that can know it — which is why this is
    #: per workflow rather than a constant somewhere.
    max_hops: int | None = None

    @property
    def is_blocking(self) -> bool:
        return self.concurrency == BLOCKING


@dataclass(slots=True, frozen=True)
class WorkflowInvocation:
    trigger: Trigger
    workflow: WorkflowDef
    version: WorkflowVersionDef
    execution_id: uuid.UUID | None = None
    """The durable execution this invocation belongs to, when there is one.

    A worker running a claimed execution knows it; a gateway running a workflow
    inline in the request does not. A step that fans work out to other workers
    needs it — to parent its children and to own the artifacts it hands them —
    and must refuse rather than degrade when it is absent."""
