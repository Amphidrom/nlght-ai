# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nlght.core.trigger.trigger import Trigger


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


@dataclass(slots=True, frozen=True)
class WorkflowInvocation:
    trigger: Trigger
    workflow: WorkflowDef
    version: WorkflowVersionDef
