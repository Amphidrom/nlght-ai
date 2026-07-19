# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from nlght.core.trigger.trigger import Trigger

if TYPE_CHECKING:
    from nlght.core.workspace.workspace import WorkspaceContext
    from nlght.ports.outbound.metering import MeteringPort
    from nlght.ports.outbound.model_client import ModelClient
    from nlght.ports.outbound.os_runtime import OsRuntime
    from nlght.ports.outbound.playbook_catalog import PlaybookCatalog
    from nlght.ports.outbound.signal_emitter import SignalEmitter
    from nlght.ports.outbound.store_coordinator import StoreCoordinator
    from nlght.ports.outbound.tool_catalog import ToolCatalog


@dataclass
class WorkflowStepContext:
    """Mutable execution context passed between steps in the step machine.

    Steps read execution state from this context and emit output via
    ``emitter`` — they do not write response fields and have no knowledge
    of the transport layer (HTTP, SSE, JSON).

    The executor creates the appropriate ``SignalEmitter`` implementation
    (buffering or streaming) before the step machine starts and injects it
    here.  See ADR-0008.
    """

    correlation_id: str
    trigger: Trigger
    model: str
    messages: list[dict[str, Any]]
    stream: bool
    emitter: SignalEmitter
    llm: ModelClient | None = None
    tools: ToolCatalog | None = None
    playbooks: PlaybookCatalog | None = None
    workspace: WorkspaceContext | None = None
    store_coordinator: StoreCoordinator | None = None
    os_runtime: OsRuntime | None = None
    metering: MeteringPort | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    ctx: WorkflowStepContext
    verdict: str | None = None


class StepBase:
    """Base class for all workflow steps."""

    TYPE: ClassVar[str]

    def __init__(self, *, config: dict[str, Any]) -> None:
        self.config = config

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        raise NotImplementedError
