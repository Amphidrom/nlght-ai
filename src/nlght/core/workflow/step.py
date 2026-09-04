# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from nlght.core.model.messages import CanonicalMessage
from nlght.core.trigger.trigger import Trigger

if TYPE_CHECKING:
    from nlght.core.workspace.workspace import WorkspaceContext
    from nlght.ports.outbound.execution_dispatcher import ExecutionDispatcher
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
    messages: list[CanonicalMessage]
    stream: bool
    emitter: SignalEmitter
    llm: ModelClient | None = None
    tools: ToolCatalog | None = None
    playbooks: PlaybookCatalog | None = None
    workspace: WorkspaceContext | None = None
    store_coordinator: StoreCoordinator | None = None
    os_runtime: OsRuntime | None = None
    metering: MeteringPort | None = None
    execution_id: uuid.UUID | None = None
    """This run's durable execution, when it has one (see ``WorkflowInvocation``)."""
    dispatcher: ExecutionDispatcher | None = None
    """Submits further executions — the seam a step uses to spread work across
    workers instead of looping over it here."""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    ctx: WorkflowStepContext
    verdict: str | None = None


@dataclass(frozen=True)
class StepOption:
    """One configuration key a step understands.

    Declaring options lets the admin UI render real inputs instead of asking an
    operator to hand-write JSON from memory. A step that declares none keeps the
    free-form JSON editor, so existing steps are unaffected.
    """

    name: str
    type: str  # "string" | "number" | "integer" | "boolean" | "object" | "array"
    description: str = ""
    required: bool = False
    default: Any = None
    choices: list[Any] | None = None
    placeholder: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("step option name must not be empty")
        if self.type not in ("string", "number", "integer", "boolean", "object", "array"):
            raise ValueError(f"unsupported step option type '{self.type}'")


class StepBase:
    """Base class for all workflow steps.

    ``config`` is the step's JSON configuration from its workflow version.
    Additional keyword arguments are runtime dependencies supplied by a bound
    ``StepLoader`` (a repository, an embedding client, …); a step that does not
    need them ignores them, exactly as ``ToolBase`` does.
    """

    TYPE: ClassVar[str]

    def __init__(self, *, config: dict[str, Any], **_: object) -> None:
        self.config = config

    @classmethod
    def options(cls) -> list[StepOption]:
        """The configuration keys this step understands.

        Override to make a step self-describing: the admin UI renders these as
        typed inputs with descriptions and defaults. The default is empty, which
        means "free-form JSON" and preserves existing behaviour.
        """
        return []

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        raise NotImplementedError
