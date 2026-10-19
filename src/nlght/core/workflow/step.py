# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from nlght.core.model.messages import CanonicalMessage
from nlght.core.model.trust import ContentLedger
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
    ``emitter``. They do not write response fields and have no knowledge
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
    """Submits further executions. This lets a step spread work across
    workers instead of looping over it here."""
    metadata: dict[str, Any] = field(default_factory=dict)
    ledger: ContentLedger = field(default_factory=ContentLedger)
    """Everything this run's model context has taken in, and where it came from.

    Monotonic for the run and shared with ``tools``: a step that trims its
    messages does not make content it already read disappear from the next
    decision (ADR-0074). Pass ``origin_of(messages)`` as ``context_origin``
    when a step executes a tool itself."""


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
    ``StepLoader`` (for example, a repository or embedding client). A step that does not
    need them ignores them, exactly as ``ToolBase`` does.
    """

    TYPE: ClassVar[str]

    class Verdict:
        """Verdicts shared by every workflow step."""

        DEFAULT = "DEFAULT"
        DONE = "done"
        FAILED = "failed"

    @classmethod
    def verdicts(cls, config: Mapping[str, Any] | None = None) -> tuple[str, ...]:
        """List verdicts available from this step before it runs.

        Args:
            config (Mapping[str, Any] | None): Persisted step configuration.

        Returns:
            tuple[str, ...]: Common and step-specific verdict values in declaration order.
        """
        values: list[str] = []
        for owner in reversed(cls.Verdict.__mro__):
            for name, value in vars(owner).items():
                if name.isupper() and isinstance(value, str) and value not in values:
                    values.append(value)
        return tuple(values)

    def __init__(self, *, config: dict[str, Any], **_: object) -> None:
        self.config = config

    @classmethod
    def options(cls) -> list[StepOption]:
        """Declare typed configuration options for a step.

        Returns:
            list[StepOption]: Empty option list until a subclass overrides it.
        """
        return []

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        raise NotImplementedError
