# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from nlght.core.entry.context import RequestContext
    from nlght.core.workspace.workspace import WorkspaceContext
    from nlght.ports.outbound.store_coordinator import StoreCoordinator


@runtime_checkable
class ToolContract(Protocol):
    """A resolvable, executable tool contract.

    Implementations hold an instance + method name and forward
    ``execute()`` to the corresponding ToolBase method.
    """

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def parameters(self) -> list[Any]: ...

    @property
    def terminal(self) -> bool:
        """True if a call to this tool ends the model turn."""
        ...

    async def execute(self, **kwargs: object) -> object: ...


@runtime_checkable
class ToolCatalog(Protocol):
    """Request-scoped lookup for active tool contracts.

    Built by the executor (from DB resources + ToolLoader) and injected
    into ``WorkflowStepContext.tools`` once the catalog wiring is
    implemented.

    Steps access tools exclusively through this port —
    never directly on ToolBase instances.
    """

    def get(self, name: str) -> ToolContract: ...

    def get_or_none(self, name: str) -> ToolContract | None: ...

    def all(self) -> list[ToolContract]: ...

    def names(self) -> list[str]: ...

    async def execute(self, tc: dict[str, Any]) -> str:
        """Executes a tool call and returns the result as a string.

        ``tc`` has the canonical form ``{"name": str, "input": dict}``.
        Returns an error string instead of raising an exception —
        timeouts must be handled by the caller via ``asyncio.wait_for``.
        """
        ...


@runtime_checkable
class ToolCatalogBuilder(Protocol):
    """Builds a request-scoped ToolCatalog.

    Extension point: wire a custom implementation via
    StepMachineWorkflowExecutor(tool_catalog_builder=...). The executor
    depends on this Protocol, not on any concrete builder class.
    """

    async def build(
        self,
        model: str | None = None,
        caller: RequestContext | None = None,
        store_coordinator: StoreCoordinator | None = None,
        workspace: WorkspaceContext | None = None,
    ) -> ToolCatalog: ...
