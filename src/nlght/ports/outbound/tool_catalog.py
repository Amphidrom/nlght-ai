# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from nlght.core.tools.action import (
    ActionSemanticsResolver,
    ExecutionCapabilities,
    JsonValue,
    ToolArgumentBinder,
)

if TYPE_CHECKING:
    from nlght.core.entry.context import RequestContext
    from nlght.core.model.trust import ContentLedger, ContentOrigin
    from nlght.core.tools.tool import ToolOutput
    from nlght.core.workspace.workspace import WorkspaceContext
    from nlght.ports.outbound.store_coordinator import StoreCoordinator


@runtime_checkable
class ToolContract(Protocol):
    """A resolvable tool contract invoked only by the authorization gate.

    Implementations hold an instance + method name.  Their private
    ``_execute_bound()`` hook is called only after the catalog has validated,
    bound and authorized the concrete action.
    """

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def parameters(self) -> list[Any]: ...

    @property
    def terminal(self) -> bool:
        """Indicate whether this tool ends the current model turn.

        Returns:
            bool: Terminal-call flag.
        """
        ...

    @property
    def action(self) -> ActionSemanticsResolver | None: ...

    @property
    def argument_binder(self) -> ToolArgumentBinder | None: ...

    @property
    def resource_address(self) -> str: ...

    @property
    def resource_config(self) -> Mapping[str, Any]: ...

    @property
    def runtime_capabilities(self) -> ExecutionCapabilities: ...

    async def _execute_bound(self, arguments: Mapping[str, JsonValue]) -> object: ...


@runtime_checkable
class ToolCatalog(Protocol):
    """Request-scoped lookup and authorization for active tool contracts.

    The executor builds the catalog from resources and injects it into
    `WorkflowStepContext.tools`. Steps access tools through this port.
    """

    def get(self, name: str) -> ToolContract: ...

    def get_or_none(self, name: str) -> ToolContract | None: ...

    def all(self) -> list[ToolContract]: ...

    def names(self) -> list[str]: ...

    @property
    def ledger(self) -> ContentLedger:
        """Expose content origins read during the workflow run.

        Returns:
            ContentLedger: Monotonic content ledger.
        """
        ...

    async def execute_with_origin(
        self,
        tc: dict[str, Any],
        *,
        context_origin: ContentOrigin | None = None,
    ) -> ToolOutput:
        """Authorize and execute a tool call with provenance.

        Args:
            tc (dict[str, Any]): Tool call to authorize and execute.
            context_origin (ContentOrigin | None): Optional origin of model-read context.

        Returns:
            ToolOutput: Tool result with recorded origin, including error output.

        Side Effects:
            Updates the ledger and may perform the authorized action.
        """
        ...

    async def execute(
        self,
        tc: dict[str, Any],
        *,
        context_origin: ContentOrigin | None = None,
    ) -> str:
        """Execute a tool call and return its text without provenance.

        Args:
            tc (dict[str, Any]): Tool call to authorize and execute.
            context_origin (ContentOrigin | None): Optional origin of model-read context.

        Returns:
            str: Result text, treated as unannotated if reused as context.

        Side Effects:
            Updates the ledger and may perform the authorized action.
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
        session_id: str | None = None,
    ) -> ToolCatalog: ...
