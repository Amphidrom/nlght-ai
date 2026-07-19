# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from nlght.core.entry.context import RequestContext
    from nlght.core.playbooks.playbook import ActivePlaybook, PlaybookDefinition
    from nlght.ports.outbound.tool_catalog import ToolCatalog


@runtime_checkable
class PlaybookCatalog(Protocol):
    """Request-scoped read-only playbook catalog.

    Built once per workflow invocation by the ``PlaybookCatalogBuilder``
    and then stored immutably in ``WorkflowStepContext.playbooks``. Steps
    call ``to_markdown()`` and place the result into the system prompt —
    they do not select playbooks manually.
    """

    def definitions(self) -> dict[str, PlaybookDefinition]: ...

    def get(self, name: str) -> ActivePlaybook | None: ...

    def all(self) -> list[ActivePlaybook]: ...

    def names(self) -> list[str]: ...

    def to_markdown(self) -> str:
        """All active playbooks as a single coherent Markdown document."""
        ...

    def playbook_markdown(self, name: str) -> str:
        """Markdown for a single playbook. Empty string if not present."""
        ...

    def to_contract(self) -> str:
        """Compact contract listing — one entry per playbook with activation criteria.

        Used by ``SystemPromptBuilder`` for the "Playbook Routing" section
        (when no playbook is active yet). Empty string if no playbooks are available.
        """
        ...

    def __len__(self) -> int: ...

    def __bool__(self) -> bool: ...


@runtime_checkable
class PlaybookCatalogBuilder(Protocol):
    """Builds a request-scoped PlaybookCatalog.

    No built-in implementation ships by default — this is a pure
    extension point. Wire a custom implementation via
    StepMachineWorkflowExecutor(playbook_catalog_builder=...). The
    executor depends on this Protocol, not on any concrete builder class.
    """

    async def build(
        self,
        tool_catalog: ToolCatalog | None,
        model: str | None = None,
        caller: RequestContext | None = None,
    ) -> PlaybookCatalog: ...
