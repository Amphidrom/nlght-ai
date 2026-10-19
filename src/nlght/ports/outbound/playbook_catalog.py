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
    may render its content as untrusted model context. The catalog grants no
    instruction authority to playbook text.
    """

    def definitions(self) -> dict[str, PlaybookDefinition]: ...

    def get(self, name: str) -> ActivePlaybook | None: ...

    def all(self) -> list[ActivePlaybook]: ...

    def names(self) -> list[str]: ...

    def to_markdown(self) -> str:
        """Render all active playbooks as one Markdown document.

        Returns:
            str: Combined Markdown text.
        """
        ...

    def playbook_markdown(self, name: str) -> str:
        """Render one active playbook as Markdown.

        Args:
            name (str): Playbook name.

        Returns:
            str: Markdown text, or empty text for an unknown name.
        """
        ...

    def to_contract(self) -> str:
        """Render playbook activation contracts for model context.

        Returns:
            str: Compact contract text, empty without available playbooks.
        """
        ...

    def __len__(self) -> int: ...

    def __bool__(self) -> bool: ...


@runtime_checkable
class PlaybookCatalogBuilder(Protocol):
    """Builds a request-scoped PlaybookCatalog.

    The built-in adapter constructs this catalog when playbooks are configured.
    A deployment can inject a custom implementation through
    StepMachineWorkflowExecutor(playbook_catalog_builder=...). The executor
    depends on this Protocol.
    """

    async def build(
        self,
        tool_catalog: ToolCatalog | None,
        model: str | None = None,
        caller: RequestContext | None = None,
    ) -> PlaybookCatalog: ...
