# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import logging
from typing import Any

from nlght.core.errors.errors import ToolActionSemanticsError, WorkflowConfigurationError
from nlght.core.tools.tool import ToolBase

logger = logging.getLogger(__name__)


class ToolLoader:
    """Registry that maps tool types onto ToolBase classes.

    The lookup key is ``(kind, provider)``. If no exact match is found,
    ``(kind, "")`` is tried as a fallback — this allows backward-compatible
    deployments when the DB provider value doesn't yet match the class
    PROVIDER.

    New tool types are registered via ``register()``. Each class should set
    both ``KIND`` **and** ``PROVIDER``::

        class CustomTool(ToolBase):
            KIND     = "custom"
            PROVIDER = "elasticsearch+mysql"

    Register before server start::

        from nlght.adapters.outbound.tools.registry import tool_registry
        from my_app.tools import MyTool

        tool_registry.register(MyTool)
    """

    def __init__(self) -> None:
        self._registry: dict[tuple[str, str], type[ToolBase]] = {}

    def register(self, tool_cls: type[ToolBase]) -> None:
        provider = getattr(tool_cls, "PROVIDER", "")
        for signature in tool_cls.signatures():
            if signature.action is None:
                raise ToolActionSemanticsError(
                    f"{tool_cls.KIND}/{provider or 'default'}::{signature.name}"
                )
        key = (tool_cls.KIND, provider)
        self._registry[key] = tool_cls
        logger.debug("tool.registered | kind=%s provider=%s", tool_cls.KIND, provider or "(default)")

    def instantiate(
        self,
        *,
        kind: str,
        provider: str = "",
        name: str,
        config: dict[str, Any],
        # Dynamic construction seam: forwarded verbatim into the tool class
        # constructor, whose keyword surface varies per tool (os_runtime,
        # store_coordinator, workspace, ...). `object` would fail to unpack
        # against the typed keywords.
        **runtime_deps: Any,  # noqa: ANN401
    ) -> ToolBase:
        cls = self.implementation_for(kind=kind, provider=provider)
        if cls is None:
            raise WorkflowConfigurationError(
                f"Unknown tool kind='{kind}' provider='{provider}'. "
                f"Registered: {sorted(self._registry)}"
            )
        logger.debug("tool.instantiate | kind=%s provider=%s name=%s", kind, provider or "(default)", name)
        return cls(name=name, config=config, **runtime_deps)

    def implementation_for(self, *, kind: str, provider: str = "") -> type[ToolBase] | None:
        """The registered class a resource of this kind and provider would build.

        Read-only, and the reason it exists: a caller that wants to know what a
        resource *offers* — its signatures are a classmethod — should not have
        to construct one to find out, and should not have to rebuild the
        provider fallback below to look the class up either. One place decides
        which implementation an address resolves to.
        """
        return self._registry.get((kind, provider)) or self._registry.get((kind, ""))

    def registered_kinds(self) -> list[str]:
        return sorted({kind for kind, _ in self._registry})

    def __contains__(self, kind: str) -> bool:
        return any(k == kind for k, _ in self._registry)
