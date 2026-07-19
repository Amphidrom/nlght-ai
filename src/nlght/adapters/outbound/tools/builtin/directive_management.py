# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, ClassVar

from nlght.core.hive_mind.directives import KnownDirective
from nlght.core.tools.tool import ToolBase, ToolParameter, ToolSignature

if TYPE_CHECKING:
    from nlght.ports.outbound.store_coordinator import StoreCoordinator

logger = logging.getLogger(__name__)


class DirectiveManagementTool(ToolBase):
    """Built-in tool that lets an LLM set behavioral directives at runtime.

    Exposes a single operation ``directive_set`` that writes a known directive
    into the session's ``StoreCoordinator``.  Unknown keys are rejected so the
    LLM cannot invent arbitrary directives.

    Requires ``store_coordinator`` to be injected — if none is available the
    tool is still instantiated but ``directive_set`` returns an error response
    instead of raising, so the LLM receives clear feedback.
    """

    KIND: ClassVar[str]     = "directive_management"
    PROVIDER: ClassVar[str] = "platform"

    def __init__(
        self,
        *,
        name: str,
        config: dict[str, Any],
        store_coordinator: StoreCoordinator | None = None,
        **_: object,
    ) -> None:
        super().__init__(name=name, config=config)
        self._store = store_coordinator

    @classmethod
    def signatures(cls) -> list[ToolSignature]:
        valid_keys = [d.key for d in KnownDirective]
        return [
            ToolSignature(
                name="directive_set",
                description=(
                    "Set a behavioral directive that controls how the agent responds. "
                    f"Valid keys: {', '.join(valid_keys)}."
                ),
                method_name="set_directive",
                parameters=[
                    ToolParameter(
                        name="key",
                        type="string",
                        description=f"Directive key. One of: {', '.join(valid_keys)}.",
                    ),
                    ToolParameter(
                        name="value",
                        type="string",
                        description="Value for the directive.",
                    ),
                ],
            ),
        ]

    async def set_directive(self, *, key: str, value: str) -> str:
        if self._store is None:
            logger.warning("directive_set: no store_coordinator available")
            return json.dumps({
                "success": False,
                "error": "No session store available — directive_set requires a StoreCoordinator.",
            })

        known = KnownDirective.for_key(key)
        if known is None:
            valid_keys = [d.key for d in KnownDirective]
            logger.warning("directive_set: unknown key %r", key)
            return json.dumps({
                "success": False,
                "key": key,
                "error": f"Unknown directive key '{key}'.",
                "valid_keys": valid_keys,
            })

        self._store.set_directive(key=key, value=value, source="user")
        logger.info("directive_set | key=%s value=%s", key, value)
        return json.dumps({
            "success": True,
            "key": key,
            "value": value,
            "prompt": known.render(value),
        })
