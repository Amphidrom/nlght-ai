# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any

from nlght.core.tools.action import (
    ActionSemanticsResolver,
    ExecutionCapabilities,
    JsonValue,
    ToolArgumentBinder,
)
from nlght.core.tools.tool import ToolBase, ToolParameter


class BoundToolContract:
    """Executable wrapper around a single ToolBase method.

    Implements the ``ToolContract`` port structurally (protocol matching).
    Created by the ``ToolCatalogBuilder`` per signature of a tool.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str,
        parameters: list[ToolParameter],
        instance: ToolBase,
        method_name: str,
        terminal: bool = False,
        action: ActionSemanticsResolver | None = None,
        argument_binder: ToolArgumentBinder | None = None,
        resource_address: str = "unknown/unknown",
        resource_config: Mapping[str, Any] | None = None,
        runtime_capabilities: ExecutionCapabilities | None = None,
    ) -> None:
        self._name = name
        self._description = description
        self._parameters = parameters
        self._instance = instance
        self._method_name = method_name
        self._terminal = terminal
        self._action = action
        self._argument_binder = argument_binder
        self._resource_address = resource_address
        self._resource_config = dict(resource_config or {})
        self._runtime_capabilities = runtime_capabilities or ExecutionCapabilities.unconfined()

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> list[ToolParameter]:
        return self._parameters

    @property
    def terminal(self) -> bool:
        return self._terminal

    @property
    def action(self) -> ActionSemanticsResolver | None:
        return self._action

    @property
    def argument_binder(self) -> ToolArgumentBinder | None:
        return self._argument_binder

    @property
    def resource_address(self) -> str:
        return self._resource_address

    @property
    def resource_config(self) -> Mapping[str, Any]:
        return self._resource_config

    @property
    def runtime_capabilities(self) -> ExecutionCapabilities:
        return self._runtime_capabilities

    async def _execute_bound(self, arguments: Mapping[str, JsonValue]) -> object:
        fn = getattr(self._instance, self._method_name)
        if inspect.iscoroutinefunction(fn):
            return await fn(**arguments)
        return fn(**arguments)


class CallbackToolContract:
    """Ephemeral Contract that invokes a callback instead of a ToolBase method.

    For protocol tools that should only exist for the duration of a single
    Step run (e.g. set_theory in the SurfaceTheorist). Registered in a
    ``TemporaryToolCatalog`` — never in the persistent ``AdapterToolCatalog``.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str,
        parameters: list[ToolParameter],
        callback: Callable[..., object],
        terminal: bool = False,
        action: ActionSemanticsResolver | None = None,
        argument_binder: ToolArgumentBinder | None = None,
        resource_address: str = "temporary/callback",
        resource_config: Mapping[str, Any] | None = None,
        runtime_capabilities: ExecutionCapabilities | None = None,
    ) -> None:
        self._name = name
        self._description = description
        self._parameters = parameters
        self._callback = callback
        self._terminal = terminal
        self._action = action
        self._argument_binder = argument_binder
        self._resource_address = resource_address
        self._resource_config = dict(resource_config or {})
        self._runtime_capabilities = runtime_capabilities or ExecutionCapabilities()

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> list[ToolParameter]:
        return self._parameters

    @property
    def terminal(self) -> bool:
        return self._terminal

    @property
    def action(self) -> ActionSemanticsResolver | None:
        return self._action

    @property
    def argument_binder(self) -> ToolArgumentBinder | None:
        return self._argument_binder

    @property
    def resource_address(self) -> str:
        return self._resource_address

    @property
    def resource_config(self) -> Mapping[str, Any]:
        return self._resource_config

    @property
    def runtime_capabilities(self) -> ExecutionCapabilities:
        return self._runtime_capabilities

    async def _execute_bound(self, arguments: Mapping[str, JsonValue]) -> object:
        result = self._callback(**arguments)
        if inspect.isawaitable(result):
            return await result
        return result
