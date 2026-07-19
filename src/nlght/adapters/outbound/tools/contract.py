# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import inspect
from collections.abc import Callable

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
    ) -> None:
        self._name = name
        self._description = description
        self._parameters = parameters
        self._instance = instance
        self._method_name = method_name
        self._terminal = terminal

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

    async def execute(self, **kwargs: object) -> object:
        fn = getattr(self._instance, self._method_name)
        if inspect.iscoroutinefunction(fn):
            return await fn(**kwargs)
        return fn(**kwargs)


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
    ) -> None:
        self._name = name
        self._description = description
        self._parameters = parameters
        self._callback = callback
        self._terminal = terminal

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

    async def execute(self, **kwargs: object) -> object:
        result = self._callback(**kwargs)
        if inspect.isawaitable(result):
            return await result
        return result
