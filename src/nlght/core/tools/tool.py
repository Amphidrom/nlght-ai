# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from nlght.ports.outbound.os_runtime import OsRuntime


@dataclass(frozen=True)
class ToolParameter:
    """Type-safe builder for a tool parameter.

    Mirrors a JSON Schema property and lowers to it losslessly (see
    ``params_to_json_schema``): ``type``/``description`` plus the JSON
    Schema constructs ``enum`` (fixed set of values) and ``items``
    (element schema for ``type="array"``). ``items`` is itself a JSON
    Schema dict, e.g. ``{"type": "integer"}`` or a nested object schema.
    """

    name: str
    type: str  # "string" | "number" | "integer" | "boolean" | "object" | "array"
    description: str = ""
    required: bool = True
    default: Any = None
    enum: list[Any] | None = None
    items: dict[str, Any] | None = None


@dataclass(frozen=True)
class ToolSignature:
    """Describes a single callable operation on a ToolBase.

    ``name`` is the identifier the LLM uses (e.g. ``"git_commit"``).
    ``method_name`` is the Python method name on the ToolBase instance.
    """

    name: str
    description: str
    method_name: str
    parameters: list[ToolParameter] = field(default_factory=list)
    terminal: bool = False
    """If True, a call to this operation ends the model turn: the
    ModelClient closes the stream after the tool call instead of waiting
    for the model's natural ``done``."""


class ToolBase:
    """Abstract base for all platform tools.

    Every subclass:
    - sets ``KIND`` as the registry key (must match the ``kind`` field in the DB)
    - overrides ``signatures()`` to declare the callable operations
    - implements the declared methods

    Minimal example::

        class MyTool(ToolBase):
            KIND = "my_tool"

            @classmethod
            def signatures(cls) -> list[ToolSignature]:
                return [
                    ToolSignature(
                        name="my_tool_greet",
                        description="Greets someone.",
                        method_name="greet",
                        parameters=[ToolParameter(name="name", type="string")],
                    )
                ]

            async def greet(self, *, name: str) -> str:
                return f"Hello, {name}!"
    """

    KIND: ClassVar[str]
    PROVIDER: ClassVar[str]

    def __init__(
        self,
        *,
        name: str,
        config: dict[str, Any],
        os_runtime: OsRuntime | None = None,
        **_: object,
    ) -> None:
        self.name = name
        self.config = config
        self.os_runtime = os_runtime

    @classmethod
    def signatures(cls) -> list[ToolSignature]:
        raise NotImplementedError
