# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ModelStreamEvent:
    """Single event yielded by a bound model client's internal stream API."""

    kind: str
    content: str = ""
    raw: dict[str, Any] | None = None


@runtime_checkable
class ModelClient(Protocol):
    """Outbound port for LLM calls.

    A bound instance is injected into ``WorkflowStepContext.llm`` by the
    executor before each step runs.  The binding resolves provider, model,
    emitter, and stream from step config and request context — step authors
    have no knowledge of these concerns.

    Multiple steps in a workflow can call ``call()`` and all output flows
    into the same open signal stream.  The stream closes only when the step
    machine reaches a terminal verdict.

    Usage in a step::

        await ctx.llm.call(ctx.messages)
    """

    async def call(self, messages: list[dict[str, Any]], *, temperature: float | None = None) -> None: ...

    def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | str | None = None,
        *,
        temperature: float | None = None,
    ) -> AsyncIterator[ModelStreamEvent]: ...

    def append_tool_turn(
        self,
        messages: list[dict[str, Any]],
        tool_calls_raw: list[dict[str, Any]],
        results: list[str],
        assistant_text: str = "",
    ) -> list[dict[str, Any]]:
        """Append assistant-tool_calls + tool-result messages in the wire format this client expects.

        Steps that drive their own tool loop must use this instead of importing
        format-specific helpers — the client knows its own wire format.
        """
        ...
