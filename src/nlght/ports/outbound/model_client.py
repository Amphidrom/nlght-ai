# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from nlght.core.model.budget import TokenBudget
from nlght.core.model.messages import CanonicalMessage, MessageLike

if TYPE_CHECKING:
    from nlght.core.tools.tool import ToolOutput


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
    emitter, and stream from step config and request context. Step authors
    have no knowledge of these concerns.

    Multiple steps in a workflow can call ``call()`` and all output flows
    into the same open signal stream.  The stream closes only when the step
    machine reaches a terminal verdict.

    Usage in a step::

        await ctx.llm.call(ctx.messages)
    """

    @property
    def token_budget(self) -> TokenBudget | None:
        """Expose the budget enforced by this bound model client.

        Returns:
            TokenBudget | None: TokenBudget or None when the provider cannot determine one.
        """
        ...

    async def call(self, messages: Sequence[MessageLike]) -> None: ...

    def stream(
        self,
        messages: Sequence[MessageLike],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | str | None = None,
    ) -> AsyncIterator[ModelStreamEvent]: ...

    def append_tool_turn(
        self,
        messages: Sequence[MessageLike],
        tool_calls_raw: list[dict[str, Any]],
        results: Sequence[str | ToolOutput],
        assistant_text: str = "",
    ) -> list[CanonicalMessage]:
        """Append typed tool calls and results without provider wiring.

        Args:
            messages (Sequence[MessageLike]): Existing conversation.
            tool_calls_raw (list[dict[str, Any]]): Raw or typed model tool calls.
            results (Sequence[str | ToolOutput]): Tool results paired with those calls.
            assistant_text (str): Assistant text accompanying the calls.

        Returns:
            list[CanonicalMessage]: Canonical conversation with an assistant and tool results.
        """
        ...
