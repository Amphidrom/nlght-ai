# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from nlght.core.model.budget import TokenBudget
from nlght.core.model.messages import CanonicalMessage, MessageLike


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

    @property
    def token_budget(self) -> TokenBudget | None:
        """This model's context budget, as the client that will make the call sees it.

        Read-only, and the point is *which* budget it is: the same object the
        safety net downstream will enforce with, derived for the model actually
        bound to this step. Anything composing a prompt has to work to the same
        number, or a prompt is built against one limit and checked against
        another — two truths that agree until the day they do not.

        `None` where the client cannot say: a provider without a known context
        window, or a test double. That is a definite answer meaning "no budget
        from here", never an invitation to invent one.
        """
        ...

    async def call(self, messages: Sequence[MessageLike], *, temperature: float | None = None) -> None: ...

    def stream(
        self,
        messages: Sequence[MessageLike],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | str | None = None,
        *,
        temperature: float | None = None,
    ) -> AsyncIterator[ModelStreamEvent]: ...

    def append_tool_turn(
        self,
        messages: Sequence[MessageLike],
        tool_calls_raw: list[dict[str, Any]],
        results: list[str],
        assistant_text: str = "",
    ) -> list[CanonicalMessage]:
        """Append typed assistant-tool_calls and genuine tool-result messages.

        Provider mapping happens only when the next request crosses the adapter.
        """
        ...
