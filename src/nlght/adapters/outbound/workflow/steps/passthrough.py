# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Forward workflow messages to a bound model.

Input: workflow step context and optional verdict configuration.
Output: a StepResult with the selected verdict.
Side effects: calls the model and emits response signals.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from nlght.core.signals.signal import Signal
from nlght.core.workflow.step import StepBase, StepOption, StepResult, WorkflowStepContext

logger = logging.getLogger(__name__)


class PassthroughStep(StepBase):
    """Forward messages and return the configured workflow verdict."""

    TYPE = "passthrough"

    @classmethod
    def verdicts(cls, config: Mapping[str, Any] | None = None) -> tuple[str, ...]:
        """Include the verdict selected by this step's configuration.

        Args:
            config (Mapping[str, Any] | None): Persisted step configuration.

        Returns:
            tuple[str, ...]: Common verdicts and the configured verdict.
        """
        configured = str((config or {}).get("verdict", cls.Verdict.DEFAULT))
        return tuple(dict.fromkeys((*super().verdicts(config), configured)))

    @classmethod
    def options(cls) -> list[StepOption]:
        """Describe the optional verdict configuration.

        Returns:
            list[StepOption]: Verdict option with ``DEFAULT`` as its fallback.
        """
        return [StepOption("verdict", "string", "Returned workflow verdict", default=cls.Verdict.DEFAULT)]

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        """Call the bound model and return the configured verdict.

        Args:
            ctx (WorkflowStepContext): Context containing messages and model binding.

        Returns:
            StepResult: Configured verdict, or ``failed`` without a bound model.

        Side Effects:
            Calls the model and emits token signals when streaming is enabled.
        """
        if ctx.llm is None:
            logger.error("[%s] PassthroughStep: no LLM available", ctx.correlation_id)
            return StepResult(ctx=ctx, verdict=self.Verdict.FAILED)

        verdict = str(self.config.get("verdict", self.Verdict.DEFAULT))

        if ctx.stream:
            await self._run_streaming(ctx)
        else:
            await ctx.llm.call(ctx.messages)

        logger.info(
            "[%s] passthrough.done | streaming=%s",
            ctx.correlation_id, ctx.stream,
        )

        return StepResult(ctx=ctx, verdict=verdict)

    async def _run_streaming(self, ctx: WorkflowStepContext) -> None:
        """Emit token events from the model stream.

        Args:
            ctx (WorkflowStepContext): Context with a bound model and emitter.

        Side Effects:
            Streams model events and emits result signals for nonempty tokens.
        """
        # Narrow the optional client after run() checks it for static typing.
        assert ctx.llm is not None
        async for event in ctx.llm.stream(ctx.messages):
            if event.kind == "token" and event.content:
                await ctx.emitter.emit(Signal(
                    role="assistant",
                    content=event.content,
                    kind="result",
                ))
            elif event.kind == "done":
                break
