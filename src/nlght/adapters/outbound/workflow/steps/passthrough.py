# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import logging

from nlght.core.signals.signal import Signal
from nlght.core.workflow.step import StepBase, StepResult, WorkflowStepContext

logger = logging.getLogger(__name__)


class PassthroughStep(StepBase):
    """Forwards the current conversation context unchanged to the LLM and returns the result.

    Streams tokens if ctx.streaming is True, otherwise calls the LLM blocking.

    Step config:
        verdict: str (optional) — overrides the returned verdict. Default: "DEFAULT"
    """

    TYPE = "passthrough"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        if ctx.llm is None:
            logger.error("[%s] PassthroughStep: no LLM available", ctx.correlation_id)
            return StepResult(ctx=ctx, verdict="failed")

        verdict = str(self.config.get("verdict", "DEFAULT"))

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
        # Only called from run() after its ctx.llm is None guard.
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