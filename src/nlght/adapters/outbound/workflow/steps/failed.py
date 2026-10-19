# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from nlght.core.workflow.step import StepBase, StepResult, WorkflowStepContext


class FailedStep(StepBase):
    """Terminal step that signals workflow failure.

    Emits nothing because the error path is a structural terminal marker.
    Preceding steps are responsible for emitting any error signals before
    transitioning here.
    """

    TYPE = "failed"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        return StepResult(ctx=ctx, verdict=self.Verdict.FAILED)
